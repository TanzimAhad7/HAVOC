"""
Module 8 — Latent Game Orchestrator (HAVOC++)
============================================

Orchestrates the closed-loop attacker–defender game.

Core principle:
---------------
The attacker adapts its search strategy based on the *current defense
intervention direction* applied in latent space.

This module ensures the defense is ACTIVE (not just diagnostic).
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import numpy as np
from tqdm import trange


from module1_Activation_Extraction import extract_activation_dynamic
from module7_controller import HAVOC_Controller, SafeRewriterLLM
from module9_adaptive_defense_policy import AdaptiveDefensePolicy
from module10_stability_controller import StabilityController
from model import HAVOCModelLoader
import torch
import torch.nn.functional as F

MODEL_NAME: str = "meta-llama/Meta-Llama-3-8B-Instruct",
DEVICE: str = "cuda"
MAX_NEW_TOKENS = 128  # example

class LatentGameOrchestrator:
    """Run the HAVOC++ attacker–defender game."""

    def __init__(
        self,
        intents: List[str],
        v_direct: np.ndarray,
        v_jb: np.ndarray,
        mu_HJ: Optional[np.ndarray],
        W: Optional[np.ndarray],
        defence_policy: AdaptiveDefensePolicy,
        stability_controller: StabilityController,
        *,
        max_iters: int = 30,
        layer: int = 20,
        attacker_seed: int = 0,
        verbose: bool = False,
    ) -> None:
        self.intents = intents
        self.v_direct = v_direct
        self.v_jb = v_jb
        self.mu_HJ = mu_HJ
        self.W = W
        self.defence_policy = defence_policy
        self.stability_controller = stability_controller
        self.max_iters = int(max_iters)
        self.layer = int(layer)
        self.seed = int(attacker_seed)
        self.verbose = bool(verbose)
        self.combined_risk = False  # Whether to combine risks
        self.model, self.tokenizer = HAVOCModelLoader(MODEL_NAME, DEVICE).load()  # To be set if needed

    # ======================================================
    # 2. SafeDecoding (logit-level intervention)
    # ======================================================
    def safe_decoding_logits( self,
        fP: np.ndarray,
        lm_head,
    ):
        """
        Apply SafeDecoding to logits.
        """
        device = next(lm_head.parameters()).device

        fP_t = torch.tensor(fP, device=device, dtype=torch.float32)

        # Alignment with harmful direction
        unsafe_alignment = max(0.0, float(np.dot(fP, self.v_direct)))

        # Raw logits from LM head
        logits = lm_head(fP_t)  # (vocab_size,)

        # SafeDecoding shift
        logits_safe = logits - self.defence_policy.strength * unsafe_alignment

        return logits_safe, unsafe_alignment

    # ======================================================
    # 3. Generate SAFE RESPONSE (autoregressive rollout)
    #    (for visualization / demo only)
    # ======================================================
    def generate_safe_response(self,
            prompt: str,
            v_direct: np.ndarray,
            strength: float,
            layer: int = 20,
            max_new_tokens: int = 64,
            device: str = "cuda",
        ):
        # model.eval()

        # Encode prompt
        context = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        generated_ids = []

        for _ in range(max_new_tokens):

            # Decode current context to text
            current_text = self.tokenizer.decode(context[0], skip_special_tokens=True)

            # Extract prompt-level activation
            fP = extract_activation_dynamic(
                current_text,
                layer=layer
            )

            # SafeDecoding
            logits_safe, _ = self.safe_decoding_logits(
                fP=fP,
                lm_head=self.model.lm_head,
                v_direct=v_direct,
                strength=strength,
            )

            # Sample next token
            probs = F.softmax(logits_safe, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append
            generated_ids.append(next_token.item())
            context = torch.cat([context, next_token.unsqueeze(0)], dim=-1)

            # Early stop on EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        # Decode safe response
        safe_response = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        return safe_response
    # ------------------------------------------------------------
    def _run_single_game(self, intent: str) -> Dict[str, Any]:
        """Run one attacker–defender game for a single intent."""

        # (1) Extract intent activation ONCE (fixed reference)
        fI = extract_activation_dynamic(intent, layer=self.layer)

        # Reset controllers (memory in Module 9 is preserved)
        self.defence_policy.reset()
        self.stability_controller.reset()

        # (2) Initialize attacker
        controller = HAVOC_Controller(
            intent=intent,
            fI=fI,
            v_direct=self.v_direct,
            v_jb=self.v_jb,
            mu_HJ=self.mu_HJ,
            W=self.W,
            max_iters=self.max_iters,
            layer=self.layer,
            seed=self.seed,
        )

        round_logs: List[Dict[str, Any]] = []
        stable = False
        round_idx = 0

        # --------------------------------------------------------
        # Main attacker–defender loop
        # --------------------------------------------------------
        while not stable:
            print(f"\n  [ROUND {round_idx+1}/{self.max_iters}] starting")
            # ==============================
            # (A) ATTACKER MOVE (Module 7)
            # ==============================
            # Attacker is conditioned on the CURRENT defense direction
            best_prompt, best_score, attack_traj = controller.run(
                steer_vector=self.defence_policy.last_delta_dir,
                steer_alpha=self.defence_policy.strength,
            )

            # Extract candidate activation
            fP = extract_activation_dynamic(best_prompt, layer=self.layer)

            # ==============================
            # (B) DEFENDER MOVE (Module 9)
            # ==============================
            risk_raw = self.defence_policy.compute_risk(fP)
            defended_fP = self.defence_policy.apply_intervention(fP)
            risk_def = self.defence_policy.compute_risk(defended_fP)

            if self.combined_risk:
                # Combine havoc risk and safeDecoding alignment-based risk
                unsafe_alignment = max(0.0, np.dot(fP, self.v_direct))
                combined_risk = max(
                    risk_def,
                    self.defence_policy.strength * unsafe_alignment
                )
                self.defence_policy.update_policy(combined_risk)
            else:
                # Update adaptive gain λ
                self.defence_policy.update_policy(risk_def)

            ## Apply SafeDecoding adjustment to logits
            safe_text = self.generate_safe_response(
                prompt=best_prompt,
                v_direct=self.v_direct,
                strength=self.defence_policy.strength,
                layer=self.layer,
                max_new_tokens=64,
                device=DEVICE,
            )
            print("SAFE RESPONSE:\n", safe_text)

            print(
                f"  [ROUND {round_idx+1}] "
                f"risk_raw={risk_raw:.4f} "
                f"risk_def={risk_def:.4f} "
                f"lambda={self.defence_policy.strength:.4f}"
            )

            # ==============================
            # (C) STABILITY CHECK (Module 10)
            # ==============================
            stable = self.stability_controller.update(
                risk_raw=risk_raw,
                risk_def=risk_def,
            )

            if stable:
                info = self.stability_controller.get_convergence_info()
                reason = info.get("reason", "unknown")
                print(
                    f"  [ROUND {round_idx+1}] CONVERGED ({reason}) — stopping game"
                )

            # ==============================
            # (D) LOGGING
            # ==============================
            round_logs.append({
                "round": round_idx,
                "best_prompt": best_prompt,
                "best_score": float(best_score),
                "r_raw": float(risk_raw),
                "r_def": float(risk_def),
                "defence_strength": float(self.defence_policy.strength),
                "attack_trajectory": attack_traj,
            })

            if self.verbose:
                print(
                    f"[Round {round_idx}] "
                    f"score={best_score:.4f} | "
                    f"r_raw={risk_raw:.4f} | "
                    f"r_def={risk_def:.4f} | "
                    f"λ={self.defence_policy.strength:.4f} | "
                    f"stable={stable}"
                )

            round_idx += 1

        # Final summary
        final_prompt = round_logs[-1]["best_prompt"] if round_logs else intent
        final_score = round_logs[-1]["best_score"] if round_logs else 0.0

        return {
            "intent": intent,
            "final_prompt": final_prompt,
            "final_score": float(final_score),
            "rounds": round_idx,
            "round_trajectories": round_logs,
        }

    # ------------------------------------------------------------
    def run(self) -> List[Dict[str, Any]]:
        """Run the latent game for all intents."""
        results = []
        for intent in self.intents:
            if self.verbose:
                print("\n=== HAVOC++ GAME START ===")
                print(intent[:120])
            results.append(self._run_single_game(intent))
        return results
