"""
Module 8 — Latent Game Orchestrator (HAVOC++)
============================================

Orchestrates the closed-loop attacker–defender game.

IMPORTANT SEMANTICS (FINAL):
----------------------------
• Exactly ONE attacker prompt is selected per round.
• Defense is applied ONLY to that prompt’s activation.
• The defended prompt is decoded from the ACTUAL defended activation
  (interpretability only, not fed back into the attacker).
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import numpy as np
from tqdm import trange

from module1_Activation_Extraction import extract_activation_dynamic, PARENT_PATH
from module7_controller import HAVOC_Controller, SafeRewriterLLM
from module9_adaptive_defense_policy import AdaptiveDefensePolicy
from module10_stability_controller import StabilityController
from model import HAVOCModelLoader
import torch
import torch.nn.functional as F
# For defended activation decoding
from module6_Steering import steer_hidden_state
from module7_controller import STEER_TEMPERATURE, STEER_TOP_P
from safe_decoding import SafeDecodingHAVOC, SafeDecodingConfig

MODEL_NAME: str = "meta-llama/Meta-Llama-3-8B-Instruct"
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

        habocLoader = HAVOCModelLoader(model_name=MODEL_NAME)
        self.tokenizer = habocLoader.load_tokenizer()
        self.model = habocLoader.load_model()

    # --------------------------------------------------
    @staticmethod
    def _l2_normalize(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
        return x / (np.linalg.norm(x) + eps)
    
    def steer_latent_safe(self, fP: np.ndarray, alpha: float, beta: float):

        fP_u = self._l2_normalize(fP)

        if self.W is not None and self.mu_HJ is not None:
            fP_s = self._l2_normalize(self.W @ (fP_u - self.mu_HJ))
            vD_s = self._l2_normalize(self.W @ self.v_direct)
            vJ_s = self._l2_normalize(self.W @ self.v_jb)
        elif self.W is not None:
            fP_s = self._l2_normalize(self.W @ fP_u)
            vD_s = self._l2_normalize(self.W @ self.v_direct)
            vJ_s = self._l2_normalize(self.W @ self.v_jb)
        else:
            fP_s = fP_u
            vD_s = self._l2_normalize(self.v_direct)
            vJ_s = self._l2_normalize(self.v_jb)

        r_harm = float(np.clip(np.dot(fP_s, vD_s), 0.0, 1.0))
        r_jb   = float(np.clip(np.dot(fP_s, vJ_s), 0.0, 1.0))

        fP_safe = (
            fP_s
            - alpha * r_jb * vJ_s
            - beta * r_harm * vD_s
        )

        fP_safe = self._l2_normalize(fP_safe)

        # 🔴 back-project to model space if needed
        if self.W is not None:
            fP_safe = np.linalg.pinv(self.W) @ fP_safe

        return fP_safe

    def make_steering_hook(self, fP_safe: torch.Tensor):

        def hook(module, inputs, output):
            if not isinstance(output, torch.Tensor):
                return output

            fP_safe_local = fP_safe.to(output.device).to(output.dtype)

            blend = 0.5
            output[:, -1, :] = (
                (1 - blend) * output[:, -1, :] + blend * fP_safe_local
            )

            return output

        return hook
    

    @torch.no_grad()
    def generate_with_latent_steering(self,
        prompt: str,
        fP_safe: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        """
        Generate text by injecting fP_safe into the model.
        """

        # --------------------
        # Tokenize
        # --------------------
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)

        # --------------------
        # Register hook
        # --------------------
        hook_handle = self.model.model.layers[self.layer].register_forward_hook(
            self.make_steering_hook(fP_safe)
        )

        # --------------------
        # Generate
        # --------------------
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # --------------------
        # Remove hook (CRITICAL)
        # --------------------
        hook_handle.remove()

        # --------------------
        # Decode
        # --------------------
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def analyze_safe_response(self, prompt: str, response: str) -> Dict[str, float]:
        """
        Extract non-invertible meta-signals from the safe response.
        """
        is_refusal = any(
            k in response.lower()
            for k in ["can't help", "not able", "cannot assist", "unable to"]
        )

        length = len(response.split())
        length_bucket = (
            0 if length < 20 else
            1 if length < 60 else
            2
        )

        # Optional: embedding distance (coarse)
        fP_prompt = extract_activation_dynamic(prompt, layer=self.layer)
        fP_resp   = extract_activation_dynamic(response, layer=self.layer)
        semantic_shift = float(1 - np.dot(fP_prompt, fP_resp))

        return {
            "is_refusal": float(is_refusal),
            "length_bucket": length_bucket,
            "semantic_shift": semantic_shift,
        }


    # ======================================================
    # 2. SafeDecoding (logit-level intervention)
    # ======================================================
    def safe_decoding_logits( self,
        fP: np.ndarray,
        lm_head,
        risk: float,
        isHavoc: bool = True,
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

        if not isHavoc:
            # SafeDecoding shift
            logits_safe = logits - max(0, self.defence_policy.strength * unsafe_alignment)
        else:
            # Havoc SafeDecoding adjustment
            logits_safe = logits - max(0, risk)

        return logits_safe, unsafe_alignment

    # ======================================================
    # 3. Generate SAFE RESPONSE (autoregressive rollout)
    #    (for visualization / demo only)
    # ======================================================
    def generate_safe_response(self,
            prompt: str,
            v_direct: np.ndarray,
            strength: float,
            risk: float,
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
                risk=risk,
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

        # (1) Fixed reference activation for intent
        fI = extract_activation_dynamic(intent, layer=self.layer)

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
            # (A) ATTACKER MOVE
            # ==============================
            best_prompt, best_score, attack_traj = controller.run(
                steer_vector=self.defence_policy.last_delta_dir,
                steer_alpha=self.defence_policy.strength,
            )

            # Activation of the ACTUAL attacker move
            fP = extract_activation_dynamic(best_prompt, layer=self.layer)

            # ==============================
            # (B) DEFENDER MOVE
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
            # safe_response1 = self.generate_safe_response(
            #     prompt=best_prompt,
            #     v_direct=self.v_direct,
            #     strength=self.defence_policy.strength,
            #     risk = risk_def,
            #     layer=self.layer,
            #     max_new_tokens=64,
            #     device=DEVICE,
            # )
            # print("SAFE RESPONSE1:\n", safe_response1)

            # cfg = SafeDecodingConfig(alpha=1.2, first_m=8, top_k=20, do_sample=False)
            # sd = SafeDecodingHAVOC(self.model, self.tokenizer, cfg=cfg, verbose=True)

            # inputs = self.tokenizer(best_prompt, return_tensors="pt")
            # safe_response1, n = sd.generate(inputs, risk=risk_def, r_harm=r_harm, r_jb=r_jb)

            # print("SAFE RESPONSE1:\n", safe_response1)

            # ------------------------------
            # Strict-mode steering escalation
            # ------------------------------
            base_alpha = 0.8
            base_beta  = 0.6

            alpha = base_alpha * (1.3 if self.defence_policy.strict_mode else 1.0)
            beta  = base_beta  * (1.3 if self.defence_policy.strict_mode else 1.0)

            fP_safe = self.steer_latent_safe(
                fP=fP,
                alpha=alpha,
                beta=beta,
            )

            # safe_response2 = self.generate_with_latent_steering(best_prompt, fP_safe=torch.tensor(fP_safe, device=DEVICE, dtype=torch.float32),
            #     max_new_tokens=128,
            #     temperature=STEER_TEMPERATURE)

            # ------------------------------
            # Response bandwidth clamp
            # ------------------------------
            effective_max_tokens = int(
                MAX_NEW_TOKENS * self.defence_policy.response_bandwidth
            )
            effective_max_tokens = max(16, effective_max_tokens)  # safety floor

            safe_response2 = self.generate_with_latent_steering(
                best_prompt,
                fP_safe=torch.tensor(fP_safe, device=DEVICE, dtype=torch.float32),
                max_new_tokens=effective_max_tokens,
                temperature=STEER_TEMPERATURE
            )
            response = safe_response2[len(best_prompt):]
            print("BEST PROMPT:\n", best_prompt)
            print("\n-------------------------------------------------\n")
            print("SAFE RESPONSE2:\n", response)
            print("\n-------------------------------------------------\n")

            response_meta = self.analyze_safe_response(best_prompt, response)
            print("SAFE RESPONSE META:", response_meta)

            # Detect attacker learning signals
            if response_meta["semantic_shift"] < 0.2:
                # Response too close to prompt → reduce informative leakage
                self.defence_policy.enable_strict_mode()

            if response_meta["length_bucket"] > 1:
                # Response too verbose → clamp future decoding
                self.defence_policy.reduce_response_bandwidth()

            if response_meta["is_refusal"] == 0:
                # Partial compliance detected → escalate faster
                self.defence_policy.increase_momentum()

            print(
                f"  [ROUND {round_idx+1}] "
                f"risk_raw={risk_raw:.4f} "
                f"risk_def={risk_def:.4f} "
                f"lambda={self.defence_policy.strength:.4f}"
            )

            # ==============================
            # (B.1) DECODE DEFENDED ACTIVATION (TRUE)
            # ==============================
            defended_prompt_text = ""
            try:
                if best_prompt and defended_fP is not None:
                    rewriter = controller.rewriter

                    # IMPORTANT:
                    # No attacker bias here — we want to observe the defense effect
                    rewriter.set_concept_bias(None)

                    # Decode using the TRUE defended delta
                    delta = defended_fP - fP

                    with steer_hidden_state(
                        model=rewriter.model,
                        layer_idx=self.layer,
                        v_comp=delta,
                        alpha=1.0,
                    ):
                        decodes = rewriter.generate(
                            seed_text=best_prompt,
                            n=1,
                            temperature=STEER_TEMPERATURE,
                            top_p=STEER_TOP_P,
                        )

                    if decodes:
                        defended_prompt_text = decodes[0]
            except Exception:
                defended_prompt_text = ""

            # ==============================
            # (B.2) LOG DEFENSE METADATA (TRUE)
            # ==============================
            defence_meta = {
                "source": self.defence_policy.last_source,   # memory | online
                "lambda_used": float(self.defence_policy.strength),
            }

            # ==============================
            # (C) STABILITY CHECK
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
            # (D) LOGGING (ONE GAME MOVE)
            # ==============================
            round_logs.append({
                "round": round_idx,
                "best_prompt": best_prompt,
                "best_score": float(best_score),
                "r_raw": float(risk_raw),
                "r_def": float(risk_def),
                "defence_strength": float(self.defence_policy.strength),
                "attack_trajectory": attack_traj,
                "defended_prompt": defended_prompt_text,
                "defence_meta": defence_meta,
            })

            # ==============================
            # (E) UPDATE λ FOR NEXT ROUND
            # ==============================
            self.defence_policy.update_policy(risk_def)

            round_idx += 1

        # Terminal outcome
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
        results = []
        for intent in self.intents:
            if self.verbose:
                print("\n=== HAVOC++ GAME START ===")
                print(intent[:120])
            results.append(self._run_single_game(intent))
        return results
