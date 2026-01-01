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

from module1_Activation_Extraction import extract_activation_dynamic
from module7_controller import HAVOC_Controller
from module9_adaptive_defense_policy import AdaptiveDefensePolicy
from module10_stability_controller import StabilityController

# For defended activation decoding
from module6_Steering import steer_hidden_state
from module7_controller import STEER_TEMPERATURE, STEER_TOP_P


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
