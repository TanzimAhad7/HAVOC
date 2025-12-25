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

from module1_Activation_Extraction import extract_activation_dynamic
from module7_controller import HAVOC_Controller
from module9_adaptive_defense_policy import AdaptiveDefensePolicy
from module10_stability_controller import StabilityController


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
        max_iters: int = 15,
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

            # Update adaptive gain λ
            self.defence_policy.update_policy(risk_def)

            # ==============================
            # (C) STABILITY CHECK (Module 10)
            # ==============================
            stable = self.stability_controller.update(risk_def)

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
