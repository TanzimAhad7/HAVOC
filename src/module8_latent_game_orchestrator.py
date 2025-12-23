"""
Module 8 — Latent Game Orchestrator
===================================

This module orchestrates the closed‑loop attacker–defender interaction
for HAVOC++.  It brings together the HAVOC attacker (Module 7), the
adaptive defence policy (Module 9) and the stability controller
(Module 10) to realise the repeated game described in the HAVOC++
design.  Each game proceeds over multiple rounds until stability is
reached or a maximum number of rounds is exceeded.

For each intent prompt in the input list, the orchestrator performs
the following steps:

1. Extract the intent activation once using Module 1.  This provides
   a fixed reference for scoring (the attacker caches this as well).
2. Initialise a ``HAVOC_Controller`` with the provided concept
   directions, harmful subspace and attacker hyperparameters.
3. Iteratively run the attacker and defender:
   a. The attacker searches activation space using fuzzing and
      steering to produce a candidate prompt and associated score.
   b. The orchestrator computes the candidate activation and the
      defence policy measures its risk and applies a corrective
      intervention in latent space.
   c. The defence policy adapts its intervention strength based on
      whether the attacker is making progress.
   d. The stability controller checks the recent risk values to
      determine whether the system has converged.  If stable or the
      maximum number of rounds has been reached, the game halts.
4. The orchestrator records per‑round statistics including prompts,
   scores, risk values and intervention strengths.

This module does not modify the original HAVOC attacker; instead it
wraps it in a higher‑level control loop.  Users may customise
hyperparameters such as the maximum number of attacker iterations per
round, the latent layer index, and verbose logging.  The defence
policy and stability controller instances may be shared across
multiple games or instantiated per‑game.

Example usage::

    from module2_concept_vector_construction import load_concepts
    from module4_direct_representaiton_space import load_direct_space
    from module9_adaptive_defense_policy import AdaptiveDefensePolicy
    from module10_stability_controller import StabilityController
    from module8_latent_game_orchestrator import LatentGameOrchestrator

    mu_B, mu_H, mu_J, v_direct, v_jb = load_concepts(layer=20)
    mu_HJ, W = load_direct_space()
    defence = AdaptiveDefensePolicy(v_direct, v_jb, mu_HJ=mu_HJ, W=W)
    stability = StabilityController(window=3, tolerance=1e-3, max_rounds=8)
    orchestrator = LatentGameOrchestrator(["attack prompt"], v_direct, v_jb,
                                          mu_HJ, W, defence, stability)
    results = orchestrator.run()

"""

from __future__ import annotations

import os
from typing import List, Dict, Any, Optional
import numpy as np

from module1_Activation_Extraction import extract_activation_dynamic
from module7_controller import HAVOC_Controller
from module9_adaptive_defense_policy import AdaptiveDefensePolicy
from module10_stability_controller import StabilityController


class LatentGameOrchestrator:
    """Orchestrate the attacker–defender game for HAVOC++.

    Args:
        intents: List of intent prompts for which to run the game.
        v_direct: direct concept vector.
        v_jb: jailbreak concept vector.
        mu_HJ: Mean of harmful subspace (from Module 4).
        W: Harmful subspace basis (rows as PCA components).
        defence_policy: Instance of ``AdaptiveDefensePolicy`` to use.
        stability_controller: Instance of ``StabilityController`` to use.
        max_iters: Maximum number of attacker iterations per round.
        layer: Transformer layer index used for activation extraction.
        attacker_seed: Random seed passed to the attacker for reproducibility.
        verbose: Whether to print per‑round progress information.
    """

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

    def _run_single_game(self, intent: str) -> Dict[str, Any]:
        """Run a single latent attacker–defender game for one intent.

        Returns a dictionary containing final and per‑round results.
        """
        # Compute the intent activation once
        fI = extract_activation_dynamic(intent, layer=self.layer)
        # Reset defence and stability controllers
        self.defence_policy.reset()
        self.stability_controller.reset()

        self.defence_policy.last_defended_fP = None

        # Instantiate the attacker
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
        round_trajectories: List[Dict[str, Any]] = []
        stable = False
        round_idx = 0
        # Loop until convergence
        while not stable:
            # ------------------------------
            # (A) ATTACKER MOVE (HAVOC Module 7)
            # ------------------------------
            # Run one full attacker search loop to propose a candidate prompt.
            #best_prompt, best_score, attack_traj = controller.run()
            # ------------------------------

            best_prompt, best_score, attack_traj = controller.run(
                steer_vector=self.defence_policy.last_defended_fP
                if hasattr(self.defence_policy, "last_defended_fP")
                else None,
                steer_alpha=self.defence_policy.strength,
            )


            # Extract the candidate prompt activation (Module 1 dynamic path).
            fP = extract_activation_dynamic(best_prompt, layer=self.layer)

            # ------------------------------
            # (B) DEFENDER MOVE (HAVOC++ Module 9)
            # ------------------------------
            if self.defence_policy is not None:
                risk_raw = self.defence_policy.compute_risk(fP)
                defended_fP = self.defence_policy.apply_intervention(fP)
                risk_def = self.defence_policy.compute_risk(defended_fP)
                self.defence_policy.update_policy(risk_def)
            else:
                risk_raw = None
                defended_fP = fP
                risk_def = None

            if self.stability_controller is not None and risk_def is not None:
                stable = self.stability_controller.update(risk_def)
            else:
                stable = (round_idx >= self.max_iters)

            # ------------------------------
            # (C) FEEDBACK: make it a real game
            # ------------------------------
            # Critical: feed defended state back to the attacker so that the attacker
            # adapts against the defence in subsequent rounds.
            # if hasattr(controller, "fI_cached"):
            #     controller.fI_cached = defended_fP
            # # (Some older variants may also use controller.fI; set it defensively.)
            # setattr(controller, "fI", defended_fP)

            # ------------------------------
            # (D) STABILITY CHECK (HAVOC++ Module 10)
            # ------------------------------
            stable = self.stability_controller.update(risk_def)

            # Record diagnostics for plotting/evaluation.
            round_info = {
                "round": round_idx,
                "best_prompt": best_prompt,
                "best_score": float(best_score),
                "r_raw": float(risk_raw),
                "r_def": float(risk_def),
                "defence_strength": float(self.defence_policy.strength),
                "attack_trajectory": attack_traj,
            }
            round_trajectories.append(round_info)

            if self.verbose:
                print(
                    f"[LatentGame] Round {round_idx}: best_score={float(best_score):.4f}, "
                    f"r_raw={float(risk_raw):.4f}, r_def={float(risk_def):.4f}, "
                    f"strength={float(self.defence_policy.strength):.4f}, stable={stable}"
                )

            round_idx += 1
        final_prompt = round_trajectories[-1]["best_prompt"] if round_trajectories else intent
        final_score = round_trajectories[-1]["best_score"] if round_trajectories else 0.0
        return {
            "intent": intent,
            "final_prompt": final_prompt,
            "final_score": float(final_score),
            "rounds": round_idx,
            "round_trajectories": round_trajectories,
        }

    def run(self) -> List[Dict[str, Any]]:
        """Run the latent game for each intent and collect results."""
        results: List[Dict[str, Any]] = []
        for intent in self.intents:
            if self.verbose:
                print("\n=== HAVOC++ latent game for intent ===\n" + intent[:120])
            res = self._run_single_game(intent)
            results.append(res)
        return results
