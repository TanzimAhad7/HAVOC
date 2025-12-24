"""
Module 9 — Adaptive Defense Policy (HAVOC++)
============================================

This module implements an adaptive latent-space defense that actively
counteracts attacker progress by adjusting the *strength* of latent
interventions over time.

Key idea:
---------
Treat the defender as a feedback controller operating on a scalar
risk signal measured from internal activations.

The controller:
- increases intervention strength when risk worsens,
- decreases strength when risk improves,
- ignores small noisy fluctuations (deadband),
- enforces hard bounds on strength (saturation).

This is a robust gain-scheduling controller suitable for unknown,
non-linear, adversarial dynamics.
"""

from __future__ import annotations
from typing import Optional
import numpy as np


class AdaptiveDefensePolicy:
    """
    Adaptive latent defense policy for HAVOC++.

    The policy monitors defended risk across rounds and adaptively
    adjusts the intervention magnitude λ (strength).

    This is NOT gradient descent.
    This is NOT model-based control.

    It is a sign-based, multiplicative gain controller designed
    for adversarial, non-stationary attackers.
    """

    def __init__(
        self,
        v_direct: np.ndarray,
        v_jb: np.ndarray,
        *,
        mu_HJ: Optional[np.ndarray] = None,
        W: Optional[np.ndarray] = None,
        risk_threshold: float = 0.4,
        base_strength: float = 0.05,
        adapt_up: float = 1.1,
        adapt_down: float = 0.9,
        # ---- NEW (control-theory hygiene) ----
        strength_min: float = 0.01,
        strength_max: float = 1.0,
        deadband: float = 1e-3,
        ema_beta: float = 0.8,
    ) -> None:
        # Normalize concept vectors once
        self.v_direct = v_direct / (np.linalg.norm(v_direct) + 1e-9)
        self.v_jb = v_jb / (np.linalg.norm(v_jb) + 1e-9)

        self.mu_HJ = mu_HJ
        self.W = W

        # Risk policy parameters
        self.risk_threshold = float(risk_threshold)

        # Adaptive gain parameters
        self.base_strength = float(base_strength)
        self.adapt_up = float(adapt_up)
        self.adapt_down = float(adapt_down)

        # Safety bounds on gain (CRITICAL for stability)
        self.strength_min = float(strength_min)
        self.strength_max = float(strength_max)

        # Ignore tiny risk fluctuations
        self.deadband = float(deadband)

        # Exponential moving average for noise suppression
        self.ema_beta = float(ema_beta)

        self.reset()

    # --------------------------------------------------
    def reset(self) -> None:
        """Reset controller state at the start of a new game."""
        self.strength = self.base_strength
        self.last_risk: Optional[float] = None
        self.smoothed_risk: Optional[float] = None

    # --------------------------------------------------
    def compute_risk(self, fP: np.ndarray) -> float:
        """
        Compute latent risk as alignment with harmful directions.

        Negative alignment (orthogonal or safe directions) is ignored.
        """
        fP_u = fP / (np.linalg.norm(fP) + 1e-9)

        r_direct = float(np.dot(fP_u, self.v_direct))
        r_jb = float(np.dot(fP_u, self.v_jb))

        # Only positive alignment contributes to risk
        return max(0.0, r_direct, r_jb)

    # --------------------------------------------------
    def get_intervention(self, fP: np.ndarray) -> np.ndarray:
        """
        Compute the latent correction vector.

        We move *away* from both harmful directions, weighted by
        their relative activation.
        """
        fP_u = fP / (np.linalg.norm(fP) + 1e-9)

        s_d = max(0.0, float(np.dot(fP_u, self.v_direct)))
        s_j = max(0.0, float(np.dot(fP_u, self.v_jb)))

        total = s_d + s_j + 1e-9
        w_d = s_d / total
        w_j = s_j / total

        delta = -(w_d * self.v_direct + w_j * self.v_jb)
        delta = delta / (np.linalg.norm(delta) + 1e-9)

        return self.strength * delta

    # --------------------------------------------------
    def apply_intervention(self, fP: np.ndarray) -> np.ndarray:
        """
        Apply intervention only when risk exceeds threshold.
        """
        risk_now = self.compute_risk(fP)

        if risk_now <= self.risk_threshold:
            vec = fP
        else:
            vec = fP + self.get_intervention(fP)

        # Optional projection into harmful subspace
        if self.W is not None and self.mu_HJ is not None:
            diff = vec - self.mu_HJ
            vec = self.mu_HJ + self.W.T @ (self.W @ diff)
        elif self.W is not None:
            vec = self.W.T @ (self.W @ vec)

        return vec / (np.linalg.norm(vec) + 1e-9)

    # --------------------------------------------------
    def update_policy(self, risk_now: float) -> None:
        """
        ADAPTATION RULE (CORE LOGIC)

        This function updates the intervention strength λ.

        Line-by-line explanation below.
        """

        # ---- Step 1: Smooth the risk signal (low-pass filter) ----
        if self.smoothed_risk is None:
            self.smoothed_risk = risk_now
        else:
            self.smoothed_risk = (
                self.ema_beta * self.smoothed_risk
                + (1.0 - self.ema_beta) * risk_now
            )

        # ---- Step 2: If no previous risk, initialize and exit ----
        if self.last_risk is None:
            self.last_risk = self.smoothed_risk
            return

        # ---- Step 3: Compute risk change ----
        delta = self.smoothed_risk - self.last_risk

        # ---- Step 4: Deadband (ignore noise) ----
        if abs(delta) <= self.deadband:
            # Do nothing: keep current strength
            pass

        # ---- Step 5: Gain scheduling ----
        elif delta > 0:
            # Risk increased → attacker is winning → strengthen defense
            self.strength *= self.adapt_up
        else:
            # Risk decreased → defense is effective → relax slightly
            self.strength *= self.adapt_down

        # ---- Step 6: Saturation (CRITICAL for stability) ----
        self.strength = float(
            np.clip(self.strength, self.strength_min, self.strength_max)
        )

        # ---- Step 7: Store for next round ----
        self.last_risk = self.smoothed_risk
