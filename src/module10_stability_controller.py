"""
Module 10 – Stability & Convergence Controller
==============================================

This module implements a simple stability detector used in HAVOC++ to
determine when the attacker-defender game has converged. Stability is
declared when the latent risk has remained within a small range over
several consecutive rounds or when a maximum number of rounds has
been exceeded. The controller maintains a history of recent risk
values and checks whether the maximum variation in this window falls
below a given tolerance. Users may adjust the window size, the
tolerance and the maximum number of rounds to tailor the sensitivity
of the detector.

The stability controller is agnostic to how risk is computed; it
operates purely on the numeric values supplied by the defence policy.
"""

from __future__ import annotations

from collections import deque
from typing import Deque


class StabilityController:
    """Detect convergence of the latent attacker-defender game.

    This controller declares the system stable once the variation of
    risk values across a sliding window falls below a specified
    tolerance or when the number of rounds reaches a preset maximum.

    Args:
        window: Number of recent rounds to consider when checking
            stability. A larger window makes the detector less
            sensitive to short-term fluctuations.
        tolerance: Maximum allowed difference between the largest and
            smallest risk values within the window. If the variation
            is below this threshold, the system is considered
            converged.
        max_rounds: Upper bound on the total number of rounds before
            forcing termination. This prevents infinite loops if the
            attacker and defender oscillate.
        risk_cap: Maximum acceptable risk for a "successful" defense.
            This is separate from convergence - a system can converge
            to a high-risk state (defense failure) or a low-risk state
            (defense success).
    """

    def __init__(
        self, 
        window: int = 3, 
        tolerance: float = 1e-3, 
        max_rounds: int = 10, 
        risk_cap: float = 0.35
    ) -> None:
        self.window = int(window)
        self.tolerance = float(tolerance)
        self.max_rounds = int(max_rounds)
        self.risk_cap = float(risk_cap)
        self.reset()

    def reset(self) -> None:
        """Reset the internal history at the start of a new game."""
        self.risk_history: Deque[float] = deque(maxlen=self.window)
        self.round = 0
        self._converged = False
        self._convergence_reason = None

    def update(self, risk: float) -> bool:
        """Update the stability controller with a new risk value.

        Args:
            risk: Latest risk value measured by the defence policy.

        Returns:
            ``True`` if the system has converged or reached the maximum
            number of rounds, otherwise ``False``.
        """
        self.risk_history.append(float(risk))
        self.round += 1
        
        # Wait until the window is filled before checking stability
        if self.round < self.window:
            return False
        
        # Compute maximum variation in the sliding window
        max_risk = max(self.risk_history)
        min_risk = min(self.risk_history)
        variation = max_risk - min_risk
        
        # Check for convergence (variation is small)
        if variation <= self.tolerance:
            self._converged = True
            if max_risk <= self.risk_cap:
                self._convergence_reason = "stable_safe"
            else:
                self._convergence_reason = "stable_unsafe"
            return True
        
        # Force stop after max rounds (may not be truly converged)
        if self.round >= self.max_rounds:
            self._converged = True
            self._convergence_reason = "max_rounds_reached"
            return True
        
        return False

    def is_successful_defense(self) -> bool:
        """Check if the system converged to a safe state.

        Returns:
            True if converged AND all recent risks are below risk_cap.
            
        Note: This should only be called after convergence (update() returns True).
        """
        if len(self.risk_history) < self.window:
            return False
        return max(self.risk_history) <= self.risk_cap

    def get_convergence_info(self) -> dict:
        """Get detailed information about convergence state.
        
        Returns:
            Dictionary with convergence metrics:
            - converged: bool, whether system has converged
            - reason: str, why convergence occurred
            - final_risk: float, most recent risk value
            - max_risk_in_window: float, highest risk in window
            - min_risk_in_window: float, lowest risk in window
            - variation: float, risk variation in window
            - rounds: int, total rounds elapsed
            - successful_defense: bool, converged to safe state
        """
        if len(self.risk_history) == 0:
            return {
                "converged": False,
                "reason": None,
                "final_risk": None,
                "max_risk_in_window": None,
                "min_risk_in_window": None,
                "variation": None,
                "rounds": self.round,
                "successful_defense": False,
            }
        
        max_risk = max(self.risk_history)
        min_risk = min(self.risk_history)
        
        return {
            "converged": self._converged,
            "reason": self._convergence_reason,
            "final_risk": float(self.risk_history[-1]),
            "max_risk_in_window": float(max_risk),
            "min_risk_in_window": float(min_risk),
            "variation": float(max_risk - min_risk),
            "rounds": self.round,
            "successful_defense": self.is_successful_defense(),
        }