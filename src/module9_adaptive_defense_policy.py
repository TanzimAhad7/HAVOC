"""
Module 9 — Adaptive + Learned Defense Policy (HAVOC++)
======================================================

This module implements a HYBRID latent defense:

(1) Online adaptive controller (feedback control)
(2) Learned memory (experience replay)

The learned memory enables rapid blocking of previously
observed jailbreak trajectories while preserving real-time
adaptation for novel attacks.
"""

from __future__ import annotations
from typing import Optional, List, Tuple
import numpy as np


class AdaptiveDefensePolicy:
    """
    Adaptive + learned latent defense policy for HAVOC++.

    The defender operates entirely in latent space and NEVER
    modifies model weights.

    Two layers of defense:
    --------------------------------
    1. Learned memory:
       - Fast reaction to known attack directions
       - No online adaptation required

    2. Online controller:
       - Adjusts intervention strength λ
       - Handles unseen or evolving attacks
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
        strength_min: float = 0.01,
        strength_max: float = 1.0,
        deadband: float = 1e-3,
        ema_beta: float = 0.8,
        memory_capacity: int = 512,
        memory_similarity: float = 0.95,
    ) -> None:

        # -----------------------------
        # Normalize concept vectors once
        # -----------------------------
        self.v_direct = v_direct / (np.linalg.norm(v_direct) + 1e-9)
        self.v_jb = v_jb / (np.linalg.norm(v_jb) + 1e-9)

        # Optional harmful subspace geometry
        self.mu_HJ = mu_HJ
        self.W = W

        # -----------------------------
        # Online controller parameters
        # -----------------------------
        self.risk_threshold = float(risk_threshold)
        self.base_strength = float(base_strength)
        self.adapt_up = float(adapt_up)
        self.adapt_down = float(adapt_down)

        # Saturation bounds (critical for stability)
        self.strength_min = float(strength_min)
        self.strength_max = float(strength_max)

        # Noise handling
        self.deadband = float(deadband)
        self.ema_beta = float(ema_beta)

        # -----------------------------
        # Learned memory parameters
        # -----------------------------
        # Stores (latent_direction, correction_vector)
        self.memory_capacity = int(memory_capacity)
        self.memory_similarity = float(memory_similarity)
        self.memory: List[Tuple[np.ndarray, np.ndarray]] = []

        self.reset()

    # --------------------------------------------------
    def reset(self) -> None:
        """
        Reset per-game controller state.

        IMPORTANT:
        - Memory is NOT cleared here
        - This allows accumulation of defense experience
        """
        self.strength = self.base_strength
        self.last_risk = None
        self.smoothed_risk = None

    # --------------------------------------------------
    def compute_risk(self, fP: np.ndarray) -> float:
        """
        Compute latent risk.

        Risk = positive alignment with harmful directions.
        Negative or orthogonal alignment is ignored.
        """
        fP_u = fP / (np.linalg.norm(fP) + 1e-9)

        r_direct = float(np.dot(fP_u, self.v_direct))
        r_jb = float(np.dot(fP_u, self.v_jb))

        return max(0.0, r_direct, r_jb)

    # --------------------------------------------------
    def _lookup_memory(self, fP: np.ndarray) -> Optional[np.ndarray]:
        """
        Check whether this activation matches a previously
        seen high-risk latent direction.

        If similarity exceeds threshold:
        → return cached correction vector
        """
        if not self.memory:
            return None

        fP_u = fP / (np.linalg.norm(fP) + 1e-9)

        for u_mem, delta_mem in self.memory:
            # Cosine similarity test
            if np.dot(fP_u, u_mem) >= self.memory_similarity:
                return delta_mem

        return None

    # --------------------------------------------------
    def _store_memory(self, fP: np.ndarray, delta: np.ndarray) -> None:
        """
        Store a new defense experience.

        Memory format:
        - u = normalized risky direction
        - delta = correction that worked
        """
        if len(self.memory) >= self.memory_capacity:
            # FIFO eviction for stability
            self.memory.pop(0)

        u = fP / (np.linalg.norm(fP) + 1e-9)
        self.memory.append((u, delta))

    # --------------------------------------------------
    def get_intervention(self, fP: np.ndarray) -> np.ndarray:
        """
        Compute latent correction vector.

        Move AWAY from harmful directions,
        weighted by their relative activation.
        """
        fP_u = fP / (np.linalg.norm(fP) + 1e-9)

        s_d = max(0.0, float(np.dot(fP_u, self.v_direct)))
        s_j = max(0.0, float(np.dot(fP_u, self.v_jb)))

        total = s_d + s_j + 1e-9

        delta = -(s_d * self.v_direct + s_j * self.v_jb) / total
        delta = delta / (np.linalg.norm(delta) + 1e-9)

        return self.strength * delta

    # --------------------------------------------------
    def apply_intervention(self, fP: np.ndarray) -> np.ndarray:
        """
        Apply defense intervention.

        Priority:
        1. Learned memory (fast path)
        2. Online controller (fallback)
        """
        risk_now = self.compute_risk(fP)

        if risk_now <= self.risk_threshold:
            vec = fP
        else:
            # Attempt fast learned response
            learned_delta = self._lookup_memory(fP)

            if learned_delta is not None:
                vec = fP + learned_delta
            else:
                # Compute new correction
                delta = self.get_intervention(fP)
                vec = fP + delta

                # Store this experience
                self._store_memory(fP, delta)

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
        Online adaptation of intervention strength λ.

        This is a sign-based gain scheduling controller.
        """

        # Smooth noisy risk signal
        if self.smoothed_risk is None:
            self.smoothed_risk = risk_now
        else:
            self.smoothed_risk = (
                self.ema_beta * self.smoothed_risk
                + (1 - self.ema_beta) * risk_now
            )

        if self.last_risk is None:
            self.last_risk = self.smoothed_risk
            return

        delta = self.smoothed_risk - self.last_risk

        # Ignore noise
        if abs(delta) > self.deadband:
            if delta > 0:
                # Attacker improving → strengthen defense
                self.strength *= self.adapt_up
            else:
                # Defense effective → relax slightly
                self.strength *= self.adapt_down

        # Enforce bounds
        self.strength = float(
            np.clip(self.strength, self.strength_min, self.strength_max)
        )

        self.last_risk = self.smoothed_risk
