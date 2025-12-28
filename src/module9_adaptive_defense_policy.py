"""
Module 9 — Adaptive + Learned Defense Policy (HAVOC++)
======================================================

This module implements a HYBRID latent defense:

(1) Online adaptive controller (feedback control)
(2) Learned memory (experience replay)

The learned memory enables rapid blocking of previously
observed jailbreak trajectories while preserving real-time
adaptation for novel attacks.

-----------------------------------------------------------------------
IMPORTANT FIX (for the attacker-defender "game" to be real)
-----------------------------------------------------------------------
In Module 7, `steer_hidden_state(..., v_comp=...)` expects a *direction vector*
in hidden space (same dimension as model hidden state).

So the defender MUST supply a *direction*, not a raw defended activation.

Therefore, this module exposes:
  - self.last_delta_dir : normalized intervention direction (recommended)
  - self.last_delta     : raw intervention vector (scaled by lambda)

Then Module 8 should pass:
  steer_vector = defence_policy.last_delta_dir
  steer_alpha  = defence_policy.strength

This makes the attacker adapt against the defender’s CURRENT action.

-----------------------------------------------------------------------
WHAT "MEMORY" MEANS HERE (NO GPU / NO MODEL-WEIGHT TRAINING)
-----------------------------------------------------------------------
This is NOT neural weights, not LoRA, not fine-tuning.

This "memory" is a small Python in-RAM buffer (a list) storing past
(high-risk direction -> correction) pairs.

It is stored in CPU RAM as numpy arrays (float32 if you choose).
It persists across intents if you do not reset it.

Memory size (rough estimate):
  - Each entry stores TWO vectors of size D:
      u_mem (D floats) + delta_mem (D floats)
  - If float32: 4 bytes per float
  - Size per entry ≈ 2 * D * 4 bytes

Example: D = 4096 (Llama hidden size)
  size/entry ≈ 2 * 4096 * 4 = 32768 bytes ≈ 32 KB
  memory_capacity = 512 entries -> ≈ 512 * 32 KB ≈ 16 MB

So it is NOT 4 MB fixed. It scales with (capacity * hidden_dim).
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
    1) Learned memory:
       - fast reaction to known risky directions
       - returns a cached correction if similar direction appears again

    2) Online controller:
       - adjusts intervention strength λ over rounds
       - handles unseen or evolving attacks

    Outputs used by the game:
    --------------------------------
    - `apply_intervention(fP)` returns defended activation
    - `update_policy(risk_def)` updates lambda
    - `last_delta_dir` is a direction for attacker steering (Module 7)
    """

    def __init__(
        self,
        v_direct: np.ndarray,
        v_jb: np.ndarray,
        *,
        mu_HJ: Optional[np.ndarray] = None,
        W: Optional[np.ndarray] = None,
        # ---------- Defense trigger ----------
        risk_threshold: float = 0.2,
        # ---------- Adaptive controller ----------
        base_strength: float = 0.05,
        adapt_up: float = 1.1,
        adapt_down: float = 0.9,
        strength_min: float = 0.01,
        strength_max: float = 1.0,
        deadband: float = 1e-3,
        ema_beta: float = 0.8,
        # ---------- Learned memory ----------
        memory_capacity: int = 512,
        memory_similarity: float = 0.95,
        # store vectors in float32 to control RAM use
        memory_dtype=np.float32,
    ) -> None:

        # =========================================================
        # (A) Normalize concept vectors ONCE
        # =========================================================
        # These represent "harmful directions" in hidden space.
        self.v_direct = (v_direct / (np.linalg.norm(v_direct) + 1e-9)).astype(memory_dtype)
        self.v_jb     = (v_jb     / (np.linalg.norm(v_jb)     + 1e-9)).astype(memory_dtype)

        # Optional harmful subspace geometry (projection)
        self.mu_HJ = None if mu_HJ is None else mu_HJ.astype(memory_dtype)
        self.W     = None if W is None else W.astype(memory_dtype)

        # =========================================================
        # (B) Online controller parameters (gain scheduling on λ)
        # =========================================================
        self.risk_threshold = float(risk_threshold)

        self.base_strength = float(base_strength)
        self.adapt_up = float(adapt_up)
        self.adapt_down = float(adapt_down)

        self.strength_min = float(strength_min)
        self.strength_max = float(strength_max)

        self.deadband = float(deadband)
        self.ema_beta = float(ema_beta)

        # =========================================================
        # (C) Learned memory parameters
        # =========================================================
        # Memory stores pairs: (u_mem, delta_mem)
        #   u_mem     = normalized risky direction
        #   delta_mem = correction vector that worked previously
        self.memory_capacity = int(memory_capacity)
        self.memory_similarity = float(memory_similarity)
        self.memory_dtype = memory_dtype

        self.memory: List[Tuple[np.ndarray, np.ndarray]] = []

        # Last-step outputs (used by Module 8 -> Module 7 coupling)
        self.last_delta: Optional[np.ndarray] = None       # raw correction applied
        self.last_delta_dir: Optional[np.ndarray] = None   # normalized direction of correction
        self.last_defended_fP: Optional[np.ndarray] = None # defended activation (optional bookkeeping)

        self.reset()

    # --------------------------------------------------
    @staticmethod
    def _l2_normalize(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
        return x / (np.linalg.norm(x) + eps)

    # --------------------------------------------------
    def reset(self) -> None:
        """
        Reset per-game controller state.

        IMPORTANT:
        - Memory is NOT cleared here (learned across intents)
        - Only the online controller state resets (λ, EMA risk)
        """
        self.strength = self.base_strength
        self.last_risk: Optional[float] = None
        self.smoothed_risk: Optional[float] = None

        # Reset last action outputs for safety (per game)
        self.last_delta = None
        self.last_delta_dir = None
        self.last_defended_fP = None

    # --------------------------------------------------
    def compute_risk(self, fP: np.ndarray) -> float:
        """
        Compute latent risk in the SAME space as Optimus-V.

        Risk = max positive cosine alignment with:
        - direct harmful direction
        - jailbreak harmful direction
        """

        # Normalize input
        fP_u = self._l2_normalize(fP.astype(self.memory_dtype))

        # --------------------------------------------------
        # PROJECT CONSISTENTLY (same rule as Optimus-V)
        # --------------------------------------------------
        if self.W is not None and self.mu_HJ is not None:
            fP_s = self._l2_normalize(self.W @ (fP_u - self.mu_HJ))
            vD_s = self._l2_normalize(self.W @ (self.v_direct - self.mu_HJ))
            vJ_s = self._l2_normalize(self.W @ (self.v_jb - self.mu_HJ))
        elif self.W is not None:
            fP_s = self._l2_normalize(self.W @ fP_u)
            vD_s = self._l2_normalize(self.W @ self.v_direct)
            vJ_s = self._l2_normalize(self.W @ self.v_jb)
        else:
            fP_s = fP_u
            vD_s = self.v_direct
            vJ_s = self.v_jb

        # --------------------------------------------------
        # RISK = worst harmful alignment
        # --------------------------------------------------
        r_direct = float(np.dot(fP_s, vD_s))
        r_jb = float(np.dot(fP_s, vJ_s))

        return max(0.0, r_direct, r_jb)


    # --------------------------------------------------
    def _lookup_memory(self, fP: np.ndarray) -> Optional[np.ndarray]:
        """
        Learned memory lookup.

        We check if the CURRENT activation direction fP is similar
        to a previously seen risky direction.

        If cosine similarity >= memory_similarity:
            return the cached delta (correction vector)
        else:
            return None
        """
        if not self.memory:
            return None

        fP_u = self._l2_normalize(fP.astype(self.memory_dtype))

        # Linear scan (capacity is small; O(M*D) is fine)
        for u_mem, delta_mem in self.memory:
            if float(np.dot(fP_u, u_mem)) >= self.memory_similarity:
                return delta_mem

        return None

    # --------------------------------------------------
    def _store_memory(self, fP: np.ndarray, delta: np.ndarray) -> None:
        """
        Store a new defense experience (experience replay).

        Stored item:
            u_mem    = normalized activation direction (what we saw)
            delta_mem= correction that we applied

        Eviction:
            FIFO eviction (pop oldest) to keep memory bounded and stable.
        """
        if len(self.memory) >= self.memory_capacity:
            self.memory.pop(0)

        u = self._l2_normalize(fP.astype(self.memory_dtype))
        d = delta.astype(self.memory_dtype)

        self.memory.append((u, d))

    # --------------------------------------------------
    def get_intervention(self, fP: np.ndarray) -> np.ndarray:
        """
        Compute a NEW latent correction vector (online policy).

        Idea:
          - measure how strongly fP aligns with v_direct and v_jb
          - move AWAY from both, weighted by the relative activation
          - scale by current strength λ

        Output:
          delta = λ * direction_away_from_harm
        """
        fP_u = self._l2_normalize(fP.astype(self.memory_dtype))

        s_d = max(0.0, float(np.dot(fP_u, self.v_direct)))
        s_j = max(0.0, float(np.dot(fP_u, self.v_jb)))

        total = s_d + s_j + 1e-9

        direction = -(s_d * self.v_direct + s_j * self.v_jb) / total
        direction = self._l2_normalize(direction.astype(self.memory_dtype))

        delta = (self.strength * direction).astype(self.memory_dtype)
        return delta
    
    def apply_intervention(self, fP: np.ndarray) -> np.ndarray:
        fP = fP.astype(self.memory_dtype)
        risk_now = self.compute_risk(fP)

        delta_used = None
        source = "none"
        vec = fP

        if risk_now > self.risk_threshold:
            learned_delta = self._lookup_memory(fP)

            if learned_delta is not None:
                delta_used = learned_delta.astype(self.memory_dtype)
                source = "memory"
            else:
                delta_used = self.get_intervention(fP)   # ← correct call
                source = "online"
                self._store_memory(fP, delta_used)

            vec = fP + delta_used

            # SINGLE, CLEAN LOG (what you asked for)
            print(f"[DEFENSE] source={source}")

        vec = self._l2_normalize(vec)

        self.last_delta = delta_used
        self.last_delta_dir = (
            self._l2_normalize(delta_used) if delta_used is not None else None
        )

        return vec

    # --------------------------------------------------
    def update_policy(self, risk_now: float) -> None:
        """
        Online adaptation of intervention strength λ.

        This is a sign-based gain scheduling controller:

        - Smooth risk with EMA
        - Compare smoothed risk to previous smoothed risk
        - If risk increases: λ <- λ * adapt_up
        - If risk decreases: λ <- λ * adapt_down
        - Ignore tiny changes (deadband)
        - Clamp λ into [strength_min, strength_max]

        This gives you:
        - stability (bounded λ)
        - robustness to noise (EMA + deadband)
        - quick escalation when attacker is winning
        """
        risk_now = float(risk_now)

        # ---- Step 1: EMA smoothing ----
        if self.smoothed_risk is None:
            self.smoothed_risk = risk_now
        else:
            self.smoothed_risk = (
                self.ema_beta * self.smoothed_risk
                + (1.0 - self.ema_beta) * risk_now
            )

        # ---- Step 2: initialize last_risk on first update ----
        if self.last_risk is None:
            self.last_risk = self.smoothed_risk
            return

        # ---- Step 3: compute change ----
        delta = self.smoothed_risk - self.last_risk

        # ---- Step 4: deadband ----
        if abs(delta) > self.deadband:
            if delta > 0.0:
                # risk increased -> attacker making progress -> strengthen defense
                self.strength *= self.adapt_up
            else:
                # risk decreased -> defense working -> relax a bit (avoid over-perturbing)
                self.strength *= self.adapt_down

        # ---- Step 5: saturation bounds ----
        self.strength = float(np.clip(self.strength, self.strength_min, self.strength_max))

        # ---- Step 6: store ----
        self.last_risk = self.smoothed_risk
