"""
Module 9 — Adaptive Defense Policy
===================================

This module implements the adaptive latent defense policy for HAVOC++.

It defines a class ``AdaptiveDefensePolicy`` that monitors latent risk and
applies interventions to keep internal representations within a stable,
refusal‑dominant region under adaptive attack.  The policy operates purely
in latent space and does not require fine‑tuning of the base model.

Key components:

* **Risk computation**: Given an activation vector ``fP``, the policy
  computes a risk score as the maximum cosine similarity to the
  ``v_direct`` and ``v_jb`` concept directions.  Higher values indicate
  closer alignment with harmful behaviours.  The cosine similarity is
  mapped from ``[-1, 1]`` to ``[0, 1]`` for ease of thresholding.

* **Intervention**: When the risk is above the configured threshold, the
  policy computes a corrective vector that points away from the
  dominant harmful concept direction and scales it by an adaptive
  strength parameter.  This vector is added to the current activation
  to push it toward safer regions.

* **Adaptation**: After each round, the policy updates the strength
  parameter.  If the risk increased relative to the previous round the
  intervention strength is amplified; if risk decreased it is slightly
  reduced.  This implements a simple proportional‑style controller and
  encourages convergence.

The policy can optionally project corrected activations into a
subspace defined by the harmful manifold (``W``, ``mu_HJ``).  This
ensures interventions respect the learned geometry of harmful/jailbreak
activations.  Users may override the risk computation or adaptation
rules by subclassing ``AdaptiveDefensePolicy``.

"""

from __future__ import annotations

from typing import Optional
import numpy as np


class AdaptiveDefensePolicy:
    """Adaptive latent defense policy for HAVOC++.

    This class encapsulates the defender component of the HAVOC++
    pipeline.  It monitors how strongly a candidate activation aligns
    with harmful or jailbreak concepts and applies a corrective vector
    to reduce that alignment.  The magnitude of the corrective action
    adapts over time in response to attacker behaviour.

    Args:
        v_direct: Normalised direct concept vector.  Representing the
            shift from benign to harmful semantics.
        v_jb: Normalised jailbreak concept vector.  Representing the
            shift from harmful to jailbreak semantics.
        mu_HJ: Mean of the harmful subspace (optional).  Providing
            centring for subspace projection.
        W: Harmful subspace basis (rows are PCA components, optional).
        risk_threshold: Maximum acceptable risk. The defense intervenes only
        when the measured risk exceeds this value.
        If the measured risk
            falls below this value the system is considered safe.
        base_strength: Initial intervention strength (step size).
        adapt_up: Multiplicative factor to increase strength when risk
            increases across rounds.
        adapt_down: Multiplicative factor to decrease strength when risk
            decreases across rounds.

    Attributes:
        strength: Current intervention strength.  Updated after each
            round based on risk dynamics.
        last_risk: Risk value from the previous round.
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
    ) -> None:
        # Normalise concept vectors
        self.v_direct = v_direct / (np.linalg.norm(v_direct) + 1e-9)
        self.v_jb = v_jb / (np.linalg.norm(v_jb) + 1e-9)
        self.mu_HJ = mu_HJ
        self.W = W
        self.risk_threshold = risk_threshold
        self.base_strength = base_strength
        self.adapt_up = adapt_up
        self.adapt_down = adapt_down
        self.reset()

    def reset(self) -> None:
        """Reset internal state at the start of a new game."""
        self.strength = self.base_strength
        self.last_risk: Optional[float] = None

    def compute_risk(self, fP: np.ndarray) -> float:
        """Compute latent risk for a candidate activation.

        Risk is defined as the maximum cosine similarity between ``fP``
        and the dangerous concept vectors.  A higher value indicates
        closer alignment with harmful behaviours.  The cosine similarity
        is mapped from ``[-1, 1]`` to ``[0, 1]``.

        Args:
            fP: Activation vector (assumed L2 normalised or arbitrary).

        Returns:
            Risk score in ``[0, 1]``.
        """
        # Ensure unit length
        fP_u = fP / (np.linalg.norm(fP) + 1e-9)

        r_direct = float(np.dot(fP_u, self.v_direct))
        r_jb = float(np.dot(fP_u, self.v_jb))

        raw = max(r_direct, r_jb)

        # IMPORTANT:
        # Ignore negatively aligned (safe or irrelevant) directions
        risk = max(0.0, raw)

        return risk


    def get_intervention(self, fP: np.ndarray) -> np.ndarray:
        """
        Compute a latent intervention vector to reduce risk.

        The intervention moves away from BOTH harmful concept directions,
        weighted by how strongly each direction is activated.
        """
        fP_u = fP / (np.linalg.norm(fP) + 1e-9)

        # Measure alignment
        score_direct = max(0.0, float(np.dot(fP_u, self.v_direct)))
        score_jb = max(0.0, float(np.dot(fP_u, self.v_jb)))

        total = score_direct + score_jb + 1e-9
        w_direct = score_direct / total
        w_jb = score_jb / total

        # Move away from both directions
        delta = -(w_direct * self.v_direct + w_jb * self.v_jb)
        delta = delta / (np.linalg.norm(delta) + 1e-9)

        return self.strength * delta


    def apply_intervention(self, fP: np.ndarray) -> np.ndarray:
        """Apply a latent intervention to ``fP`` when risk is high.

        If the current risk is below ``risk_threshold``, no intervention is
        applied (we only normalise / optionally project). This makes the
        defence a *policy* rather than an always-on perturbation, and aligns
        with the HAVOC++ claim: intervene when the attacker pushes risk upward.

        Args:
            fP: Activation vector.

        Returns:
            Defended and L2‑normalised activation vector.
        """
        # Measure risk on the incoming activation
        risk_now = self.compute_risk(fP)

        # If already low-risk, do not perturb (but keep representation in the same space)
        if risk_now <= self.risk_threshold:
            vec_new = fP
        else:
            delta = self.get_intervention(fP)
            vec_new = fP + delta

        # If a subspace basis is provided, project the result into it
        if self.W is not None and self.mu_HJ is not None:
            diff = vec_new - self.mu_HJ
            vec_proj = self.mu_HJ + self.W.T @ (self.W @ diff)
        elif self.W is not None:
            # Project without mean centring if ``mu_HJ`` is unavailable
            vec_proj = self.W.T @ (self.W @ vec_new)
        else:
            vec_proj = vec_new

        # Normalise
        vec_proj = vec_proj / (np.linalg.norm(vec_proj) + 1e-9)
        return vec_proj

    def update_policy(self, risk_now: float) -> None:
        """Update the intervention strength based on recent risk.

        If risk increases relative to the previous round, strengthen the
        intervention; if risk decreases, weaken it slightly.  This
        implements a simple proportional‑style controller.

        Args:
            risk_now: Newly measured risk value.
        """
        if self.last_risk is not None:
            if risk_now > self.last_risk + 1e-6:
                # Attackers made progress → strengthen defence
                self.strength *= self.adapt_up
            else:
                # Defence is effective → reduce strength gently
                self.strength *= self.adapt_down
        self.last_risk = risk_now