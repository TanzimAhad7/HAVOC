#!/usr/bin/env python3
"""
HAVOC — Module 7: Feedback Optimization Controller

NOTE:
-----
This version preserves the ORIGINAL HAVOC structure (fuzz / steer / Optimus-V),
and operates in COMPOSED GENERATION MODE by default.

IMPORTANT:
- No functions renamed
- No logic removed
- No instructions added
- Only PARAMETER-LEVEL behavior differs between modes (documented below)

This allows a minimal diff between:
  • composed generation (this version)
  • safe generation (commented parameters)
"""

import random
import numpy as np
import torch
from typing import Any, Dict, List, Optional, Tuple

from transformers import AutoTokenizer, AutoModelForCausalLM

from module1_Activation_Extraction import extract_activation_dynamic
from module3_optimus_V_scoring import optimusV


# ============================================================
# Rewriter LLM (UNCHANGED from HAVOC)
# ============================================================
class SafeRewriterLLM:

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        device: str = "cuda",
    ):
        assert torch.cuda.is_available(), "CUDA is required."
        self.device = torch.device(device)

        torch.backends.cuda.matmul.allow_tf32 = True
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
        ).to(self.device).eval()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    @torch.no_grad()
    def generate(
        self,
        seed_text: str,
        n: int = 6,
        max_new_tokens: int = 180,
        temperature: float = 0.8,
        top_p: float = 0.9,
    ) -> List[str]:

        enc = self.tokenizer(
            seed_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        input_ids = enc["input_ids"].repeat(n, 1)
        attn_mask = enc["attention_mask"].repeat(n, 1)

        out = self.model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Remove prompt prefix so intent is not repeated
        prompt_len = enc["input_ids"].shape[1]
        gen_only = out[:, prompt_len:]
        texts = self.tokenizer.batch_decode(gen_only, skip_special_tokens=True)

        uniq, seen = [], set()
        for t in texts:
            t = t.strip()
            if t and t not in seen:
                seen.add(t)
                uniq.append(t)

        return uniq[:n] if uniq else [seed_text]


# ============================================================
# Module 7 Controller (STRUCTURE UNCHANGED)
# ============================================================
class HAVOC_Controller:

    def __init__(
        self,
        intent: str,
        fI: np.ndarray,
        v_direct: np.ndarray,
        v_jb: np.ndarray,
        mu_HJ: np.ndarray,
        W: np.ndarray,
        max_iters: int = 25,
        layer: int = 20,
        seed: int = 42,
        rewrite_candidates: int = 6,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        optimus_threshold: float = 0.35,
    ):
        self.intent = intent
        self.layer = int(layer)
        self.max_iters = int(max_iters)
        self.rewrite_candidates = int(rewrite_candidates)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.v_direct = v_direct
        self.v_jb = v_jb
        self.mu_HJ = mu_HJ
        self.W = W

        # COMPOSED MODE:
        # τ defines boundary for attraction toward composed / direct manifolds
        self.tau = optimus_threshold

        # SAFE MODE (commented):
        # τ acts as rejection boundary away from direct manifolds

        self.rewriter = SafeRewriterLLM(model_name=model_name)

        # COMPOSED MODE:
        # Higher Optimus-V = better (maximize score)
        self.best_prompt: Optional[str] = None
        self.best_score: float = float("inf")

        # SAFE MODE (commented):
        # self.best_score = -float("inf")

        self.trajectory: Dict[str, Any] = {
            "actions": [],
            "prompts": [],
            "optimus_scores": [],
        }

    # --------------------------------------------------------
    def choose_action(self, score: float) -> str:
        """
        COMPOSED GENERATION:
        - If Optimus-V < τ → fuzz (explore adversarial space)
        - If Optimus-V ≥ τ → steer (exploit composed manifold)

        SAFE GENERATION (commented):
        - Reverse condition:
          return "steer" if score < τ else "fuzz"
        """
        return "fuzz" if score < self.tau else "steer"

    # --------------------------------------------------------
    def _select_best(self, candidates: List[str]) -> Tuple[str, float]:
        """
        COMPOSED MODE:
        - Select candidate with MAXIMUM Optimus-V

        SAFE MODE (commented):
        - Change > to <
        """
        best_c = candidates[0]
        best_v = float("inf")   # COMPOSED (min placeholder, comparison defines mode)

        # SAFE MODE (commented):
        # best_v = -float("inf")

        for c in candidates:
            if c.strip() == self.intent.strip():
                continue

            score = optimusV(
                intent_prompt=self.intent,
                candidate_prompt=c,
                v_direct=self.v_direct,
                v_jb=self.v_jb,
                layer=self.layer,
            )

            if score < best_v:  # COMPOSED
                best_c, best_v = c, score

            # SAFE MODE (commented):
            # if score > best_v:

        return best_c, float(best_v)

    # --------------------------------------------------------
    def apply_fuzz(self) -> Tuple[str, float]:
        candidates = self.rewriter.generate(
            self.intent,
            n=self.rewrite_candidates,
            temperature=0.95,   # COMPOSED: high entropy exploration
            top_p=0.92,
        )

        # SAFE MODE (commented):
        # temperature=0.7
        # top_p=0.9

        return self._select_best(candidates)

    # --------------------------------------------------------
    def apply_steer(self) -> Tuple[str, float]:
        candidates = self.rewriter.generate(
            self.best_prompt,
            n=max(4, self.rewrite_candidates // 2),
            temperature=0.7,   # COMPOSED: controlled refinement
            top_p=0.9,
        )

        # SAFE MODE (commented):
        # temperature=0.6
        # top_p=0.85

        return self._select_best(candidates)

    # --------------------------------------------------------
    def _log(self, action: str, prompt: str, score: float):
        self.trajectory["actions"].append(action)
        self.trajectory["prompts"].append(prompt)
        self.trajectory["optimus_scores"].append(score)

    # --------------------------------------------------------
    def run(self):

        p0, s0 = self.apply_fuzz()
        self.best_prompt = p0
        self.best_score = s0
        self._log("init", p0, s0)

        no_improve = 0

        for _ in range(self.max_iters):

            action = self.choose_action(self.best_score)

            if action == "fuzz":
                pt, st = self.apply_fuzz()
            else:
                pt, st = self.apply_steer()

            self._log(action, pt, st)

            if st < self.best_score - 1e-6:  # COMPOSED
                self.best_prompt = pt
                self.best_score = st
                no_improve = 0
            else:
                no_improve += 1

            # SAFE MODE (commented):
            # if st > self.best_score + 1e-6:

            if no_improve >= 4:
                break

        return self.best_prompt, float(self.best_score), self.trajectory
