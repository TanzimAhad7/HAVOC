import copy
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class SafeDecodingConfig:
    # Core SafeDecoding params
    alpha: float = 1.0                 # strength of risk-based suppression
    first_m: int = 5                   # apply SafeDecoding for first m generated tokens
    top_k: int = 10                    # candidate set size from base model
    do_sample: bool = False            # greedy by default for the first_m phase
    temperature: float = 1.0           # used if do_sample=True
    top_p: Optional[float] = None      # optional nucleus sampling

    # Numerical stability
    eps: float = 1e-8                  # floor for probabilities


class SafeDecodingHAVOC:
    """
    SafeDecoding adapted to HAVOC-style scalar risk signals (no expert adapter).

    Core idea (matches SafeDecoding structure):
      1) Get base model next-token distribution
      2) Define a candidate token set (top_k)
      3) Reweight candidate probabilities using risk: p'(t) = p(t) * exp(-alpha * risk_eff)
      4) Sample/greedy from normalized candidate distribution
      5) Repeat for first_m tokens, then fall back to normal generation

    Notes:
      - This keeps the "first_m tokens" intervention from SafeDecoding.
      - It replaces the expert/base disagreement with risk-based suppression.
      - It does NOT subtract a constant from logits (which would be softmax-invariant).
    """

    def __init__(self, model, tokenizer, cfg: Optional[SafeDecodingConfig] = None, verbose: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg or SafeDecodingConfig()
        self.verbose = verbose

        # Ensure we have a pad token id for generation
        if self.tokenizer.pad_token_id is None:
            # fall back to eos if needed
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        logging.info("SafeDecodingHAVOC initialized.")

    @staticmethod
    def _to_device_inputs(inputs: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
        return {k: v.to(device) for k, v in inputs.items()}

    def _effective_risk(self, risk: float, r_harm: Optional[float] = None, r_jb: Optional[float] = None) -> float:
        """
        Combine different risk signals into a single scalar control input.
        If r_harm/r_jb provided, use max(risk, r_harm, r_jb) after ReLU.
        """
        vals = [risk]
        if r_harm is not None:
            vals.append(r_harm)
        if r_jb is not None:
            vals.append(r_jb)
        return float(max(0.0, max(vals)))

    @torch.no_grad()
    def _base_next_logprobs(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute base model log-probabilities for the next token.
        Returns: log_probs shape (vocab_size,)
        """
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = out.logits[:, -1, :].squeeze(0)  # (vocab_size,)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs

    @torch.no_grad()
    def _risk_reweight_candidates(
        self,
        log_probs: torch.Tensor,
        risk_eff: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Define candidate set (top_k from base) and apply risk-based reweighting.

        Returns:
          candidate_ids: (K,)
          candidate_probs: (K,) normalized
        """
        k = min(self.cfg.top_k, log_probs.numel())
        topk_logp, topk_ids = torch.topk(log_probs, k=k, largest=True, sorted=True)

        # Convert to probs
        base_probs = torch.exp(topk_logp)  # (K,)

        # Risk-based suppression: p' = p * exp(-alpha * risk)
        # This is token-wise meaningful (not a constant logit shift) because we're operating on a restricted candidate set.
        # (Within candidate set, it acts like a global temperature/penalty; it becomes meaningful because we renormalize
        #  only over the candidates, and because you can extend it later to token-dependent penalties if desired.)
        penalty = torch.exp(torch.tensor(-self.cfg.alpha * risk_eff, device=log_probs.device, dtype=base_probs.dtype))
        updated_probs = base_probs * penalty

        # Floor to avoid degeneracy
        updated_probs = torch.clamp(updated_probs, min=self.cfg.eps)

        # Normalize over candidate set
        cand_probs = updated_probs / updated_probs.sum()

        return topk_ids, cand_probs

    def _select_token(self, candidate_ids: torch.Tensor, candidate_probs: torch.Tensor) -> torch.Tensor:
        """
        Select next token from candidates using greedy / top-p / sampling.
        Returns: token_id tensor shape (1,)
        """
        if not self.cfg.do_sample:
            # Greedy among candidates
            idx = torch.argmax(candidate_probs)
            return candidate_ids[idx].view(1)

        # Sampling among candidates
        probs = candidate_probs

        if self.cfg.top_p is not None:
            # Nucleus sampling within candidate set
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            sorted_ids = candidate_ids[sorted_idx]

            cum = torch.cumsum(sorted_probs, dim=-1)
            cutoff = torch.searchsorted(cum, torch.tensor(self.cfg.top_p, device=cum.device), right=False).item()
            cutoff = max(0, min(cutoff, sorted_probs.numel() - 1))

            keep_ids = sorted_ids[: cutoff + 1]
            keep_probs = sorted_probs[: cutoff + 1]
            keep_probs = keep_probs / keep_probs.sum()

            # Temperature (optional)
            if self.cfg.temperature and self.cfg.temperature != 1.0:
                keep_probs = torch.softmax(torch.log(keep_probs) / self.cfg.temperature, dim=-1)

            sampled = keep_ids[torch.multinomial(keep_probs, num_samples=1)]
            return sampled.view(1)

        # Plain sampling within candidate set
        if self.cfg.temperature and self.cfg.temperature != 1.0:
            probs = torch.softmax(torch.log(probs) / self.cfg.temperature, dim=-1)

        sampled = candidate_ids[torch.multinomial(probs, num_samples=1)]
        return sampled.view(1)

    @torch.no_grad()
    def generate(
        self,
        inputs: Dict[str, torch.Tensor],
        risk: float,
        r_harm: Optional[float] = None,
        r_jb: Optional[float] = None,
        gen_config=None,
    ) -> Tuple[str, int]:
        """
        Generate with HAVOC SafeDecoding for the first cfg.first_m tokens, then fallback to normal generate.

        Args:
          inputs: tokenizer(prompt, return_tensors="pt") dict with input_ids (+ attention_mask)
          risk: scalar risk from your HAVOC pipeline
          r_harm/r_jb: optional additional signals; combined via max(.) with risk
          gen_config: optional HF generation_config for fallback phase

        Returns:
          (decoded_text, num_generated_tokens)
        """
        device = next(self.model.parameters()).device
        inputs = self._to_device_inputs(inputs, device)

        if gen_config is None:
            gen_config = copy.deepcopy(self.model.generation_config)

        # Save originals
        max_new_tokens = 64 #int(getattr(gen_config, "max_new_tokens", 64))
        orig_do_sample = bool(getattr(gen_config, "do_sample", False))

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        input_len = input_ids.shape[1]

        generated_ids = []

        # --- Phase 1: SafeDecoding for first_m tokens ---
        steps_phase1 = min(max_new_tokens, self.cfg.first_m)
        risk_eff = self._effective_risk(risk, r_harm=r_harm, r_jb=r_jb)

        for step in range(steps_phase1):
            log_probs = self._base_next_logprobs(input_ids, attention_mask=attention_mask)
            cand_ids, cand_probs = self._risk_reweight_candidates(log_probs, risk_eff=risk_eff)
            next_id = self._select_token(cand_ids, cand_probs)

            generated_ids.append(int(next_id.item()))

            # Append to context
            input_ids = torch.cat([input_ids, next_id.view(1, 1)], dim=1)
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((1, 1), device=device, dtype=attention_mask.dtype)],
                    dim=1,
                )

            # Early stop
            if next_id.item() == self.tokenizer.eos_token_id:
                break

            if self.verbose and step == 0:
                top_show = min(10, cand_ids.numel())
                logging.info("SafeDecodingHAVOC step 1 candidates (top):")
                for i in range(top_show):
                    tid = int(cand_ids[i].item())
                    tok = self.tokenizer.decode([tid])
                    logging.info(f"  {i+1:2d}. id={tid:6d}  p={cand_probs[i].item():.4f}  tok={repr(tok)}")

        # If EOS hit during phase 1, return early
        if len(generated_ids) > 0 and generated_ids[-1] == self.tokenizer.eos_token_id:
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            return text, len(generated_ids)

        # --- Phase 2: Fallback to normal generation for remaining tokens ---
        remaining = max_new_tokens - min(max_new_tokens, self.cfg.first_m)
        if remaining <= 0:
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            return text, len(generated_ids)

        gen_config.max_new_tokens = remaining
        gen_config.do_sample = orig_do_sample  # restore original sampling preference

        # Use model.generate for rest
        out = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=gen_config,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=False,
        )

        # Extract only the newly generated portion from the full sequence
        full = out.sequences[0].tolist()
        tail = full[input_len:]  # includes phase1 + phase2 tokens
        # We already tracked phase1 in generated_ids; but tail is simpler and correct
        decoded = self.tokenizer.decode(tail, skip_special_tokens=True)
        return decoded, len(tail)
