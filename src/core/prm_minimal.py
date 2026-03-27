from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


@dataclass
class PRMScoreResult:
    step_scores: List[float]
    last_score: float
    min_score: float
    mean_score: float
    prod_score: float


class BasePRMScorer:
    def score_paths(self, question: str, steps_batch: Sequence[Sequence[str]]) -> List[PRMScoreResult]:
        raise NotImplementedError


class QwenPRMScorer(BasePRMScorer):
    """
    本地 HF 版本的 Qwen2.5-Math-PRM-7B scorer。
    这个类只应该在 PRM 专用环境里使用。
    """

    def __init__(
        self,
        model_ckpt: str,
        device: str = "cuda:0",
        torch_dtype: str = "bfloat16",
        batch_size: int = 4,
        max_length: int = 4096,
        system_prompt: str = "Please reason step by step, and put your final answer within \\boxed{}.",
    ):
        self.model_ckpt = model_ckpt
        self.device = torch.device(device)
        self.batch_size = max(1, int(batch_size))
        self.max_length = int(max_length)
        self.system_prompt = system_prompt

        dtype = getattr(torch, torch_dtype) if isinstance(torch_dtype, str) else torch_dtype

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_ckpt,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModel.from_pretrained(
            model_ckpt,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(self.device).eval()

        step_sep_ids = self.tokenizer.encode("<extra_0>", add_special_tokens=False)
        if len(step_sep_ids) != 1:
            raise ValueError(f"<extra_0> should map to exactly 1 token, got: {step_sep_ids}")
        self.step_sep_id = step_sep_ids[0]

    def _build_conversation(self, question: str, steps: Sequence[str]) -> str:
        clean_steps = []
        for s in steps:
            s = (s or "").strip()
            if s:
                clean_steps.append(s)

        if not clean_steps:
            clean_steps = [""]

        assistant_content = "<extra_0>".join(clean_steps) + "<extra_0>"

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question.strip()},
            {"role": "assistant", "content": assistant_content},
        ]

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    def _make_step_rewards(self, logits: torch.Tensor, token_masks: torch.Tensor) -> List[List[float]]:
        probabilities = F.softmax(logits, dim=-1)

        all_scores = []
        for i in range(probabilities.size(0)):
            sample_probs = probabilities[i]
            sample_mask = token_masks[i]
            step_probs = sample_probs[sample_mask]

            if step_probs.numel() == 0:
                all_scores.append([0.5])
                continue

            positive_probs = step_probs[:, 1]
            all_scores.append(positive_probs.detach().float().cpu().tolist())

        return all_scores

    @torch.inference_mode()
    def score_paths(self, question: str, steps_batch: Sequence[Sequence[str]]) -> List[PRMScoreResult]:
        conversations = [self._build_conversation(question, steps) for steps in steps_batch]
        outputs: List[PRMScoreResult] = []

        for start in range(0, len(conversations), self.batch_size):
            cur_convs = conversations[start:start + self.batch_size]

            enc = self.tokenizer(
                cur_convs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)

            model_outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            logits = model_outputs[0] if isinstance(model_outputs, (tuple, list)) else model_outputs.logits
            token_masks = (input_ids == self.step_sep_id)
            batch_step_scores = self._make_step_rewards(logits, token_masks)

            for step_scores in batch_step_scores:
                step_scores = [float(x) for x in step_scores] or [0.5]
                prod = 1.0
                for x in step_scores:
                    prod *= x
                outputs.append(
                    PRMScoreResult(
                        step_scores=step_scores,
                        last_score=step_scores[-1],
                        min_score=min(step_scores),
                        mean_score=sum(step_scores) / len(step_scores),
                        prod_score=prod,
                    )
                )

        return outputs
