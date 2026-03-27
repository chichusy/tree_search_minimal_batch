from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import List, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


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


class RemotePRMScorer(BasePRMScorer):
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:18080",
        timeout_seconds: float = 120.0,
        num_retries: int = 2,
        retry_sleep_seconds: float = 1.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self.num_retries = max(0, int(num_retries))
        self.retry_sleep_seconds = float(retry_sleep_seconds)

    def _post_json(self, path: str, payload: dict) -> dict:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        last_err = None
        for attempt in range(self.num_retries + 1):
            try:
                with urlopen(req, timeout=self.timeout_seconds) as resp:
                    body = resp.read().decode("utf-8")
                return json.loads(body)
            except HTTPError as e:
                detail = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
                raise RuntimeError(f"PRM server HTTP {e.code}: {detail}") from e
            except (URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
                last_err = e
                if attempt < self.num_retries:
                    time.sleep(self.retry_sleep_seconds)
                else:
                    break
        raise RuntimeError(f"PRM server request failed after retries: {last_err}") from last_err

    def health_check(self) -> dict:
        return self._post_json("/health", {})

    def score_paths(self, question: str, steps_batch: Sequence[Sequence[str]]) -> List[PRMScoreResult]:
        payload = {
            "question": question,
            "steps_batch": [list(x) for x in steps_batch],
        }
        res = self._post_json("/score", payload)
        results = res.get("results", None)
        if not isinstance(results, list):
            raise RuntimeError(f"Invalid PRM server response: {res}")
        parsed: List[PRMScoreResult] = []
        for item in results:
            parsed.append(
                PRMScoreResult(
                    step_scores=[float(x) for x in item.get("step_scores", [])],
                    last_score=float(item.get("last_score", 0.0)),
                    min_score=float(item.get("min_score", 0.0)),
                    mean_score=float(item.get("mean_score", 0.0)),
                    prod_score=float(item.get("prod_score", 0.0)),
                )
            )
        return parsed
