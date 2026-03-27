from __future__ import annotations

import argparse
import json
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict

from no_vllm.src.core.prm_minimal import QwenPRMScorer


class PRMHTTPServer(ThreadingHTTPServer):
    scorer: QwenPRMScorer


class Handler(BaseHTTPRequestHandler):
    server_version = "PRMHTTP/0.1"

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length > 0 else b"{}"
        return json.loads(raw.decode("utf-8"))

    def _write_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        try:
            if self.path == "/health":
                self._write_json(200, {"ok": True, "model_ckpt": self.server.scorer.model_ckpt})
                return

            if self.path != "/score":
                self._write_json(404, {"ok": False, "error": f"unknown path: {self.path}"})
                return

            payload = self._read_json()
            question = str(payload.get("question", "") or "")
            steps_batch = payload.get("steps_batch", [])
            if not isinstance(steps_batch, list):
                raise ValueError("steps_batch must be a list")

            results = self.server.scorer.score_paths(question, steps_batch)
            self._write_json(200, {
                "ok": True,
                "results": [
                    {
                        "step_scores": r.step_scores,
                        "last_score": r.last_score,
                        "min_score": r.min_score,
                        "mean_score": r.mean_score,
                        "prod_score": r.prod_score,
                    }
                    for r in results
                ],
            })
        except Exception as e:
            self._write_json(500, {
                "ok": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
            })

    def log_message(self, format: str, *args):
        return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--prm_model_ckpt", type=str, required=True)
    ap.add_argument("--prm_device", type=str, default="cuda:0")
    ap.add_argument("--prm_dtype", type=str, default="bfloat16")
    ap.add_argument("--prm_batch_size", type=int, default=4)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument(
        "--prm_system_prompt",
        type=str,
        default="Please reason step by step, and put your final answer within \\boxed{}.",
    )
    args = ap.parse_args()

    scorer = QwenPRMScorer(
        model_ckpt=args.prm_model_ckpt,
        device=args.prm_device,
        torch_dtype=args.prm_dtype,
        batch_size=args.prm_batch_size,
        max_length=args.max_model_len,
        system_prompt=args.prm_system_prompt,
    )

    server = PRMHTTPServer((args.host, args.port), Handler)
    server.scorer = scorer
    print(f"[PRM SERVER] listening on http://{args.host}:{args.port}")
    print(f"[PRM SERVER] model={args.prm_model_ckpt}")
    server.serve_forever()


if __name__ == "__main__":
    main()
