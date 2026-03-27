from __future__ import annotations

import json
import os
from typing import Any


class JsonlWriter:
    def __init__(self, out_path: str, mode: str = "w"):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        self.out_path = out_path
        self._f = open(out_path, mode, encoding="utf-8")

    def write(self, obj: Any) -> None:
        if obj is None:
            return
        if hasattr(obj, "to_dict"):
            obj = obj.to_dict()
        self._f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None