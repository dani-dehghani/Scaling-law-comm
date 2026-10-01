from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_json_dump(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name, dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.remove(tmp_name)
        except FileNotFoundError:
            pass
        raise
