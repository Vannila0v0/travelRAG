from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_RUNTIME_BLOCK_RE = re.compile(r"```json\s+skill-runtime\s*\n(.*?)\n```", re.DOTALL)


@dataclass(frozen=True)
class MarkdownSkillSpec:
    name: str
    path: Path
    config: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "MarkdownSkillSpec":
        skill_path = Path(path)
        text = skill_path.read_text(encoding="utf-8")
        match = _RUNTIME_BLOCK_RE.search(text)
        if not match:
            raise ValueError(f"Missing json skill-runtime block in {skill_path}")

        config = json.loads(match.group(1))
        name = str(config.get("name") or skill_path.stem)
        return cls(name=name, path=skill_path, config=config)

