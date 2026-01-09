from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VersionBundle:
    """버전 정보를 한 곳에 묶어 저장하기 위한 구조체입니다."""

    router_version: str
    schema_version: str
    provider_version: str
    preprocessor_version: str


def ensure_logs_dir(project_root: str | Path) -> Path:
    """로그 디렉토리를 생성하고 Path를 반환합니다.

    Args:
        project_root: TradingAgent_v02 루트(또는 실행 기준 루트)

    Returns:
        Path: logs 디렉토리 경로
    """

    root = Path(project_root)
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def make_run_id(prefix: str = "run") -> str:
    """재현성 있는 추적을 위해 run_id를 생성합니다.

    Args:
        prefix: run_id 접두어

    Returns:
        str: 예) run_20251220T091500Z_12345
    """

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    pid = os.getpid()
    return f"{prefix}_{ts}_{pid}"


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    """jsonl 파일에 레코드 1줄을 append합니다.

    Args:
        path: jsonl 파일 경로
        record: 저장할 레코드(dict)
    """

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")





