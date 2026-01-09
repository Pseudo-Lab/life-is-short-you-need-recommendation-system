from __future__ import annotations

import sys
from pathlib import Path


def pytest_configure() -> None:
    """pytest 실행 시 프로젝트 루트 경로를 sys.path에 추가합니다.

    uv 환경에서 pytest가 CWD 루트를 sys.path에 포함하지 않는 케이스가 있어,
    로컬 패키지(`agents`, `data_providers`, `data_pipeline`) import가 안정적으로 되도록 보정합니다.
    """

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))





