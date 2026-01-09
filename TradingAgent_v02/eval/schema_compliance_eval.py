from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


def main(logs_dir: str = "logs") -> None:
    """report_events.jsonl을 기반으로 카테고리별 schema compliance를 요약합니다.

    Args:
        logs_dir: logs 디렉토리 경로
    """

    path = Path(logs_dir) / "report_events.jsonl"
    if not path.exists():
        print(f"[schema_compliance_eval] missing log: {path}")
        return

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        print("[schema_compliance_eval] no records")
        return

    missing_by_cat = defaultdict(list)
    for r in records:
        report = r.get("report_json", {}) or {}
        cat = str(report.get("domain_category", "Others"))
        missing = report.get("missing_fields", []) or []
        missing_by_cat[cat].append(len(missing))

    print("=== schema_compliance_eval summary ===")
    for cat, arr in sorted(missing_by_cat.items(), key=lambda kv: sum(kv[1]) / max(1, len(kv[1]))):
        avg_missing = sum(arr) / float(len(arr))
        print(f"- {cat}: runs={len(arr)} avg_missing_fields={avg_missing:.2f}")


if __name__ == "__main__":
    main()





