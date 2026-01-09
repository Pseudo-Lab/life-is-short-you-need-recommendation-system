from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def main(logs_dir: str = "logs") -> None:
    """classification_events.jsonl을 기반으로 커버리지/드리프트/분포를 요약합니다.

    Args:
        logs_dir: logs 디렉토리 경로
    """

    path = Path(logs_dir) / "classification_events.jsonl"
    if not path.exists():
        print(f"[classification_eval] missing log: {path}")
        return

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        print("[classification_eval] no records")
        return

    domains = [r.get("classification", {}).get("domain_category", "Others") for r in records]
    confs = [float(r.get("classification", {}).get("confidence", 0.0)) for r in records]

    coverage_others = domains.count("Others") / float(len(domains))
    domain_counts = Counter(domains)

    # drift: ticker별 domain 변경 횟수
    by_ticker = defaultdict(list)
    for r in records:
        by_ticker[str(r.get("ticker", ""))].append(str(r.get("classification", {}).get("domain_category", "Others")))

    drift_count = 0
    for t, seq in by_ticker.items():
        uniq = list(dict.fromkeys(seq))
        if len(uniq) > 1:
            drift_count += 1

    print("=== classification_eval summary ===")
    print(f"total_runs: {len(records)}")
    print(f"others_ratio: {coverage_others:.3f}")
    print(f"tickers_with_drift: {drift_count}")
    print(f"confidence_avg: {sum(confs)/len(confs):.3f}")
    print("domain_counts:")
    for k, v in domain_counts.most_common():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()





