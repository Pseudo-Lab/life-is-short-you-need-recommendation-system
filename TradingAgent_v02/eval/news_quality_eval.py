from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def main(logs_dir: str = "logs") -> None:
    """news_preprocess_events.jsonl 기반으로 뉴스 입력 품질을 요약합니다.

    Args:
        logs_dir: logs 디렉토리 경로
    """

    path = Path(logs_dir) / "news_preprocess_events.jsonl"
    if not path.exists():
        print(f"[news_quality_eval] missing log: {path}")
        return

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        print("[news_quality_eval] no records")
        return

    removed_rates = []
    avg_scores = []
    event_counter = Counter()

    for r in records:
        stats = r.get("stats", {}) or {}
        dedupe = stats.get("dedupe_stats", {}) or {}
        rel = stats.get("relevance_stats", {}) or {}
        ev = stats.get("event_stats", {}) or {}

        before = float(dedupe.get("before", 0))
        after = float(dedupe.get("after", 0))
        removed = float(dedupe.get("removed", 0))
        if before > 0:
            removed_rates.append(removed / before)

        avg_scores.append(float(rel.get("avg_score", 0.0)))

        for k, v in ev.items():
            event_counter[str(k)] += int(v)

    print("=== news_quality_eval summary ===")
    if removed_rates:
        print(f"dedupe_removed_rate_avg: {sum(removed_rates)/len(removed_rates):.3f}")
    if avg_scores:
        print(f"relevance_avg_score_avg: {sum(avg_scores)/len(avg_scores):.3f}")
    print("event_label_counts:")
    for k, v in event_counter.most_common():
        print(f"  - {k}: {v}")

    total_events = sum(event_counter.values()) or 1
    rumor_ratio = event_counter.get("RUMOR", 0) / float(total_events)
    other_ratio = event_counter.get("OTHER", 0) / float(total_events)
    if rumor_ratio > 0.5 or other_ratio > 0.8:
        print("[WARN] event_label distribution looks suspicious (too many RUMOR/OTHER).")


if __name__ == "__main__":
    main()





