from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents.logging_utils import append_jsonl, ensure_logs_dir


def enforce_report_schema(
    *,
    agent_name: str,
    ticker: str,
    domain_category: str,
    analysis_schema_id: str,
    analysis_schema: dict[str, Any],
    raw_text_report: str,
    key_evidence: list[dict[str, Any]] | None = None,
    extra_fields: dict[str, Any] | None = None,
    run_id: str,
    as_of: str,
    versions: dict[str, Any],
) -> dict[str, Any]:
    """Analyst 결과를 공통 report schema로 감싸고 compliance를 계산합니다.

    공통 스키마 필드:
    - ticker, domain_category, analysis_schema_id
    - assumptions, key_evidence
    - required_metrics_filled, answers_to_questions
    - invalidation_triggers, uncertainty_top3, confidence, missing_fields

    compliance 규칙:
    - required_metrics 중 null/미기입 -> missing_fields 추가
    - analysis_questions 중 미응답 -> missing_fields 추가
    - missing_fields 개수에 따라 confidence 하향

    Args:
        agent_name: 에이전트 이름(예: "News Analyst")
        ticker: 티커
        domain_category: 도메인 카테고리
        analysis_schema_id: 스키마 ID
        analysis_schema: 스키마 객체(dict)
        raw_text_report: 기존 텍스트/마크다운 리포트
        key_evidence: 증거 목록(옵션)
        extra_fields: agent별 추가 필드(옵션)
        run_id: 런 식별자
        as_of: ISO-8601 datetime
        versions: 버전 정보(dict)

    Returns:
        dict[str, Any]: 구조화 리포트(dict)
    """

    required_metrics = list(analysis_schema.get("required_metrics", []) or [])
    questions = list(analysis_schema.get("analysis_questions", []) or [])
    invalidation_triggers = list(analysis_schema.get("invalidation_triggers", []) or [])

    required_metrics_filled = {m: None for m in required_metrics}
    answers_to_questions = {q: "" for q in questions}

    missing_fields: list[str] = []
    for m, v in required_metrics_filled.items():
        if v is None:
            missing_fields.append(f"required_metric:{m}")
    for q, ans in answers_to_questions.items():
        if not str(ans).strip():
            missing_fields.append(f"analysis_question:{q}")

    base_confidence = 0.7
    conf_penalty = _confidence_penalty(len(missing_fields))
    confidence = max(0.05, min(1.0, base_confidence + conf_penalty))

    report: dict[str, Any] = {
        "ticker": ticker,
        "domain_category": domain_category,
        "analysis_schema_id": analysis_schema_id,
        "assumptions": [
            "뉴스 입력은 Internal News Provider(v1)에서 수집/전처리된 결과를 사용한다.",
            "v1에서는 재무/가격 데이터 벤더가 완전 연동되지 않아 일부 메트릭은 공란일 수 있다.",
        ],
        "key_evidence": key_evidence or [],
        "required_metrics_filled": required_metrics_filled,
        "answers_to_questions": answers_to_questions,
        "invalidation_triggers": invalidation_triggers,
        "uncertainty_top3": [
            "내부 뉴스 입력의 커버리지/지연",
            "카테고리 분류 규칙의 드리프트 가능성",
            "재무/가격 데이터 미연동으로 인한 정량 근거 부족",
        ],
        "confidence": confidence,
        "missing_fields": missing_fields,
        "raw_text_report": raw_text_report,
    }

    if extra_fields:
        report.update(extra_fields)

    _log_report_event(
        run_id=run_id,
        ticker=ticker,
        as_of=as_of,
        agent_name=agent_name,
        report=report,
        versions=versions,
        compliance={"missing_fields": missing_fields, "confidence": confidence},
    )

    return report


def _confidence_penalty(missing_count: int) -> float:
    """missing_fields 개수에 따른 confidence 패널티를 반환합니다."""

    if missing_count == 0:
        return 0.0
    if 1 <= missing_count <= 2:
        return -0.1
    if 3 <= missing_count <= 5:
        return -0.2
    return -0.35


def _log_report_event(
    *,
    run_id: str,
    ticker: str,
    as_of: str,
    agent_name: str,
    report: dict[str, Any],
    versions: dict[str, Any],
    compliance: dict[str, Any],
) -> None:
    """report_events.jsonl에 구조화 리포트와 compliance를 저장합니다."""

    project_root = Path(__file__).resolve().parents[1]
    logs_dir = ensure_logs_dir(project_root)
    append_jsonl(
        logs_dir / "report_events.jsonl",
        {
            "run_id": run_id,
            "ticker": ticker,
            "as_of": as_of or datetime.now(timezone.utc).isoformat(),
            "version": versions,
            "agent_name": agent_name,
            "report_json": report,
            "compliance": compliance,
        },
    )





