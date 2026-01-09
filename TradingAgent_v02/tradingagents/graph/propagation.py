# TradingAgents/graph/propagation.py

from typing import Dict, Any
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from agents.category_router import load_analysis_schema, load_taxonomy, route_category
from agents.logging_utils import append_jsonl, ensure_logs_dir, make_run_id
from data_pipeline.news_preprocessor import preprocess_news
from data_providers.internal_news_provider import InternalNewsProvider
from data_providers.security_master import get_security_master
from tradingagents.agents.utils.agent_states import (
    AgentState,
    InvestDebateState,
    RiskDebateState,
)


class Propagator:
    """Handles state initialization and propagation through the graph."""

    def __init__(self, max_recur_limit=100):
        """Initialize with configuration parameters."""
        self.max_recur_limit = max_recur_limit

    def create_initial_state(
        self, company_name: str, trade_date: str, user_profile: str = ""
    ) -> Dict[str, Any]:
        """분류/스키마/뉴스 전처리를 포함해 초기 상태를 구성합니다."""

        project_root = Path(__file__).resolve().parents[2]
        logs_dir = ensure_logs_dir(project_root)

        run_id = make_run_id()
        as_of = datetime.now(timezone.utc).isoformat()

        trade_dt = _safe_parse_date(trade_date)
        news_start = trade_dt - timedelta(days=7)
        news_end = trade_dt

        taxonomy = load_taxonomy(project_root / "config" / "taxonomy_v1.yaml")
        analysis_schema = load_analysis_schema(project_root / "config" / "analysis_schema_v1.json")
        versions = {
            "router": taxonomy.get("version", "taxonomy_v1.0.0"),
            "schema": analysis_schema.get("version", "analysis_schema_v1.0.0"),
            "provider": "internal_provider_v1.1.0",
            "preprocessor": "news_preprocessor_v1.0.0",
        }

        security_master = get_security_master(company_name)

        news_provider = InternalNewsProvider(config={"fallback_company_news": [], "fallback_macro_news": []})
        company_news = news_provider.get_company_news(company_name, news_start, news_end, limit=50)
        macro_news = news_provider.get_macro_news(news_start, news_end, topics=None, limit=50)

        news_texts = _news_texts(company_news + macro_news)
        routing = route_category(
            ticker=company_name,
            security_master=security_master,
            news_texts=news_texts,
            news_items=company_news + macro_news,
            taxonomy=taxonomy,
            analysis_schema=analysis_schema,
        )
        selected_schema = (analysis_schema.get("schemas") or {}).get(
            routing.analysis_schema_id, (analysis_schema.get("schemas") or {}).get("Default_v1", {})
        )

        news_bundle = preprocess_news(
            ticker=company_name,
            domain_category=routing.domain_category,
            company_news=company_news,
            macro_news=macro_news,
        )

        append_jsonl(
            logs_dir / "news_fetch_events.jsonl",
            {
                "run_id": run_id,
                "ticker": company_name,
                "as_of": as_of,
                "window": {"start": news_start.isoformat(), "end": news_end.isoformat()},
                "counts": {"company": len(company_news), "macro": len(macro_news)},
                "version": versions,
            },
        )

        append_jsonl(
            logs_dir / "classification_events.jsonl",
            {
                "run_id": run_id,
                "ticker": company_name,
                "as_of": as_of,
                "version": {"router": versions["router"], "schema": versions["schema"]},
                "security_master": asdict(security_master),
                "news_context": {
                    "company_count": len(company_news),
                    "macro_count": len(macro_news),
                    "used_text_fields": ["title", "summary", "body"],
                },
                "classification": routing.__dict__,
            },
        )

        append_jsonl(
            logs_dir / "news_preprocess_events.jsonl",
            {
                "run_id": run_id,
                "ticker": company_name,
                "as_of": as_of,
                "version": versions,
                "dedupe_stats": news_bundle.get("dedupe_stats", {}),
                "relevance_stats": news_bundle.get("relevance_stats", {}),
                "event_stats": news_bundle.get("event_stats", {}),
            },
        )

        return {
            "messages": [("human", company_name)],
            "company_of_interest": company_name,
            "trade_date": str(trade_date),
            "run_id": run_id,
            "as_of": as_of,
            "versions": versions,
            "security_master": security_master,
            "asset_type": routing.asset_type,
            "domain_category": routing.domain_category,
            "analysis_schema_id": routing.analysis_schema_id,
            "analysis_schema": selected_schema,
            "classification_evidence": routing.evidence,
            "classification_candidates": routing.candidates_topk,
            "news_bundle": news_bundle,
            "user_profile": user_profile,
            "investment_debate_state": InvestDebateState(
                {"history": "", "current_response": "", "count": 0}
            ),
            "risk_debate_state": RiskDebateState(
                {
                    "history": "",
                    "current_risky_response": "",
                    "current_safe_response": "",
                    "current_neutral_response": "",
                    "count": 0,
                }
            ),
            "market_report": "",
            "fundamentals_report": "",
            "sentiment_report": "",
            "news_report": "",
        }

    def get_graph_args(self) -> Dict[str, Any]:
        """Get arguments for the graph invocation."""
        return {
            "stream_mode": "values",
            "config": {"recursion_limit": self.max_recur_limit},
        }


def _safe_parse_date(trade_date: str) -> datetime:
    """trade_date 문자열을 datetime으로 변환합니다(실패 시 today UTC)."""

    try:
        return datetime.fromisoformat(str(trade_date))
    except Exception:
        return datetime.now(timezone.utc)


def _news_texts(items: list[Any]) -> list[str]:
    """뉴스 아이템 리스트에서 title/summary/body 텍스트를 추출합니다."""

    texts: list[str] = []
    for it in items or []:
        for field in [getattr(it, "title", None), getattr(it, "summary", None), getattr(it, "body", None)]:
            if field:
                texts.append(str(field))
    return texts
