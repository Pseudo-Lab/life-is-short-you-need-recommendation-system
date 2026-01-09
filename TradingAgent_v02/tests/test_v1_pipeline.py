from __future__ import annotations

from pathlib import Path

from agents.category_router import load_analysis_schema, load_taxonomy, route_category
from data_pipeline.news_preprocessor import preprocess_news
from data_providers.internal_news_provider import InternalNewsProvider
from data_providers.security_master import get_security_master


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FALLBACK_COMPANY = [
    {
        "id": "fallback_company_1",
        "published_at": "2025-12-05T00:00:00Z",
        "source": "fallback",
        "title": "JPM beats earnings expectations",
        "summary": "Banks earnings highlight",
        "body": "JPM earnings beat analyst expectations.",
        "url": "https://example.com/jpm",
        "tickers": ["JPM"],
        "topics": ["Banks"],
        "lang": "en",
        "meta": {"provider": "fallback"},
    }
]
FALLBACK_MACRO = [
    {
        "id": "fallback_macro_1",
        "published_at": "2025-12-05T00:00:00Z",
        "source": "fallback",
        "title": "Fed hints at rate pause",
        "summary": "Macro headline",
        "body": "Fed signals potential rate pause.",
        "url": "https://example.com/fed",
        "tickers": [],
        "topics": ["macro"],
        "lang": "en",
        "meta": {"provider": "fallback"},
    }
]


def _load_cfg():
    taxonomy = load_taxonomy(PROJECT_ROOT / "config" / "taxonomy_v1.yaml")
    schema_cfg = load_analysis_schema(PROJECT_ROOT / "config" / "analysis_schema_v1.json")
    return taxonomy, schema_cfg


def test_classification_router_cases():
    taxonomy, schema_cfg = _load_cfg()

    cases = [
        ("O", "REIT"),
        ("JPM", "Banks"),
        ("XOM", "Energy_EP"),
        ("CRM", "SaaS"),
        ("MRNA", "Biotech"),
        ("UNKNOWN", "Others"),
    ]

    for ticker, expected_domain in cases:
        sec = get_security_master(ticker)
        res = route_category(
            ticker=ticker,
            security_master=sec,
            taxonomy=taxonomy,
            analysis_schema=schema_cfg,
        )
        assert res.domain_category == expected_domain
        assert res.asset_type
        assert res.evidence and len(res.evidence) >= 1
        if res.domain_category == "Others":
            assert "fallback" in (res.evidence[-1].get("field") or "")


def test_internal_news_provider_schema_and_empty_safe():
    provider = InternalNewsProvider(config={"fallback_company_news": FALLBACK_COMPANY, "fallback_macro_news": FALLBACK_MACRO})
    assert provider.healthcheck() is True

    # empty-window should produce empty lists without crashing
    from datetime import datetime, timezone

    start = datetime(1990, 1, 1, tzinfo=timezone.utc)
    end = datetime(1990, 1, 2, tzinfo=timezone.utc)
    company_news = provider.get_company_news("JPM", start, end, limit=10)
    macro_news = provider.get_macro_news(start, end, topics=None, limit=10)
    bundle = preprocess_news(
        ticker="JPM",
        domain_category="Banks",
        company_news=company_news,
        macro_news=macro_news,
    )
    assert "company_news" in bundle and "macro_news" in bundle
    assert isinstance(bundle["company_news"], list)
    assert isinstance(bundle["macro_news"], list)


def test_news_item_has_required_fields():
    provider = InternalNewsProvider(config={"fallback_company_news": FALLBACK_COMPANY, "fallback_macro_news": FALLBACK_MACRO})
    from datetime import datetime, timezone

    start = datetime(2025, 12, 1, tzinfo=timezone.utc)
    end = datetime(2025, 12, 20, tzinfo=timezone.utc)
    items = provider.get_company_news("JPM", start, end, limit=5)
    assert items

    it = items[0]
    assert it.id
    assert it.published_at
    assert it.source
    assert it.title
    assert isinstance(it.tickers, list)
    assert isinstance(it.topics, list)
    assert it.lang
    assert isinstance(it.meta, dict)





