from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from data_providers.internal_news_provider import NewsItem


@dataclass(frozen=True)
class NewsPreprocessorConfig:
    """뉴스 전처리 설정을 담는 데이터클래스입니다."""

    company_top_k: int = 5
    macro_top_k: int = 3
    relevance_threshold: float = 0.2


_EVENT_RULES: dict[str, list[str]] = {
    "EARNINGS": ["earnings", "results", "quarter", "q1", "q2", "q3", "q4"],
    "GUIDANCE": ["guidance", "outlook", "forecast"],
    "M&A": ["acquisition", "merge", "merger", "buyout", "deal"],
    "REGULATION": ["regulation", "regulatory", "fed", "sec", "doj", "fda"],
    "PRODUCT": ["launch", "product", "release", "feature"],
    "MACRO": ["macro", "inflation", "cpi", "pce", "unemployment", "rates", "yield"],
    "LEGAL": ["lawsuit", "legal", "court", "settlement"],
    "SECURITY": ["breach", "hack", "security incident", "cyber"],
    "RUMOR": ["rumor", "reportedly", "sources say"],
}


_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Banks": ["nim", "deposit", "credit", "loan", "fed", "rate"],
    "REIT": ["ffo", "noi", "occupancy", "cap rate", "lease", "tenant"],
    "Energy_EP": ["wti", "brent", "opec", "production", "capex", "upstream"],
    "SaaS": ["arr", "nrr", "subscription", "churn", "cloud", "margin"],
    "Biotech": ["fda", "trial", "phase", "pdufa", "approval", "clinical"],
}


def preprocess_news(
    *,
    ticker: str,
    domain_category: str,
    company_news: list[NewsItem],
    macro_news: list[NewsItem],
    config: NewsPreprocessorConfig | None = None,
) -> dict[str, Any]:
    """뉴스를 중복 제거/관련도 점수화/이벤트 라벨링 후 LLM 입력용 번들로 패킹합니다.

    - DEDUPE: title/url/body_hash 기반 중복 제거
    - RECENCY RANK: 최근 기사 우선 정렬
    - RELEVANCE SCORE: ticker/카테고리 키워드 기반 단순 점수
    - EVENT LABELING: rule 기반 라벨 부여
    - OUTPUT PACKING: company top_k, macro top_k로 제한

    Args:
        ticker: 대상 티커
        domain_category: 도메인 카테고리(예: "Banks")
        company_news: 회사 뉴스 목록(정규화된 NewsItem)
        macro_news: 거시 뉴스 목록(정규화된 NewsItem)
        config: 전처리 설정(옵션)

    Returns:
        dict[str, Any]: 전처리 결과 번들(company/macro + 통계)
    """

    cfg = config or NewsPreprocessorConfig()
    before = len(company_news) + len(macro_news)

    company_deduped = _dedupe_news(company_news)
    macro_deduped = _dedupe_news(macro_news)

    company_scored = [
        _with_scores(item=x, ticker=ticker, domain_category=domain_category) for x in company_deduped
    ]
    macro_scored = [
        _with_scores(item=x, ticker=ticker, domain_category=domain_category) for x in macro_deduped
    ]

    company_scored.sort(key=lambda x: (x["relevance_score"], x["published_at"]), reverse=True)
    macro_scored.sort(key=lambda x: (x["relevance_score"], x["published_at"]), reverse=True)

    company_kept = [x for x in company_scored if x["relevance_score"] >= cfg.relevance_threshold]
    macro_kept = [x for x in macro_scored if x["relevance_score"] >= cfg.relevance_threshold]

    company_final = company_kept[: cfg.company_top_k]
    macro_final = macro_kept[: cfg.macro_top_k]

    after = len(company_final) + len(macro_final)
    avg_score = 0.0
    if company_final or macro_final:
        avg_score = sum([x["relevance_score"] for x in company_final + macro_final]) / float(after)

    event_stats: dict[str, int] = {}
    for x in company_final + macro_final:
        label = x["event_label"]
        event_stats[label] = event_stats.get(label, 0) + 1

    return {
        "company_news": [x["news_item"] for x in company_final],
        "macro_news": [x["news_item"] for x in macro_final],
        "company_news_cards": [
            {
                "news_id": x["news_item"].id,
                "published_at": x["news_item"].published_at,
                "title": x["news_item"].title,
                "source": x["news_item"].source,
                "relevance_score": x["relevance_score"],
                "event_label": x["event_label"],
            }
            for x in company_final
        ],
        "macro_news_cards": [
            {
                "news_id": x["news_item"].id,
                "published_at": x["news_item"].published_at,
                "title": x["news_item"].title,
                "source": x["news_item"].source,
                "relevance_score": x["relevance_score"],
                "event_label": x["event_label"],
            }
            for x in macro_final
        ],
        "dedupe_stats": {"before": before, "after": after, "removed": max(0, before - after)},
        "relevance_stats": {"kept_threshold": cfg.relevance_threshold, "avg_score": avg_score},
        "event_stats": event_stats,
    }


def _dedupe_news(items: list[NewsItem]) -> list[NewsItem]:
    """뉴스 아이템을 중복 제거합니다.

    Args:
        items: 뉴스 목록

    Returns:
        list[NewsItem]: 중복 제거된 뉴스 목록
    """

    seen: set[str] = set()
    out: list[NewsItem] = []
    for it in items:
        key = _dedupe_key(it)
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def _dedupe_key(item: NewsItem) -> str:
    """중복 판정을 위한 키를 생성합니다."""

    title = (item.title or "").strip().lower()
    url = (item.url or "").strip().lower()
    body = (item.body or item.summary or "").strip().lower()
    body_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()[:12] if body else ""
    return f"{title}||{url}||{body_hash}"


def _with_scores(*, ticker: str, domain_category: str, item: NewsItem) -> dict[str, Any]:
    """뉴스 아이템에 relevance_score와 event_label을 계산해 합칩니다."""

    score = _relevance_score(item=item, ticker=ticker, domain_category=domain_category)
    label = _event_label(item=item)
    published_at = _safe_datetime(item.published_at)
    return {
        "news_item": item,
        "relevance_score": score,
        "event_label": label,
        "published_at": published_at,
    }


def _safe_datetime(iso_s: str) -> datetime:
    """ISO-8601 문자열을 datetime으로 변환합니다(실패 시 최소값)."""

    try:
        return datetime.fromisoformat(iso_s.replace("Z", "+00:00"))
    except Exception:
        return datetime.min


def _relevance_score(*, item: NewsItem, ticker: str, domain_category: str) -> float:
    """뉴스 관련도 점수를 계산합니다.

    점수는 0~1 범위의 휴리스틱으로 구성합니다.
    - 티커 직접 매칭: +0.6
    - 제목/본문에서 티커 출현 빈도: 최대 +0.2
    - 카테고리 키워드 매칭: 최대 +0.2

    Args:
        item: 뉴스 아이템
        ticker: 대상 티커
        domain_category: 도메인 카테고리

    Returns:
        float: 관련도 점수(0~1)
    """

    t = ticker.upper().strip()
    text = " ".join([(item.title or ""), (item.summary or ""), (item.body or "")]).lower()

    score = 0.0
    if t in {x.upper() for x in item.tickers}:
        score += 0.6

    # ticker 빈도(단순)
    hits = len(re.findall(rf"\b{re.escape(t.lower())}\b", text))
    score += min(0.2, 0.05 * float(hits))

    keywords = _CATEGORY_KEYWORDS.get(domain_category, [])
    kw_hits = sum(1 for kw in keywords if kw.lower() in text)
    score += min(0.2, 0.05 * float(kw_hits))

    return max(0.0, min(1.0, score))


def _event_label(*, item: NewsItem) -> str:
    """이벤트 라벨을 rule 기반으로 부여합니다."""

    text = " ".join([(item.title or ""), (item.summary or ""), (item.body or "")]).lower()
    for label, needles in _EVENT_RULES.items():
        if any(n in text for n in needles):
            return label
    return "OTHER"


