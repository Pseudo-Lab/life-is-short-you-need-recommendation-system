from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

# 기존 뉴스 벤더 라우터를 재사용하여 실데이터를 가져온다.
from tradingagents.dataflows.interface import route_to_vendor


@dataclass(frozen=True)
class NewsItem:
    """내부 뉴스 정규화 스키마(NewsItem)를 표현합니다."""

    id: str
    published_at: str
    source: str
    title: str
    body: str | None
    summary: str | None
    url: str | None
    tickers: list[str]
    topics: list[str]
    lang: str
    meta: dict[str, Any]


def _stable_hash_id(parts: list[str]) -> str:
    """입력 문자열 조각을 기반으로 안정적인 해시 ID를 생성합니다.

    Args:
        parts: 해시를 구성할 문자열 목록(예: title, url, published_at)

    Returns:
        str: SHA-256 기반의 짧은 식별자(앞 16자)
    """

    joined = "||".join([p for p in parts if p])
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return digest[:16]


def _normalize_news_item(raw: dict[str, Any]) -> NewsItem:
    """원본 raw dict를 NewsItem 스키마로 정규화합니다.

    Args:
        raw: 내부 소스에서 가져온 원본 뉴스 레코드

    Returns:
        NewsItem: 정규화된 뉴스 아이템
    """

    published_at = str(raw.get("published_at", "")).strip()
    title = str(raw.get("title", "")).strip()
    url = raw.get("url")
    raw_id = str(raw.get("id", "")).strip()

    if not raw_id:
        raw_id = _stable_hash_id([published_at, title, str(url or "")])

    tickers = [str(t).upper().strip() for t in (raw.get("tickers") or []) if str(t).strip()]
    topics = [str(x).strip() for x in (raw.get("topics") or []) if str(x).strip()]

    return NewsItem(
        id=raw_id,
        published_at=published_at,
        source=str(raw.get("source", "")).strip() or "internal",
        title=title,
        body=(str(raw.get("body")).strip() if raw.get("body") is not None else None),
        summary=(str(raw.get("summary")).strip() if raw.get("summary") is not None else None),
        url=(str(url).strip() if url is not None else None),
        tickers=tickers,
        topics=topics,
        lang=str(raw.get("lang", "en")).strip() or "en",
        meta=raw.get("meta") if isinstance(raw.get("meta"), dict) else {"provider": "internal", "raw": {}},
    )


class InternalNewsProvider:
    """Internal News Provider v1.1

    - v1.1: 기존 get_news/get_global_news 벤더 라우터를 사용해 실데이터를 수집.
    - 샘플 파일 의존성을 제거하고, 필요 시 config로 전달된 fallback 데이터만 사용.
    """

    def __init__(self, config: dict[str, Any]):
        """Provider를 초기화합니다.

        Args:
            config: 실행 구성(dict).
                - fallback_company_news: 벤더 실패 시 사용할 사전 로드 회사 뉴스(list[dict])
                - fallback_macro_news: 벤더 실패 시 사용할 사전 로드 거시 뉴스(list[dict])
        """

        self._config = config
        self._fallback_company = config.get("fallback_company_news") or []
        self._fallback_macro = config.get("fallback_macro_news") or []

    def healthcheck(self) -> bool:
        """Provider 상태를 점검합니다.

        - 기본: 벤더 라우터가 호출 가능하면 True
        - 폴백: fallback 데이터가 있으면 True
        """

        try:
            _ = route_to_vendor("get_news", "AAPL", "2024-01-01", "2024-01-05")
            return True
        except Exception:
            return bool(self._fallback_company or self._fallback_macro)

    def get_company_news(
        self,
        ticker: str,
        start_dt: datetime,
        end_dt: datetime,
        limit: int = 50,
    ) -> list[NewsItem]:
        """특정 티커 관련 뉴스를 조회합니다.

        Args:
            ticker: 대상 티커
            start_dt: 시작 시각(datetime)
            end_dt: 종료 시각(datetime)
            limit: 최대 반환 개수

        Returns:
            list[NewsItem]: 정규화된 회사 뉴스 목록
        """

        raw_items = self._fetch_company_news_raw(ticker=ticker, start_dt=start_dt, end_dt=end_dt)
        items = [_normalize_news_item(x) for x in raw_items if isinstance(x, dict)]
        items = self._filter_by_time(items, start_dt, end_dt)
        t = ticker.upper().strip()
        items = [x for x in items if not x.tickers or t in x.tickers]
        return items[: max(0, int(limit))]

    def get_macro_news(
        self,
        start_dt: datetime,
        end_dt: datetime,
        topics: list[str] | None = None,
        limit: int = 50,
    ) -> list[NewsItem]:
        """거시/글로벌 뉴스를 조회합니다.

        Args:
            start_dt: 시작 시각(datetime)
            end_dt: 종료 시각(datetime)
            topics: 토픽 필터(옵션)
            limit: 최대 반환 개수

        Returns:
            list[NewsItem]: 정규화된 거시 뉴스 목록
        """

        raw_items = self._fetch_macro_news_raw(start_dt=start_dt, end_dt=end_dt, topics=topics)
        items = [_normalize_news_item(x) for x in raw_items if isinstance(x, dict)]
        items = self._filter_by_time(items, start_dt, end_dt)

        if topics:
            want = {str(t).strip().lower() for t in topics if str(t).strip()}
            items = [x for x in items if any(tp.lower() in want for tp in x.topics)]

        return items[: max(0, int(limit))]

    # ------------------------------------------------------------------ #
    # 벤더 라우터를 통한 실데이터 수집
    # ------------------------------------------------------------------ #
    def _fetch_company_news_raw(
        self, *, ticker: str, start_dt: datetime, end_dt: datetime
    ) -> list[dict[str, Any]]:
        start = start_dt.date().isoformat()
        end = end_dt.date().isoformat()
        try:
            raw = route_to_vendor("get_news", ticker, start, end)
            return self._coerce_raw_news(raw, default_ticker=ticker)
        except Exception:
            return [x for x in self._fallback_company if isinstance(x, dict)]

    def _fetch_macro_news_raw(
        self, *, start_dt: datetime, end_dt: datetime, topics: list[str] | None
    ) -> list[dict[str, Any]]:
        curr = end_dt.date().isoformat()
        look_back_days = max(1, (end_dt.date() - start_dt.date()).days)
        try:
            raw = route_to_vendor("get_global_news", curr, look_back_days, 20)
            return self._coerce_raw_news(raw, default_topics=topics or [])
        except Exception:
            return [x for x in self._fallback_macro if isinstance(x, dict)]

    @staticmethod
    def _coerce_raw_news(
        raw: Any, *, default_ticker: str | None = None, default_topics: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """route_to_vendor 결과를 NewsItem dict 리스트로 변환."""

        default_topics = default_topics or []

        # 1) 이미 list[dict]인 경우 그대로
        if isinstance(raw, list) and all(isinstance(x, dict) for x in raw):
            return raw

        # 2) JSON 문자열 파싱 시도
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                raw = parsed
            except Exception:
                parsed = None

        # 3) Alpha Vantage NEWS_SENTIMENT 응답(dict with "feed")
        if isinstance(raw, dict) and isinstance(raw.get("feed"), list):
            out: list[dict[str, Any]] = []
            for item in raw["feed"]:
                out.append(
                    {
                        "id": item.get("uuid") or item.get("id"),
                        "published_at": item.get("time_published"),
                        "source": item.get("source"),
                        "title": item.get("title"),
                        "summary": item.get("summary"),
                        "body": item.get("summary"),
                        "url": item.get("url"),
                        "tickers": [
                            x.get("ticker", "").upper()
                            for x in item.get("ticker_sentiment", [])
                            if x.get("ticker")
                        ],
                        "topics": [x.get("topic", "") for x in item.get("topics", []) if x.get("topic")],
                        "lang": item.get("language", "en"),
                        "meta": {"provider": "alpha_vantage", "raw": item},
                    }
                )
            return out

        # 4) 단일 문자열 등 비정형 응답 → 최소 정보로 1건 생성
        if isinstance(raw, str):
            return [
                {
                    "id": _stable_hash_id([raw]),
                    "published_at": datetime.utcnow().isoformat(),
                    "source": "vendor_raw",
                    "title": raw[:120],
                    "summary": raw,
                    "body": raw,
                    "url": None,
                    "tickers": [default_ticker] if default_ticker else [],
                    "topics": default_topics,
                    "lang": "en",
                    "meta": {"provider": "vendor_raw"},
                }
            ]

        # 5) 알 수 없는 형태 → 빈 리스트
        return []

    @staticmethod
    def _filter_by_time(items: list[NewsItem], start_dt: datetime, end_dt: datetime) -> list[NewsItem]:
        """published_at을 기준으로 기간 필터링합니다.

        Args:
            items: 뉴스 목록
            start_dt: 시작 시각
            end_dt: 종료 시각

        Returns:
            list[NewsItem]: 기간 내 뉴스 목록
        """

        def _parse_iso(s: str) -> datetime | None:
            try:
                return datetime.fromisoformat(s.replace("Z", "+00:00"))
            except Exception:
                return None

        out: list[NewsItem] = []
        for it in items:
            dt = _parse_iso(it.published_at)
            if dt is None:
                continue
            if start_dt <= dt <= end_dt:
                out.append(it)
        return out




