from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class SecurityMaster:
    """Security Master(종목 메타) 최소 스키마를 표현합니다."""

    ticker: str
    exchange: str
    name: str
    asset_type_hint: str | None
    industry_classification: str | None
    business_description: str | None
    source_map: dict[str, Any]


def get_security_master(ticker: str) -> SecurityMaster:
    """티커에 대한 최소 Security Master 정보를 반환합니다.

    v1에서는 외부 벤더가 미확정이므로, 테스트/개발을 위한 mock 데이터를 제공합니다.

    Args:
        ticker: 자산 심볼(예: "JPM")

    Returns:
        SecurityMaster: 최소 필드를 포함한 종목 메타
    """

    t = ticker.upper().strip()
    now_iso = datetime.now(timezone.utc).isoformat()

    # 매우 단순한 mock 매핑(v1)
    mock = {
        "JPM": ("NYSE", "JPMorgan Chase & Co.", "bank financial services deposits loans"),
        "O": ("NYSE", "Realty Income", "reit net lease retail occupancy ffo noi"),
        "XOM": ("NYSE", "Exxon Mobil", "oil gas upstream production capex opec"),
        "CRM": ("NYSE", "Salesforce", "saas subscription cloud enterprise arr nrr"),
        "MRNA": ("NASDAQ", "Moderna", "biotech clinical trial fda phase pdufa"),
    }

    exchange, name, desc = mock.get(t, ("UNKNOWN", t, ""))
    return SecurityMaster(
        ticker=t,
        exchange=exchange,
        name=name,
        asset_type_hint="Equity",
        industry_classification=None,
        business_description=desc or None,
        source_map={
            "ticker": {"source": "mock", "updated_at": now_iso},
            "exchange": {"source": "mock", "updated_at": now_iso},
            "name": {"source": "mock", "updated_at": now_iso},
            "business_description": {"source": "mock", "updated_at": now_iso},
        },
    )





