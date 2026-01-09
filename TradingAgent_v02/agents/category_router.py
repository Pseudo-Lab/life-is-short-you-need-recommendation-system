from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from data_providers.security_master import SecurityMaster


@dataclass(frozen=True)
class ClassificationEvidence:
    """분류 근거(evidence) 단위입니다.

    evidence는 재현성을 위해 source/field/rule/matched_text/ts를 항상 포함합니다.
    """

    source: str
    field: str
    rule: str
    matched_text: str
    ts: str


@dataclass(frozen=True)
class CategoryRoutingResult:
    """2층 분류 + 스키마 라우팅 결과입니다."""

    asset_type: str
    domain_category: str
    confidence: float
    candidates_topk: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    version: str
    analysis_schema_id: str


def load_taxonomy(taxonomy_path: str | Path) -> dict[str, Any]:
    """taxonomy_v1.yaml을 로드합니다.

    Args:
        taxonomy_path: taxonomy YAML 경로

    Returns:
        dict[str, Any]: taxonomy 설정(dict)
    """

    p = Path(taxonomy_path)
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_analysis_schema(schema_json_path: str | Path) -> dict[str, Any]:
    """analysis_schema_v1.json을 로드합니다.

    Args:
        schema_json_path: 스키마 JSON 경로

    Returns:
        dict[str, Any]: schema 설정(dict)
    """

    p = Path(schema_json_path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def route_category(
    *,
    ticker: str,
    security_master: SecurityMaster,
    news_texts: list[str] | None = None,
    news_items: list[Any] | None = None,  # NewsItem 리스트 (topics 필드 활용용)
    taxonomy: dict[str, Any],
    analysis_schema: dict[str, Any],
) -> CategoryRoutingResult:
    """뉴스/메타를 기반으로 Asset Type + Domain Category + schema_id를 결정합니다.

    v1에서는 LLM에 의존하지 않고 아래 다단계로 분류합니다.
    - deterministic rules(강한 규칙): 티커 override
    - taxonomy lookup: 키워드/티커 리스트 기반
    - text fallback: business_description + (옵션) news_texts 기반 키워드 스코어링
    - topics 필드 활용: 뉴스의 topics 필드에서 직접 섹터/산업 태그 매칭
    - 회사명 매칭: security_master.name과 뉴스 텍스트에서 회사명 매칭
    - Others fallback: 항상 근거/사유를 기록

    핵심: `news_texts`는 "뉴스를 가져온 뒤 분류한다" 요구를 반영하기 위한 입력이며,
    title/summary/body 등 사람이 읽는 텍스트를 그대로 넣으면 됩니다.
    `news_items`는 topics 필드와 회사명 매칭을 위해 NewsItem 리스트를 전달합니다.

    Args:
        ticker: 티커(예: "JPM")
        security_master: 종목 메타
        news_texts: 내부 뉴스에서 추출한 텍스트 목록(예: title/summary/body). 없으면 None
        news_items: NewsItem 리스트(topics 필드 활용용). 없으면 None
        taxonomy: taxonomy 설정(dict)
        analysis_schema: analysis_schema 설정(dict)

    Returns:
        CategoryRoutingResult: 분류 결과 + schema_id
    """

    now = datetime.now(timezone.utc).isoformat()
    t = ticker.upper().strip()

    version = str(taxonomy.get("version", "taxonomy_unknown"))
    asset_type = (security_master.asset_type_hint or taxonomy.get("asset_type_fallback") or "Equity").strip()

    evidence: list[ClassificationEvidence] = []
    domain: str = "Others"  # 안전한 기본값

    # 1) deterministic ticker override
    overrides = taxonomy.get("ticker_domain_overrides", {}) or {}
    if t in overrides:
        domain = str(overrides[t]).strip()
        evidence.append(
            ClassificationEvidence(
                source="taxonomy",
                field="ticker_domain_overrides",
                rule="deterministic_override",
                matched_text=f"{t}->{domain}",
                ts=now,
            )
        )
        candidates = [{"domain_category": domain, "score": 1.0}, {"domain_category": "Others", "score": 0.0}]
        schema_id = _domain_to_schema_id(domain=domain, analysis_schema=analysis_schema)
        return CategoryRoutingResult(
            asset_type=asset_type,
            domain_category=domain,
            confidence=0.95,
            candidates_topk=candidates[:3],
            evidence=[e.__dict__ for e in evidence],
            version=version,
            analysis_schema_id=schema_id,
        )

    # 2) keyword scoring (taxonomy lookup + text-based)
    domain_cfg = taxonomy.get("domain_categories", {}) or {}
    desc = (security_master.business_description or "").lower()
    name = (security_master.name or "").lower()
    news_bag = " ".join([str(x) for x in (news_texts or []) if str(x).strip()]).lower()
    bag = f"{t.lower()} {name} {desc} {news_bag}"

    scores: dict[str, float] = {}
    for domain, cfg in domain_cfg.items():
        if domain == "Others":
            continue
        keywords = [str(x).lower() for x in (cfg or {}).get("keywords", [])]
        tickers = {str(x).upper() for x in (cfg or {}).get("tickers", [])}

        score = 0.0
        if t in tickers:
            score += 0.7
            evidence.append(
                ClassificationEvidence(
                    source="taxonomy",
                    field=f"domain_categories.{domain}.tickers",
                    rule="ticker_in_list",
                    matched_text=t,
                    ts=now,
                )
            )

        kw_hits = [kw for kw in keywords if kw and kw in bag]
        if kw_hits:
            score += min(0.3, 0.05 * float(len(kw_hits)))
            evidence.append(
                ClassificationEvidence(
                    source="security_master",
                    field="business_description",
                    rule=f"keyword_match:{domain}",
                    matched_text=", ".join(kw_hits[:5]),
                    ts=now,
                )
            )

        # 뉴스 텍스트에 대한 추가 근거(있을 때만)
        if news_bag:
            news_kw_hits = [kw for kw in keywords if kw and kw in news_bag]
            if news_kw_hits:
                score += min(0.2, 0.05 * float(len(news_kw_hits)))
                evidence.append(
                    ClassificationEvidence(
                        source="internal_news",
                        field="title/summary/body",
                        rule=f"keyword_match_news:{domain}",
                        matched_text=", ".join(news_kw_hits[:5]),
                        ts=now,
                    )
                )

        # topics 필드 직접 활용(뉴스에 섹터/산업 태그가 있는 경우)
        if news_items:
            for item in news_items:
                item_topics = [str(x).lower().strip() for x in (getattr(item, "topics", []) or [])]
                domain_lower = domain.lower()
                # topics에 카테고리명이 직접 포함되거나, 키워드 매칭
                if domain_lower in item_topics or any(kw in item_topics for kw in keywords):
                    score += 0.3  # topics 필드는 강한 신호
                    evidence.append(
                        ClassificationEvidence(
                            source="internal_news",
                            field="topics",
                            rule=f"topic_tag_match:{domain}",
                            matched_text=f"news_id={getattr(item, 'id', 'unknown')}, topics={item_topics}",
                            ts=now,
                        )
                    )
                    break  # 첫 매칭만 기록

        # 회사명 매칭: security_master.name과 뉴스 텍스트에서 회사명 찾기
        if name and name.strip() and news_bag:
            # 간단한 회사명 매칭(띄어쓰기 제거 후 부분 문자열 매칭)
            name_clean = name.replace(" ", "").replace(".", "").replace(",", "")
            news_bag_clean = news_bag.replace(" ", "").replace(".", "").replace(",", "")
            if name_clean and len(name_clean) >= 3 and name_clean in news_bag_clean:
                score += 0.15
                evidence.append(
                    ClassificationEvidence(
                        source="internal_news",
                        field="company_name_match",
                        rule="company_name_in_news",
                        matched_text=f"name={name} found in news text",
                        ts=now,
                    )
                )

        if score > 0:
            scores[domain] = score

    if not scores:
        # 4) Corpus-based fallback: 도메인 전체 키워드로 넓게 매칭 시도
        corpus_parts: list[str] = []
        if security_master.business_description:
            corpus_parts.append(security_master.business_description.lower())
        if news_texts:
            corpus_parts.extend([t.lower() for t in news_texts if t])
        corpus = " ".join(corpus_parts)

        fallback_scores: dict[str, float] = {}
        for dom, cfg in domain_cfg.items():
            if dom == "Others":
                continue
            kw_list = cfg.get("keywords") or []
            hit = sum(1 for kw in kw_list if kw and kw.lower() in corpus)
            if hit > 0:
                sc = min(0.4, hit * 0.05)  # 간단 가중치
                fallback_scores[dom] = sc
                evidence.append(
                    ClassificationEvidence(
                        source="fallback_corpus",
                        field="business_description+news",
                        rule=f"keywords_hit:{hit}",
                        matched_text=f"{dom} hits={hit}",
                        ts=now,
                    )
                )

        if fallback_scores:
            scores.update(fallback_scores)
        else:
            # 5) Others fallback + reason
            evidence.append(
                ClassificationEvidence(
                    source="router",
                    field="fallback",
                    rule="no_match",
                    matched_text="No deterministic/taxonomy/text match; fallback to Others",
                    ts=now,
                )
            )
            domain = "Others"
            schema_id = _domain_to_schema_id(domain=domain, analysis_schema=analysis_schema)
            return CategoryRoutingResult(
                asset_type=asset_type,
                domain_category=domain,
                confidence=0.2,
                candidates_topk=[{"domain_category": "Others", "score": 0.2}],
                evidence=[e.__dict__ for e in evidence],
                version=version,
                analysis_schema_id=schema_id,
            )

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    domain = ranked[0][0]
    best = float(ranked[0][1])
    second = float(ranked[1][1]) if len(ranked) > 1 else 0.0
    margin = best - second

    candidates = [{"domain_category": d, "score": float(s)} for d, s in ranked[:3]]

    if not evidence:
        evidence.append(
            ClassificationEvidence(
                source="router",
                field="fallback",
                rule="score_without_evidence",
                matched_text="Score computed but no evidence captured; forced evidence",
                ts=now,
            )
        )

    # 상대적 margin 기반 confidence (sigmoid)
    alpha = 4.0  # margin 민감도
    conf = 1.0 / (1.0 + math.exp(-alpha * margin))
    conf = max(0.2, min(0.95, conf))

    schema_id = _domain_to_schema_id(domain=domain, analysis_schema=analysis_schema)
    return CategoryRoutingResult(
        asset_type=asset_type,
        domain_category=domain,
        confidence=conf,
        candidates_topk=candidates,
        evidence=[e.__dict__ for e in evidence],
        version=version,
        analysis_schema_id=schema_id,
    )


def _domain_to_schema_id(*, domain: str, analysis_schema: dict[str, Any]) -> str:
    """도메인 카테고리에서 스키마 ID를 찾습니다(없으면 Default_v1)."""

    m = analysis_schema.get("domain_to_schema", {}) or {}
    return str(m.get(domain, m.get("Others", "Default_v1")))


