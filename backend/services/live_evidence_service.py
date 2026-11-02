"""TruthLens live-source discovery adapters (hardened query and status handling).

Google Fact Check returns existing ClaimReview metadata.
BBC and optional GDELT return news discovery leads, not verification evidence.
Wikidata returns entity matches, not claim-level evidence.
"""
from __future__ import annotations

import hashlib
import os
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

CONNECT_TIMEOUT = 5
READ_TIMEOUT = 35
USER_AGENT = "TruthLensAI/1.0 (academic research prototype)"
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
BBC_SEARCH_URL = "https://www.bbc.com/search"
BBC_BASE_URL = "https://www.bbc.com"
GOOGLE_FACTCHECK_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
WIKIDATA_SEARCH_URL = "https://www.wikidata.org/w/api.php"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_id(prefix: str, value: str) -> str:
    return f"{prefix}-{hashlib.sha256(value.encode('utf-8')).hexdigest()[:12]}"


def _safe_text(value: Any, limit: int = 1200) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip()[:limit]


def _get_json(url: str, params: dict, *, headers: dict | None = None,
              timeout: tuple[int, int] = (CONNECT_TIMEOUT, READ_TIMEOUT)) -> dict:
    response = requests.get(
        url,
        params=params,
        headers={"User-Agent": USER_AGENT, **(headers or {})},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("API returned an unexpected JSON structure.")
    return payload


def _extract_entity_query(claim: str) -> str:
    """Extract a short capitalized entity phrase; fall back to compact keywords."""
    text = _safe_text(claim, 500)
    # Multi-word proper names such as Narendra Modi, New Delhi, World Health Organization.
    matches = re.findall(
        r"\b(?:[A-Z][a-z]+|[A-Z]{2,})(?:\s+(?:[A-Z][a-z]+|[A-Z]{2,})){0,4}\b",
        text,
    )
    stop = {"The", "This", "That", "India", "Prime Minister"}
    candidates = [m.strip() for m in matches if m.strip() not in stop]
    # Prefer the longest candidate; for the supplied example this yields Narendra Modi.
    if candidates:
        candidates.sort(key=lambda x: (len(x.split()), len(x)), reverse=True)
        return candidates[0][:100]
    words = re.findall(r"[A-Za-z]{3,}", text)
    return " ".join(words[:5])[:100]


def search_google_fact_checks(claim: str, *, page_size: int = 5) -> list[dict]:
    api_key = os.getenv("GOOGLE_FACT_CHECK_API_KEY", "").strip()
    if not api_key:
        return []
    payload = _get_json(GOOGLE_FACTCHECK_URL, {
        "query": claim[:500], "languageCode": "en",
        "pageSize": max(1, min(int(page_size), 10)), "key": api_key,
    })
    results = []
    for claim_item in payload.get("claims", []) or []:
        claim_text = _safe_text(claim_item.get("text"), 500)
        for review in claim_item.get("claimReview", []) or []:
            publisher = review.get("publisher") or {}
            url = _safe_text(review.get("url"), 1000)
            if not url or urlparse(url).scheme != "https":
                continue
            results.append({
                "id": _stable_id("GFACT", url),
                "claim_text": claim_text,
                "title": _safe_text(review.get("title"), 300) or "Fact-check review",
                "publisher": _safe_text(publisher.get("name"), 160) or "Unknown publisher",
                "rating": _safe_text(review.get("textualRating"), 160) or "Rating not supplied",
                "review_url": url,
                "review_date": _safe_text(review.get("reviewDate"), 80) or None,
                "source_type": "fact_check_review", "retrieved_at": _now(),
                "interpretation_note": "Inspect the original review, date, scope and cited sources before relying on its rating.",
            })
    return results



def _get_html(url: str, params: dict, *, timeout: tuple[int, int] = (CONNECT_TIMEOUT, 15)) -> str:
    """Fetch an HTML page for best-effort public news discovery."""
    response = requests.get(
        url,
        params=params,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.text


def _bbc_article_url(href: str) -> str:
    url = urljoin(BBC_BASE_URL, href)
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in {"www.bbc.com", "bbc.com", "www.bbc.co.uk", "bbc.co.uk"}:
        return ""
    if "/news/articles/" not in parsed.path:
        return ""
    return url


def search_bbc_news(claim: str, *, max_records: int = 8) -> list[dict]:
    """Scrape BBC search cards as discovery leads, never as verification evidence.

    BBC does not expose a documented public search API here; this adapter is
    best-effort and may need updates if BBC changes its HTML structure.
    """
    query = _safe_text(claim, 300)
    if not query:
        return []

    html = _get_html(BBC_SEARCH_URL, {"q": query}, timeout=(CONNECT_TIMEOUT, 15))
    soup = BeautifulSoup(html, "html.parser")
    main = soup.find("main")
    if main is None:
        return []

    results: list[dict] = []
    seen_urls: set[str] = set()
    cards = main.select('[data-testid="newport-card"]')
    # Fallback for a possible BBC markup change while retaining URL validation.
    if not cards:
        cards = main.select('a[href*="/news/articles/"]')

    for card in cards:
        anchor = card if getattr(card, "name", None) == "a" else None
        if anchor is None:
            anchor = next(
                (a for a in card.find_all("a", href=True) if _bbc_article_url(a.get("href", ""))),
                None,
            )
        if anchor is None:
            continue

        url = _bbc_article_url(anchor.get("href", ""))
        if not url or url in seen_urls:
            continue

        headline_node = card.select_one('[data-testid="card-headline"]') if hasattr(card, "select_one") else None
        title = _safe_text(headline_node.get_text(" ", strip=True) if headline_node else anchor.get_text(" ", strip=True), 300)
        if not title:
            continue

        date_node = card.find("time") if hasattr(card, "find") else None
        published_at = ""
        if date_node is not None:
            published_at = _safe_text(date_node.get("datetime") or date_node.get_text(" ", strip=True), 80)

        summary = ""
        if hasattr(card, "find_all"):
            for paragraph in card.find_all("p"):
                candidate = _safe_text(paragraph.get_text(" ", strip=True), 600)
                if candidate and candidate != title:
                    summary = candidate
                    break

        # BBC cards sometimes render a visible relative date without a <time> node.
        if not published_at and hasattr(card, "get_text"):
            card_text = _safe_text(card.get_text(" ", strip=True), 1200)
            date_match = re.search(
                r"\b(?:\d{1,2}\s+[A-Z][a-z]{2}\s+\d{4}|\d+\s+(?:mins?|hours?|hrs?|days?|weeks?)\s+ago|\d+\s+hours?\s+ago)\b",
                card_text,
                flags=re.IGNORECASE,
            )
            if date_match:
                published_at = date_match.group(0)

        seen_urls.add(url)
        results.append({
            "id": _stable_id("BBC", url),
            "title": title,
            "summary": summary or None,
            "url": url,
            "domain": urlparse(url).hostname or "www.bbc.com",
            "published_at": published_at or None,
            "source_name": "BBC News",
            "source_type": "news_discovery",
            "retrieved_at": _now(),
            "interpretation_note": (
                "BBC search result is a discovery lead, not proof that the claim is true or false. "
                "Open the original article and check its date, context and cited sources."
            ),
        })
        if len(results) >= max(1, min(int(max_records), 20)):
            break

    return results

def search_gdelt_news(claim: str, *, max_records: int = 8) -> list[dict]:
    """Search with compact unquoted keywords; return discovery leads only."""
    text = _safe_text(claim, 500)
    if not text:
        return []
    # Remove common grammatical filler and avoid brittle exact-phrase searches.
    tokens = re.findall(r"[A-Za-z0-9]+", text)
    stop = {"is", "are", "was", "were", "the", "a", "an", "in", "on", "of", "to", "for", "as", "since", "during", "2026"}
    keywords = [t for t in tokens if t.lower() not in stop]
    query = " ".join(keywords[:8])
    if not query:
        return []
    payload = _get_json(GDELT_URL, {
        "query": query,
        "mode": "ArtList", "format": "json",
        "maxrecords": max(1, min(int(max_records), 25)),
        "sort": "HybridRel",
    }, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
    results = []
    for article in payload.get("articles", []) or []:
        title = _safe_text(article.get("title"), 300)
        url = _safe_text(article.get("url"), 1000)
        if not title or not url or urlparse(url).scheme != "https":
            continue
        results.append({
            "id": _stable_id("GDELT", url), "title": title, "url": url,
            "domain": _safe_text(article.get("domain"), 200),
            "published_at": _safe_text(article.get("seendate"), 40) or None,
            "language": _safe_text(article.get("language"), 40) or None,
            "source_country": _safe_text(article.get("sourcecountry"), 80) or None,
            "source_type": "news_discovery", "retrieved_at": _now(),
            "interpretation_note": "News coverage is a discovery lead, not evidence by itself and not proof that the claim is true or false.",
        })
    return results


def search_wikidata_entities(claim: str, *, limit: int = 5) -> list[dict]:
    query = _extract_entity_query(claim)
    if not query:
        return []
    payload = _get_json(WIKIDATA_SEARCH_URL, {
        "action": "wbsearchentities", "search": query,
        "language": "en", "uselang": "en", "type": "item",
        "limit": max(1, min(int(limit), 10)), "format": "json",
    }, headers={"Accept": "application/json"}, timeout=(CONNECT_TIMEOUT, 20))
    results = []
    for entity in payload.get("search", []) or []:
        entity_id = _safe_text(entity.get("id"), 40)
        label = _safe_text(entity.get("label"), 200)
        if not entity_id or not label:
            continue
        results.append({
            "id": _stable_id("WD", entity_id), "entity_id": entity_id,
            "label": label, "description": _safe_text(entity.get("description"), 500),
            "url": f"https://www.wikidata.org/wiki/{entity_id}",
            "source_type": "knowledge_graph_entity", "retrieved_at": _now(),
            "search_query": query,
            "interpretation_note": "Entity search identifies related entities; it does not establish whether the submitted claim is true.",
        })
    return results


def search_live_sources(claim: str) -> dict:
    if not isinstance(claim, str) or len(claim.strip()) < 8:
        raise ValueError("Please provide a claim with at least 8 characters.")
    output = {
        "query": claim.strip(), "retrieved_at": _now(),
        "fact_checks": [], "news_leads": [], "knowledge_graph_entities": [],
        "provider_errors": {}, "provider_status": {},
        "limitations": [
            "Search results are not automatically proof.",
            "Fact-check ratings require inspection of original review, date and scope.",
            "BBC and GDELT results are news discovery leads, not verified evidence.",
            "Wikidata entity matches are not claim-level verification.",
            "No result does not mean a claim is false.",
        ],
    }
    providers = [
        ("google_fact_check", "fact_checks", search_google_fact_checks),
        ("bbc_search", "news_leads", search_bbc_news),
        ("wikidata", "knowledge_graph_entities", search_wikidata_entities),
    ]
    # GDELT is opt-in because it is unreachable from some networks, including
    # the user's current WSL environment. It remains available for later testing.
    gdelt_enabled = os.getenv("ENABLE_GDELT", "false").strip().lower() in {"1", "true", "yes", "on"}
    if gdelt_enabled:
        providers.append(("gdelt", "news_leads", search_gdelt_news))
    else:
        output["provider_status"]["gdelt"] = "SKIPPED — disabled (set ENABLE_GDELT=true to enable)"

    for name, key, provider in providers:
        if name == "google_fact_check" and not os.getenv("GOOGLE_FACT_CHECK_API_KEY", "").strip():
            output["provider_status"][name] = "SKIPPED — API key not configured"
            continue
        try:
            provider_results = provider(claim)
            if key == "news_leads":
                output[key].extend(provider_results)
            else:
                output[key] = provider_results
            output["provider_status"][name] = f"SUCCESS — {len(provider_results)} result(s)"
        except requests.RequestException as exc:
            output["provider_errors"][name] = f"Network/API error: {type(exc).__name__}: {str(exc)[:160]}"
            output["provider_status"][name] = "ERROR — request failed"
        except Exception as exc:
            output["provider_errors"][name] = f"Provider error: {type(exc).__name__}: {str(exc)[:160]}"
            output["provider_status"][name] = "ERROR — provider response could not be processed"
    return output
