"""
src/evidence.py: External evidence retrieval using public Google News RSS.

Retrieves public news items related to input claims, strips HTML formatting,
ranks articles by MiniLM semantic similarity to the claim title, and marks
relevance levels.

Operational safeguards:
- Non-blocking external retrieval: does not throw unhandled exceptions.
- Handles timeouts, DNS/network failures, empty feeds, and malformed responses.
- Explicit non-endorsement: retrieved articles are contextual evidence, not verified truth.
"""

import html
import re
import urllib.parse
from typing import Any, Dict, List, Optional
import requests
import feedparser

from src.inconsistency import load_models


def _normalize_text_for_comparison(text: str) -> str:
    """Normalizes text by lowercasing, stripping punctuation, and collapsing whitespace."""
    if not text:
        return ""
    no_punct = re.sub(r"[^\w\s]", " ", text.lower())
    return re.sub(r"\s+", " ", no_punct).strip()


def _clean_html(raw_html: Optional[str], title: str = "", publisher: str = "") -> str:
    """
    Strips HTML tags, decodes entities, collapses whitespace, and returns at most 300 characters.
    If the excerpt only repeats the title and/or publisher without new content, returns an empty string.
    """
    if not raw_html:
        return ""
    # Strip HTML tags
    cleaned = re.sub(r"<[^>]+>", " ", raw_html)
    # Decode HTML entities (&amp;, &quot;, &#39;, etc.)
    cleaned = html.unescape(cleaned)
    # Collapse multiple whitespaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Compare normalized versions
    norm_cleaned = _normalize_text_for_comparison(cleaned)
    norm_title = _normalize_text_for_comparison(title)
    norm_pub = _normalize_text_for_comparison(publisher)

    title_main = title.rsplit(" - ", 1)[0] if " - " in title else title
    norm_title_main = _normalize_text_for_comparison(title_main)

    # Detect if cleaned excerpt merely duplicates the headline / publisher metadata
    if not norm_cleaned:
        return ""
    if norm_cleaned in (norm_title, norm_title_main):
        return ""
    if norm_pub and norm_cleaned in (f"{norm_title_main} {norm_pub}", f"{norm_title} {norm_pub}"):
        return ""

    return cleaned[:300]


def _extract_publisher(entry: Any) -> str:
    """Extracts publisher name from feed entry source or title suffix."""
    source = entry.get("source")
    if isinstance(source, dict) and source.get("title"):
        return str(source["title"]).strip()
    title = entry.get("title", "")
    if " - " in title:
        parts = title.rsplit(" - ", 1)
        if len(parts) == 2 and parts[1].strip():
            return parts[1].strip()
    return "Unknown"


def search_evidence(
    claim: str,
    max_results: int = 5,
    timeout: float = 8.0,
    relevance_threshold: float = 0.40
) -> Dict[str, Any]:
    """
    Searches Google News RSS for articles related to the given claim.

    Parameters:
    -----------
    claim : str
        The textual claim or assertion to investigate.
    max_results : int
        Maximum number of ranked results to return (default: 5).
    timeout : float
        HTTP request timeout in seconds (default: 8.0).
    relevance_threshold : float
        MiniLM cosine similarity cutoff below which an article is flagged as low relevance.

    Returns:
    --------
    dict with:
        - 'status': one of 'related sources found', 'low-relevance results only',
                    'no useful evidence retrieved', or 'retrieval failed: <reason>'
        - 'claim': query string
        - 'relevance_threshold': float, heuristic similarity threshold
        - 'relevance_threshold_description': string describing threshold as a heuristic
        - 'results': list of dicts with title, publisher, published, excerpt, url, similarity, is_low_relevance, relevance
        - 'note': explicit notice that retrieved articles are not verified truth
    """
    note = "Retrieved articles are contextual external sources, not verified truth. They do not imply verification or contradiction."

    if not isinstance(claim, str) or not claim.strip():
        return {
            "status": "no useful evidence retrieved",
            "claim": claim if isinstance(claim, str) else "",
            "relevance_threshold": relevance_threshold,
            "relevance_threshold_description": f"Heuristic similarity threshold ({relevance_threshold:.2f}) for MiniLM semantic similarity ranking",
            "results": [],
            "note": note
        }

    encoded_query = urllib.parse.quote(claim.strip())
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }

    # 1. Fetch RSS feed with robust exception handling
    try:
        response = requests.get(rss_url, headers=headers, timeout=timeout)
        if response.status_code != 200:
            return {
                "status": f"retrieval failed: HTTP {response.status_code}",
                "claim": claim,
                "relevance_threshold": relevance_threshold,
                "relevance_threshold_description": f"Heuristic similarity threshold ({relevance_threshold:.2f}) for MiniLM semantic similarity ranking",
                "results": [],
                "note": note
            }
        feed = feedparser.parse(response.content)
    except requests.exceptions.Timeout:
        return {
            "status": f"retrieval failed: request timed out after {timeout}s",
            "claim": claim,
            "relevance_threshold": relevance_threshold,
            "relevance_threshold_description": f"Heuristic similarity threshold ({relevance_threshold:.2f}) for MiniLM semantic similarity ranking",
            "results": [],
            "note": note
        }
    except requests.exceptions.RequestException as e:
        return {
            "status": f"retrieval failed: network error ({type(e).__name__})",
            "claim": claim,
            "relevance_threshold": relevance_threshold,
            "relevance_threshold_description": f"Heuristic similarity threshold ({relevance_threshold:.2f}) for MiniLM semantic similarity ranking",
            "results": [],
            "note": note
        }
    except Exception as e:
        return {
            "status": f"retrieval failed: unexpected error ({type(e).__name__}: {str(e)})",
            "claim": claim,
            "relevance_threshold": relevance_threshold,
            "relevance_threshold_description": f"Heuristic similarity threshold ({relevance_threshold:.2f}) for MiniLM semantic similarity ranking",
            "results": [],
            "note": note
        }

    # 2. Check for empty or malformed feeds
    entries = getattr(feed, "entries", [])
    if not entries:
        return {
            "status": "no useful evidence retrieved",
            "claim": claim,
            "relevance_threshold": relevance_threshold,
            "relevance_threshold_description": f"Heuristic similarity threshold ({relevance_threshold:.2f}) for MiniLM semantic similarity ranking",
            "results": [],
            "note": note
        }

    # 3. Extract items
    extracted_items: List[Dict[str, Any]] = []
    for entry in entries:
        title = entry.get("title", "").strip()
        if not title:
            continue
        publisher = _extract_publisher(entry)
        published = entry.get("published", "Unknown date")
        summary_raw = entry.get("summary") or entry.get("description") or ""
        excerpt = _clean_html(summary_raw, title=title, publisher=publisher)
        url = entry.get("link", "")
        extracted_items.append({
            "title": title,
            "publisher": publisher,
            "published": published,
            "excerpt": excerpt,
            "url": url
        })

    if not extracted_items:
        return {
            "status": "no useful evidence retrieved",
            "claim": claim,
            "relevance_threshold": relevance_threshold,
            "relevance_threshold_description": f"Heuristic similarity threshold ({relevance_threshold:.2f}) for MiniLM semantic similarity ranking",
            "results": [],
            "note": note
        }

    # 4. Rank by MiniLM cosine similarity between claim and title
    try:
        models = load_models()
        minilm = models["minilm_model"]
        claim_emb = minilm.encode(claim, normalize_embeddings=True)
        title_embs = minilm.encode([item["title"] for item in extracted_items], normalize_embeddings=True)
        similarities = (title_embs @ claim_emb).tolist()
    except Exception as e:
        similarities = [0.0] * len(extracted_items)

    for item, sim in zip(extracted_items, similarities):
        item["similarity"] = float(sim)
        item["is_low_relevance"] = bool(sim < relevance_threshold)
        item["relevance"] = "related" if sim >= relevance_threshold else "low relevance"

    # Sort descending by semantic similarity
    extracted_items.sort(key=lambda x: x["similarity"], reverse=True)
    top_results = extracted_items[:max_results]

    # 5. Determine overall retrieval status
    has_relevant = any(not item["is_low_relevance"] for item in top_results)
    if has_relevant:
        status = "related sources found"
    else:
        status = "low-relevance results only"

    return {
        "status": status,
        "claim": claim,
        "relevance_threshold": relevance_threshold,
        "relevance_threshold_description": f"Heuristic similarity threshold ({relevance_threshold:.2f}) for MiniLM semantic similarity ranking",
        "results": top_results,
        "note": note
    }
