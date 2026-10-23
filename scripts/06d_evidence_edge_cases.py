#!/usr/bin/env python3
"""
scripts/06d_evidence_edge_cases.py: Test search_evidence on three edge-case queries:
(a) "something happened in the city yesterday"
(b) "Chandrayaan-3 landed on Mars"
(c) "a dog running across a grassy park"
"""

import os
import sys

# Add repository root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evidence import search_evidence


def main():
    print("=" * 80)
    print("PHASE 4A.3: Evidence Retrieval Edge-Case Analysis")
    print("=" * 80)

    queries = [
        ("(a) Vague Query", "something happened in the city yesterday"),
        ("(b) Contrafactual Claim", "Chandrayaan-3 landed on Mars"),
        ("(c) Everyday Scene", "a dog running across a grassy park")
    ]

    for label, q in queries:
        print("\n" + "=" * 80)
        print(f"Query {label}: \"{q}\"")
        print("=" * 80)

        res = search_evidence(claim=q, max_results=5, timeout=8.0, relevance_threshold=0.40)

        print(f"Status:              {res['status']}")
        print(f"Relevance Threshold: {res['relevance_threshold']} ({res['relevance_threshold_description']})")
        print(f"Total Results:       {len(res['results'])}")
        print(f"Note:                {res['note']}")

        if not res["results"]:
            print("  [No candidate results found]")
            continue

        print("\nCandidate Results:")
        for idx, item in enumerate(res["results"], 1):
            title = item.get("title")
            sim = item.get("similarity")
            rel = item.get("relevance")
            pub = item.get("publisher")
            pub_date = item.get("published")
            excerpt = item.get("excerpt")
            excerpt_str = f"\"{excerpt}\"" if excerpt else "[EMPTY - summary repeats headline/source]"

            print(f"  [{idx}] Title:      {title}")
            print(f"      Similarity: {sim:.4f}")
            print(f"      Relevance:  {rel}")
            print(f"      Publisher:  {pub} | Published: {pub_date}")
            print(f"      Excerpt:    {excerpt_str}")
            print(f"      URL:        {item.get('url')}\n")

    print("=" * 80)
    print("EDGE-CASE EVALUATION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
