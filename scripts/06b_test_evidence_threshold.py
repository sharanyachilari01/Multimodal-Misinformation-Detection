#!/usr/bin/env python3
"""
scripts/06b_test_evidence_threshold.py: Test evidence retrieval thresholds,
heuristic descriptions, excerpt deduplication, and relevance tagging.
"""

import os
import sys

# Add repository root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evidence import search_evidence


def run_tests():
    print("=" * 80)
    print("PHASE 4A.2: Testing Evidence Retrieval Thresholds & Deduplication")
    print("=" * 80)

    queries = [
        ("Query 1 (Real / Well-Known)", "Chandrayaan-3 landed near the Moon's south pole"),
        ("Query 2 (Real / Well-Known)", "James Webb Space Telescope captures deep field image"),
        ("Query 3 (Nonsense / Synthetic)", "zxvbnm98745lkjhgfdsapoiuytrewq nonexistent event xyz 12345")
    ]

    for label, q in queries:
        print("\n" + "=" * 80)
        print(f"{label}: \"{q}\"")
        print("=" * 80)

        output = search_evidence(claim=q, max_results=5, timeout=8.0, relevance_threshold=0.40)

        print(f"Status:                    {output['status']}")
        print(f"Relevance Threshold:       {output['relevance_threshold']} ({output['relevance_threshold_description']})")
        print(f"Note:                      {output['note']}")
        print(f"Results Retrieved:         {len(output['results'])}")

        if not output["results"]:
            print("  (No candidate articles retrieved)")
            continue

        for idx, item in enumerate(output["results"], 1):
            rel_label = item.get("relevance", "unknown")
            is_low = item.get("is_low_relevance", False)
            excerpt_str = item.get("excerpt", "")
            excerpt_display = f"\"{excerpt_str}\"" if excerpt_str else "[EMPTY - omitted because it repeated title/publisher]"

            print(f"\n  [{idx}] Title:      {item['title']}")
            print(f"      Publisher:  {item['publisher']} | Published: {item['published']}")
            print(f"      Similarity: {item['similarity']:.4f}")
            print(f"      Relevance:  {rel_label} (is_low_relevance={is_low})")
            print(f"      Excerpt:    {excerpt_display}")
            print(f"      URL:        {item['url']}")

    print("\n" + "=" * 80)
    print("EVIDENCE RETRIEVAL THRESHOLD TESTS COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    run_tests()
