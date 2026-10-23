#!/usr/bin/env python3
"""
scripts/06_test_phase4a.py: Validation script for Phase 4A components.
Tests:
1. Live multimodal classification (predict) on two examples from examples/pairs.csv.
2. External evidence retrieval (search_evidence) on two real claims and one nonsense query.
3. Resilience to simulated network failure (timeout=0.001).
"""

import os
import sys
import json
import pandas as pd

# Add repository root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.classify import predict
from src.evidence import search_evidence


def test_classification():
    print("=" * 80)
    print("STEP 1: Multimodal Classification (predict) on examples/pairs.csv")
    print("=" * 80)

    pairs_csv = "examples/pairs.csv"
    if not os.path.exists(pairs_csv):
        print(f"Notice: '{pairs_csv}' does not exist. Skipping classification test as specified.")
        return

    df = pd.read_csv(pairs_csv).dropna(how="all")
    print(f"Loaded '{pairs_csv}' ({len(df)} rows). Testing first 2 pairs:\n")

    test_rows = df.head(2)
    for idx, row in test_rows.iterrows():
        img_val = row.get("image") or row.get("image_path")
        text_val = row.get("text") or row.get("claim") or row.get("caption")
        csv_label = row.get("label", "unknown")
        csv_note = row.get("note", "")

        # Resolve image path relative to examples/ if needed
        if not os.path.isabs(img_val) and not os.path.exists(img_val):
            candidate = os.path.join("examples", img_val)
            if os.path.exists(candidate):
                img_val = candidate

        print(f"--- Pair #{idx + 1} ---")
        print(f"  Image:       {img_val}")
        print(f"  Text:        {text_val}")
        print(f"  CSV Label:   {csv_label} (Note: {csv_note})")

        res = predict(text=text_val, image=img_val)

        print(f"  Prediction Results:")
        print(f"    • Model Score:       {res['model_score']:.4f} ({res['score_description']})")
        print(f"    • Classification:    {res['label']} (Threshold: {res['threshold']})")
        print(f"    • CLIP Cosine:       {res['clip_cosine']:.4f}")
        print(f"    • Seed Scores:       seed_42={res['seed_scores']['seed_42']:.4f}, "
              f"seed_43={res['seed_scores']['seed_43']:.4f}, seed_44={res['seed_scores']['seed_44']:.4f}")
        print(f"    • Limitations:       {res['limitations']}\n")


def test_evidence_retrieval():
    print("=" * 80)
    print("STEP 2: External Evidence Retrieval (search_evidence)")
    print("=" * 80)

    test_queries = [
        ("Claim 1 (Real)", "Chandrayaan-3 landed near the Moon's south pole"),
        ("Claim 2 (Real)", "James Webb Space Telescope captures deep field image"),
        ("Query 3 (Nonsense)", "zxvbnm98745lkjhgfdsapoiuytrewq nonexistent event xyz 12345")
    ]

    for category, query in test_queries:
        print(f"\n--- {category}: \"{query}\" ---")
        output = search_evidence(claim=query, max_results=5, timeout=8.0)
        print(f"  Status:  {output['status']}")
        print(f"  Note:    {output['note']}")
        print(f"  Results: {len(output['results'])} items retrieved")

        for i, item in enumerate(output["results"]):
            relevance_flag = "[LOW RELEVANCE]" if item["is_low_relevance"] else "[RELATED]"
            print(f"\n    [{i+1}] {relevance_flag} {item['title']}")
            print(f"        Publisher:  {item['publisher']} | Published: {item['published']}")
            print(f"        Similarity: {item['similarity']:.4f}")
            print(f"        Excerpt:    {item['excerpt']}")
            print(f"        URL:        {item['url']}")


def test_network_failure_resilience():
    print("\n" + "=" * 80)
    print("STEP 3: Network Failure Simulation (timeout=0.001)")
    print("=" * 80)

    claim = "Chandrayaan-3 landed near the Moon's south pole"
    print(f"Testing simulated network timeout on query: \"{claim}\"")
    output = search_evidence(claim=claim, timeout=0.001)

    print(f"  Status returned: \"{output['status']}\"")
    print(f"  Results count:   {len(output['results'])}")
    print(f"  Note:            \"{output['note']}\"")

    assert output["status"].startswith("retrieval failed"), (
        f"Expected status to start with 'retrieval failed', got: {output['status']}"
    )
    assert len(output["results"]) == 0, "Expected empty results on retrieval failure"
    print("✓ Assert passed: Simulated network timeout returned clean failure status without raising unhandled exceptions.")


def test_predict_works_when_retrieval_fails():
    print("\n" + "=" * 80)
    print("STEP 4: Predict Works When Retrieval Fails (Non-Blocking Decoupling)")
    print("=" * 80)

    test_claim = "Chandrayaan-3 landed near the Moon's south pole"
    test_image = "examples/image.jpg"
    print(f"Testing claim: \"{test_claim}\" with image: \"{test_image}\"")

    # 1. Force external retrieval to fail with simulated timeout
    print("  1. Simulating external retrieval failure (timeout=0.0001) ...")
    retrieval_res = search_evidence(claim=test_claim, timeout=0.0001)
    print(f"     Retrieval status: \"{retrieval_res['status']}\"")
    assert retrieval_res["status"].startswith("retrieval failed")

    # 2. Confirm that classifier predict() succeeds independently
    print("  2. Executing predict() to confirm classifier runs unhindered ...")
    pred_res = predict(text=test_claim, image=test_image)
    print(f"     Classification Succeeded:")
    print(f"       • Model Score:    {pred_res['model_score']:.4f} ({pred_res['score_description']})")
    print(f"       • Label:          {pred_res['label']}")
    print(f"       • CLIP Cosine:    {pred_res['clip_cosine']:.4f}")
    assert "model_score" in pred_res
    assert pred_res["label"] in ["likely out-of-context pairing", "no out-of-context signal"]
    print("✓ Confirmed: Multimodal classification operates independently and succeeds even when retrieval fails.")


def main():
    print("================================================================================")
    print("PHASE 4A: Classifier and External Retrieval Integration Test")
    print("================================================================================\n")

    test_classification()
    test_evidence_retrieval()
    test_network_failure_resilience()
    test_predict_works_when_retrieval_fails()

    print("\n" + "=" * 80)
    print("PHASE 4A TEST COMPLETED SUCCESSFULLY")
    print("================================================================================")


if __name__ == "__main__":
    main()
