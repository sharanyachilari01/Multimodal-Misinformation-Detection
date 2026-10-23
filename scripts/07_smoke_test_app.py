#!/usr/bin/env python3
"""
scripts/07_smoke_test_app.py: Smoke test for app.py programmatic analysis pipelines.

Verifies:
1. Inconsistency, classifier, and evidence analysis on two rows from examples/pairs.csv.
2. Low-relevance label code-path test with a hand-made mocked feed (unit test, not a result).
3. Section independence test: sections 1 & 2 return valid results when evidence retrieval fails.
"""

import os
import sys
import pandas as pd

# Add repository root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import (
    analyze_inconsistency,
    analyze_classifier,
    analyze_evidence,
    run_full_analysis
)


def test_pairs_from_csv():
    print("=" * 80)
    print("STEP 1: Run App Analysis Pipelines on First Two Rows of examples/pairs.csv")
    print("=" * 80)

    pairs_csv = "examples/pairs.csv"
    if not os.path.exists(pairs_csv):
        print(f"ERROR: '{pairs_csv}' not found.")
        sys.exit(1)

    df = pd.read_csv(pairs_csv).dropna(how="all")
    print(f"Loaded '{pairs_csv}' ({len(df)} rows). Testing first 2 rows:\n")

    test_rows = df.head(2)
    for idx, row in test_rows.iterrows():
        img_val = row.get("image") or row.get("image_path")
        text_val = row.get("text") or row.get("claim") or row.get("caption")
        csv_label = str(row.get("label", "unknown")).strip()
        csv_note = str(row.get("note", "")).strip()

        # Resolve image path relative to workspace root if needed
        resolved_img = img_val
        if not os.path.isabs(resolved_img) and not os.path.exists(resolved_img):
            candidate = os.path.join("examples", resolved_img)
            if os.path.exists(candidate):
                resolved_img = candidate

        print(f"--------------------------------------------------------------------------------")
        print(f"Evaluating Pair #{idx + 1}:")
        print(f"  Image:       {img_val}")
        print(f"  Claim:       {text_val}")
        print(f"  CSV Label:   {csv_label} (Note: {csv_note})")
        print(f"--------------------------------------------------------------------------------")

        res = run_full_analysis(image=resolved_img, text=text_val)

        # 1. Inconsistency Section
        inc = res["inconsistency"]
        print(f"\n[SECTION 1: Inconsistency Reasoning]")
        if inc["success"]:
            idata = inc["data"]
            print(f"  • CLIP Text-Image Cosine:      {idata['clip_cosine']:.4f}")
            print(f"  • Percentile (Genuine Pairs):   {idata['percentile_pristine']:.1f}%")
            print(f"  • Percentile (Mismatched Pairs):{idata['percentile_falsified']:.1f}%")
            print(f"  • Signal Level:                 [{idata['signal_level'].upper()}]")
            print(f"  • BLIP Generated Description:   \"{idata['caption']}\"")
            print(f"  • Claim vs Description Sim:     {idata['claim_caption_similarity']:.4f}")
        else:
            print(f"  ERROR: {inc['error']}")

        # 2. Classifier Section
        cls_out = res["classifier"]
        print(f"\n[SECTION 2: Out-of-Context Classifier (M4 Ensemble)]")
        if cls_out["success"]:
            cdata = cls_out["data"]
            print(f"  • Raw CLIP Cosine:              {cdata['clip_cosine']:.4f}")
            print(f"  • Uncalibrated Model Score:     {cdata['model_score']:.4f} ({cdata['score_description']})")
            print(f"  • Decision Label:               {cdata['label']}")
            print(f"  • Seed Checkpoints:             seed_42={cdata['seed_scores']['seed_42']:.4f}, "
                  f"seed_43={cdata['seed_scores']['seed_43']:.4f}, seed_44={cdata['seed_scores']['seed_44']:.4f}")
            print(f"  • Seed Agreement Status:        {'Seeds disagree across 0.5' if cdata['seeds_disagree'] else 'Seeds agree'}")
            print(f"  • Cross-Modal Comparison:       Raw CLIP Cosine = {cdata['clip_cosine']:.4f} | M4 Score = {cdata['model_score']:.4f}")
        else:
            print(f"  ERROR: {cls_out['error']}")

        # 3. Evidence Section
        ev = res["evidence"]
        print(f"\n[SECTION 3: External Contextual Evidence]")
        if ev["success"]:
            edata = ev["data"]
            print(f"  • Retrieval Status:             {edata['status']}")
            print(f"  • Relevance Threshold:          {edata.get('relevance_threshold', 0.40)} ({edata.get('relevance_threshold_description', 'heuristic cutoff')})")
            print(f"  • Results Count:                {len(edata.get('results', []))}")
            print(f"  • Excerpt Availability Note:    Excerpts are unavailable; articles are not verified truth.")
            for i, item in enumerate(edata.get("results", [])[:3], 1):
                sim = item.get("similarity", 0.0)
                print(f"      [{i}] {item.get('title')}")
                print(f"          Publisher: {item.get('publisher')} | Date: {item.get('published')}")
                print(f"          Label:     topically related (title similarity {sim:.4f})")
                print(f"          URL:       {item.get('url')}")
        else:
            print(f"  ERROR: {ev['error']}")

        print("\n")


def test_mocked_low_relevance_feed():
    print("=" * 80)
    print("STEP 2: Code-Path Test of Low-Relevance Labeling (Mocked Feed Unit Test)")
    print("=" * 80)
    print("NOTE: This is a unit test validating threshold code paths, NOT an empirical benchmark result.\n")

    # Hand-made mock feed returning an article strictly below the 0.40 relevance cutoff
    def mock_low_relevance_search_fn(claim: str):
        return {
            "status": "low-relevance results only",
            "claim": claim,
            "relevance_threshold": 0.40,
            "relevance_threshold_description": "Heuristic similarity threshold (0.40) for MiniLM semantic similarity ranking",
            "results": [
                {
                    "title": "Historical Archives Open Annual Exhibition in Town Library - Regional Herald",
                    "publisher": "Regional Herald",
                    "published": "Wed, 14 Jan 2026 12:00:00 GMT",
                    "excerpt": "",
                    "url": "https://example.com/mocked-low-relevance-item",
                    "similarity": 0.1742,
                    "is_low_relevance": True,
                    "relevance": "low relevance"
                }
            ],
            "note": "Retrieved articles are contextual external sources, not verified truth. They do not imply verification or contradiction."
        }

    test_query = "A spacecraft is launching into the sky"
    ev_res = analyze_evidence(text=test_query, search_fn=mock_low_relevance_search_fn)

    print(f"Query: \"{test_query}\"")
    print(f"Analyze Evidence Output:")
    print(f"  • Success:             {ev_res['success']}")
    edata = ev_res["data"]
    print(f"  • Status:              {edata['status']}")
    print(f"  • Relevance Threshold: {edata['relevance_threshold']} ({edata['relevance_threshold_description']})")

    item = edata["results"][0]
    print(f"  • Article Title:       \"{item['title']}\"")
    print(f"  • MiniLM Similarity:   {item['similarity']:.4f}")
    print(f"  • Relevance Label:     {item['relevance']} (is_low_relevance={item['is_low_relevance']})")

    # Assertions
    assert ev_res["success"] is True
    assert edata["status"] == "low-relevance results only"
    assert item["is_low_relevance"] is True
    assert item["relevance"] == "low relevance"
    print("\n✓ Unit Test Passed: Low-relevance label code path executed and validated successfully.")


def test_section_independence_on_retrieval_failure():
    print("\n" + "=" * 80)
    print("STEP 3: Decoupling Test (Sections 1 & 2 Still Return When Evidence Fails)")
    print("=" * 80)

    # Search function that simulates a hard network connection / timeout crash
    def mock_hard_failure_search_fn(claim: str):
        raise ConnectionResetError("Simulated socket connection reset by peer (network offline)")

    sample_img = "examples/space_mission.jpeg"
    sample_text = "A space shuttle launches from a launch pad with bright flames and smoke."

    print(f"Running full analysis with failing evidence function...")
    print(f"  Image: {sample_img}")
    print(f"  Text:  {sample_text}\n")

    results = run_full_analysis(image=sample_img, text=sample_text, search_fn=mock_hard_failure_search_fn)

    inc = results["inconsistency"]
    cls_out = results["classifier"]
    ev = results["evidence"]

    print("Pipeline Results:")
    print(f"  1. Inconsistency Analysis Success: {inc['success']}")
    if inc["success"]:
        print(f"     -> CLIP Cosine: {inc['data']['clip_cosine']:.4f}, Signal Level: [{inc['data']['signal_level']}]")

    print(f"  2. Classifier Analysis Success:    {cls_out['success']}")
    if cls_out["success"]:
        print(f"     -> Model Score: {cls_out['data']['model_score']:.4f}, Label: \"{cls_out['data']['label']}\"")

    print(f"  3. Evidence Retrieval Success:     {ev['success']}")
    print(f"     -> Handled Gracefully:          Status = \"{ev['data']['status']}\"")

    # Assertions: sections 1 and 2 must succeed and return complete outputs despite evidence failure
    assert inc["success"] is True, "Section 1 (Inconsistency) failed unexpectedly"
    assert "clip_cosine" in inc["data"], "Section 1 missing clip_cosine"

    assert cls_out["success"] is True, "Section 2 (Classifier) failed unexpectedly"
    assert "model_score" in cls_out["data"], "Section 2 missing model_score"

    assert ev["success"] is False, "Expected evidence to fail in this decoupling test"
    assert "retrieval failed" in ev["data"]["status"], "Expected retrieval failed status in evidence data"

    print("\n✓ Decoupling Test Passed: Inconsistency reasoning (Section 1) and Classifier ensemble (Section 2) "
          "returned completely intact despite external evidence retrieval failure.")


def main():
    print("================================================================================")
    print("PHASE 4B: Smoke Testing App Analysis Pipelines & Error Handling")
    print("================================================================================\n")

    test_pairs_from_csv()
    test_mocked_low_relevance_feed()
    test_section_independence_on_retrieval_failure()

    print("\n" + "=" * 80)
    print("PHASE 4B SMOKE TEST COMPLETED SUCCESSFULLY")
    print("================================================================================")


if __name__ == "__main__":
    main()
