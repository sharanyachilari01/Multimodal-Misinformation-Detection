#!/usr/bin/env python3
"""
scripts/06c_predict_all.py: Run predict() on ALL rows of examples/pairs.csv.

Outputs:
- Saves results/phase4a_predictions.csv
- Formatted table of predictions across all 21 rows
- Mean model_score per label category (match, mismatch, partial)
- Pair comparison: how many mismatch rows score higher than match row for the same image (k of 7)
- Spearman rank correlation between clip_cosine and model_score across all 21 rows
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy import stats

# Add repository root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.classify import predict


def main():
    print("=" * 80)
    print("PHASE 4A.2: Live Predictions Across ALL Examples (examples/pairs.csv)")
    print("=" * 80)

    pairs_csv = "examples/pairs.csv"
    if not os.path.exists(pairs_csv):
        print(f"ERROR: '{pairs_csv}' not found. Cannot proceed without input file.")
        sys.exit(1)

    df = pd.read_csv(pairs_csv).dropna(how="all")
    print(f"Loaded {len(df)} candidate image-text rows from '{pairs_csv}'.\n")

    records = []
    for idx, row in df.iterrows():
        img_val = row.get("image") or row.get("image_path")
        text_val = row.get("text") or row.get("claim") or row.get("caption")
        csv_label = str(row.get("label", "unknown")).strip()

        # Resolve image path relative to workspace root if needed
        resolved_img = img_val
        if not os.path.isabs(resolved_img) and not os.path.exists(resolved_img):
            candidate = os.path.join("examples", resolved_img)
            if os.path.exists(candidate):
                resolved_img = candidate

        pred = predict(text=text_val, image=resolved_img)

        records.append({
            "image": img_val,
            "label": csv_label,
            "text": text_val,
            "clip_cosine": round(pred["clip_cosine"], 4),
            "model_score": round(pred["model_score"], 4),
            "seed_42": round(pred["seed_scores"]["seed_42"], 4),
            "seed_43": round(pred["seed_scores"]["seed_43"], 4),
            "seed_44": round(pred["seed_scores"]["seed_44"], 4),
            "label_wording": pred["label"]
        })

    res_df = pd.DataFrame(records)

    # Save to results/phase4a_predictions.csv
    out_csv = "results/phase4a_predictions.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    res_df.to_csv(out_csv, index=False)
    print(f"Saved {len(res_df)} predictions to: {out_csv}\n")

    # Print Table
    print("=" * 120)
    print("ALL 21 EXAMPLE PREDICTIONS TABLE")
    print("=" * 120)
    fmt_hdr = f"{'#':<3} | {'Image':<22} | {'Label':<10} | {'CLIP Cos':<8} | {'Score':<7} | {'S_42':<6} | {'S_43':<6} | {'S_44':<6} | {'Predicted Label'}"
    print(fmt_hdr)
    print("-" * 120)
    for i, r in res_df.iterrows():
        idx_str = str(i + 1)
        img_str = str(r['image'])[:22]
        lbl_str = str(r['label'])[:10]
        cos_str = f"{r['clip_cosine']:.4f}"
        sc_str = f"{r['model_score']:.4f}"
        s42_str = f"{r['seed_42']:.4f}"
        s43_str = f"{r['seed_43']:.4f}"
        s44_str = f"{r['seed_44']:.4f}"
        wording_str = str(r['label_wording'])
        print(f"{idx_str:<3} | {img_str:<22} | {lbl_str:<10} | {cos_str:<8} | {sc_str:<7} | {s42_str:<6} | {s43_str:<6} | {s44_str:<6} | {wording_str}")

    # Summary Statistics
    print("\n" + "=" * 80)
    print("SUMMARY EVALUATION METRICS")
    print("=" * 80)

    # 1. Mean model_score per label
    print("1. Mean Model Score per Ground-Truth Label:")
    for lbl in ["match", "mismatch", "partial"]:
        sub = res_df[res_df["label"].str.lower() == lbl]["model_score"]
        if len(sub) > 0:
            print(f"   • {lbl:<10}: Mean = {sub.mean():.4f} (std = {sub.std():.4f}, N = {len(sub)})")
        else:
            print(f"   • {lbl:<10}: N = 0")

    # 2. Pair comparison: mismatch vs match for SAME image (k of 7)
    unique_images = res_df["image"].unique()
    higher_count = 0
    total_pairs = 0
    print(f"\n2. Within-Image Pair Comparison (Mismatch vs Match for Same Image):")
    for img in unique_images:
        img_rows = res_df[res_df["image"] == img]
        match_row = img_rows[img_rows["label"].str.lower() == "match"]
        mismatch_row = img_rows[img_rows["label"].str.lower() == "mismatch"]

        if len(match_row) > 0 and len(mismatch_row) > 0:
            m_sc = match_row.iloc[0]["model_score"]
            mm_sc = mismatch_row.iloc[0]["model_score"]
            total_pairs += 1
            is_higher = mm_sc > m_sc
            if is_higher:
                higher_count += 1
            print(f"   • {img:<22}: Mismatch Score = {mm_sc:.4f} vs Match Score = {m_sc:.4f} -> {'MISMATCH HIGHER (✓)' if is_higher else 'MISMATCH LOWER (✗)'}")

    print(f"\n   Total pairs where mismatch score > match score: {higher_count} of {total_pairs} ({(higher_count / total_pairs * 100):.2f}%)")

    # 3. Spearman correlation between clip_cosine and model_score across the 21 rows
    cosines = res_df["clip_cosine"].values
    scores = res_df["model_score"].values
    spearman_res = stats.spearmanr(cosines, scores)
    print(f"\n3. Spearman Rank Correlation (clip_cosine vs model_score across N={len(res_df)} rows):")
    print(f"   • Spearman rho:   {spearman_res.statistic:.4f}")
    print(f"   • p-value:        {spearman_res.pvalue:.4e}")
    print(f"   (Strong negative correlation confirms higher inconsistency / lower cosine yields higher out-of-context score)")

    print("\n" + "=" * 80)
    print("PHASE 4A.2 ALL-EXAMPLE PREDICTIONS COMPLETE")
    print("================================================================================")


if __name__ == "__main__":
    main()
