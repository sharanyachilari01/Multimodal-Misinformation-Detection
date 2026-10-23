#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
import torch
from PIL import Image

# Ensure src module can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inconsistency import explain, clip_reference_stats, clip_similarity, load_models

def main():
    print("=" * 80)
    print("PHASE 3B: Testing Live Multimodal Inconsistency Module")
    print("=" * 80)

    pairs_csv_path = "examples/pairs.csv"
    
    # Requirement: If examples/pairs.csv is missing, stop and report; do not invent examples.
    if not os.path.exists(pairs_csv_path):
        print(f"\nERROR: Required input file '{pairs_csv_path}' is missing.")
        print("As instructed by operational rules, stopping execution immediately without inventing examples, scores, or outcomes.")
        print(f"Please provide '{pairs_csv_path}' to execute live evaluations on candidate image-text pairs.")
        sys.exit(1)

    print(f"Found '{pairs_csv_path}'. Loading examples ...")
    df = pd.read_csv(pairs_csv_path).dropna(how="all")
    print(f"Loaded {len(df)} rows from {pairs_csv_path}. Columns: {list(df.columns)}")

    results = []
    for idx, row in df.iterrows():
        img_val = row.get("image") or row.get("image_path")
        text_val = row.get("text") or row.get("claim") or row.get("caption")
        label_val = row.get("label", "unknown")

        # Resolve image path
        if not os.path.isabs(img_val) and not os.path.exists(img_val):
            # Check relative to examples/
            candidate = os.path.join("examples", img_val)
            if os.path.exists(candidate):
                img_val = candidate

        try:
            exp = explain(text=text_val, image=img_val)
            results.append({
                "image": row.get("image", img_val),
                "label": label_val,
                "claim": text_val,
                "clip_cosine": exp["clip_cosine"],
                "percentile_pristine": exp["percentile_pristine"],
                "percentile_falsified": exp["percentile_falsified"],
                "blip_caption": exp["caption"],
                "claim_caption_similarity": exp["claim_caption_similarity"],
                "signal_level": exp["signal_level"]
            })
        except Exception as e:
            print(f"Row {idx} evaluation error ({img_val}): {e}")
            results.append({
                "image": row.get("image", img_val),
                "label": label_val,
                "claim": text_val,
                "clip_cosine": np.nan,
                "percentile_pristine": np.nan,
                "percentile_falsified": np.nan,
                "blip_caption": f"ERROR: {e}",
                "claim_caption_similarity": np.nan,
                "signal_level": "error"
            })

    res_df = pd.DataFrame(results)

    # Verification before table
    print("\n" + "=" * 90)
    print("CLIP FEATURE VECTOR SHAPE & MANUAL PROJECTION PATH VERIFICATION")
    print("=" * 90)
    first_row = df.iloc[0]
    sample_img_path = first_row.get("image") or first_row.get("image_path")
    sample_text = first_row.get("text") or first_row.get("claim") or first_row.get("caption")
    if not os.path.isabs(sample_img_path) and not os.path.exists(sample_img_path):
        cand = os.path.join("examples", sample_img_path)
        if os.path.exists(cand):
            sample_img_path = cand

    print(f"Sample 0 input image: '{sample_img_path}'")
    print(f"Sample 0 input text:  '{sample_text}'")
    sample_img = Image.open(sample_img_path).convert("RGB")
    _, test_img_emb, test_txt_emb = clip_similarity(sample_img, sample_text)

    print(f"Returned CLIP image feature shape: {test_img_emb.shape}")
    print(f"Returned CLIP text feature shape:  {test_txt_emb.shape}")
    assert test_img_emb.shape == (512,), f"Expected image feature shape (512,), got {test_img_emb.shape}"
    assert test_txt_emb.shape == (512,), f"Expected text feature shape (512,), got {test_txt_emb.shape}"
    print("✓ Assert passed: Both CLIP image and text features are 512-dimensional.")

    # Confirm features equal manual path: vision_model -> visual_projection and text_model -> text_projection
    models = load_models()
    clip_model = models["clip_model"]
    clip_processor = models["clip_processor"]
    device = models["device"]

    inputs = clip_processor(text=[sample_text], images=sample_img, return_tensors="pt", padding=True, truncation=True, max_length=77)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        v_out = clip_model.vision_model(pixel_values=inputs["pixel_values"])
        v_proj = clip_model.visual_projection(v_out.pooler_output)
        v_norm = v_proj / v_proj.norm(dim=-1, keepdim=True)
        v_manual = v_norm.cpu().numpy()[0].astype(np.float32)

        t_out = clip_model.text_model(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"))
        t_proj = clip_model.text_projection(t_out.pooler_output)
        t_norm = t_proj / t_proj.norm(dim=-1, keepdim=True)
        t_manual = t_norm.cpu().numpy()[0].astype(np.float32)

    img_diff = float(np.max(np.abs(test_img_emb - v_manual)))
    txt_diff = float(np.max(np.abs(test_txt_emb - t_manual)))
    print(f"Max absolute diff with manual vision_model -> visual_projection: {img_diff:.2e}")
    print(f"Max absolute diff with manual text_model -> text_projection:     {txt_diff:.2e}")
    assert np.allclose(test_img_emb, v_manual, atol=1e-5), f"Image features differ from manual path: {img_diff}"
    assert np.allclose(test_txt_emb, t_manual, atol=1e-5), f"Text features differ from manual path: {txt_diff}"
    print("✓ Confirmed: Features equal manual path (vision_model -> visual_projection, text_model -> text_projection).")

    # Print Signal Determination Rules
    ref_stats = clip_reference_stats()
    m0_thresh = ref_stats["m0_threshold"]
    low_cutoff = ref_stats["pristine_mean"] - 1.5 * ref_stats["pristine_std"]
    print("\n" + "=" * 90)
    print("INCONSISTENCY SIGNAL DETERMINATION RULES")
    print("=" * 90)
    print(f"  • If CLIP cosine >= M0 threshold ({m0_thresh:.4f}):")
    print(f"      -> 'no strong signal'")
    print(f"  • If CLIP cosine < M0 threshold ({m0_thresh:.4f}):")
    print(f"      - If CLIP cosine < (pristine_mean - 1.5*std) ({low_cutoff:.4f}) OR MiniLM sim < 0.25:")
    print(f"          -> 'possible mismatch'")
    print(f"      - Else:")
    print(f"          -> 'weak evidence'")

    # Print Table
    print("\n" + "=" * 125)
    print("LIVE INCONSISTENCY EVALUATION TABLE")
    print("=" * 125)
    fmt_hdr = f"{'Image':<22} | {'Label':<10} | {'CLIP Cos':<8} | {'Pct(Pristine)':<13} | {'Pct(Falsified)':<14} | {'MiniLM Sim':<10} | {'Signal Level':<20} | {'BLIP Caption'}"
    print(fmt_hdr)
    print("-" * 135)
    for _, r in res_df.iterrows():
        img_str = str(r['image'])[:22]
        lbl_str = str(r['label'])[:10]
        cos_str = f"{r['clip_cosine']:.4f}" if pd.notna(r['clip_cosine']) else "N/A"
        pct_p_str = f"{r['percentile_pristine']:.1f}%" if pd.notna(r['percentile_pristine']) else "N/A"
        pct_f_str = f"{r['percentile_falsified']:.1f}%" if pd.notna(r['percentile_falsified']) else "N/A"
        sim_str = f"{r['claim_caption_similarity']:.4f}" if pd.notna(r['claim_caption_similarity']) else "N/A"
        sig_str = str(r['signal_level'])[:20]
        cap_str = str(r['blip_caption'])
        print(f"{img_str:<22} | {lbl_str:<10} | {cos_str:<8} | {pct_p_str:<13} | {pct_f_str:<14} | {sim_str:<10} | {sig_str:<20} | {cap_str}")

    # Save to results/phase3b_examples.csv
    out_csv = "results/phase3b_examples.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    res_df.to_csv(out_csv, index=False)
    print(f"\nSaved results to: {out_csv}")

    # Summary Lines
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY & COMPARISONS")
    print("=" * 80)
    
    # 1. Mean CLIP cosine for match vs mismatch vs partial
    labels_present = res_df["label"].unique()
    print("Mean CLIP Cosine by Label:")
    for lbl in ["match", "mismatch", "partial"]:
        sub = res_df[res_df["label"].str.lower() == lbl]["clip_cosine"].dropna()
        if len(sub) > 0:
            print(f"  • {lbl:<10}: Mean = {sub.mean():.4f} (N={len(sub)})")
        else:
            print(f"  • {lbl:<10}: None present in file")

    # 2. How many mismatch rows have lower cosine than their matching row for the SAME image
    mismatch_rows = res_df[res_df["label"].str.lower() == "mismatch"]
    match_rows = res_df[res_df["label"].str.lower() == "match"].set_index("image")
    
    lower_count = 0
    comparable_count = 0
    for _, m_row in mismatch_rows.iterrows():
        img_name = m_row["image"]
        if img_name in match_rows.index:
            matching_cos = match_rows.loc[img_name]["clip_cosine"]
            if pd.notna(m_row["clip_cosine"]) and pd.notna(matching_cos):
                comparable_count += 1
                if m_row["clip_cosine"] < matching_cos:
                    lower_count += 1
                    
    print(f"\nPair Comparison for Same Image:")
    print(f"  • Mismatch rows with lower cosine than matching row for the SAME image: "
          f"{lower_count} / {comparable_count} ({(lower_count / comparable_count * 100):.2f}%)" if comparable_count > 0 else "  • No matching/mismatch pairs sharing the same image found.")

    # 3. Range of cosines vs Dataset range
    ref_stats = clip_reference_stats()
    valid_cos = res_df["clip_cosine"].dropna()
    print(f"\nCosine Range Comparison:")
    print(f"  • Examples Range:   Min = {valid_cos.min():.4f}, Max = {valid_cos.max():.4f}, Mean = {valid_cos.mean():.4f} (N={len(valid_cos)})")
    print(f"  • Validation Range: Min = {ref_stats['val_min']:.4f}, Max = {ref_stats['val_max']:.4f}, Mean = {ref_stats['val_mean']:.4f} (N=7,024)")
    print(f"    - Pristine Val Mean:  {ref_stats['pristine_mean']:.4f} (±{ref_stats['pristine_std']:.4f})")
    print(f"    - Falsified Val Mean: {ref_stats['falsified_mean']:.4f} (±{ref_stats['falsified_std']:.4f})")
    print(f"    - M0 Threshold:       {ref_stats['m0_threshold']:.4f}")

if __name__ == "__main__":
    main()
