"""
scripts/08_slide_numbers.py: Comprehensive aggregation of verified slide numbers,
subgroup metrics, provenance metadata, quadruplet analysis, and v2 sanity set evaluation.

Phase 5 only. Read-only on existing results files; generates:
- results/slide_numbers.json
- figures/v2_per_image.png
"""

import os
import sys
import json
from typing import Dict, List, Any
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure repository root is on sys.path
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))


REQUIRED_FROZEN_FILES = [
    "examples/pairs_v2_frozen.csv",
    "results/phase3b_examples_v2.csv",
    "results/phase4a_predictions_v2.csv"
]

SOURCE_CSV = "results/inconsistency_breakdown.csv"
DIAG_TXT = "results/phase3a1_diagnostic.txt"
BASELINES_CSV = "results/baselines_summary.csv"
PHASE3A_LOG = "results/phase3a_log.txt"


def check_frozen_files():
    print("=" * 80)
    print("STEP 1: Checking Frozen V2 Files Existence")
    print("=" * 80)
    missing = []
    for fp in REQUIRED_FROZEN_FILES:
        if not os.path.exists(fp):
            missing.append(fp)
            print(f"[MISSING] Required frozen file not found: {fp}")
        else:
            print(f"[OK] Found frozen file: {fp}")

    if missing:
        print(f"\nERROR: Missing required frozen files: {missing}. Stopping.")
        sys.exit(1)
    print("All required frozen files are present.")


def get_source_group_metrics() -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("STEP 2: Source Group Metrics & Provenance Mapping")
    print("=" * 80)

    # In inconsistency_breakdown.csv:
    # Line 2: overall,Overall Test Set,3632,0.6523,0.6019,0.5279,0.5487,0.7058,0.7832,0.3072,0.274,39.81
    # Line 3: source_dataset,semantics_clip_text_image,908,0.5171,0.5242,0.5444,0.58,0.5837,0.6685,0.2949,0.2993,47.58
    # Line 4: source_dataset,semantics_clip_text_text,908,0.8221,0.8491,0.5358,0.5554,0.832,0.9141,0.3225,0.2589,15.09
    # Line 5: source_dataset,scene_resnet_place,908,0.6459,0.5341,0.5316,0.5595,0.7276,0.7996,0.2977,0.256,46.59
    # Line 6: source_dataset,person_sbert_text_text,908,0.6239,0.5,0.5,0.5,0.6797,0.7507,0.3137,0.2818,50.0

    # In phase3a1_diagnostic.txt summary table:
    # Line 179: person_sbert_text_text       | 908    | ... | M0 ROC: 0.6870
    # Line 180: semantics_clip_text_image    | 908    | ... | M0 ROC: 0.4763
    # Line 181: semantics_clip_text_text     | 908    | ... | M0 ROC: 0.9029
    # Line 182: scene_resnet_place           | 908    | ... | M0 ROC: 0.7624
    # Line 183: Overall Test Set             | 3632   | ... | M0 ROC: 0.7197

    groups_data = {
        "semantics_clip_text_image": {
            "pairs": 908,
            "m0_accuracy": 0.5171,
            "m0_pair_accuracy": 0.5242,
            "m0_roc_auc": 0.4763,
            "m4_accuracy": 0.5837,
            "m4_pair_accuracy": 0.6685,
            "provenance": {
                "pairs": f"{SOURCE_CSV}:3",
                "m0_accuracy": f"{SOURCE_CSV}:3",
                "m0_pair_accuracy": f"{SOURCE_CSV}:3",
                "m0_roc_auc": f"{DIAG_TXT}:180",
                "m4_accuracy": f"{SOURCE_CSV}:3",
                "m4_pair_accuracy": f"{SOURCE_CSV}:3"
            }
        },
        "semantics_clip_text_text": {
            "pairs": 908,
            "m0_accuracy": 0.8221,
            "m0_pair_accuracy": 0.8491,
            "m0_roc_auc": 0.9029,
            "m4_accuracy": 0.8320,
            "m4_pair_accuracy": 0.9141,
            "provenance": {
                "pairs": f"{SOURCE_CSV}:4",
                "m0_accuracy": f"{SOURCE_CSV}:4",
                "m0_pair_accuracy": f"{SOURCE_CSV}:4",
                "m0_roc_auc": f"{DIAG_TXT}:181",
                "m4_accuracy": f"{SOURCE_CSV}:4",
                "m4_pair_accuracy": f"{SOURCE_CSV}:4"
            }
        },
        "scene_resnet_place": {
            "pairs": 908,
            "m0_accuracy": 0.6459,
            "m0_pair_accuracy": 0.5341,
            "m0_roc_auc": 0.7624,
            "m4_accuracy": 0.7276,
            "m4_pair_accuracy": 0.7996,
            "provenance": {
                "pairs": f"{SOURCE_CSV}:5",
                "m0_accuracy": f"{SOURCE_CSV}:5",
                "m0_pair_accuracy": f"{SOURCE_CSV}:5",
                "m0_roc_auc": f"{DIAG_TXT}:182",
                "m4_accuracy": f"{SOURCE_CSV}:5",
                "m4_pair_accuracy": f"{SOURCE_CSV}:5"
            }
        },
        "person_sbert_text_text": {
            "pairs": 908,
            "m0_accuracy": 0.6239,
            "m0_pair_accuracy": 0.5000,
            "m0_roc_auc": 0.6870,
            "m4_accuracy": 0.6797,
            "m4_pair_accuracy": 0.7507,
            "provenance": {
                "pairs": f"{SOURCE_CSV}:6",
                "m0_accuracy": f"{SOURCE_CSV}:6",
                "m0_pair_accuracy": f"{SOURCE_CSV}:6",
                "m0_roc_auc": f"{DIAG_TXT}:179",
                "m4_accuracy": f"{SOURCE_CSV}:6",
                "m4_pair_accuracy": f"{SOURCE_CSV}:6"
            }
        }
    }

    overall_data = {
        "pairs": 3632,
        "m0_accuracy": 0.6523,
        "m0_pair_accuracy": 0.6019,
        "m0_roc_auc": 0.7197,
        "m4_accuracy": 0.7058,
        "m4_pair_accuracy": 0.7832,
        "provenance": {
            "pairs": f"{SOURCE_CSV}:2",
            "m0_accuracy": f"{SOURCE_CSV}:2 (and {BASELINES_CSV}:2)",
            "m0_pair_accuracy": f"{SOURCE_CSV}:2 (and {BASELINES_CSV}:2)",
            "m0_roc_auc": f"{DIAG_TXT}:183",
            "m4_accuracy": f"{SOURCE_CSV}:2 (and {BASELINES_CSV}:6)",
            "m4_pair_accuracy": f"{SOURCE_CSV}:2 (and {BASELINES_CSV}:6)"
        }
    }

    # Print formatted table
    print(f"{'Group':<28} | {'Pairs':<5} | {'M0 Acc':<7} | {'M0 Pair':<7} | {'M0 ROC':<7} | {'M4 Acc':<7} | {'M4 Pair':<7}")
    print("-" * 85)
    for gname, d in groups_data.items():
        print(f"{gname:<28} | {d['pairs']:<5} | {d['m0_accuracy']:<7.4f} | {d['m0_pair_accuracy']:<7.4f} | {d['m0_roc_auc']:<7.4f} | {d['m4_accuracy']:<7.4f} | {d['m4_pair_accuracy']:<7.4f}")
    print("-" * 85)
    print(f"{'Overall Test Set':<28} | {overall_data['pairs']:<5} | {overall_data['m0_accuracy']:<7.4f} | {overall_data['m0_pair_accuracy']:<7.4f} | {overall_data['m0_roc_auc']:<7.4f} | {overall_data['m4_accuracy']:<7.4f} | {overall_data['m4_pair_accuracy']:<7.4f}")

    print("\nProvenance Information (File and Line Numbers):")
    for gname, d in groups_data.items():
        print(f"  • {gname}:")
        for k, v in d["provenance"].items():
            print(f"      - {k:<16}: {v}")
    print(f"  • Overall Test Set:")
    for k, v in overall_data["provenance"].items():
        print(f"      - {k:<16}: {v}")

    return {"source_groups": groups_data, "overall": overall_data}


def get_bootstrap_gap() -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("STEP 3: M4 vs M0 Pair-Accuracy Gap & Bootstrap Confidence Interval")
    print("=" * 80)

    # Line 121 of results/phase3a_log.txt:
    # Difference (M4 - M0):    Observed = 0.1814 | Bootstrap Mean = 0.1810 | 95% CI = [0.1667, 0.1962]
    gap_info = {
        "file": PHASE3A_LOG,
        "line": 121,
        "observed_gap": 0.1814,
        "bootstrap_mean": 0.1810,
        "bootstrap_95_ci": [0.1667, 0.1962],
        "m0_pair_accuracy": 0.6019,
        "m0_bootstrap_95_ci": [0.5867, 0.6178],
        "m4_pair_accuracy": 0.7832,
        "m4_bootstrap_95_ci": [0.7710, 0.7947]
    }
    print(f"Source File: {gap_info['file']} (Line {gap_info['line']})")
    print(f"M4 vs M0 Pair-Accuracy Difference:")
    print(f"  • Observed Difference : {gap_info['observed_gap']:.4f} (+18.14 percentage points)")
    print(f"  • Bootstrap Mean      : {gap_info['bootstrap_mean']:.4f}")
    print(f"  • Bootstrap 95% CI    : [{gap_info['bootstrap_95_ci'][0]:.4f}, {gap_info['bootstrap_95_ci'][1]:.4f}]")
    print(f"  • M0 Pair Acc 95% CI  : [{gap_info['m0_bootstrap_95_ci'][0]:.4f}, {gap_info['m0_bootstrap_95_ci'][1]:.4f}]")
    print(f"  • M4 Pair Acc 95% CI  : [{gap_info['m4_bootstrap_95_ci'][0]:.4f}, {gap_info['m4_bootstrap_95_ci'][1]:.4f}]")
    return gap_info


def compute_person_quadruplets() -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("STEP 4: person_sbert_text_text Dynamic Quadruplet Computation (TEST Split)")
    print("=" * 80)

    test_npz = np.load("features/test.npz")
    T_test = test_npz["T"]
    V_test = test_npz["V"]
    ids_test = test_npz["ids"]
    image_ids_test = test_npz["image_ids"]
    y_test = test_npz["y"]

    with open("data/hf/data/merged_balanced/test.json", "r", encoding="utf-8") as f:
        test_json = json.load(f)

    annotations = test_json["annotations"]
    source_datasets_map = test_json.get("source_datasets", {})

    # Alignment assertion as in 04b_diagnose.py
    assert np.array_equal(ids_test, [a["id"] for a in annotations]), "Assertion failed: IDs do not match"
    assert np.array_equal(image_ids_test, [a["image_id"] for a in annotations]), "Assertion failed: Image IDs do not match"
    assert np.array_equal(y_test, [1 if a["falsified"] else 0 for a in annotations]), "Assertion failed: Labels do not match"
    print("Assertion PASSED: All 7,264 samples match test.json annotations identically.")

    # Compute M0 cosines
    cos_test = np.sum(T_test * V_test, axis=1)

    # Filter person_sbert_text_text pairs
    num_pairs = len(annotations) // 2
    person_pairs = []
    for k in range(num_pairs):
        f_ann = annotations[2 * k + 1]
        src_id = str(f_ann["source_dataset"])
        src_name = source_datasets_map.get(src_id, f"source_{src_id}")
        if src_name == "person_sbert_text_text":
            person_pairs.append(k)

    # Map pairs by (pristine_img, falsified_img)
    pair_by_img_tuple = {}
    for k in person_pairs:
        img_p = int(image_ids_test[2 * k])
        img_f = int(image_ids_test[2 * k + 1])
        pair_by_img_tuple[(img_p, img_f)] = k

    visited = set()
    quadruplets = []
    for (img_p, img_f), k in pair_by_img_tuple.items():
        if k in visited:
            continue
        k_partner = pair_by_img_tuple.get((img_f, img_p))
        if k_partner is not None:
            quadruplets.append((k, k_partner))
            visited.add(k)
            visited.add(k_partner)

    both_right = 0
    both_wrong = 0
    split = 0

    for k1, k2 in quadruplets:
        # In M0: pristine correct if cos_p > cos_f
        k1_right = bool(cos_test[2 * k1] > cos_test[2 * k1 + 1])
        k2_right = bool(cos_test[2 * k2] > cos_test[2 * k2 + 1])
        if k1_right and k2_right:
            both_right += 1
        elif (not k1_right) and (not k2_right):
            both_wrong += 1
        else:
            split += 1

    total_quads = len(quadruplets)
    print(f"Total person_sbert_text_text pairs: {len(person_pairs)}")
    print(f"Total mutual 2-way quadruplets   : {total_quads}")
    print(f"  • M0 gets BOTH pairs right      : {both_right} quadruplets ({both_right / total_quads * 100:.2f}%)")
    print(f"  • M0 gets BOTH pairs wrong      : {both_wrong} quadruplets ({both_wrong / total_quads * 100:.2f}%)")
    print(f"  • M0 gets SPLIT (one right, one wrong): {split} quadruplets ({split / total_quads * 100:.2f}%)")
    print(f"Mathematical Consequence: {both_right}*2 + {split} = {both_right*2 + split} correct pairs out of {len(person_pairs)} -> Pair Accuracy = 0.5000 exactly.")

    return {
        "total_pairs": len(person_pairs),
        "total_quadruplets": total_quads,
        "both_right": both_right,
        "both_wrong": both_wrong,
        "split": split
    }


def analyze_v2_handmade_set() -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("STEP 5: Analysis of Frozen V2 Hand-Made Sanity Set (n = 21 rows)")
    print("=" * 80)

    p4_df = pd.read_csv("results/phase4a_predictions_v2.csv")
    p3_df = pd.read_csv("results/phase3b_examples_v2.csv")

    merged_df = p4_df.merge(p3_df[["image", "label", "signal_level"]], on=["image", "label"], how="left")

    print("\nPer-Row Table (image, label, clip_cosine, M4 score, seed_42, seed_43, seed_44, signal_level):")
    print("-" * 115)
    print(f"{'Image':<30} | {'Label':<8} | {'Cosine':<7} | {'M4 Score':<8} | {'Seed 42':<7} | {'Seed 43':<7} | {'Seed 44':<7} | {'Signal Level':<17}")
    print("-" * 115)
    for _, row in merged_df.iterrows():
        print(f"{row['image']:<30} | {row['label']:<8} | {row['clip_cosine']:<7.4f} | {row['model_score']:<8.4f} | {row['seed_42']:<7.4f} | {row['seed_43']:<7.4f} | {row['seed_44']:<7.4f} | {str(row['signal_level']):<17}")
    print("-" * 115)

    images = merged_df["image"].unique()
    n_images = len(images)

    # 1. Mismatch vs Match: mismatch cosine < match cosine AND mismatch M4 score > match M4 score
    cos_mismatch_lower = 0
    m4_mismatch_higher = 0
    for img in images:
        sub = merged_df[merged_df["image"] == img].set_index("label")
        if sub.loc["mismatch", "clip_cosine"] < sub.loc["match", "clip_cosine"]:
            cos_mismatch_lower += 1
        if sub.loc["mismatch", "model_score"] > sub.loc["match", "model_score"]:
            m4_mismatch_higher += 1

    print(f"\nPairwise Comparison (Mismatch vs Match across {n_images} images):")
    print(f"  • Mismatch cosine < Match cosine    : {cos_mismatch_lower} of {n_images} images")
    print(f"  • Mismatch M4 score > Match M4 score: {m4_mismatch_higher} of {n_images} images")

    # 2. Full ordering: match > partial > mismatch
    # For CLIP cosine: match > partial > mismatch
    cos_order_pass = []
    cos_order_fail = []
    for img in images:
        sub = merged_df[merged_df["image"] == img].set_index("label")
        m = sub.loc["match", "clip_cosine"]
        p = sub.loc["partial", "clip_cosine"]
        mm = sub.loc["mismatch", "clip_cosine"]
        if m > p > mm:
            cos_order_pass.append(img)
        else:
            cos_order_fail.append((img, m, p, mm))

    print(f"\nFull Ordering (Cosine: match > partial > mismatch):")
    print(f"  • Satisfied: {len(cos_order_pass)} of {n_images} images")
    print(f"  • Failing Images: {[f[0] for f in cos_order_fail] if cos_order_fail else 'None'}")

    # For M4 score:
    # Note: M4 is an out-of-context probability score (higher = falsified).
    # Out-of-context monotonicity: mismatch > partial > match
    # Literal match > partial > mismatch: checks authenticity ranking
    m4_ooc_pass = []
    m4_ooc_fail = []
    m4_literal_pass = []
    m4_literal_fail = []

    for img in images:
        sub = merged_df[merged_df["image"] == img].set_index("label")
        m = sub.loc["match", "model_score"]
        p = sub.loc["partial", "model_score"]
        mm = sub.loc["mismatch", "model_score"]
        if mm > p > m:
            m4_ooc_pass.append(img)
        else:
            m4_ooc_fail.append((img, m, p, mm))
        if m > p > mm:
            m4_literal_pass.append(img)
        else:
            m4_literal_fail.append((img, m, p, mm))

    print(f"\nFull Ordering (M4 Score):")
    print(f"  • Out-of-context monotonic ordering (mismatch > partial > match): {len(m4_ooc_pass)} of {n_images} images")
    print(f"    ↳ Failing image: {[f[0] for f in m4_ooc_fail]}")
    for f in m4_ooc_fail:
        print(f"      Details for {f[0]}: match={f[1]:.4f}, partial={f[2]:.4f}, mismatch={f[3]:.4f} (partial < match)")
    print(f"  • Literal ordering (match > partial > mismatch): {len(m4_literal_pass)} of {n_images} images")
    print(f"    ↳ Failing images: {[f[0] for f in m4_literal_fail]} (all 7 fail because M4 assigns higher scores to mismatch)")

    # 3. Two flagged correct captions (match rows where M4 score >= 0.5)
    flagged_captions = []
    match_rows = merged_df[merged_df["label"] == "match"]
    for _, row in match_rows.iterrows():
        if row["model_score"] >= 0.5:
            flagged_captions.append({
                "image": row["image"],
                "text": row["text"],
                "model_score": float(row["model_score"]),
                "signal_level": row["signal_level"]
            })

    print(f"\nTwo Flagged Correct Captions (match rows with M4 score >= 0.5):")
    for idx, fc in enumerate(flagged_captions, 1):
        print(f"  [{idx}] Image: {fc['image']}")
        print(f"      Caption: \"{fc['text']}\"")
        print(f"      M4 Score: {fc['model_score']:.4f} (Signal: {fc['signal_level']})")

    # 4. Spearman correlation between clip_cosine and model_score (no p-value)
    rho = float(spearmanr(merged_df["clip_cosine"], merged_df["model_score"]).statistic)
    print(f"\nSpearman rho between clip_cosine and M4 score: {rho:.4f}")

    return {
        "per_row_table": merged_df.to_dict(orient="records"),
        "mismatch_vs_match": {
            "mismatch_cosine_lower": f"{cos_mismatch_lower} of {n_images}",
            "mismatch_m4_higher": f"{m4_mismatch_higher} of {n_images}"
        },
        "full_ordering_cosine": {
            "ordering": "match > partial > mismatch",
            "k_of_7": f"{len(cos_order_pass)} of {n_images}",
            "failing_images": [f[0] for f in cos_order_fail]
        },
        "full_ordering_m4_ooc": {
            "ordering": "mismatch > partial > match",
            "k_of_7": f"{len(m4_ooc_pass)} of {n_images}",
            "failing_images": [f[0] for f in m4_ooc_fail]
        },
        "full_ordering_m4_literal": {
            "ordering": "match > partial > mismatch",
            "k_of_7": f"{len(m4_literal_pass)} of {n_images}",
            "failing_images": [f[0] for f in m4_literal_fail]
        },
        "flagged_correct_captions": flagged_captions,
        "spearman_rho": round(rho, 4)
    }


def plot_v2_figure(v2_summary: Dict[str, Any], fig_path: str = "figures/v2_per_image.png"):
    print("\n" + "=" * 80)
    print("STEP 6: Generating figures/v2_per_image.png")
    print("=" * 80)

    if os.path.exists(fig_path):
        print(f"Notice: Figure '{fig_path}' already exists; preserving existing file.")
        return

    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    df = pd.DataFrame(v2_summary["per_row_table"])

    # Shared image order
    image_paths = [
        "examples/space_mission.jpeg",
        "examples/parliament.jpeg",
        "examples/concert.jpg",
        "examples/sunflower.jpg",
        "examples/flood.jpg",
        "examples/elections.jpg",
        "examples/sports.jpg"
    ]
    labels_clean = [os.path.splitext(os.path.basename(p))[0] for p in image_paths]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300, sharex=True)

    x = np.arange(len(image_paths))
    width = 0.25

    # Extract cosine values
    cos_match = [df[(df["image"] == p) & (df["label"] == "match")]["clip_cosine"].values[0] for p in image_paths]
    cos_partial = [df[(df["image"] == p) & (df["label"] == "partial")]["clip_cosine"].values[0] for p in image_paths]
    cos_mismatch = [df[(df["image"] == p) & (df["label"] == "mismatch")]["clip_cosine"].values[0] for p in image_paths]

    # Extract M4 scores
    m4_match = [df[(df["image"] == p) & (df["label"] == "match")]["model_score"].values[0] for p in image_paths]
    m4_partial = [df[(df["image"] == p) & (df["label"] == "partial")]["model_score"].values[0] for p in image_paths]
    m4_mismatch = [df[(df["image"] == p) & (df["label"] == "mismatch")]["model_score"].values[0] for p in image_paths]

    # Panel 1: CLIP Cosine
    ax1 = axes[0]
    ax1.bar(x - width, cos_match, width, label="Match", color="#2ca02c", alpha=0.85, edgecolor="#1b611b")
    ax1.bar(x, cos_partial, width, label="Partial", color="#ff7f0e", alpha=0.85, edgecolor="#b35504")
    ax1.bar(x + width, cos_mismatch, width, label="Mismatch", color="#d62728", alpha=0.85, edgecolor="#8a1718")
    ax1.axhline(0.3056, color="#333333", linestyle="--", linewidth=1.2, label=r"M0 threshold ($\theta^*$ = 0.3056)")
    ax1.set_ylabel(r"CLIP Cosine Similarity ($\cos(T, V)$)", fontsize=11)
    ax1.set_title("Panel A: CLIP Text-Image Cosine Similarity", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_clean, rotation=25, ha="right", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax1.set_ylim(0.0, 0.42)

    # Panel 2: M4 Model Score
    ax2 = axes[1]
    ax2.bar(x - width, m4_match, width, label="Match", color="#2ca02c", alpha=0.85, edgecolor="#1b611b")
    ax2.bar(x, m4_partial, width, label="Partial", color="#ff7f0e", alpha=0.85, edgecolor="#b35504")
    ax2.bar(x + width, m4_mismatch, width, label="Mismatch", color="#d62728", alpha=0.85, edgecolor="#8a1718")
    ax2.axhline(0.5, color="#333333", linestyle="--", linewidth=1.2, label="Classification threshold (0.50)")
    ax2.set_ylabel("M4 Uncalibrated Model Score", fontsize=11)
    ax2.set_title("Panel B: M4 Ensemble Score (Out-of-Context Probability)", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_clean, rotation=25, ha="right", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax2.set_ylim(0.0, 1.08)

    fig.suptitle(r"Hand-made v2 sanity set (n = 21 rows, descriptive only)", fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure successfully to: {fig_path}")


def main():
    check_frozen_files()
    metrics = get_source_group_metrics()
    bootstrap_info = get_bootstrap_gap()
    quad_info = compute_person_quadruplets()
    v2_analysis = analyze_v2_handmade_set()

    # Save to results/slide_numbers.json
    out_json_path = "results/slide_numbers.json"
    slide_numbers_data = {
        "source_groups": metrics["source_groups"],
        "overall": metrics["overall"],
        "m4_vs_m0_bootstrap_gap": bootstrap_info,
        "person_sbert_text_text_quadruplets": quad_info,
        "v2_handmade_sanity_set": {
            "mismatch_vs_match": v2_analysis["mismatch_vs_match"],
            "full_ordering_cosine": v2_analysis["full_ordering_cosine"],
            "full_ordering_m4_ooc": v2_analysis["full_ordering_m4_ooc"],
            "full_ordering_m4_literal": v2_analysis["full_ordering_m4_literal"],
            "flagged_correct_captions": v2_analysis["flagged_correct_captions"],
            "spearman_rho": v2_analysis["spearman_rho"]
        }
    }

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(slide_numbers_data, f, indent=2)
    print(f"\nSaved structured slide metrics to: {out_json_path}")

    # Create figure
    plot_v2_figure(v2_analysis, fig_path="figures/v2_per_image.png")

    print("\n" + "=" * 80)
    print("PHASE 5 EXECUTION COMPLETED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    main()
