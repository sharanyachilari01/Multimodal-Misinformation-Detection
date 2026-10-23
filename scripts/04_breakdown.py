#!/usr/bin/env python3
import os
import sys
import json
import gc
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
import torch

# Ensure src module can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import MLPClassifier, build_features
from src.train import set_seed, get_device, predict_scores, evaluate_predictions

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class DualLogger:
    """Tee stdout to both terminal and a file."""
    def __init__(self, filepath: str):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.log_file = open(filepath, "w", encoding="utf-8")
        
    def write(self, message: str):
        self.terminal.write(message)
        self.terminal.flush()
        self.log_file.write(message)
        self.log_file.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def compute_tie_adjusted_pair_scores(pristine_scores: np.ndarray, falsified_scores: np.ndarray) -> np.ndarray:
    """
    Returns array of pair scores where:
    - falsified > pristine: 1.0
    - falsified == pristine: 0.5 (tie)
    - falsified < pristine: 0.0
    """
    return (falsified_scores > pristine_scores).astype(float) + 0.5 * (falsified_scores == pristine_scores).astype(float)

def plot_cosine_distribution(val_npz_path: str, m0_thresh_path: str, save_path: str):
    """Plot histogram of validation cosines (pristine vs falsified) with M0 threshold line."""
    val_data = np.load(val_npz_path)
    T_val, V_val, y_val = val_data["T"], val_data["V"], val_data["y"]
    cos_val = np.sum(T_val * V_val, axis=1)
    
    cos_pristine = cos_val[y_val == 0]
    cos_falsified = cos_val[y_val == 1]
    
    with open(m0_thresh_path, "r", encoding="utf-8") as f:
        threshold = json.load(f)["optimal_threshold"]
        
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    
    bins = np.linspace(min(cos_val.min(), threshold - 0.1), max(cos_val.max(), threshold + 0.1), 60)
    
    ax.hist(
        cos_pristine,
        bins=bins,
        alpha=0.55,
        color="#1f77b4",
        edgecolor="#154f78",
        label=f"Pristine (N={len(cos_pristine):,}, Mean={cos_pristine.mean():.4f})",
        density=True
    )
    ax.hist(
        cos_falsified,
        bins=bins,
        alpha=0.55,
        color="#d62728",
        edgecolor="#8c1b1b",
        label=f"Falsified (N={len(cos_falsified):,}, Mean={cos_falsified.mean():.4f})",
        density=True
    )
    
    ax.axvline(
        x=threshold,
        color="#111111",
        linestyle="--",
        linewidth=2.2,
        label=f"M0 Optimal Threshold ($\theta^*={threshold:.4f}$)"
    )
    
    ax.set_title("Validation Set CLIP Cosine Similarity Distribution: Pristine vs. Falsified", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("CLIP Cosine Similarity: $\cos(T, V)$", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

def main():
    log_path = "results/phase3a_log.txt"
    sys.stdout = DualLogger(log_path)
    
    print("=" * 80)
    print("PHASE 3A: Fine-Grained Breakdown & Consistency Analysis")
    print("=" * 80)
    
    config = load_config("config.json")
    seed_base = config.get("seed", 42)
    seeds_for_runs = config.get("seeds_for_runs", [42, 43, 44])
    batch_size = int(config.get("batch_size", 256))
    device = get_device()
    print(f"Device: {device} | Seeds: {seeds_for_runs}")

    # =========================================================================
    # Step 1: Load and Assert Order
    # =========================================================================
    print("\n--- Step 1: Loading test.npz and test.json (Asserting Exact Alignment) ---")
    test_npz = np.load("features/test.npz")
    T_test = test_npz["T"]
    V_test = test_npz["V"]
    y_test = test_npz["y"]
    ids_test = test_npz["ids"]
    image_ids_test = test_npz["image_ids"]
    
    with open("data/hf/data/merged_balanced/test.json", "r", encoding="utf-8") as f:
        test_json = json.load(f)
        
    source_datasets_map = test_json.get("source_datasets", {})
    annotations = test_json["annotations"]
    
    json_ids = np.array([a["id"] for a in annotations], dtype=np.int64)
    json_image_ids = np.array([a["image_id"] for a in annotations], dtype=np.int64)
    json_y = np.array([1 if a["falsified"] else 0 for a in annotations], dtype=np.int64)
    
    if not (len(ids_test) == len(json_ids) == len(annotations)):
        print(f"ERROR: Count mismatch! npz={len(ids_test)}, json={len(json_ids)}. Stopping.")
        sys.exit(1)
        
    if not np.array_equal(ids_test, json_ids):
        print("ERROR: Caption IDs do not match between test.npz and test.json! Stopping.")
        sys.exit(1)
        
    if not np.array_equal(image_ids_test, json_image_ids):
        print("ERROR: Image IDs do not match between test.npz and test.json! Stopping.")
        sys.exit(1)
        
    if not np.array_equal(y_test, json_y):
        print("ERROR: Binary labels do not match between test.npz and test.json! Stopping.")
        sys.exit(1)
        
    print(f"Assertion PASSED: All {len(ids_test):,} samples match exactly in sequence, IDs, image IDs, and labels.")
    print(f"Source datasets mapping decoded: {source_datasets_map}")
    
    num_pairs = len(annotations) // 2
    
    # Load M0 threshold
    with open("models/M0_threshold.json", "r", encoding="utf-8") as f:
        m0_threshold_info = json.load(f)
    m0_thresh = m0_threshold_info["optimal_threshold"]
    print(f"Loaded M0 optimal threshold: {m0_thresh:.6f}")

    # =========================================================================
    # Step 2: Recompute Tie-Adjusted Pair Accuracy for All Models
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: Recomputing Pair Accuracy with Ties Counted as 0.5")
    print("=" * 80)
    
    models_all = ["M0", "M1", "M2", "M3", "M4"]
    trained_models = ["M1", "M2", "M3", "M4"]
    
    # Precompute test predictions/scores for each model & seed
    model_predictions = {}
    
    # M0 evaluation
    cos_test = np.sum(T_test * V_test, axis=1)
    cos_p = cos_test[0::2]
    cos_f = cos_test[1::2]
    # In M0, score = 1 - cosine (higher means more falsified)
    score_p_m0 = 1.0 - cos_p
    score_f_m0 = 1.0 - cos_f
    m0_pair_scores = compute_tie_adjusted_pair_scores(score_p_m0, score_f_m0)
    m0_tie_adj_acc = float(np.mean(m0_pair_scores))
    m0_strict_acc = float(np.mean(score_f_m0 > score_p_m0))
    m0_pred = (cos_test < m0_thresh).astype(int)
    m0_metrics = evaluate_predictions(y_test, 1.0 - cos_test, ids_test, threshold=1.0 - m0_thresh)
    m0_metrics["accuracy"] = float(np.mean(m0_pred == y_test))
    m0_metrics["pair_accuracy"] = m0_tie_adj_acc
    m0_metrics["strict_pair_accuracy"] = m0_strict_acc
    model_predictions["M0"] = {"scores": 1.0 - cos_test, "pair_scores": m0_pair_scores, "preds": m0_pred}
    
    print(f"M0: Strict Pair Acc={m0_strict_acc:.4f}, Tie-Adjusted Pair Acc={m0_tie_adj_acc:.4f}")

    # M1 to M4 evaluations
    per_seed_tie_adj = []
    per_seed_tie_adj.append({
        "model": "M0",
        "seed": "none",
        **{k: m0_metrics[k] for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pair_accuracy"]}
    })
    
    for m in trained_models:
        in_dim = 512 if m in ["M1", "M2"] else (1024 if m == "M3" else 2049)
        X_test_m = build_features(m, T_test, V_test)
        model_predictions[m] = {}
        
        for seed in seeds_for_runs:
            ckpt_path = f"models/{m}_seed{seed}.pt"
            model = MLPClassifier(in_features=in_dim, hidden_dim=256, dropout=0.3)
            model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
            model.to(device)
            
            probs = predict_scores(model, X_test_m, batch_size=batch_size, device=device)
            p0 = probs[0::2]
            p1 = probs[1::2]
            pair_scores = compute_tie_adjusted_pair_scores(p0, p1)
            tie_acc = float(np.mean(pair_scores))
            strict_acc = float(np.mean(p1 > p0))
            ties_count = int(np.sum(p1 == p0))
            
            preds = (probs >= 0.5).astype(int)
            base_metrics = evaluate_predictions(y_test, probs, ids_test, threshold=0.5)
            base_metrics["pair_accuracy"] = tie_acc
            base_metrics["strict_pair_accuracy"] = strict_acc
            
            model_predictions[m][seed] = {
                "probs": probs,
                "preds": preds,
                "pair_scores": pair_scores,
                "metrics": base_metrics
            }
            
            print(f"{m} (Seed {seed}): Strict Pair Acc={strict_acc:.4f}, "
                  f"Tie-Adjusted Pair Acc={tie_acc:.4f} (Ties={ties_count})")
            
            per_seed_tie_adj.append({
                "model": m,
                "seed": seed,
                **{k: base_metrics[k] for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pair_accuracy"]}
            })
            
            del model
            if device.type == "mps":
                torch.mps.empty_cache()
        del X_test_m
        gc.collect()

    # Generate results/baselines_summary_tieadjusted.csv
    df_per_seed_adj = pd.DataFrame(per_seed_tie_adj)
    metrics_to_agg = ["accuracy", "precision", "recall", "f1", "roc_auc", "pair_accuracy"]
    summary_adj_rows = []
    
    # M0 row
    m0_data = df_per_seed_adj[df_per_seed_adj["model"] == "M0"].iloc[0]
    m0_sum = {"model": "M0"}
    for k in metrics_to_agg:
        m0_sum[f"{k}_mean"] = float(m0_data[k])
        m0_sum[f"{k}_std"] = 0.0
    summary_adj_rows.append(m0_sum)
    
    for m in trained_models:
        sub = df_per_seed_adj[df_per_seed_adj["model"] == m]
        m_sum = {"model": m}
        for k in metrics_to_agg:
            m_sum[f"{k}_mean"] = float(sub[k].mean())
            m_sum[f"{k}_std"] = float(sub[k].std(ddof=1)) if len(sub) > 1 else 0.0
        summary_adj_rows.append(m_sum)
        
    df_summary_adj = pd.DataFrame(summary_adj_rows)
    adj_csv_path = "results/baselines_summary_tieadjusted.csv"
    df_summary_adj.to_csv(adj_csv_path, index=False)
    print(f"\nSaved tie-adjusted summary to: {adj_csv_path}")

    # =========================================================================
    # Step 3 & 4: Breakdown by Source Dataset and Similarity Score
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 3 & 4: Breakdown by Source Dataset & Similarity Score")
    print("=" * 80)
    print("Grouping Method Explanation:")
    print("  - Each consecutive pair (2k, 2k+1) is grouped by the FALSIFIED sample's attributes:")
    print("    1) 'source_dataset': Decoded via source_datasets metadata into 4 retrieval methods:")
    print("       'semantics_clip_text_image', 'semantics_clip_text_text', 'scene_resnet_place', 'person_sbert_text_text'")
    print("    2) 'similarity_score': Continuous retrieval score binned into 4 quartiles (Q1..Q4) across the test set.")
    print("  - Both samples of each pair (pristine & falsified) are evaluated within each group.")

    # Collect pair-level metadata
    pair_source_names = []
    pair_sim_scores = []
    for k in range(num_pairs):
        falsified_ann = annotations[2 * k + 1]
        src_id = str(falsified_ann["source_dataset"])
        pair_source_names.append(source_datasets_map.get(src_id, f"source_{src_id}"))
        pair_sim_scores.append(float(falsified_ann["similarity_score"]))
        
    pair_source_names = np.array(pair_source_names)
    pair_sim_scores = np.array(pair_sim_scores)
    
    # Quartile edges for similarity score
    q_edges = [
        np.min(pair_sim_scores),
        np.percentile(pair_sim_scores, 25),
        np.percentile(pair_sim_scores, 50),
        np.percentile(pair_sim_scores, 75),
        np.max(pair_sim_scores)
    ]
    
    groups_to_evaluate = []
    
    # 1. Overall
    groups_to_evaluate.append({
        "type": "overall",
        "name": "Overall Test Set",
        "pair_mask": np.ones(num_pairs, dtype=bool)
    })
    
    # 2. Source Dataset groups
    for src_id in sorted(source_datasets_map.keys(), key=lambda x: int(x)):
        name = source_datasets_map[src_id]
        mask = (pair_source_names == name)
        groups_to_evaluate.append({
            "type": "source_dataset",
            "name": name,
            "pair_mask": mask
        })
        
    # 3. Similarity score quartiles
    for q_idx in range(4):
        low, high = q_edges[q_idx], q_edges[q_idx + 1]
        if q_idx == 0:
            mask = (pair_sim_scores >= low) & (pair_sim_scores <= high)
        else:
            mask = (pair_sim_scores > low) & (pair_sim_scores <= high)
        groups_to_evaluate.append({
            "type": "similarity_score_quartile",
            "name": f"Q{q_idx+1}_[{low:.4f}_{high:.4f}]",
            "pair_mask": mask
        })

    breakdown_results = []
    
    print("\n--- Evaluation Across Groups (M0, M2, M4) ---")
    for grp in groups_to_evaluate:
        gtype = grp["type"]
        gname = grp["name"]
        mask = grp["pair_mask"]
        pair_indices = np.where(mask)[0]
        n_pairs_g = len(pair_indices)
        
        # All individual sample indices for this group (2k and 2k+1)
        sample_indices = []
        for k in pair_indices:
            sample_indices.extend([2 * k, 2 * k + 1])
        sample_indices = np.array(sample_indices)
        y_g = y_test[sample_indices]
        
        # Cosines for M0
        cos_p_g = cos_p[pair_indices]
        cos_f_g = cos_f[pair_indices]
        mean_cos_p = float(np.mean(cos_p_g))
        mean_cos_f = float(np.mean(cos_f_g))
        clip_prefers_wrong_frac = float(np.mean(cos_f_g > cos_p_g))
        
        # M0 metrics on this group
        m0_preds_g = model_predictions["M0"]["preds"][sample_indices]
        m0_acc_g = float(np.mean(m0_preds_g == y_g))
        m0_pair_scores_g = model_predictions["M0"]["pair_scores"][pair_indices]
        m0_pair_acc_g = float(np.mean(m0_pair_scores_g))
        
        # M2 metrics averaged over 3 seeds
        m2_accs, m2_pair_accs = [], []
        for seed in seeds_for_runs:
            m2_preds_g = model_predictions["M2"][seed]["preds"][sample_indices]
            m2_accs.append(float(np.mean(m2_preds_g == y_g)))
            m2_pair_scores_g = model_predictions["M2"][seed]["pair_scores"][pair_indices]
            m2_pair_accs.append(float(np.mean(m2_pair_scores_g)))
        m2_acc_mean = float(np.mean(m2_accs))
        m2_pair_acc_mean = float(np.mean(m2_pair_accs))
        
        # M4 metrics averaged over 3 seeds
        m4_accs, m4_pair_accs = [], []
        for seed in seeds_for_runs:
            m4_preds_g = model_predictions["M4"][seed]["preds"][sample_indices]
            m4_accs.append(float(np.mean(m4_preds_g == y_g)))
            m4_pair_scores_g = model_predictions["M4"][seed]["pair_scores"][pair_indices]
            m4_pair_accs.append(float(np.mean(m4_pair_scores_g)))
        m4_acc_mean = float(np.mean(m4_accs))
        m4_pair_acc_mean = float(np.mean(m4_pair_accs))
        
        print(f"\nGroup [{gtype.upper()}]: {gname}")
        print(f"  Number of Pairs: {n_pairs_g:,} (Total Samples: {len(sample_indices):,})")
        print(f"  CLIP Cosines: Pristine Mean={mean_cos_p:.4f} | Falsified Mean={mean_cos_f:.4f}")
        print(f"  CLIP prefers wrong image (cos_falsified > cos_pristine): {clip_prefers_wrong_frac:.4f} ({clip_prefers_wrong_frac*100:.2f}%)")
        print(f"  M0: Accuracy={m0_acc_g:.4f} | Tie-adjusted Pair Acc={m0_pair_acc_g:.4f}")
        print(f"  M2: Accuracy={m2_acc_mean:.4f} | Tie-adjusted Pair Acc={m2_pair_acc_mean:.4f}")
        print(f"  M4: Accuracy={m4_acc_mean:.4f} | Tie-adjusted Pair Acc={m4_pair_acc_mean:.4f}")
        
        breakdown_results.append({
            "group_type": gtype,
            "group_name": gname,
            "num_pairs": n_pairs_g,
            "m0_accuracy": round(m0_acc_g, 4),
            "m0_pair_accuracy": round(m0_pair_acc_g, 4),
            "m2_accuracy": round(m2_acc_mean, 4),
            "m2_pair_accuracy": round(m2_pair_acc_mean, 4),
            "m4_accuracy": round(m4_acc_mean, 4),
            "m4_pair_accuracy": round(m4_pair_acc_mean, 4),
            "pristine_mean_cosine": round(mean_cos_p, 4),
            "falsified_mean_cosine": round(mean_cos_f, 4),
            "clip_prefers_wrong_pct": round(clip_prefers_wrong_frac * 100, 2)
        })

    # Save results/inconsistency_breakdown.csv
    df_breakdown = pd.DataFrame(breakdown_results)
    breakdown_csv = "results/inconsistency_breakdown.csv"
    df_breakdown.to_csv(breakdown_csv, index=False)
    print(f"\nSaved inconsistency breakdown to: {breakdown_csv}")

    # =========================================================================
    # Step 5: Bootstrap 95% Confidence Intervals
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: Bootstrap 95% Confidence Intervals (1,000 resamples over pairs, seed 42)")
    print("=" * 80)
    
    # Per-pair scores across all 3632 pairs
    pair_scores_m0 = model_predictions["M0"]["pair_scores"]
    
    # M4 seed-averaged pair score per pair
    m4_seed_scores = [model_predictions["M4"][s]["pair_scores"] for s in seeds_for_runs]
    pair_scores_m4_avg = np.mean(m4_seed_scores, axis=0)
    
    rng = np.random.default_rng(seed_base)
    n_resamples = 1000
    boot_m0_accs = []
    boot_m4_accs = []
    boot_diffs = []
    
    for _ in range(n_resamples):
        boot_idx = rng.choice(num_pairs, size=num_pairs, replace=True)
        b_m0 = np.mean(pair_scores_m0[boot_idx])
        b_m4 = np.mean(pair_scores_m4_avg[boot_idx])
        boot_m0_accs.append(b_m0)
        boot_m4_accs.append(b_m4)
        boot_diffs.append(b_m4 - b_m0)
        
    boot_m0_accs = np.array(boot_m0_accs)
    boot_m4_accs = np.array(boot_m4_accs)
    boot_diffs = np.array(boot_diffs)
    
    ci_m0 = np.percentile(boot_m0_accs, [2.5, 97.5])
    ci_m4 = np.percentile(boot_m4_accs, [2.5, 97.5])
    ci_diff = np.percentile(boot_diffs, [2.5, 97.5])
    
    obs_m0 = float(np.mean(pair_scores_m0))
    obs_m4 = float(np.mean(pair_scores_m4_avg))
    obs_diff = obs_m4 - obs_m0
    
    print(f"M0 Pair Accuracy:        Observed = {obs_m0:.4f} | Bootstrap Mean = {np.mean(boot_m0_accs):.4f} | 95% CI = [{ci_m0[0]:.4f}, {ci_m0[1]:.4f}]")
    print(f"M4 (Seed-Avg) Pair Acc:  Observed = {obs_m4:.4f} | Bootstrap Mean = {np.mean(boot_m4_accs):.4f} | 95% CI = [{ci_m4[0]:.4f}, {ci_m4[1]:.4f}]")
    print(f"Difference (M4 - M0):    Observed = {obs_diff:.4f} | Bootstrap Mean = {np.mean(boot_diffs):.4f} | 95% CI = [{ci_diff[0]:.4f}, {ci_diff[1]:.4f}]")

    # =========================================================================
    # Step 6: Plot Validation Cosine Distribution
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 6: Generating Validation Cosine Distribution Figure")
    print("=" * 80)
    fig_path = "figures/cosine_distribution.png"
    plot_cosine_distribution("features/val.npz", "models/M0_threshold.json", fig_path)
    print(f"Saved figure to: {fig_path}")

    # =========================================================================
    # Final Presentation Tables
    # =========================================================================
    print("\n" + "=" * 90)
    print("TIE-ADJUSTED SUMMARY TABLE (Test Set)")
    print("=" * 90)
    hdr = f"{'Model':<8} | {'Accuracy':<17} | {'Precision':<17} | {'Recall':<17} | {'F1':<17} | {'ROC-AUC':<17} | {'Tie-Adj Pair Acc':<17}"
    print(hdr)
    print("-" * len(hdr))
    for _, r in df_summary_adj.iterrows():
        m = r["model"]
        if m == "M0":
            print(f"{m:<8} | {r['accuracy_mean']:.4f}            | {r['precision_mean']:.4f}            | {r['recall_mean']:.4f}            | "
                  f"{r['f1_mean']:.4f}            | {r['roc_auc_mean']:.4f}            | {r['pair_accuracy_mean']:.4f}")
        else:
            print(f"{m:<8} | {r['accuracy_mean']:.4f} ± {r['accuracy_std']:.4f} | {r['precision_mean']:.4f} ± {r['precision_std']:.4f} | "
                  f"{r['recall_mean']:.4f} ± {r['recall_std']:.4f} | {r['f1_mean']:.4f} ± {r['f1_std']:.4f} | "
                  f"{r['roc_auc_mean']:.4f} ± {r['roc_auc_std']:.4f} | {r['pair_accuracy_mean']:.4f} ± {r['pair_accuracy_std']:.4f}")

    print("\n" + "=" * 90)
    print("INCONSISTENCY BREAKDOWN TABLE BY RETRIEVAL METHOD & SIMILARITY QUARTILE")
    print("=" * 90)
    bhdr = f"{'Group':<28} | {'Pairs':<6} | {'Pristine Cos':<12} | {'Falsified Cos':<13} | {'CLIP Prefers Wrong':<19} | {'M0 Acc / Pair':<15} | {'M2 Acc / Pair':<15} | {'M4 Acc / Pair':<15}"
    print(bhdr)
    print("-" * len(bhdr))
    for _, r in df_breakdown.iterrows():
        g_name = r["group_name"]
        print(f"{g_name:<28} | {r['num_pairs']:<6} | {r['pristine_mean_cosine']:<12.4f} | {r['falsified_mean_cosine']:<13.4f} | "
              f"{r['clip_prefers_wrong_pct']:>5.2f}%              | {r['m0_accuracy']:.2f} / {r['m0_pair_accuracy']:.2f}      | "
              f"{r['m2_accuracy']:.2f} / {r['m2_pair_accuracy']:.2f}      | {r['m4_accuracy']:.2f} / {r['m4_pair_accuracy']:.2f}")

    print("\nPhase 3A analysis completed successfully.")

if __name__ == "__main__":
    main()
