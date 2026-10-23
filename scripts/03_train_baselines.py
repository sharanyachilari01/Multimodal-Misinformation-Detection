#!/usr/bin/env python3
import os
import sys
import json
import gc
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import torch

# Ensure src module can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import MLPClassifier, build_features
from src.train import set_seed, get_device, train_model, predict_scores, evaluate_predictions, compute_pair_accuracy

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

def find_m0_threshold(T_val: np.ndarray, V_val: np.ndarray, y_val: np.ndarray):
    """
    Find threshold on validation set that maximizes accuracy.
    Score = cosine(T, V). Predict falsified (1) when cosine < threshold.
    """
    cos_val = np.sum(T_val * V_val, axis=1)
    sorted_cos = np.sort(np.unique(cos_val))
    
    candidates = np.concatenate([
        [sorted_cos[0] - 1e-5],
        (sorted_cos[:-1] + sorted_cos[1:]) / 2.0,
        [sorted_cos[-1] + 1e-5]
    ])
    
    # Vectorized search over candidate thresholds
    # Falsified when cos < candidate threshold
    preds = cos_val[:, None] < candidates[None, :]
    accs = np.mean(preds == y_val[:, None], axis=0)
    best_idx = np.argmax(accs)
    best_acc = float(accs[best_idx])
    best_thresh = float(candidates[best_idx])
    
    return best_thresh, best_acc

def plot_confusion_matrices(cm_dict: Dict[str, Any], save_path: str):
    """Plot confusion matrices for M0 and trained models."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    models_to_plot = ["M0", "M1", "M2", "M3", "M4"]
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), dpi=300)
    
    for idx, m in enumerate(models_to_plot):
        ax = axes[idx]
        cm = np.array(cm_dict[m]["representative"])
        
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(f"{m} Confusion Matrix", fontsize=12, fontweight="bold")
        tick_marks = np.arange(2)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(["Pristine (0)", "Falsified (1)"], fontsize=9)
        ax.set_yticklabels(["Pristine (0)", "Falsified (1)"], fontsize=9)
        ax.set_ylabel("True label", fontsize=10)
        ax.set_xlabel("Predicted label", fontsize=10)
        
        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                val = int(round(cm[i, j]))
                ax.text(
                    j, i, f"{val:,}",
                    horizontalalignment="center",
                    verticalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=11,
                    fontweight="semibold"
                )
                
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

def plot_val_loss_curves(loss_histories: Dict[str, Dict[int, List[float]]], save_path: str):
    """Plot validation loss curves over epochs for all trained models and seeds."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    colors = {
        "M1": "#2b5c8f",
        "M2": "#d95f02",
        "M3": "#7570b3",
        "M4": "#1b9e77"
    }
    line_styles = {42: "-", 43: "--", 44: ":"}
    
    for model_name, seeds_dict in loss_histories.items():
        for seed, val_losses in seeds_dict.items():
            epochs = list(range(1, len(val_losses) + 1))
            ax.plot(
                epochs,
                val_losses,
                label=f"{model_name} (seed {seed})",
                color=colors.get(model_name, "black"),
                linestyle=line_styles.get(seed, "-"),
                linewidth=1.8
            )
            
    ax.set_title("Validation Loss Curves across Epochs", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Validation Loss (BCEWithLogits)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=10)
    
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

def main():
    log_path = "results/phase2_log.txt"
    dual_logger = DualLogger(log_path)
    sys.stdout = dual_logger

    config = load_config("config.json")
    seed_base = config.get("seed", 42)
    seeds_for_runs = config.get("seeds_for_runs", [42, 43, 44])
    lr = float(config.get("lr", 0.001))
    batch_size = int(config.get("batch_size", 256))
    epochs_max = int(config.get("epochs_max", 30))
    early_stopping_patience = int(config.get("early_stopping_patience", 4))

    device = get_device()
    print("=" * 70)
    print("PHASE 2: Baseline Models Training & Evaluation")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Hyperparameters: lr={lr}, batch_size={batch_size}, epochs_max={epochs_max}, "
          f"early_stopping_patience={early_stopping_patience}")
    print(f"Seeds for runs: {seeds_for_runs}")

    # Load dataset features
    print("\nLoading precomputed features from features/*.npz ...")
    train_data = np.load("features/train.npz")
    val_data = np.load("features/val.npz")
    test_data = np.load("features/test.npz")

    T_train, V_train, y_train = train_data["T"], train_data["V"], train_data["y"]
    T_val, V_val, y_val = val_data["T"], val_data["V"], val_data["y"]
    T_test, V_test, y_test = test_data["T"], test_data["V"], test_data["y"]
    ids_test = test_data["ids"]

    # Print dataset sizes and class balances
    print("\n--- Dataset Summary & Class Balance ---")
    for split_name, y_split in [("Train", y_train), ("Validation", y_val), ("Test", y_test)]:
        total = len(y_split)
        falsified_cnt = int(np.sum(y_split == 1))
        pristine_cnt = int(np.sum(y_split == 0))
        pct_falsified = (falsified_cnt / total) * 100
        pct_pristine = (pristine_cnt / total) * 100
        print(f"  {split_name:10s}: Total={total:,} | Pristine={pristine_cnt:,} ({pct_pristine:.2f}%) | "
              f"Falsified={falsified_cnt:,} ({pct_falsified:.2f}%)")

    # Storage for per-seed results
    per_seed_results = []
    confusion_matrices_dict = {}
    val_loss_histories = {"M1": {}, "M2": {}, "M3": {}, "M4": {}}

    # ==========================================
    # M0: CLIP-similarity Baseline (No training)
    # ==========================================
    print("\n" + "=" * 70)
    print("EVALUATING M0: CLIP-Similarity Baseline (No training)")
    print("=" * 70)
    best_thresh, val_acc_m0 = find_m0_threshold(T_val, V_val, y_val)
    print(f"Optimal threshold chosen on validation set: {best_thresh:.6f} (Val Accuracy: {val_acc_m0:.4f})")

    # Save threshold
    os.makedirs("models", exist_ok=True)
    m0_threshold_data = {
        "optimal_threshold": best_thresh,
        "val_accuracy": val_acc_m0,
        "criterion": "max_val_accuracy",
        "decision_rule": "falsified if cosine(T, V) < threshold"
    }
    with open("models/M0_threshold.json", "w", encoding="utf-8") as f:
        json.dump(m0_threshold_data, f, indent=2)
    print("Saved M0 threshold to models/M0_threshold.json")

    # Evaluate M0 on test set
    cos_test = np.sum(T_test * V_test, axis=1)
    # Score for ROC-AUC and pair accuracy: higher means more falsified
    # (1 - cosine) has exact 1-1 monotonic relationship with -cosine
    score_test_m0 = 1.0 - cos_test
    # Binary prediction: cosine < threshold <=> (1 - cosine) > (1 - threshold)
    pred_test_m0 = (cos_test < best_thresh).astype(int)
    
    # Calculate metrics
    m0_metrics = evaluate_predictions(y_test, score_test_m0, ids_test, threshold=1.0 - best_thresh)
    # Ensure binary predictions strictly match cosine < threshold
    m0_metrics["accuracy"] = float(np.mean(pred_test_m0 == y_test))
    
    print(f"M0 Test Metrics: Accuracy={m0_metrics['accuracy']:.4f}, Precision={m0_metrics['precision']:.4f}, "
          f"Recall={m0_metrics['recall']:.4f}, F1={m0_metrics['f1']:.4f}, "
          f"ROC-AUC={m0_metrics['roc_auc']:.4f}, Pair Accuracy={m0_metrics['pair_accuracy']:.4f}")

    per_seed_results.append({
        "model": "M0",
        "seed": "none",
        **{k: m0_metrics[k] for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pair_accuracy", "tn", "fp", "fn", "tp"]}
    })
    
    confusion_matrices_dict["M0"] = {
        "none": m0_metrics["confusion_matrix"],
        "representative": m0_metrics["confusion_matrix"]
    }

    # ==========================================
    # M1 to M4: MLP Baseline Models
    # ==========================================
    trained_models = ["M1", "M2", "M3", "M4"]

    for model_name in trained_models:
        print("\n" + "=" * 70)
        print(f"PROCESSING MODEL: {model_name}")
        print("=" * 70)

        # Build train and validation features
        print(f"Building {model_name} features for train and val...")
        X_train = build_features(model_name, T_train, V_train)
        X_val = build_features(model_name, T_val, V_val)
        in_features = X_train.shape[1]
        print(f"Feature dimension for {model_name}: {in_features}")

        # Note: Do NOT touch test features until training is completed for each seed!
        model_cms = []

        for seed in seeds_for_runs:
            print(f"\n--- Training {model_name} | Seed {seed} ---")
            mlp_model = MLPClassifier(in_features=in_features, hidden_dim=256, dropout=0.3)
            
            trained_mlp, history, best_epoch, best_val_loss = train_model(
                model=mlp_model,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                lr=lr,
                batch_size=batch_size,
                epochs_max=epochs_max,
                early_stopping_patience=early_stopping_patience,
                seed=seed,
                device=device
            )
            print(f"Training completed at epoch {len(history['train_loss'])}. "
                  f"Best epoch: {best_epoch} with Val Loss: {best_val_loss:.6f}")
            
            val_loss_histories[model_name][seed] = history["val_loss"]

            # Save model checkpoint
            model_save_path = f"models/{model_name}_seed{seed}.pt"
            torch.save(trained_mlp.state_dict(), model_save_path)
            print(f"Saved weights to {model_save_path}")

            # Model selection complete. Now evaluate on test set EXACTLY ONCE.
            # Build test features just in time
            X_test = build_features(model_name, T_test, V_test)
            test_scores = predict_scores(trained_mlp, X_test, batch_size=batch_size, device=device)
            test_metrics = evaluate_predictions(y_test, test_scores, ids_test, threshold=0.5)

            print(f"Test Evaluation ({model_name}, Seed {seed}): "
                  f"Acc={test_metrics['accuracy']:.4f}, Prec={test_metrics['precision']:.4f}, "
                  f"Rec={test_metrics['recall']:.4f}, F1={test_metrics['f1']:.4f}, "
                  f"ROC-AUC={test_metrics['roc_auc']:.4f}, Pair Acc={test_metrics['pair_accuracy']:.4f}")

            per_seed_results.append({
                "model": model_name,
                "seed": seed,
                **{k: test_metrics[k] for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pair_accuracy", "tn", "fp", "fn", "tp"]}
            })
            model_cms.append(test_metrics["confusion_matrix"])

            del trained_mlp, X_test, test_scores
            if device.type == "mps":
                torch.mps.empty_cache()
            gc.collect()

        confusion_matrices_dict[model_name] = {
            f"seed_{seed}": cm for seed, cm in zip(seeds_for_runs, model_cms)
        }
        # Representative confusion matrix: average over seeds
        mean_cm = np.mean(np.array(model_cms), axis=0).tolist()
        confusion_matrices_dict[model_name]["representative"] = mean_cm

        del X_train, X_val
        gc.collect()

    # ==========================================
    # Results Processing & Aggregation
    # ==========================================
    print("\n" + "=" * 70)
    print("SAVING RESULTS AND GENERATING FIGURES")
    print("=" * 70)

    # 1. results/baselines_per_seed.csv
    df_per_seed = pd.DataFrame(per_seed_results)
    per_seed_csv = "results/baselines_per_seed.csv"
    df_per_seed.to_csv(per_seed_csv, index=False)
    print(f"Saved per-seed results to: {per_seed_csv}")

    # 2. results/baselines_summary.csv
    metrics_to_agg = ["accuracy", "precision", "recall", "f1", "roc_auc", "pair_accuracy"]
    summary_rows = []

    # M0 row
    m0_row_data = df_per_seed[df_per_seed["model"] == "M0"].iloc[0]
    m0_summary = {"model": "M0"}
    for m in metrics_to_agg:
        m0_summary[f"{m}_mean"] = float(m0_row_data[m])
        m0_summary[f"{m}_std"] = 0.0
    summary_rows.append(m0_summary)

    # M1 - M4 rows
    for model_name in trained_models:
        sub_df = df_per_seed[df_per_seed["model"] == model_name]
        m_summary = {"model": model_name}
        for m in metrics_to_agg:
            m_summary[f"{m}_mean"] = float(sub_df[m].mean())
            m_summary[f"{m}_std"] = float(sub_df[m].std(ddof=1)) if len(sub_df) > 1 else 0.0
        summary_rows.append(m_summary)

    df_summary = pd.DataFrame(summary_rows)
    summary_csv = "results/baselines_summary.csv"
    df_summary.to_csv(summary_csv, index=False)
    print(f"Saved summary results to: {summary_csv}")

    # 3. results/confusion_matrices.json
    cm_json_path = "results/confusion_matrices.json"
    with open(cm_json_path, "w", encoding="utf-8") as f:
        json.dump(confusion_matrices_dict, f, indent=2)
    print(f"Saved confusion matrices to: {cm_json_path}")

    # 4. figures/confusion_matrices.png
    cm_fig_path = "figures/confusion_matrices.png"
    plot_confusion_matrices(confusion_matrices_dict, cm_fig_path)
    print(f"Saved confusion matrix plot to: {cm_fig_path}")

    # 5. figures/val_loss_curves.png
    loss_fig_path = "figures/val_loss_curves.png"
    plot_val_loss_curves(val_loss_histories, loss_fig_path)
    print(f"Saved validation loss curves to: {loss_fig_path}")

    # ==========================================
    # Final Results Table & Anomaly Inspection
    # ==========================================
    print("\n" + "=" * 80)
    print("FINAL SUMMARY RESULTS TABLE (Test Set)")
    print("=" * 80)
    
    header = f"{'Model':<8} | {'Accuracy':<16} | {'Precision':<16} | {'Recall':<16} | {'F1':<16} | {'ROC-AUC':<16} | {'Pair Acc':<16}"
    print(header)
    print("-" * len(header))
    for _, row in df_summary.iterrows():
        m = row["model"]
        if m == "M0":
            acc_str = f"{row['accuracy_mean']:.4f}"
            prec_str = f"{row['precision_mean']:.4f}"
            rec_str = f"{row['recall_mean']:.4f}"
            f1_str = f"{row['f1_mean']:.4f}"
            auc_str = f"{row['roc_auc_mean']:.4f}"
            pair_str = f"{row['pair_accuracy_mean']:.4f}"
        else:
            acc_str = f"{row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f}"
            prec_str = f"{row['precision_mean']:.4f} ± {row['precision_std']:.4f}"
            rec_str = f"{row['recall_mean']:.4f} ± {row['recall_std']:.4f}"
            f1_str = f"{row['f1_mean']:.4f} ± {row['f1_std']:.4f}"
            auc_str = f"{row['roc_auc_mean']:.4f} ± {row['roc_auc_std']:.4f}"
            pair_str = f"{row['pair_accuracy_mean']:.4f} ± {row['pair_accuracy_std']:.4f}"
        print(f"{m:<8} | {acc_str:<16} | {prec_str:<16} | {rec_str:<16} | {f1_str:<16} | {auc_str:<16} | {pair_str:<16}")

    print("\n--- Model Behavior & Sanity Observations ---")
    m1_acc = df_summary[df_summary["model"] == "M1"]["accuracy_mean"].values[0]
    m2_acc = df_summary[df_summary["model"] == "M2"]["accuracy_mean"].values[0]
    print(f"M1 (Text-only) Mean Accuracy:  {m1_acc:.4f}")
    print(f"M2 (Image-only) Mean Accuracy: {m2_acc:.4f}")
    if abs(m1_acc - 0.50) < 0.05:
        print("Note on M1: M1 accuracy is close to 50% (chance level). This is theoretically expected "
              "because pristine and falsified pairs in NewsCLIPpings merged_balanced share identical caption text, "
              "making text-only classification non-informative without leakage.")
    else:
        print(f"Warning on M1: M1 accuracy ({m1_acc:.4f}) deviates noticeably from 50% chance level.")

    print("\nPhase 2 training and evaluation finished successfully.")

if __name__ == "__main__":
    main()
