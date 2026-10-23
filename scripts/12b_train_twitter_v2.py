#!/usr/bin/env python3
"""
scripts/12b_train_twitter_v2.py: Phase 7C.2 Twitter / MediaEval 2015 Model Training & Evaluation (v2).

1. Validation split GROUPED BY image_id (never split one image across train/val), 15% of image IDs,
   stratified so BOTH classes are present; prints counts per class, and fails loudly if either class is missing.
   Drops validation tweets whose cleaned text also appears in train (reconstructed from raw tweets file by tweet_id)
   and prints the count.
2. Retrains M1-M4 (same code, hyperparameters, 3 seeds [42, 43, 44], early stopping on val) and picks M0 threshold
   on this validation set. Runs both humor variants:
     - Variant A: Humor counted as fake (fake=1, humor=1, real=0)
     - Variant B: Humor dropped from train and validation (test has none)
3. Test report per model: accuracy, fake-F1, precision, recall, ROC-AUC, PLUS the confusion matrix (TP/FP/TN/FN)
   and the predicted-fake rate. Prints majority and always-fake baselines. Image-cluster bootstrap CIs as before.
4. Prints per-event test confusion counts for M0, M2, and M4 across the 4 test events.
5. Saves results/twitter_v2_results.csv and results/twitter_v2_training.log.
"""

import os
import sys
import re
import csv
import json
import random
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

# Ensure project root is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.models import MLPClassifier, build_features
from src.train import set_seed, get_device, train_model, predict_scores


class DualLogger:
    """Tee stdout to both terminal and a log file."""
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
    full_path = os.path.join(ROOT_DIR, config_path)
    with open(full_path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_text(raw_text: str) -> str:
    """Removes t.co/http URLs and collapses whitespace; keeps everything else."""
    no_urls = re.sub(r"https?://\S+", "", raw_text)
    collapsed = " ".join(no_urls.split())
    return collapsed


def find_m0_threshold(T_val: np.ndarray, V_val: np.ndarray, y_val: np.ndarray) -> Tuple[float, float]:
    """
    Find threshold on validation set that maximizes accuracy.
    Score = cosine(T, V). Predict falsified/fake (1) when cosine < threshold.
    """
    cos_val = np.sum(T_val * V_val, axis=1)
    sorted_cos = np.sort(np.unique(cos_val))

    candidates = np.concatenate([
        [sorted_cos[0] - 1e-5],
        (sorted_cos[:-1] + sorted_cos[1:]) / 2.0,
        [sorted_cos[-1] + 1e-5]
    ])

    preds = cos_val[:, None] < candidates[None, :]
    accs = np.mean(preds == y_val[:, None], axis=0)
    best_idx = np.argmax(accs)
    best_acc = float(accs[best_idx])
    best_thresh = float(candidates[best_idx])

    return best_thresh, best_acc


def compute_metrics(y_true: np.ndarray, preds: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
    """Compute binary classification metrics for fake class (1), including confusion matrix and predicted-fake rate."""
    acc = float(accuracy_score(y_true, preds))
    prec = float(precision_score(y_true, preds, pos_label=1, zero_division=0))
    rec = float(recall_score(y_true, preds, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, preds, pos_label=1, zero_division=0))
    try:
        roc_auc = float(roc_auc_score(y_true, scores))
    except ValueError:
        roc_auc = float("nan")

    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp = int(cm[0][0]), int(cm[0][1])
    fn, tp = int(cm[1][0]), int(cm[1][1])
    pred_fake_rate = float(np.mean(preds == 1))

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "pred_fake_rate": pred_fake_rate,
    }


def run_cluster_bootstrap(
    y_test: np.ndarray,
    preds: np.ndarray,
    image_ids: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 42,
) -> Tuple[Tuple[float, float], Tuple[float, float], float, float]:
    """
    Cluster bootstrap resampling by image ID.
    Returns:
      ((acc_ci_lower, acc_ci_upper), (f1_ci_lower, f1_ci_upper), acc_boot_mean, f1_boot_mean)
    """
    unique_imgs = np.array(sorted(list(set(image_ids.tolist()))))
    n_clusters = len(unique_imgs)
    img_to_indices = {img: np.where(image_ids == img)[0] for img in unique_imgs}

    rng = np.random.default_rng(seed)
    boot_accs = []
    boot_f1s = []

    for _ in range(n_resamples):
        sampled_imgs = rng.choice(unique_imgs, size=n_clusters, replace=True)
        sampled_indices = np.concatenate([img_to_indices[img] for img in sampled_imgs])
        y_boot = y_test[sampled_indices]
        p_boot = preds[sampled_indices]

        acc = accuracy_score(y_boot, p_boot)
        f1 = f1_score(y_boot, p_boot, pos_label=1, zero_division=0)
        boot_accs.append(acc)
        boot_f1s.append(f1)

    boot_accs = np.array(boot_accs)
    boot_f1s = np.array(boot_f1s)

    ci_acc = np.percentile(boot_accs, [2.5, 97.5])
    ci_f1 = np.percentile(boot_f1s, [2.5, 97.5])

    return (
        (float(ci_acc[0]), float(ci_acc[1])),
        (float(ci_f1[0]), float(ci_f1[1])),
        float(np.mean(boot_accs)),
        float(np.mean(boot_f1s)),
    )


def main():
    log_file = os.path.join(ROOT_DIR, "results", "twitter_v2_training.log")
    logger = DualLogger(log_file)
    sys.stdout = logger

    config = load_config("config.json")
    seeds_for_runs = config.get("seeds_for_runs", [42, 43, 44])
    lr = float(config.get("lr", 0.001))
    batch_size = int(config.get("batch_size", 256))
    epochs_max = int(config.get("epochs_max", 30))
    early_stopping_patience = int(config.get("early_stopping_patience", 4))
    device = get_device()

    print("=" * 80)
    print("PHASE 7C.2: TWITTER / MEDIAEVAL 2015 MODEL TRAINING & EVALUATION (v2)")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Hyperparameters: lr={lr}, batch_size={batch_size}, epochs_max={epochs_max}, "
          f"early_stopping_patience={early_stopping_patience}")
    print(f"Seeds for runs: {seeds_for_runs}")

    # =========================================================================
    # 1. Load features/twitter_dev.npz and twitter_test.npz
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: LOAD FEATURES AND PRINT SHAPES & LABEL COUNTS")
    print("=" * 80)

    dev_path = os.path.join(ROOT_DIR, "features", "twitter_dev.npz")
    test_path = os.path.join(ROOT_DIR, "features", "twitter_test.npz")

    dev_data = np.load(dev_path, allow_pickle=True)
    test_data = np.load(test_path, allow_pickle=True)

    T_dev = dev_data["T"]
    V_dev = dev_data["V"]
    y_dev = dev_data["y"]
    raw_label_dev = dev_data["raw_label"]
    is_humor_dev = dev_data["is_humor"]
    tweet_id_dev = dev_data["tweet_id"]
    image_id_dev = dev_data["image_id"]
    event_dev = dev_data["event"]
    cos_dev = dev_data["cosine"]

    T_test = test_data["T"]
    V_test = test_data["V"]
    y_test = test_data["y"]
    raw_label_test = test_data["raw_label"]
    is_humor_test = test_data["is_humor"]
    tweet_id_test = test_data["tweet_id"]
    image_id_test = test_data["image_id"]
    event_test = test_data["event"]
    cos_test = test_data["cosine"]

    print("Devset Feature Arrays:")
    print(f"  • T (Text embeddings)  : shape = {T_dev.shape}, dtype = {T_dev.dtype}")
    print(f"  • V (Image embeddings) : shape = {V_dev.shape}, dtype = {V_dev.dtype}")
    print(f"  • y (Binary labels)    : shape = {y_dev.shape}, dtype = {y_dev.dtype}")
    print(f"  • raw_label            : shape = {raw_label_dev.shape}, dtype = {raw_label_dev.dtype}")
    print(f"  • is_humor             : shape = {is_humor_dev.shape}, dtype = {is_humor_dev.dtype}")
    print(f"  • tweet_id             : shape = {tweet_id_dev.shape}, dtype = {tweet_id_dev.dtype}")
    print(f"  • image_id             : shape = {image_id_dev.shape}, dtype = {image_id_dev.dtype}")
    print(f"  • event                : shape = {event_dev.shape}, dtype = {event_dev.dtype}")
    print(f"  • cosine               : shape = {cos_dev.shape}, dtype = {cos_dev.dtype}")

    print("\nTestset Feature Arrays:")
    print(f"  • T (Text embeddings)  : shape = {T_test.shape}, dtype = {T_test.dtype}")
    print(f"  • V (Image embeddings) : shape = {V_test.shape}, dtype = {V_test.dtype}")
    print(f"  • y (Binary labels)    : shape = {y_test.shape}, dtype = {y_test.dtype}")
    print(f"  • raw_label            : shape = {raw_label_test.shape}, dtype = {raw_label_test.dtype}")
    print(f"  • is_humor             : shape = {is_humor_test.shape}, dtype = {is_humor_test.dtype}")
    print(f"  • tweet_id             : shape = {tweet_id_test.shape}, dtype = {tweet_id_test.dtype}")
    print(f"  • image_id             : shape = {image_id_test.shape}, dtype = {image_id_test.dtype}")
    print(f"  • event                : shape = {event_test.shape}, dtype = {event_test.dtype}")
    print(f"  • cosine               : shape = {cos_test.shape}, dtype = {cos_test.dtype}")

    dev_raw_counter = Counter(raw_label_dev)
    dev_binary_counter = Counter(y_dev)
    test_raw_counter = Counter(raw_label_test)
    test_binary_counter = Counter(y_test)

    print("\nLabel Counts:")
    print(f"  • Devset Raw Labels    : {dict(dev_raw_counter)}")
    print(f"  • Devset Binary (0/1)  : {dict(dev_binary_counter)} (0=real, 1=fake+humor)")
    print(f"  • Testset Raw Labels   : {dict(test_raw_counter)}")
    print(f"  • Testset Binary (0/1) : {dict(test_binary_counter)} (0=real, 1=fake)")

    # =========================================================================
    # 2. Validation split GROUPED BY image_id (15%, stratified)
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: VALIDATION SPLIT GROUPED BY IMAGE ID (15%, STRATIFIED)")
    print("=" * 80)

    unique_dev_images = sorted(list(set(image_id_dev.tolist())))
    print(f"Total Unique Devset Images: {len(unique_dev_images)}")

    # Map each image_id to its binary label
    img_to_label: Dict[str, int] = {}
    for img in unique_dev_images:
        idx = np.where(image_id_dev == img)[0]
        # Verify all tweets for this image share the same binary label
        labs = set(y_dev[idx].tolist())
        if len(labs) > 1:
            raise RuntimeError(f"Image {img} has mixed binary labels: {labs}!")
        img_to_label[img] = int(y_dev[idx[0]])

    img_labels = [img_to_label[img] for img in unique_dev_images]
    img_label_counter = Counter(img_labels)
    print(f"Devset Images Class Balance: Real(0) = {img_label_counter[0]}, Fake(1) = {img_label_counter[1]}")

    # Stratified 15% split on image IDs
    train_images, val_images, y_train_img, y_val_img = train_test_split(
        unique_dev_images,
        img_labels,
        test_size=0.15,
        stratify=img_labels,
        random_state=42
    )

    train_images_set = set(train_images)
    val_images_set = set(val_images)

    # Strictly check that no image crosses train and val
    overlap_images = train_images_set.intersection(val_images_set)
    if overlap_images:
        raise RuntimeError(f"FATAL: Image overlap detected between train and val: {overlap_images}")

    print(f"\nImage-Level Split (15% validation, random_state=42):")
    print(f"  • Train Images : {len(train_images)} (Real={Counter(y_train_img)[0]}, Fake={Counter(y_train_img)[1]})")
    print(f"  • Val Images   : {len(val_images)} (Real={Counter(y_val_img)[0]}, Fake={Counter(y_val_img)[1]})")

    # Check that both classes are present at image level
    if Counter(y_val_img)[0] == 0 or Counter(y_val_img)[1] == 0:
        raise RuntimeError(f"FATAL: Missing class in validation images! Counts: {Counter(y_val_img)}")

    # Initial tweet index masks
    val_mask_initial = np.isin(image_id_dev, val_images)
    train_mask_initial = ~val_mask_initial

    train_indices_initial = np.where(train_mask_initial)[0]
    val_indices_initial = np.where(val_mask_initial)[0]

    print(f"\nTweet Counts before text deduplication:")
    print(f"  • Train Tweets : {len(train_indices_initial):,} | Real(0)={np.sum(y_dev[train_indices_initial]==0):,} | Fake(1)={np.sum(y_dev[train_indices_initial]==1):,}")
    print(f"  • Val Tweets   : {len(val_indices_initial):,} | Real(0)={np.sum(y_dev[val_indices_initial]==0):,} | Fake(1)={np.sum(y_dev[val_indices_initial]==1):,}")

    # Reconstruct cleaned text via tweet_id from raw tweets file
    raw_tweets_file = os.path.join(ROOT_DIR, "data", "twitter_raw", "mediaeval2015", "devset", "tweets.txt")
    tweet_id_to_clean_text: Dict[str, str] = {}
    with open(raw_tweets_file, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f, delimiter="\t")
        _ = next(reader)
        for row in reader:
            if len(row) >= 2:
                tid = row[0].strip()
                ttext = row[1]
                tweet_id_to_clean_text[tid] = clean_text(ttext)

    # Cross-split duplicate text audit
    train_clean_texts = set(tweet_id_to_clean_text[str(tweet_id_dev[idx])] for idx in train_indices_initial)
    print(f"\nUnique cleaned texts in training split: {len(train_clean_texts):,}")

    val_cross_duplicate_indices = [
        idx for idx in val_indices_initial
        if tweet_id_to_clean_text[str(tweet_id_dev[idx])] in train_clean_texts
    ]
    print(f"Validation tweets before duplicate dropping : {len(val_indices_initial):,}")
    print(f"Cross-split cleaned-text duplicates detected : {len(val_cross_duplicate_indices):,}")

    # Drop validation tweets whose cleaned text also appears in train
    val_indices_clean = np.array([idx for idx in val_indices_initial if idx not in val_cross_duplicate_indices], dtype=np.int64)
    train_indices_clean = np.array(train_indices_initial, dtype=np.int64)

    val_classes_clean = set(y_dev[val_indices_clean].tolist())
    val_real_cnt_clean = int(np.sum(y_dev[val_indices_clean] == 0))
    val_fake_cnt_clean = int(np.sum(y_dev[val_indices_clean] == 1))

    print(f"Validation tweets after duplicate dropping  : {len(val_indices_clean):,}")
    print(f"  ↳ Validation Class Counts: Real(0) = {val_real_cnt_clean:,}, Fake(1) = {val_fake_cnt_clean:,}")

    # Fail loudly if either class is missing
    if 0 not in val_classes_clean or 1 not in val_classes_clean:
        raise RuntimeError(f"FATAL: Missing class in cleaned validation set! Classes present: {val_classes_clean}")
    if val_real_cnt_clean == 0 or val_fake_cnt_clean == 0:
        raise RuntimeError(f"FATAL: Validation class count is zero! Real={val_real_cnt_clean}, Fake={val_fake_cnt_clean}")

    print("Validation split sanity check PASSED: both classes are well represented and strictly non-empty.")

    # =========================================================================
    # Baseline on test split
    # =========================================================================
    print("\n" + "=" * 80)
    print("TEST SPLIT MAJORITY-CLASS & ALWAYS-FAKE BASELINES")
    print("=" * 80)
    n_test = len(y_test)
    test_real_cnt = int(np.sum(y_test == 0))
    test_fake_cnt = int(np.sum(y_test == 1))

    # Majority baseline (predict 0 always)
    maj_preds = np.zeros(n_test, dtype=int)
    maj_cm = confusion_matrix(y_test, maj_preds, labels=[0, 1])
    maj_tn, maj_fp = int(maj_cm[0][0]), int(maj_cm[0][1])
    maj_fn, maj_tp = int(maj_cm[1][0]), int(maj_cm[1][1])
    maj_acc = test_real_cnt / n_test
    maj_fake_rate = 0.0

    # Always-fake baseline (predict 1 always)
    always_fake_preds = np.ones(n_test, dtype=int)
    af_cm = confusion_matrix(y_test, always_fake_preds, labels=[0, 1])
    af_tn, af_fp = int(af_cm[0][0]), int(af_cm[0][1])
    af_fn, af_tp = int(af_cm[1][0]), int(af_cm[1][1])
    always_fake_acc = test_fake_cnt / n_test
    always_fake_f1 = f1_score(y_test, always_fake_preds, pos_label=1, zero_division=0)
    always_fake_prec = precision_score(y_test, always_fake_preds, pos_label=1, zero_division=0)
    always_fake_rec = recall_score(y_test, always_fake_preds, pos_label=1, zero_division=0)
    af_fake_rate = 1.0

    print(f"Test Split Size: {n_test:,} | Real (0) = {test_real_cnt:,} ({test_real_cnt/n_test*100:.2f}%) | "
          f"Fake (1) = {test_fake_cnt:,} ({test_fake_cnt/n_test*100:.2f}%)")
    print(f"  • Majority-Class Baseline (Always Real=0) : Acc = {maj_acc:.4f} ({maj_acc*100:.2f}%) | "
          f"TP={maj_tp}, FP={maj_fp}, TN={maj_tn}, FN={maj_fn} | Pred-Fake Rate = {maj_fake_rate:.4f}")
    print(f"  • Always-Fake Baseline (Always Fake=1)     : Acc = {always_fake_acc:.4f}, Fake-F1 = {always_fake_f1:.4f}, "
          f"Prec = {always_fake_prec:.4f}, Rec = {always_fake_rec:.4f} | TP={af_tp}, FP={af_fp}, TN={af_tn}, FN={af_fn} | "
          f"Pred-Fake Rate = {af_fake_rate:.4f}")

    results_records: List[Dict[str, Any]] = []

    # Store baseline rows
    results_records.append({
        "variant": "all",
        "model": "Majority_Class",
        "seed": "none",
        "accuracy": maj_acc,
        "fake_f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "roc_auc": 0.5,
        "tp": maj_tp,
        "fp": maj_fp,
        "tn": maj_tn,
        "fn": maj_fn,
        "pred_fake_rate": maj_fake_rate,
        "boot_acc_ci_lower": np.nan,
        "boot_acc_ci_upper": np.nan,
        "boot_f1_ci_lower": np.nan,
        "boot_f1_ci_upper": np.nan,
    })
    results_records.append({
        "variant": "all",
        "model": "Always_Fake",
        "seed": "none",
        "accuracy": always_fake_acc,
        "fake_f1": always_fake_f1,
        "precision": always_fake_prec,
        "recall": always_fake_rec,
        "roc_auc": 0.5,
        "tp": af_tp,
        "fp": af_fp,
        "tn": af_tn,
        "fn": af_fn,
        "pred_fake_rate": af_fake_rate,
        "boot_acc_ci_lower": np.nan,
        "boot_acc_ci_upper": np.nan,
        "boot_f1_ci_lower": np.nan,
        "boot_f1_ci_upper": np.nan,
    })

    # =========================================================================
    # 3. Model Training & Evaluation for Both Variants
    # =========================================================================
    variants = [
        ("humor_as_fake", "Variant A: Humor Counted as Fake (fake=1, humor=1, real=0)"),
        ("humor_dropped", "Variant B: Humor Dropped from Train and Validation"),
    ]

    # Storage for per-event test breakdown
    event_breakdowns: Dict[str, Dict[str, Any]] = {}

    for var_key, var_title in variants:
        print("\n" + "=" * 80)
        print(f"STARTING {var_title.upper()}")
        print("=" * 80)

        # Set up train and val subsets for this variant
        if var_key == "humor_as_fake":
            tr_idx = train_indices_clean
            va_idx = val_indices_clean
        else:
            # Drop humor tweets from train and validation
            tr_idx = np.array([idx for idx in train_indices_clean if not is_humor_dev[idx]], dtype=np.int64)
            va_idx = np.array([idx for idx in val_indices_clean if not is_humor_dev[idx]], dtype=np.int64)

            # Re-check cross-split text leakage after dropping humor
            train_clean_texts_b = set(tweet_id_to_clean_text[str(tweet_id_dev[idx])] for idx in tr_idx)
            val_dups_b = [idx for idx in va_idx if tweet_id_to_clean_text[str(tweet_id_dev[idx])] in train_clean_texts_b]
            va_idx = np.array([idx for idx in va_idx if idx not in val_dups_b], dtype=np.int64)

        T_tr, V_tr, y_tr = T_dev[tr_idx], V_dev[tr_idx], y_dev[tr_idx]
        T_va, V_va, y_va = T_dev[va_idx], V_dev[va_idx], y_dev[va_idx]
        cos_va = cos_dev[va_idx]

        # Fail loudly if either class is missing
        if 0 not in set(y_va.tolist()) or 1 not in set(y_va.tolist()):
            raise RuntimeError(f"FATAL: Missing class in validation set for {var_key}! Classes: {set(y_va.tolist())}")

        print(f"Split sizes for {var_key}:")
        print(f"  • Train : N = {len(y_tr):,} | Real(0) = {np.sum(y_tr==0):,} | Fake(1) = {np.sum(y_tr==1):,}")
        print(f"  • Val   : N = {len(y_va):,} | Real(0) = {np.sum(y_va==0):,} | Fake(1) = {np.sum(y_va==1):,}")
        print(f"  • Test  : N = {len(y_test):,} | Real(0) = {np.sum(y_test==0):,} | Fake(1) = {np.sum(y_test==1):,}")

        # ---------------------------------------------------------------------
        # M0: CLIP Cosine Threshold Baseline
        # ---------------------------------------------------------------------
        print("\n--- Model M0: CLIP Cosine Baseline ---")
        best_thresh, val_acc_m0 = find_m0_threshold(T_va, V_va, y_va)
        print(f"Optimal threshold chosen on validation split: {best_thresh:.6f} (Val Accuracy: {val_acc_m0:.4f})")

        # Evaluate M0 on test split
        preds_test_m0 = (cos_test < best_thresh).astype(int)
        scores_test_m0 = 1.0 - cos_test
        m0_metrics = compute_metrics(y_test, preds_test_m0, scores_test_m0)

        # Bootstrap 95% CI resampling by image ID
        (ci_acc_m0, ci_f1_m0, mean_boot_acc_m0, mean_boot_f1_m0) = run_cluster_bootstrap(
            y_test=y_test,
            preds=preds_test_m0,
            image_ids=image_id_test,
            n_resamples=1000,
            seed=42
        )

        print(f"M0 Test Metrics: Accuracy = {m0_metrics['accuracy']:.4f}, Fake-F1 = {m0_metrics['f1']:.4f}, "
              f"Precision = {m0_metrics['precision']:.4f}, Recall = {m0_metrics['recall']:.4f}, "
              f"ROC-AUC = {m0_metrics['roc_auc']:.4f}")
        print(f"  Confusion Matrix : TP = {m0_metrics['tp']:,}, FP = {m0_metrics['fp']:,}, "
              f"TN = {m0_metrics['tn']:,}, FN = {m0_metrics['fn']:,} | Predicted-Fake Rate = {m0_metrics['pred_fake_rate']:.4f}")
        print(f"M0 Image-Cluster Bootstrap (1,000 resamples, seed 42):")
        print(f"  • Accuracy 95% CI : [{ci_acc_m0[0]:.4f}, {ci_acc_m0[1]:.4f}] (Bootstrap Mean = {mean_boot_acc_m0:.4f})")
        print(f"  • Fake-F1  95% CI : [{ci_f1_m0[0]:.4f}, {ci_f1_m0[1]:.4f}] (Bootstrap Mean = {mean_boot_f1_m0:.4f})")

        results_records.append({
            "variant": var_key,
            "model": "M0",
            "seed": "none",
            "accuracy": m0_metrics["accuracy"],
            "fake_f1": m0_metrics["f1"],
            "precision": m0_metrics["precision"],
            "recall": m0_metrics["recall"],
            "roc_auc": m0_metrics["roc_auc"],
            "tp": m0_metrics["tp"],
            "fp": m0_metrics["fp"],
            "tn": m0_metrics["tn"],
            "fn": m0_metrics["fn"],
            "pred_fake_rate": m0_metrics["pred_fake_rate"],
            "boot_acc_ci_lower": ci_acc_m0[0],
            "boot_acc_ci_upper": ci_acc_m0[1],
            "boot_f1_ci_lower": ci_f1_m0[0],
            "boot_f1_ci_upper": ci_f1_m0[1],
        })

        # ---------------------------------------------------------------------
        # M1, M2, M3, M4: Baseline MLPs
        # ---------------------------------------------------------------------
        models_to_train = ["M1", "M2", "M3", "M4"]
        model_seed_predictions = defaultdict(dict)
        model_seed_probs = defaultdict(dict)

        for model_name in models_to_train:
            print(f"\n--- Model {model_name} ---")
            X_train = build_features(model_name, T_tr, V_tr)
            X_val = build_features(model_name, T_va, V_va)
            X_test = build_features(model_name, T_test, V_test)
            in_features = X_train.shape[1]
            print(f"Feature dimension for {model_name}: {in_features}")

            seed_metrics_list = []
            seed_probs_list = []
            seed_preds_list = []

            for seed in seeds_for_runs:
                mlp_model = MLPClassifier(in_features=in_features, hidden_dim=256, dropout=0.3)
                trained_mlp, hist, best_ep, best_loss = train_model(
                    model=mlp_model,
                    X_train=X_train,
                    y_train=y_tr,
                    X_val=X_val,
                    y_val=y_va,
                    lr=lr,
                    batch_size=batch_size,
                    epochs_max=epochs_max,
                    early_stopping_patience=early_stopping_patience,
                    seed=seed,
                    device=device,
                )
                print(f"  Seed {seed}: stopped at epoch {len(hist['train_loss'])}, best epoch {best_ep} "
                      f"(best val BCE loss = {best_loss:.6f})")

                # Predict continuous probabilities on test set
                test_probs = predict_scores(trained_mlp, X_test, batch_size=batch_size, device=device)
                test_preds = (test_probs >= 0.5).astype(int)

                metrics = compute_metrics(y_test, test_preds, test_probs)
                seed_metrics_list.append(metrics)
                seed_probs_list.append(test_probs)
                seed_preds_list.append(test_preds)

                # Bootstrap for this individual seed run
                (s_ci_acc, s_ci_f1, _, _) = run_cluster_bootstrap(
                    y_test=y_test,
                    preds=test_preds,
                    image_ids=image_id_test,
                    n_resamples=1000,
                    seed=42
                )

                results_records.append({
                    "variant": var_key,
                    "model": model_name,
                    "seed": seed,
                    "accuracy": metrics["accuracy"],
                    "fake_f1": metrics["f1"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "roc_auc": metrics["roc_auc"],
                    "tp": metrics["tp"],
                    "fp": metrics["fp"],
                    "tn": metrics["tn"],
                    "fn": metrics["fn"],
                    "pred_fake_rate": metrics["pred_fake_rate"],
                    "boot_acc_ci_lower": s_ci_acc[0],
                    "boot_acc_ci_upper": s_ci_acc[1],
                    "boot_f1_ci_lower": s_ci_f1[0],
                    "boot_f1_ci_upper": s_ci_f1[1],
                })

                model_seed_predictions[model_name][seed] = test_preds
                model_seed_probs[model_name][seed] = test_probs

            # Compute seed-averaged metrics (mean and sample std, ddof=1)
            accs = [m["accuracy"] for m in seed_metrics_list]
            f1s = [m["f1"] for m in seed_metrics_list]
            precs = [m["precision"] for m in seed_metrics_list]
            recs = [m["recall"] for m in seed_metrics_list]
            aucs = [m["roc_auc"] for m in seed_metrics_list]
            tps = [m["tp"] for m in seed_metrics_list]
            fps = [m["fp"] for m in seed_metrics_list]
            tns = [m["tn"] for m in seed_metrics_list]
            fns = [m["fn"] for m in seed_metrics_list]
            fake_rates = [m["pred_fake_rate"] for m in seed_metrics_list]

            mean_acc, std_acc = float(np.mean(accs)), float(np.std(accs, ddof=1))
            mean_f1, std_f1 = float(np.mean(f1s)), float(np.std(f1s, ddof=1))
            mean_prec, std_prec = float(np.mean(precs)), float(np.std(precs, ddof=1))
            mean_rec, std_rec = float(np.mean(recs)), float(np.std(recs, ddof=1))
            mean_auc, std_auc = float(np.mean(aucs)), float(np.std(aucs, ddof=1))
            mean_tp, std_tp = float(np.mean(tps)), float(np.std(tps, ddof=1))
            mean_fp, std_fp = float(np.mean(fps)), float(np.std(fps, ddof=1))
            mean_tn, std_tn = float(np.mean(tns)), float(np.std(tns, ddof=1))
            mean_fn, std_fn = float(np.mean(fns)), float(np.std(fns, ddof=1))
            mean_fake_rate, std_fake_rate = float(np.mean(fake_rates)), float(np.std(fake_rates, ddof=1))

            # Compute seed-averaged ensemble prediction
            avg_probs = np.mean(seed_probs_list, axis=0)
            avg_preds = (avg_probs >= 0.5).astype(int)
            ens_metrics = compute_metrics(y_test, avg_preds, avg_probs)

            (avg_ci_acc, avg_ci_f1, avg_b_acc, avg_b_f1) = run_cluster_bootstrap(
                y_test=y_test,
                preds=avg_preds,
                image_ids=image_id_test,
                n_resamples=1000,
                seed=42
            )

            print(f"{model_name} Test Summary (3 seeds):")
            print(f"  • Accuracy  : {mean_acc:.4f} ± {std_acc:.4f}")
            print(f"  • Fake-F1   : {mean_f1:.4f} ± {std_f1:.4f}")
            print(f"  • Precision : {mean_prec:.4f} ± {std_prec:.4f}")
            print(f"  • Recall    : {mean_rec:.4f} ± {std_rec:.4f}")
            print(f"  • ROC-AUC   : {mean_auc:.4f} ± {std_auc:.4f}")
            print(f"  • Confusion Matrix (Mean ± Std) : TP = {mean_tp:.1f} ± {std_tp:.1f}, FP = {mean_fp:.1f} ± {std_fp:.1f}, "
                  f"TN = {mean_tn:.1f} ± {std_tn:.1f}, FN = {mean_fn:.1f} ± {std_fn:.1f}")
            print(f"  • Predicted-Fake Rate           : {mean_fake_rate:.4f} ± {std_fake_rate:.4f}")
            print(f"  • Ensemble (Seed-Averaged Probability) : Acc = {ens_metrics['accuracy']:.4f}, Fake-F1 = {ens_metrics['f1']:.4f}, "
                  f"TP = {ens_metrics['tp']}, FP = {ens_metrics['fp']}, TN = {ens_metrics['tn']}, FN = {ens_metrics['fn']}, "
                  f"Pred-Fake Rate = {ens_metrics['pred_fake_rate']:.4f}")
            print(f"  • Bootstrap 95% CI (Seed-Averaged Ensemble, image-cluster, seed 42):")
            print(f"    ↳ Accuracy 95% CI : [{avg_ci_acc[0]:.4f}, {avg_ci_acc[1]:.4f}] (Bootstrap Mean = {avg_b_acc:.4f})")
            print(f"    ↳ Fake-F1  95% CI : [{avg_ci_f1[0]:.4f}, {avg_ci_f1[1]:.4f}] (Bootstrap Mean = {avg_b_f1:.4f})")

            # Add summary rows
            results_records.append({
                "variant": var_key,
                "model": model_name,
                "seed": "seed_mean",
                "accuracy": mean_acc,
                "fake_f1": mean_f1,
                "precision": mean_prec,
                "recall": mean_rec,
                "roc_auc": mean_auc,
                "tp": mean_tp,
                "fp": mean_fp,
                "tn": mean_tn,
                "fn": mean_fn,
                "pred_fake_rate": mean_fake_rate,
                "boot_acc_ci_lower": avg_ci_acc[0],
                "boot_acc_ci_upper": avg_ci_acc[1],
                "boot_f1_ci_lower": avg_ci_f1[0],
                "boot_f1_ci_upper": avg_ci_f1[1],
            })
            results_records.append({
                "variant": var_key,
                "model": model_name,
                "seed": "seed_std",
                "accuracy": std_acc,
                "fake_f1": std_f1,
                "precision": std_prec,
                "recall": std_rec,
                "roc_auc": std_auc,
                "tp": std_tp,
                "fp": std_fp,
                "tn": std_tn,
                "fn": std_fn,
                "pred_fake_rate": std_fake_rate,
                "boot_acc_ci_lower": np.nan,
                "boot_acc_ci_upper": np.nan,
                "boot_f1_ci_lower": np.nan,
                "boot_f1_ci_upper": np.nan,
            })

        # Store predictions for per-event breakdown
        event_breakdowns[var_key] = {
            "m0_preds": preds_test_m0,
            "seed_preds": model_seed_predictions,
            "seed_probs": model_seed_probs,
        }

    # =========================================================================
    # 4. Per-Event Test Confusion Counts for M0, M2, and M4
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: PER-EVENT TEST CONFUSION COUNTS (M0, M2, M4)")
    print("=" * 80)

    unique_test_events = sorted(list(set(event_test.tolist())))
    print(f"Testset Events ({len(unique_test_events)}): {unique_test_events}\n")

    for var_key, var_title in variants:
        print(f"--- {var_title.upper()} ---")
        m0_preds = event_breakdowns[var_key]["m0_preds"]
        seed_preds = event_breakdowns[var_key]["seed_preds"]
        seed_probs = event_breakdowns[var_key]["seed_probs"]

        # Models to report per-event confusion counts
        target_models = ["M0", "M2", "M4"]

        for m_name in target_models:
            print(f"\nModel {m_name} Per-Event Confusion Counts:")
            print(f"{'Event':<10s} | {'Total':<5s} | {'Real':<5s} | {'Fake':<5s} | "
                  f"{'TP':<5s} | {'FP':<5s} | {'TN':<5s} | {'FN':<5s} | {'Acc':<7s} | {'Pred-Fake Rate':<14s}")
            print("-" * 88)

            for e in unique_test_events:
                mask_e = (event_test == e)
                n_e = int(mask_e.sum())
                r_e = int(np.sum((event_test == e) & (y_test == 0)))
                f_e = int(np.sum((event_test == e) & (y_test == 1)))

                if m_name == "M0":
                    p_e = m0_preds[mask_e]
                else:
                    # Use seed-averaged ensemble predictions
                    avg_p = np.mean([seed_probs[m_name][s][mask_e] for s in seeds_for_runs], axis=0)
                    p_e = (avg_p >= 0.5).astype(int)

                cm_e = confusion_matrix(y_test[mask_e], p_e, labels=[0, 1])
                tn_e, fp_e = int(cm_e[0][0]), int(cm_e[0][1])
                fn_e, tp_e = int(cm_e[1][0]), int(cm_e[1][1])
                acc_e = float(accuracy_score(y_test[mask_e], p_e))
                fake_rate_e = float(np.mean(p_e == 1))

                print(f"{e:<10s} | {n_e:5d} | {r_e:5d} | {f_e:5d} | "
                      f"{tp_e:5d} | {fp_e:5d} | {tn_e:5d} | {fn_e:5d} | {acc_e:7.4f} | {fake_rate_e:14.4f}")

            # Overall row
            if m_name == "M0":
                p_all = m0_preds
            else:
                avg_p_all = np.mean([seed_probs[m_name][s] for s in seeds_for_runs], axis=0)
                p_all = (avg_p_all >= 0.5).astype(int)

            cm_all = confusion_matrix(y_test, p_all, labels=[0, 1])
            tn_all, fp_all = int(cm_all[0][0]), int(cm_all[0][1])
            fn_all, tp_all = int(cm_all[1][0]), int(cm_all[1][1])
            acc_all = float(accuracy_score(y_test, p_all))
            fake_rate_all = float(np.mean(p_all == 1))

            print("-" * 88)
            print(f"{'OVERALL':<10s} | {n_test:5d} | {test_real_cnt:5d} | {test_fake_cnt:5d} | "
                  f"{tp_all:5d} | {fp_all:5d} | {tn_all:5d} | {fn_all:5d} | {acc_all:7.4f} | {fake_rate_all:14.4f}")
        print()

    # =========================================================================
    # 5. Save Results CSV
    # =========================================================================
    print("=" * 80)
    print("STEP 5: SAVING RESULTS CSV & LOG")
    print("=" * 80)

    results_df = pd.DataFrame(results_records)
    csv_out_path = os.path.join(ROOT_DIR, "results", "twitter_v2_results.csv")
    results_df.to_csv(csv_out_path, index=False)
    print(f"Saved results table to: {csv_out_path} ({len(results_df)} rows)")

    # Print final summary table to stdout
    print("\n" + "=" * 135)
    print("FINAL SUMMARY: TEST SET PERFORMANCE PER MODEL (SEED-AVERAGED)")
    print("=" * 135)
    print(f"{'Variant':<16s} | {'Model':<15s} | {'Accuracy':<17s} | {'Fake-F1':<17s} | {'Precision':<17s} | {'Recall':<17s} | {'ROC-AUC':<17s} | {'TP':<6s} | {'FP':<6s} | {'TN':<6s} | {'FN':<6s} | {'Pred-Fake Rate':<14s}")
    print("-" * 165)

    # Baselines
    maj_row = results_df[results_df["model"] == "Majority_Class"].iloc[0]
    af_row = results_df[results_df["model"] == "Always_Fake"].iloc[0]
    print(f"{'baseline':<16s} | {'Majority_Class':<15s} | {maj_row['accuracy']:8.4f}           | {maj_row['fake_f1']:8.4f}           | {maj_row['precision']:8.4f}           | {maj_row['recall']:8.4f}           | {maj_row['roc_auc']:8.4f}           | {int(maj_row['tp']):6d} | {int(maj_row['fp']):6d} | {int(maj_row['tn']):6d} | {int(maj_row['fn']):6d} | {maj_row['pred_fake_rate']:14.4f}")
    print(f"{'baseline':<16s} | {'Always_Fake':<15s} | {af_row['accuracy']:8.4f}           | {af_row['fake_f1']:8.4f}           | {af_row['precision']:8.4f}           | {af_row['recall']:8.4f}           | {af_row['roc_auc']:8.4f}           | {int(af_row['tp']):6d} | {int(af_row['fp']):6d} | {int(af_row['tn']):6d} | {int(af_row['fn']):6d} | {af_row['pred_fake_rate']:14.4f}")
    print("-" * 165)

    for var_key, _ in variants:
        sub_var = results_df[results_df["variant"] == var_key]

        # M0
        m0_sub = sub_var[sub_var["model"] == "M0"].iloc[0]
        print(f"{var_key:<16s} | {'M0':<15s} | {m0_sub['accuracy']:8.4f}           | {m0_sub['fake_f1']:8.4f}           | {m0_sub['precision']:8.4f}           | {m0_sub['recall']:8.4f}           | {m0_sub['roc_auc']:8.4f}           | {int(m0_sub['tp']):6d} | {int(m0_sub['fp']):6d} | {int(m0_sub['tn']):6d} | {int(m0_sub['fn']):6d} | {m0_sub['pred_fake_rate']:14.4f}")

        # Trained models M1..M4
        for m_name in ["M1", "M2", "M3", "M4"]:
            mean_r = sub_var[(sub_var["model"] == m_name) & (sub_var["seed"] == "seed_mean")].iloc[0]
            std_r = sub_var[(sub_var["model"] == m_name) & (sub_var["seed"] == "seed_std")].iloc[0]
            acc_str = f"{mean_r['accuracy']:.4f} ± {std_r['accuracy']:.4f}"
            f1_str = f"{mean_r['fake_f1']:.4f} ± {std_r['fake_f1']:.4f}"
            prec_str = f"{mean_r['precision']:.4f} ± {std_r['precision']:.4f}"
            rec_str = f"{mean_r['recall']:.4f} ± {std_r['recall']:.4f}"
            auc_str = f"{mean_r['roc_auc']:.4f} ± {std_r['roc_auc']:.4f}"
            tp_str = f"{mean_r['tp']:.0f}"
            fp_str = f"{mean_r['fp']:.0f}"
            tn_str = f"{mean_r['tn']:.0f}"
            fn_str = f"{mean_r['fn']:.0f}"
            fake_rate_str = f"{mean_r['pred_fake_rate']:.4f} ± {std_r['pred_fake_rate']:.4f}"
            print(f"{var_key:<16s} | {m_name:<15s} | {acc_str:<17s} | {f1_str:<17s} | {prec_str:<17s} | {rec_str:<17s} | {auc_str:<17s} | {tp_str:<6s} | {fp_str:<6s} | {tn_str:<6s} | {fn_str:<6s} | {fake_rate_str:<14s}")
        print("-" * 165)

    print("\nExecution complete. All outputs saved to:")
    print(f"  • {csv_out_path}")
    print(f"  • {log_file}")
    print("=" * 135)


if __name__ == "__main__":
    main()
