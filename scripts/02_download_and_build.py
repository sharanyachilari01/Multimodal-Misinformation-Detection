#!/usr/bin/env python3
import os
import sys
import gc
import json
import pickle
import numpy as np
from huggingface_hub import hf_hub_download

def load_config():
    config_path = "config.json"
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"seed": 42}

def download_files():
    repo_id = "g-luo/news-clippings"
    repo_type = "model"
    local_dir = "data/hf"
    
    files_to_download = [
        "data/merged_balanced/train.json",
        "data/merged_balanced/val.json",
        "data/merged_balanced/test.json",
        "embeddings/clip_image_embeddings/clip_image_embeddings_train.pkl",
        "embeddings/clip_image_embeddings/clip_image_embeddings_val.pkl",
        "embeddings/clip_image_embeddings/clip_image_embeddings_test.pkl",
        "embeddings/clip_text_embeddings/clip_text_embeddings_train.pkl",
        "embeddings/clip_text_embeddings/clip_text_embeddings_val.pkl",
        "embeddings/clip_text_embeddings/clip_text_embeddings_test.pkl",
    ]
    
    print("=" * 60)
    print("STEP 1: Downloading / Verifying Required Dataset Files")
    print("=" * 60)
    
    downloaded_paths = {}
    for rel_path in files_to_download:
        dest_path = os.path.join(local_dir, rel_path)
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
            size_bytes = os.path.getsize(dest_path)
            size_mb = size_bytes / (1024 * 1024)
            print(f"[EXISTS] {rel_path} ({size_mb:.2f} MB, {size_bytes:,} bytes)")
            downloaded_paths[rel_path] = dest_path
        else:
            print(f"[DOWNLOADING] {rel_path} ...")
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=rel_path,
                repo_type=repo_type,
                local_dir=local_dir
            )
            size_bytes = os.path.getsize(downloaded)
            size_mb = size_bytes / (1024 * 1024)
            print(f"[DOWNLOADED] {rel_path} ({size_mb:.2f} MB, {size_bytes:,} bytes)")
            downloaded_paths[rel_path] = downloaded
            
    return downloaded_paths

def main():
    config = load_config()
    seed = config.get("seed", 42)
    rng = np.random.default_rng(seed)
    
    print(f"Loaded config: seed={seed}")
    
    # 1. Download/Verify files
    download_files()
    
    splits = ["train", "val", "test"]
    annotations_per_split = {}
    caption_ids_per_split = {}
    image_ids_per_split = {}
    split_summaries = {}
    
    print("\n" + "=" * 60)
    print("STEP 2: Inspecting Annotation JSONs")
    print("=" * 60)
    
    for split in splits:
        ann_path = os.path.join("data/hf/data/merged_balanced", f"{split}.json")
        with open(ann_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        annotations = data.get("annotations", [])
        annotations_per_split[split] = annotations
        
        sample_keys = list(annotations[0].keys()) if annotations else []
        n_samples = len(annotations)
        
        falsified_true = sum(1 for a in annotations if a.get("falsified") is True)
        falsified_false = sum(1 for a in annotations if a.get("falsified") is False)
        pct_true = (falsified_true / n_samples * 100) if n_samples > 0 else 0.0
        pct_false = (falsified_false / n_samples * 100) if n_samples > 0 else 0.0
        
        print(f"\n--- Split: {split} ---")
        print(f"Top-level JSON keys: {list(data.keys())}")
        print(f"Annotation record keys: {sample_keys}")
        print(f"Total samples: {n_samples:,}")
        print(f"Class Balance: Falsified=True: {falsified_true:,} ({pct_true:.2f}%), "
              f"Falsified=False: {falsified_false:,} ({pct_false:.2f}%)")
        
        c_ids = [a["id"] for a in annotations]
        img_ids = [a["image_id"] for a in annotations]
        caption_ids_per_split[split] = set(c_ids)
        image_ids_per_split[split] = set(img_ids)
        
        # Structure checks per split
        n_pairs = n_samples // 2
        matching_pairs = 0
        for k in range(n_pairs):
            a0 = annotations[2 * k]
            a1 = annotations[2 * k + 1]
            if a0.get("falsified") is False and a1.get("falsified") is True and a0.get("id") == a1.get("id"):
                matching_pairs += 1
                
        frac_consecutive = (matching_pairs / n_pairs) if n_pairs > 0 else 0.0
        print(f"Structure check: consecutive pairs (False, True with same caption id): "
              f"{matching_pairs:,} / {n_pairs:,} (fraction: {frac_consecutive:.4f})")
        print(f"Unique caption ids: {len(caption_ids_per_split[split]):,}")
        print(f"Unique image ids: {len(image_ids_per_split[split]):,}")
        
        split_summaries[split] = {
            "num_samples": n_samples,
            "falsified_true_count": falsified_true,
            "falsified_false_count": falsified_false,
            "falsified_true_pct": round(pct_true, 2),
            "falsified_false_pct": round(pct_false, 2),
            "consecutive_pairs_total": n_pairs,
            "consecutive_pairs_matching": matching_pairs,
            "consecutive_pairs_fraction": round(frac_consecutive, 4),
            "unique_caption_ids": len(caption_ids_per_split[split]),
            "unique_image_ids": len(image_ids_per_split[split])
        }

    print("\n" + "=" * 60)
    print("STEP 3: Data Leakage Overlap Checks Across Splits")
    print("=" * 60)
    
    pairs_to_check = [("train", "val"), ("train", "test"), ("val", "test")]
    overlap_results = {}
    for s1, s2 in pairs_to_check:
        cap_overlap = len(caption_ids_per_split[s1].intersection(caption_ids_per_split[s2]))
        img_overlap = len(image_ids_per_split[s1].intersection(image_ids_per_split[s2]))
        pair_key = f"{s1}_{s2}"
        overlap_results[pair_key] = {
            "caption_id_overlap": cap_overlap,
            "image_id_overlap": img_overlap
        }
        print(f"Overlap between {s1} and {s2}:")
        print(f"  Caption ID overlap: {cap_overlap:,}")
        print(f"  Image ID overlap:   {img_overlap:,}")

    print("\n" + "=" * 60)
    print("STEP 4: Loading Pickles, Embedding Checks, and Feature Extraction")
    print("=" * 60)
    
    os.makedirs("features", exist_ok=True)
    
    for split in splits:
        print(f"\n>>> Processing split: {split}")
        annotations = annotations_per_split[split]
        n_samples = len(annotations)
        
        txt_pkl_path = os.path.join(
            "data/hf/embeddings/clip_text_embeddings",
            f"clip_text_embeddings_{split}.pkl"
        )
        img_pkl_path = os.path.join(
            "data/hf/embeddings/clip_image_embeddings",
            f"clip_image_embeddings_{split}.pkl"
        )
        
        print(f"Loading text embeddings from: {txt_pkl_path}")
        with open(txt_pkl_path, "rb") as f:
            text_dict = pickle.load(f)
            
        print(f"Loading image embeddings from: {img_pkl_path}")
        with open(img_pkl_path, "rb") as f:
            image_dict = pickle.load(f)
            
        # Inspect text embeddings
        txt_keys = list(text_dict.keys())
        first_txt_vec = text_dict[txt_keys[0]]
        print(f"CLIP Text Embeddings - Number of keys: {len(txt_keys):,}, "
              f"Array shape: {first_txt_vec.shape}, Dtype: {first_txt_vec.dtype}")
        
        sample_size_txt = min(1000, len(txt_keys))
        sampled_txt_keys = rng.choice(txt_keys, size=sample_size_txt, replace=False)
        txt_sample_vecs = np.stack([text_dict[k] for k in sampled_txt_keys])
        txt_l2_norms = np.linalg.norm(txt_sample_vecs, axis=1)
        mean_txt_norm = float(np.mean(txt_l2_norms))
        print(f"CLIP Text Embeddings - Mean L2 norm of {sample_size_txt} sampled vectors: {mean_txt_norm:.6f}")
        
        # Inspect image embeddings
        img_keys = list(image_dict.keys())
        first_img_vec = image_dict[img_keys[0]]
        print(f"CLIP Image Embeddings - Number of keys: {len(img_keys):,}, "
              f"Array shape: {first_img_vec.shape}, Dtype: {first_img_vec.dtype}")
        
        sample_size_img = min(1000, len(img_keys))
        sampled_img_keys = rng.choice(img_keys, size=sample_size_img, replace=False)
        img_sample_vecs = np.stack([image_dict[k] for k in sampled_img_keys])
        img_l2_norms = np.linalg.norm(img_sample_vecs, axis=1)
        mean_img_norm = float(np.mean(img_l2_norms))
        print(f"CLIP Image Embeddings - Mean L2 norm of {sample_size_img} sampled vectors: {mean_img_norm:.6f}")
        
        # Check missing ids
        missing_txt_ids = []
        missing_img_ids = []
        
        for a in annotations:
            cid = a["id"]
            iid = a["image_id"]
            if cid not in text_dict:
                missing_txt_ids.append(cid)
            if iid not in image_dict:
                missing_img_ids.append(iid)
                
        print(f"Missing caption IDs in text embeddings: {len(missing_txt_ids)}")
        print(f"Missing image IDs in image embeddings: {len(missing_img_ids)}")
        
        if len(missing_txt_ids) > 0 or len(missing_img_ids) > 0:
            if missing_txt_ids:
                print(f"ERROR: Example missing caption ID: {missing_txt_ids[0]}")
            if missing_img_ids:
                print(f"ERROR: Example missing image ID: {missing_img_ids[0]}")
            print("Stopping execution as required; not dropping samples silently.")
            sys.exit(1)
            
        # Build arrays
        dim = first_txt_vec.shape[0]
        T = np.empty((n_samples, dim), dtype=np.float32)
        V = np.empty((n_samples, dim), dtype=np.float32)
        y = np.empty(n_samples, dtype=np.int64)
        ids_arr = np.empty(n_samples, dtype=np.int64)
        image_ids_arr = np.empty(n_samples, dtype=np.int64)
        
        for idx, a in enumerate(annotations):
            T[idx] = text_dict[a["id"]]
            V[idx] = image_dict[a["image_id"]]
            y[idx] = 1 if a.get("falsified") is True else 0
            ids_arr[idx] = a["id"]
            image_ids_arr[idx] = a["image_id"]
            
        # Free memory of pickles immediately
        del text_dict
        del image_dict
        gc.collect()
        print("Pickle dictionaries deleted and memory collected.")
        
        # Ensure L2-normalization
        T_norms = np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
        V_norms = np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
        T = (T / T_norms).astype(np.float32)
        V = (V / V_norms).astype(np.float32)
        
        final_T_norm = float(np.mean(np.linalg.norm(T[:1000], axis=1)))
        final_V_norm = float(np.mean(np.linalg.norm(V[:1000], axis=1)))
        
        npz_out = os.path.join("features", f"{split}.npz")
        np.savez_compressed(
            npz_out,
            T=T,
            V=V,
            y=y,
            ids=ids_arr,
            image_ids=image_ids_arr
        )
        npz_size_mb = os.path.getsize(npz_out) / (1024 * 1024)
        print(f"Saved {split} features to {npz_out} ({npz_size_mb:.2f} MB)")
        print(f"  Arrays: T={T.shape} float32, V={V.shape} float32, y={y.shape} int64, "
              f"ids={ids_arr.shape}, image_ids={image_ids_arr.shape}")
        print(f"  Verification mean L2 norm: T={final_T_norm:.6f}, V={final_V_norm:.6f}")
        
        split_summaries[split]["raw_text_norm_sampled"] = round(mean_txt_norm, 6)
        split_summaries[split]["raw_image_norm_sampled"] = round(mean_img_norm, 6)
        split_summaries[split]["final_T_norm_sampled"] = round(final_T_norm, 6)
        split_summaries[split]["final_V_norm_sampled"] = round(final_V_norm, 6)
        split_summaries[split]["embedding_dim"] = dim
        split_summaries[split]["dtype"] = "float32"
        split_summaries[split]["npz_path"] = npz_out
        split_summaries[split]["npz_size_mb"] = round(npz_size_mb, 2)
        
        # Free feature arrays
        del T, V, y, ids_arr, image_ids_arr
        gc.collect()

    # Save summary to results/dataset_summary.json
    summary_out = {
        "dataset": "NewsCLIPpings",
        "seed": seed,
        "splits": split_summaries,
        "leakage_overlap": overlap_results
    }
    
    os.makedirs("results", exist_ok=True)
    summary_path = os.path.join("results", "dataset_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2)
        
    print("\n" + "=" * 60)
    print("EXECUTION SUMMARY")
    print("=" * 60)
    print(f"Script executed: scripts/02_download_and_build.py")
    print(f"Features saved to: features/train.npz, features/val.npz, features/test.npz")
    print(f"Dataset summary saved to: {summary_path}")

if __name__ == "__main__":
    main()
