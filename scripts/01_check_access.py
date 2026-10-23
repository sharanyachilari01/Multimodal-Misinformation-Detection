#!/usr/bin/env python3
import os
import sys
import csv
from huggingface_hub import HfApi

def main():
    repo_id = "g-luo/news-clippings"
    api = HfApi()

    repo_info = None
    working_repo_type = None

    print(f"Checking repository access for: {repo_id}")
    
    # Attempt 1: dataset
    print("\n--- Attempting repo_type='dataset' ---")
    try:
        repo_info = api.repo_info(repo_id, repo_type="dataset", files_metadata=True)
        working_repo_type = "dataset"
        print("Success accessing repo as 'dataset'.")
    except Exception as e:
        print(f"Failed to access as 'dataset': {type(e).__name__}: {e}")

    # Attempt 2: model if dataset failed
    if repo_info is None:
        print("\n--- Attempting repo_type='model' ---")
        try:
            repo_info = api.repo_info(repo_id, repo_type="model", files_metadata=True)
            working_repo_type = "model"
            print("Success accessing repo as 'model'.")
        except Exception as e:
            print(f"Failed to access as 'model': {type(e).__name__}: {e}")

    # If neither worked, stop
    if repo_info is None:
        print("\nERROR: Could not access the repository with repo_type 'dataset' or 'model'. Stopping.")
        sys.exit(1)

    print(f"\nWorking repo type: {working_repo_type}")
    
    # Extract file list and metadata
    siblings = repo_info.siblings or []
    print(f"Number of files: {len(siblings)}")

    file_entries = []
    for s in siblings:
        rfilename = s.rfilename
        size_bytes = s.size if s.size is not None else 0
        size_mb = size_bytes / (1024 * 1024)
        file_entries.append((rfilename, size_bytes, size_mb))

    # Sort by path
    file_entries.sort(key=lambda x: x[0])

    print("\nFile listing (sorted by path):")
    for path, _, size_mb in file_entries:
        print(f"  {path} ({size_mb:.2f} MB)")

    # Save to results/hf_file_listing.csv
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "hf_file_listing.csv")

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "size_mb"])
        for path, _, size_mb in file_entries:
            writer.writerow([path, f"{size_mb:.2f}"])

    print(f"\nSaved file listing to: {csv_path}")

    # Verification checks
    paths = [p for p, _, _ in file_entries]

    # 1. Annotation JSON files (e.g. containing "merged_balanced")
    has_merged_balanced = any("merged_balanced" in p and p.endswith(".json") for p in paths)
    
    # 2. CLIP image embeddings for train, val, test
    has_clip_img_train = any("clip_image_embeddings" in p and "train" in p and p.endswith(".pkl") for p in paths)
    has_clip_img_val = any("clip_image_embeddings" in p and "val" in p and p.endswith(".pkl") for p in paths)
    has_clip_img_test = any("clip_image_embeddings" in p and "test" in p and p.endswith(".pkl") for p in paths)
    has_clip_img = has_clip_img_train and has_clip_img_val and has_clip_img_test

    # 3. CLIP text embeddings for train, val, test
    has_clip_txt_train = any("clip_text_embeddings" in p and "train" in p and p.endswith(".pkl") for p in paths)
    has_clip_txt_val = any("clip_text_embeddings" in p and "val" in p and p.endswith(".pkl") for p in paths)
    has_clip_txt_test = any("clip_text_embeddings" in p and "test" in p and p.endswith(".pkl") for p in paths)
    has_clip_txt = has_clip_txt_train and has_clip_txt_val and has_clip_txt_test

    print("\n--- File Checks ---")
    print(f"Are there annotation JSON files (e.g. anything containing \"merged_balanced\")? {'Yes' if has_merged_balanced else 'No'}")
    print(f"Are there CLIP image embedding files for train, val and test? {'Yes' if has_clip_img else 'No'}")
    print(f"Are there CLIP text embedding files for train, val and test? {'Yes' if has_clip_txt else 'No'}")

if __name__ == "__main__":
    main()
