"""
scripts/11_build_twitter_features.py: Feature extraction and dataset preparation
for MediaEval 2015 Twitter Image Verification corpus.

Builds L2-normalized 512-d CLIP embeddings (ViT-B/32) for tweets and associated images,
performs text cleaning, leakage auditing, and saves features/twitter_dev.npz and
features/twitter_test.npz.
"""

import os
import re
import csv
import sys
import time
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any, Set, Optional

import numpy as np
import torch
from PIL import Image

# Ensure project root is in path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.inconsistency import load_models, get_target_device

DEV_TWEETS_FILE = os.path.join(ROOT_DIR, "data", "twitter_raw", "mediaeval2015", "devset", "tweets.txt")
DEV_IMAGES_FILE = os.path.join(ROOT_DIR, "data", "twitter_raw", "mediaeval2015", "devset", "images.txt")
TEST_TWEETS_FILE = os.path.join(ROOT_DIR, "data", "twitter_raw", "mediaeval2015", "testset", "tweets.txt")

DEV_IMG_DIR = os.path.join(ROOT_DIR, "data", "twitter_images", "dev")
TEST_IMG_DIR = os.path.join(ROOT_DIR, "data", "twitter_images", "test")

FEATURES_DIR = os.path.join(ROOT_DIR, "features")
DEV_NPZ_OUT = os.path.join(FEATURES_DIR, "twitter_dev.npz")
TEST_NPZ_OUT = os.path.join(FEATURES_DIR, "twitter_test.npz")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif"}
URL_REGEX = re.compile(r"https?://\S+")
BATCH_SIZE = 256


def scan_image_directory(dir_path: str, split_name: str) -> Tuple[Dict[str, str], List[str], Counter, Counter, int]:
    """
    Recursively scans image directory, collects valid image files by lower-case stem,
    records non-image files, counts extensions, PIL modes, and animated images.
    """
    stem_map: Dict[str, List[str]] = defaultdict(list)
    ignored_files: List[str] = []
    ext_counts: Counter = Counter()
    mode_counts: Counter = Counter()
    animated_count: int = 0

    p_dir = Path(dir_path)
    if not p_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {dir_path}")

    for file_path in p_dir.rglob("*"):
        if not file_path.is_file():
            continue
        ext = file_path.suffix.lower()
        if ext in VALID_EXTENSIONS:
            stem = file_path.stem.lower()
            stem_map[stem].append(str(file_path))
            ext_counts[ext] += 1
            try:
                with Image.open(file_path) as img:
                    mode_counts[img.mode] += 1
                    if getattr(img, "is_animated", False):
                        animated_count += 1
            except Exception as e:
                print(f"[scan] Warning: Error probing image {file_path}: {e}")
        else:
            ignored_files.append(str(file_path))

    # Check for stem collision (multiple files with identical stem)
    final_stem_map: Dict[str, str] = {}
    collision_count = 0
    for stem, path_list in stem_map.items():
        if len(path_list) > 1:
            collision_count += 1
            print(f"[scan] Notice: Stem '{stem}' in {split_name} matches {len(path_list)} files: {path_list}")
        final_stem_map[stem] = path_list[0]

    return final_stem_map, ignored_files, ext_counts, mode_counts, animated_count


def load_dev_events_map(images_file: str) -> Dict[str, str]:
    """
    Reads devset/images.txt to map lower-case image_id to event name.
    """
    event_map: Dict[str, str] = {}
    if not os.path.isfile(images_file):
        raise FileNotFoundError(f"Devset images.txt not found: {images_file}")

    with open(images_file, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i == 0 or not line.strip():
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                img_id = parts[0].strip().lower()
                if len(parts) == 4:
                    event_map[img_id] = parts[3].strip()
                elif len(parts) == 3:
                    event_map[img_id] = parts[2].strip()
                else:
                    event_map[img_id] = "unknown"
    return event_map


def clean_text(raw_text: str) -> str:
    """
    Removes t.co/http URLs and collapses whitespace; keeps everything else.
    """
    no_urls = URL_REGEX.sub("", raw_text)
    collapsed = " ".join(no_urls.split())
    return collapsed


def parse_and_filter_split(
    tweet_file: str,
    stem_map: Dict[str, str],
    split_name: str,
    dev_events_map: Optional[Dict[str, str]] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Parses tweets file, maps to image files, handles multi-image tweets,
    cleans text, assigns labels and events, and filters tweets lacking image files.
    """
    with open(tweet_file, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        raw_rows = list(reader)

    total_tweets = len(raw_rows)
    multi_image_tweets = 0
    all_referenced_ids: Set[str] = set()
    matched_ids: Set[str] = set()
    unmatched_ids: Set[str] = set()

    kept_records: List[Dict[str, Any]] = []
    dropped_tweets_count = 0
    missing_id_tweet_counts: Counter = Counter()
    raw_label_counts_all: Counter = Counter()
    raw_label_counts_kept: Counter = Counter()
    binary_label_counts_kept: Counter = Counter()
    humor_counts_kept: Counter = Counter()
    empty_texts_count = 0

    for row in raw_rows:
        if len(row) < 7:
            continue
        tweet_id = row[0].strip()
        raw_text = row[1]
        image_field = row[3].strip()
        raw_label = row[6].strip()

        raw_label_counts_all[raw_label] += 1

        parts = [p.strip() for p in image_field.replace(",", " ").split() if p.strip()]
        if len(parts) > 1:
            multi_image_tweets += 1
        for p in parts:
            all_referenced_ids.add(p)

        first_image_id = parts[0] if parts else ""
        first_stem = first_image_id.lower()

        # Check image file presence
        if first_stem in stem_map:
            image_path = stem_map[first_stem]
            matched_ids.add(first_image_id)

            # Determine event
            if split_name == "dev":
                event_name = dev_events_map.get(first_stem, first_image_id.split("_")[0]) if dev_events_map else first_image_id.split("_")[0]
            else:
                event_name = first_image_id.split("_")[0]

            # Clean text
            cleaned_t = clean_text(raw_text)
            if not cleaned_t:
                empty_texts_count += 1

            # Binary label & humor flag
            # primary binary label: fake=1 (fake + humor), real=0
            is_humor_flag = (raw_label.lower() == "humor")
            binary_label = 1 if (raw_label.lower() in {"fake", "humor"}) else 0

            raw_label_counts_kept[raw_label] += 1
            binary_label_counts_kept[binary_label] += 1
            humor_counts_kept[is_humor_flag] += 1

            kept_records.append({
                "tweet_id": tweet_id,
                "raw_text": raw_text,
                "cleaned_text": cleaned_t,
                "image_id": first_image_id,
                "image_path": image_path,
                "event": event_name,
                "raw_label": raw_label,
                "binary_label": binary_label,
                "is_humor": is_humor_flag
            })
        else:
            dropped_tweets_count += 1
            unmatched_ids.add(first_image_id)
            missing_id_tweet_counts[first_image_id] += 1

    stats = {
        "split_name": split_name,
        "total_tweets": total_tweets,
        "multi_image_tweets": multi_image_tweets,
        "unique_referenced_ids_all_slots": len(all_referenced_ids),
        "unique_referenced_ids_first_slot": len(matched_ids) + len(unmatched_ids),
        "matched_unique_ids_count": len(matched_ids),
        "unmatched_unique_ids_count": len(unmatched_ids),
        "unmatched_unique_ids": sorted(list(unmatched_ids)),
        "dropped_tweets_count": dropped_tweets_count,
        "missing_id_tweet_counts": dict(missing_id_tweet_counts),
        "kept_tweets_count": len(kept_records),
        "raw_label_counts_all": dict(raw_label_counts_all),
        "raw_label_counts_kept": dict(raw_label_counts_kept),
        "binary_label_counts_kept": dict(binary_label_counts_kept),
        "humor_counts_kept": dict(humor_counts_kept),
        "empty_texts_count": empty_texts_count
    }

    return kept_records, stats


def compute_clip_embeddings(
    records: List[Dict[str, Any]],
    tokenizer: Any,
    processor: Any,
    model: Any,
    device: torch.device,
    split_name: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Computes CLIP embeddings:
    - Once per unique image file
    - Once per unique cleaned text
    Skips and counts unreadable images; never crashes.
    Reports truncated texts exceeding 77 tokens.
    Returns:
        T: (N, 512) L2-normalized text features
        V: (N, 512) L2-normalized image features
        cosine: (N,) cosine similarities
        unreadable_images_count: int
        truncated_texts_count: int
    """
    # 1. Unique images
    unique_image_paths = sorted(list(set(r["image_path"] for r in records)))
    print(f"\n[{split_name.upper()}] Encoding {len(unique_image_paths)} unique image files with CLIP ViT-B/32 on {device}...")
    image_cache: Dict[str, np.ndarray] = {}
    unreadable_count = 0

    t_img_start = time.time()
    for idx, img_path in enumerate(unique_image_paths, 1):
        try:
            with Image.open(img_path) as raw_img:
                rgb_img = raw_img.convert("RGB")
                inputs = processor(images=rgb_img, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(device)
                with torch.no_grad():
                    img_out = model.get_image_features(pixel_values=pixel_values)
                    feat = img_out.pooler_output if hasattr(img_out, "pooler_output") and img_out.pooler_output is not None else img_out
                    norm = feat / feat.norm(dim=-1, keepdim=True)
                    emb = norm.cpu().numpy()[0].astype(np.float32)
                image_cache[img_path] = emb
        except Exception as e:
            unreadable_count += 1
            print(f"[{split_name.upper()}] Error reading image {img_path}: {e}")
            # Fallback zero vector in case of catastrophic read failure
            image_cache[img_path] = np.zeros(512, dtype=np.float32)

    t_img_end = time.time()
    print(f"[{split_name.upper()}] Image encoding complete in {t_img_end - t_img_start:.2f}s. Unreadable images: {unreadable_count}")

    # 2. Unique texts and token truncation check
    unique_texts = sorted(list(set(r["cleaned_text"] for r in records)))
    print(f"[{split_name.upper()}] Encoding {len(unique_texts)} unique cleaned tweet texts (batch size {BATCH_SIZE})...")

    truncated_texts_count = 0
    for txt in unique_texts:
        tokens = tokenizer.encode(txt if txt else " ", add_special_tokens=True)
        if len(tokens) > 77:
            truncated_texts_count += 1

    text_cache: Dict[str, np.ndarray] = {}
    t_txt_start = time.time()

    for i in range(0, len(unique_texts), BATCH_SIZE):
        batch = unique_texts[i:i + BATCH_SIZE]
        # Use single space fallback for empty text to maintain tokenizer compatibility
        batch_input = [txt if txt else " " for txt in batch]
        tokens = tokenizer(batch_input, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
        with torch.no_grad():
            txt_out = model.get_text_features(**tokens)
            feat = txt_out.pooler_output if hasattr(txt_out, "pooler_output") and txt_out.pooler_output is not None else txt_out
            norm = feat / feat.norm(dim=-1, keepdim=True)
            embs = norm.cpu().numpy().astype(np.float32)

        for txt, emb in zip(batch, embs):
            text_cache[txt] = emb

    t_txt_end = time.time()
    print(f"[{split_name.upper()}] Text encoding complete in {t_txt_end - t_txt_start:.2f}s. Unique texts truncated (>77 tokens): {truncated_texts_count}")

    # 3. Assemble full feature matrices
    N = len(records)
    T = np.zeros((N, 512), dtype=np.float32)
    V = np.zeros((N, 512), dtype=np.float32)

    for i, r in enumerate(records):
        T[i] = text_cache[r["cleaned_text"]]
        V[i] = image_cache[r["image_path"]]

    cosine = np.sum(T * V, axis=-1).astype(np.float32)
    return T, V, cosine, unreadable_count, truncated_texts_count


def run_leakage_checks(
    dev_records: List[Dict[str, Any]],
    test_records: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Checks exact duplicate cleaned texts within dev, within test,
    and shared between dev and test.
    """
    dev_texts = [r["cleaned_text"] for r in dev_records]
    test_texts = [r["cleaned_text"] for r in test_records]

    dev_unique = set(dev_texts)
    test_unique = set(test_texts)

    dev_internal_dups = len(dev_texts) - len(dev_unique)
    test_internal_dups = len(test_texts) - len(test_unique)

    shared_texts = dev_unique.intersection(test_unique)

    return {
        "dev_total_texts": len(dev_texts),
        "dev_unique_texts": len(dev_unique),
        "dev_exact_duplicates": dev_internal_dups,
        "test_total_texts": len(test_texts),
        "test_unique_texts": len(test_unique),
        "test_exact_duplicates": test_internal_dups,
        "shared_unique_texts_count": len(shared_texts),
        "shared_unique_texts": sorted(list(shared_texts))
    }


def main():
    print("=" * 80)
    print("PHASE 7B: BUILD TWITTER MULTIMODAL FEATURES (MEDIAEVAL 2015)")
    print("=" * 80)

    # 1. Scan image directories
    print("\n--- [Step 1] Scanning Extracted Image Archives ---")
    dev_stem_map, dev_ignored, dev_exts, dev_modes, dev_anim = scan_image_directory(DEV_IMG_DIR, "dev")
    test_stem_map, test_ignored, test_exts, test_modes, test_anim = scan_image_directory(TEST_IMG_DIR, "test")

    print(f"Dev Image Archive ({DEV_IMG_DIR}):")
    print(f"  • Image files found: {len(dev_stem_map)}")
    print(f"  • File extension breakdown: {dict(dev_exts)}")
    print(f"  • PIL Image modes: {dict(dev_modes)}")
    print(f"  • Animated images detected: {dev_anim}")
    print(f"  • Non-image files ignored: {len(dev_ignored)} {dev_ignored}")

    print(f"\nTest Image Archive ({TEST_IMG_DIR}):")
    print(f"  • Image files found: {len(test_stem_map)}")
    print(f"  • File extension breakdown: {dict(test_exts)}")
    print(f"  • PIL Image modes: {dict(test_modes)}")
    print(f"  • Animated images detected: {test_anim}")
    print(f"  • Non-image files ignored: {len(test_ignored)} {test_ignored}")

    # 2. Load Dev events mapping
    dev_events_map = load_dev_events_map(DEV_IMAGES_FILE)
    print(f"\nLoaded devset event annotations for {len(dev_events_map)} images from devset/images.txt.")

    # 3. Parse and filter tweets for dev and test
    print("\n--- [Step 2] Mapping Tweet Records & Image Filtering ---")
    dev_records, dev_stats = parse_and_filter_split(DEV_TWEETS_FILE, dev_stem_map, "dev", dev_events_map)
    test_records, test_stats = parse_and_filter_split(TEST_TWEETS_FILE, test_stem_map, "test")

    print(f"Devset Tweets Mapping:")
    print(f"  • Total tweets in file: {dev_stats['total_tweets']:,}")
    print(f"  • Multi-image tweets (using first image): {dev_stats['multi_image_tweets']}")
    print(f"  • Unique image IDs in all slots: {dev_stats['unique_referenced_ids_all_slots']}")
    print(f"  • Matched unique image IDs: {dev_stats['matched_unique_ids_count']}")
    print(f"  • Unmatched unique image IDs: {dev_stats['unmatched_unique_ids_count']} -> {dev_stats['unmatched_unique_ids']}")
    print(f"  • Tweets referencing missing image files: {dev_stats['dropped_tweets_count']} (dropped from saved npz)")
    print(f"    ↳ Breakdown by missing ID: {dev_stats['missing_id_tweet_counts']}")
    print(f"  • Kept tweets with valid image files: {dev_stats['kept_tweets_count']:,}")
    print(f"  • Texts becoming empty after URL removal: {dev_stats['empty_texts_count']}")
    print(f"  • Raw label distribution (all):  {dev_stats['raw_label_counts_all']}")
    print(f"  • Raw label distribution (kept): {dev_stats['raw_label_counts_kept']}")
    print(f"  • Binary label distribution (0=real, 1=fake+humor): {dev_stats['binary_label_counts_kept']}")
    print(f"  • Humor flag count (is_humor=True): {dev_stats['humor_counts_kept'].get(True, 0)}")

    print(f"\nTestset Tweets Mapping:")
    print(f"  • Total tweets in file: {test_stats['total_tweets']:,}")
    print(f"  • Multi-image tweets (using first image): {test_stats['multi_image_tweets']}")
    print(f"  • Unique image IDs in all slots: {test_stats['unique_referenced_ids_all_slots']}")
    print(f"  • Matched unique image IDs: {test_stats['matched_unique_ids_count']}")
    print(f"  • Unmatched unique image IDs: {test_stats['unmatched_unique_ids_count']} -> {test_stats['unmatched_unique_ids']}")
    print(f"  • Tweets referencing missing image files: {test_stats['dropped_tweets_count']} (dropped from saved npz)")
    print(f"    ↳ Breakdown by missing ID: {test_stats['missing_id_tweet_counts']}")
    print(f"  • Kept tweets with valid image files: {test_stats['kept_tweets_count']:,}")
    print(f"  • Texts becoming empty after URL removal: {test_stats['empty_texts_count']}")
    print(f"  • Raw label distribution (all):  {test_stats['raw_label_counts_all']}")
    print(f"  • Raw label distribution (kept): {test_stats['raw_label_counts_kept']}")
    print(f"  • Binary label distribution (0=real, 1=fake+humor): {test_stats['binary_label_counts_kept']}")
    print(f"  • Humor flag count (is_humor=True): {test_stats['humor_counts_kept'].get(True, 0)}")

    # 4. Leakage checks
    print("\n--- [Step 3] Leakage Checks (Exact Cleaned Texts) ---")
    leak_info = run_leakage_checks(dev_records, test_records)
    print(f"• Dev split cleaned texts : {leak_info['dev_total_texts']:,} total | {leak_info['dev_unique_texts']:,} unique | {leak_info['dev_exact_duplicates']:,} internal duplicates")
    print(f"• Test split cleaned texts: {leak_info['test_total_texts']:,} total | {leak_info['test_unique_texts']:,} unique | {leak_info['test_exact_duplicates']:,} internal duplicates")
    print(f"• Shared cleaned texts between dev and test: {leak_info['shared_unique_texts_count']}")
    if leak_info['shared_unique_texts_count'] > 0:
        print(f"  ↳ Sample shared texts: {leak_info['shared_unique_texts'][:5]}")
    else:
        print(f"  ↳ No shared cleaned texts detected between dev and test splits.")

    # 5. Load CLIP and extract features
    print("\n--- [Step 4] Extracting CLIP Features ---")
    models = load_models()
    clip_model = models["clip_model"]
    clip_processor = models["clip_processor"]
    clip_tokenizer = models["clip_tokenizer"]
    device = models["device"]

    T_dev, V_dev, cos_dev, dev_unread, dev_trunc = compute_clip_embeddings(
        dev_records, clip_tokenizer, clip_processor, clip_model, device, "dev"
    )
    T_test, V_test, cos_test, test_unread, test_trunc = compute_clip_embeddings(
        test_records, clip_tokenizer, clip_processor, clip_model, device, "test"
    )

    print(f"\nDevset CLIP Truncation Summary: {dev_trunc} unique texts truncated at 77 tokens.")
    print(f"Testset CLIP Truncation Summary: {test_trunc} unique texts truncated at 77 tokens.")

    # 6. Save features to NPZ
    print("\n--- [Step 5] Saving NPZ Archives ---")
    os.makedirs(FEATURES_DIR, exist_ok=True)

    y_dev = np.array([r["binary_label"] for r in dev_records], dtype=np.int64)
    raw_label_dev = np.array([r["raw_label"] for r in dev_records], dtype=object)
    is_humor_dev = np.array([r["is_humor"] for r in dev_records], dtype=bool)
    tweet_id_dev = np.array([r["tweet_id"] for r in dev_records], dtype=object)
    image_id_dev = np.array([r["image_id"] for r in dev_records], dtype=object)
    event_dev = np.array([r["event"] for r in dev_records], dtype=object)

    np.savez_compressed(
        DEV_NPZ_OUT,
        T=T_dev,
        V=V_dev,
        y=y_dev,
        raw_label=raw_label_dev,
        is_humor=is_humor_dev,
        tweet_id=tweet_id_dev,
        image_id=image_id_dev,
        event=event_dev,
        cosine=cos_dev
    )
    print(f"Saved Dev Features: {DEV_NPZ_OUT} ({os.path.getsize(DEV_NPZ_OUT):,} bytes)")
    print(f"  ↳ T: {T_dev.shape}, V: {V_dev.shape}, y: {y_dev.shape}, cosine: {cos_dev.shape}")

    y_test = np.array([r["binary_label"] for r in test_records], dtype=np.int64)
    raw_label_test = np.array([r["raw_label"] for r in test_records], dtype=object)
    is_humor_test = np.array([r["is_humor"] for r in test_records], dtype=bool)
    tweet_id_test = np.array([r["tweet_id"] for r in test_records], dtype=object)
    image_id_test = np.array([r["image_id"] for r in test_records], dtype=object)
    event_test = np.array([r["event"] for r in test_records], dtype=object)

    np.savez_compressed(
        TEST_NPZ_OUT,
        T=T_test,
        V=V_test,
        y=y_test,
        raw_label=raw_label_test,
        is_humor=is_humor_test,
        tweet_id=tweet_id_test,
        image_id=image_id_test,
        event=event_test,
        cosine=cos_test
    )
    print(f"Saved Test Features: {TEST_NPZ_OUT} ({os.path.getsize(TEST_NPZ_OUT):,} bytes)")
    print(f"  ↳ T: {T_test.shape}, V: {V_test.shape}, y: {y_test.shape}, cosine: {cos_test.shape}")

    # 7. Descriptive statistics (CLIP cosine by label)
    print("\n" + "=" * 80)
    print("STEP 6: MEAN CLIP COSINE SIMILARITY (DESCRIPTIVE ONLY)")
    print("=" * 80)

    # Dev statistics
    dev_fake_mask = (y_dev == 1)
    dev_real_mask = (y_dev == 0)
    dev_humor_mask = is_humor_dev
    dev_pure_fake_mask = (raw_label_dev == "fake")

    print(f"Devset (N = {len(y_dev):,}):")
    print(f"  • Real (y = 0)         : count = {dev_real_mask.sum():,}, mean CLIP cosine = {cos_dev[dev_real_mask].mean():.4f} (std = {cos_dev[dev_real_mask].std():.4f})")
    print(f"  • Fake (y = 1, all)    : count = {dev_fake_mask.sum():,}, mean CLIP cosine = {cos_dev[dev_fake_mask].mean():.4f} (std = {cos_dev[dev_fake_mask].std():.4f})")
    print(f"    ↳ Pure Fake          : count = {dev_pure_fake_mask.sum():,}, mean CLIP cosine = {cos_dev[dev_pure_fake_mask].mean():.4f} (std = {cos_dev[dev_pure_fake_mask].std():.4f})")
    print(f"    ↳ Humor (subset)     : count = {dev_humor_mask.sum():,}, mean CLIP cosine = {cos_dev[dev_humor_mask].mean():.4f} (std = {cos_dev[dev_humor_mask].std():.4f})")

    # Test statistics
    test_fake_mask = (y_test == 1)
    test_real_mask = (y_test == 0)

    print(f"\nTestset (N = {len(y_test):,}):")
    print(f"  • Real (y = 0)         : count = {test_real_mask.sum():,}, mean CLIP cosine = {cos_test[test_real_mask].mean():.4f} (std = {cos_test[test_real_mask].std():.4f})")
    print(f"  • Fake (y = 1)         : count = {test_fake_mask.sum():,}, mean CLIP cosine = {cos_test[test_fake_mask].mean():.4f} (std = {cos_test[test_fake_mask].std():.4f})")
    print("=" * 80)


if __name__ == "__main__":
    main()
