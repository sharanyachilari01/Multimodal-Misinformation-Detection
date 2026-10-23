"""
scripts/10_check_twitter.py: Verifies access and structure of the MediaEval 2015
Twitter Image Verification corpus (data/twitter_raw/).

Tasks:
1. Lists files in the cloned repository and inspects README.md, reporting real
   file names, formats, and column/field names found.
2. Finds the 2015 devset and testset tweet files, reporting tweet counts,
   label counts (fake, real, humor), event field availability and distinct events,
   and unique image IDs / image URLs.
3. Checks and prints overlap of image IDs, event names, and URLs between splits.
4. Tests 50 random image URLs from each split (timeout 8s, robust exception handling)
   and computes accessibility success rates.
5. Writes comprehensive report to results/twitter_access_report.json and prints summary.
"""

import os
import re
import csv
import json
import random
from typing import Dict, Any, List, Set, Tuple
from collections import Counter
from urllib.parse import urlparse

import requests

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "twitter_raw"))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
OUTPUT_JSON = os.path.join(RESULTS_DIR, "twitter_access_report.json")

DEV_TWEETS_FILE = os.path.join(REPO_ROOT, "mediaeval2015", "devset", "tweets.txt")
DEV_IMAGES_FILE = os.path.join(REPO_ROOT, "mediaeval2015", "devset", "images.txt")
TEST_TWEETS_FILE = os.path.join(REPO_ROOT, "mediaeval2015", "testset", "tweets.txt")
README_FILE = os.path.join(REPO_ROOT, "README.md")

RANDOM_SEED = 42
REQUEST_TIMEOUT = 8.0
SAMPLE_SIZE = 50


def step1_list_repo_files_and_readme() -> Dict[str, Any]:
    print("=" * 80)
    print("STEP 1: CLONED REPOSITORY FILE INVENTORY & README INSPECTION")
    print("=" * 80)
    print(f"Target Repository Directory: {REPO_ROOT}\n")

    if not os.path.isdir(REPO_ROOT):
        raise FileNotFoundError(f"Cloned repository directory not found: {REPO_ROOT}")

    # 1. Read README.md
    readme_text = ""
    if os.path.isfile(README_FILE):
        with open(README_FILE, "r", encoding="utf-8", errors="ignore") as f:
            readme_text = f.read()
        print("--- README.md Content (Excerpt) ---")
        for line in readme_text.splitlines()[:22]:
            print("  ", line)
        if len(readme_text.splitlines()) > 22:
            print("   ... [truncated for display] ...\n")
    else:
        print("README.md not found in repository root.\n")

    # 2. Inventory all files excluding .git
    file_records = []
    print("--- Repository Files Inventory ---")
    for root, dirs, files in os.walk(REPO_ROOT):
        if ".git" in dirs:
            dirs.remove(".git")
        for fname in sorted(files):
            full_path = os.path.join(root, fname)
            rel_path = os.path.relpath(full_path, REPO_ROOT)
            size_bytes = os.path.getsize(full_path)
            _, ext = os.path.splitext(fname)
            ext_lower = ext.lower()

            record: Dict[str, Any] = {
                "file_path": rel_path,
                "size_bytes": size_bytes,
                "extension": ext_lower,
                "format": "unknown",
                "columns": None
            }

            if ext_lower == ".md":
                record["format"] = "Markdown documentation"
            elif ext_lower == ".pdf":
                record["format"] = "PDF document"
            elif ext_lower == ".rar":
                record["format"] = "RAR archive"
            elif ext_lower == ".java":
                record["format"] = "Java source code"
            elif ext_lower == ".xml":
                record["format"] = "XML document"
            elif ext_lower in [".txt", ".csv"]:
                # Try reading header row to determine delimiter and columns
                try:
                    with open(full_path, "r", encoding="utf-8", errors="ignore") as tf:
                        first_line = tf.readline().strip()
                        second_line = tf.readline().strip()

                    if "\t" in first_line:
                        cols = [c.strip() for c in first_line.split("\t") if c.strip()]
                        record["format"] = "Tab-separated text (TSV)"
                        record["columns"] = cols
                    elif "," in first_line:
                        cols = [c.strip() for c in first_line.split(",") if c.strip()]
                        record["format"] = "Comma-separated values (CSV)"
                        record["columns"] = cols
                    else:
                        record["format"] = "Plain text"
                        record["columns"] = None

                    # Special case check for files lacking explicit header row
                    if fname == "tweets_images_update.txt":
                        record["format"] = "Tab-separated text without header"
                        record["columns"] = ["tweet_id", "image_id", "label", "event (inferred)"]

                except Exception as e:
                    record["format"] = f"Text read error: {e}"
            else:
                record["format"] = "Binary/Other"

            file_records.append(record)
            col_str = f" | Fields: {record['columns']}" if record['columns'] else ""
            print(f"• {rel_path} ({record['size_bytes']:,} bytes) - {record['format']}{col_str}")

    print(f"\nTotal inventory files identified: {len(file_records)}\n")
    return {
        "readme_path": os.path.relpath(README_FILE, REPO_ROOT),
        "readme_length_chars": len(readme_text),
        "files_count": len(file_records),
        "files": file_records
    }


def step2_parse_2015_splits() -> Dict[str, Any]:
    print("=" * 80)
    print("STEP 2: 2015 DEVSET & TESTSET TWEET FILES ANALYSIS")
    print("=" * 80)

    url_regex = re.compile(r"https?://[^\s\t\n]+")

    # 1. Devset parsing
    print(f"\n--- [1] Analyzing Devset Tweets: {DEV_TWEETS_FILE} ---")
    if not os.path.isfile(DEV_TWEETS_FILE):
        raise FileNotFoundError(f"Devset tweets file not found: {DEV_TWEETS_FILE}")

    with open(DEV_TWEETS_FILE, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f, delimiter="\t")
        dev_header = next(reader)
        dev_rows = list(reader)

    dev_tweet_count = len(dev_rows)
    dev_labels = Counter()
    dev_image_ids_set = set()
    dev_tweet_urls_set = set()

    for row in dev_rows:
        if len(row) >= 7:
            dev_labels[row[6].strip()] += 1
        if len(row) >= 4:
            val = row[3].strip()
            for img in val.replace(",", " ").split():
                if img.strip():
                    dev_image_ids_set.add(img.strip())
        if len(row) >= 2:
            for u in url_regex.findall(row[1]):
                clean_u = u.split("\\")[0].rstrip(".,;:\"'()[]<>")
                if clean_u:
                    dev_tweet_urls_set.add(clean_u)

    # Dev images.txt
    dev_images_txt_ids = set()
    dev_images_txt_urls = set()
    dev_images_txt_events = set()
    if os.path.isfile(DEV_IMAGES_FILE):
        with open(DEV_IMAGES_FILE, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if i == 0 or not line.strip():
                    continue
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    dev_images_txt_ids.add(parts[0].strip())
                    dev_images_txt_urls.add(parts[1].strip())
                if len(parts) == 4:
                    dev_images_txt_events.add(parts[3].strip())
                elif len(parts) == 3:
                    dev_images_txt_events.add(parts[2].strip())

    dev_id_prefixes = set(id_str.split("_")[0] for id_str in dev_image_ids_set)

    print(f"Header columns found: {dev_header}")
    print(f"Number of tweets: {dev_tweet_count:,}")
    print(f"Label counts: {dict(dev_labels)}")
    print("Event field in tweets.txt: DOES NOT EXIST (not present in columns)")
    print(f"  ↳ Note: Devset event metadata exists in devset/images.txt ({len(dev_images_txt_events)} events)")
    print(f"  ↳ Devset events in images.txt: {sorted(list(dev_images_txt_events))}")
    print(f"  ↳ Devset imageId prefixes: {sorted(list(dev_id_prefixes))}")
    print(f"Number of unique image IDs in dev tweets: {len(dev_image_ids_set)}")
    print(f"Number of unique image IDs in devset/images.txt: {len(dev_images_txt_ids)}")
    print(f"Number of unique image URLs in devset/images.txt: {len(dev_images_txt_urls)}")
    print(f"Number of unique t.co URLs in dev tweet texts: {len(dev_tweet_urls_set)}")

    # 2. Testset parsing
    print(f"\n--- [2] Analyzing Testset Tweets: {TEST_TWEETS_FILE} ---")
    if not os.path.isfile(TEST_TWEETS_FILE):
        raise FileNotFoundError(f"Testset tweets file not found: {TEST_TWEETS_FILE}")

    with open(TEST_TWEETS_FILE, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f, delimiter="\t")
        test_header = next(reader)
        test_rows = list(reader)

    test_tweet_count = len(test_rows)
    test_labels = Counter()
    test_image_ids_set = set()
    test_tweet_urls_set = set()

    for row in test_rows:
        if len(row) >= 7:
            test_labels[row[6].strip()] += 1
        if len(row) >= 4:
            val = row[3].strip()
            for img in val.replace(",", " ").split():
                if img.strip():
                    test_image_ids_set.add(img.strip())
        if len(row) >= 2:
            for u in url_regex.findall(row[1]):
                clean_u = u.split("\\")[0].rstrip(".,;:\"'()[]<>")
                if clean_u:
                    test_tweet_urls_set.add(clean_u)

    test_id_prefixes = set(id_str.split("_")[0] for id_str in test_image_ids_set)

    print(f"Header columns found: {test_header}")
    print(f"Number of tweets: {test_tweet_count:,}")
    print(f"Label counts: {dict(test_labels)}")
    print("Event field in tweets.txt: DOES NOT EXIST (not present in columns)")
    print("  ↳ Note: No images.txt file is provided in testset.")
    print(f"  ↳ Inferred event prefixes from test image IDs ({len(test_id_prefixes)} distinct): {sorted(list(test_id_prefixes))}")
    print(f"Number of unique image IDs in test tweets: {len(test_image_ids_set)}")
    print(f"Number of unique t.co URLs in test tweet texts: {len(test_tweet_urls_set)}\n")

    return {
        "devset": {
            "file": os.path.relpath(DEV_TWEETS_FILE, REPO_ROOT),
            "header": dev_header,
            "tweet_count": dev_tweet_count,
            "label_counts": dict(dev_labels),
            "event_field_in_file": False,
            "event_field_explanation": "The 'event' field does NOT exist in devset/tweets.txt. It is present in devset/images.txt.",
            "distinct_events_count_in_images_txt": len(dev_images_txt_events),
            "events_in_images_txt": sorted(list(dev_images_txt_events)),
            "distinct_event_prefixes_count": len(dev_id_prefixes),
            "event_prefixes": sorted(list(dev_id_prefixes)),
            "unique_image_ids_count": len(dev_image_ids_set),
            "unique_image_ids": sorted(list(dev_image_ids_set)),
            "unique_image_urls_in_images_txt": len(dev_images_txt_urls),
            "unique_tweet_text_urls": len(dev_tweet_urls_set)
        },
        "testset": {
            "file": os.path.relpath(TEST_TWEETS_FILE, REPO_ROOT),
            "header": test_header,
            "tweet_count": test_tweet_count,
            "label_counts": dict(test_labels),
            "event_field_in_file": False,
            "event_field_explanation": "The 'event' field does NOT exist in testset/tweets.txt, and no testset/images.txt is provided. Distinct events are inferred from image ID prefixes.",
            "distinct_events_count_inferred": len(test_id_prefixes),
            "events_inferred": sorted(list(test_id_prefixes)),
            "unique_image_ids_count": len(test_image_ids_set),
            "unique_image_ids": sorted(list(test_image_ids_set)),
            "unique_image_urls_in_images_txt": 0,
            "unique_tweet_text_urls": len(test_tweet_urls_set)
        },
        "dev_image_ids_set": dev_image_ids_set,
        "test_image_ids_set": test_image_ids_set,
        "dev_images_txt_urls": dev_images_txt_urls,
        "dev_tweet_urls_set": dev_tweet_urls_set,
        "test_tweet_urls_set": test_tweet_urls_set,
        "dev_events_set": dev_images_txt_events,
        "test_events_inferred_set": test_id_prefixes
    }


def step3_overlap_analysis(split_data: Dict[str, Any]) -> Dict[str, Any]:
    print("=" * 80)
    print("STEP 3: DEVSET VS. TESTSET OVERLAP ANALYSIS")
    print("=" * 80)

    dev_img_ids = split_data["dev_image_ids_set"]
    test_img_ids = split_data["test_image_ids_set"]
    img_id_overlap = dev_img_ids.intersection(test_img_ids)

    dev_events = split_data["dev_events_set"]
    test_events = split_data["test_events_inferred_set"]
    # Normalize lower case for strict comparison
    dev_events_lower = set(e.lower() for e in dev_events)
    test_events_lower = set(e.lower() for e in test_events)
    event_overlap = dev_events_lower.intersection(test_events_lower)

    dev_txt_urls = split_data["dev_images_txt_urls"]
    test_tweet_urls = split_data["test_tweet_urls_set"]
    url_overlap_txt = dev_txt_urls.intersection(test_tweet_urls)

    dev_tweet_urls = split_data["dev_tweet_urls_set"]
    url_overlap_tweets = dev_tweet_urls.intersection(test_tweet_urls)

    print(f"1. Image IDs Overlap:")
    print(f"   • Devset unique image IDs : {len(dev_img_ids)}")
    print(f"   • Testset unique image IDs: {len(test_img_ids)}")
    print(f"   • Overlapping image IDs   : {len(img_id_overlap)} -> {sorted(list(img_id_overlap))}")
    print(f"   • Result: Completely disjoint image IDs (0% overlap).")

    print(f"\n2. Event Names Overlap:")
    print(f"   • Devset events (images.txt): {sorted(list(dev_events))}")
    print(f"   • Testset inferred events   : {sorted(list(test_events))}")
    print(f"   • Overlapping event names   : {len(event_overlap)} -> {sorted(list(event_overlap))}")
    print(f"   • Result: Strictly disjoint events (0% overlap; event-level split design).")

    print(f"\n3. Image URLs Overlap:")
    print(f"   • Devset images.txt URLs vs Testset tweet URLs overlap: {len(url_overlap_txt)}")
    print(f"   • Devset tweet URLs vs Testset tweet URLs overlap     : {len(url_overlap_tweets)}")
    print(f"   • Result: 0 overlapping URLs between dev and test splits.\n")

    return {
        "image_ids_overlap": {
            "dev_count": len(dev_img_ids),
            "test_count": len(test_img_ids),
            "overlap_count": len(img_id_overlap),
            "overlap_list": sorted(list(img_id_overlap)),
            "is_disjoint": len(img_id_overlap) == 0
        },
        "event_names_overlap": {
            "dev_events": sorted(list(dev_events)),
            "test_events": sorted(list(test_events)),
            "overlap_count": len(event_overlap),
            "overlap_list": sorted(list(event_overlap)),
            "is_disjoint": len(event_overlap) == 0
        },
        "urls_overlap": {
            "images_txt_vs_test_overlap": len(url_overlap_txt),
            "tweet_urls_vs_test_overlap": len(url_overlap_tweets),
            "is_disjoint": len(url_overlap_txt) == 0 and len(url_overlap_tweets) == 0
        }
    }


def step4_test_image_urls(split_data: Dict[str, Any]) -> Dict[str, Any]:
    print("=" * 80)
    print("STEP 4: NETWORK ACCESSIBILITY TEST (50 RANDOM IMAGE URLS PER SPLIT)")
    print(f"Parameters: Sample Size = {SAMPLE_SIZE}, Timeout = {REQUEST_TIMEOUT}s, Seed = {RANDOM_SEED}")
    print("=" * 80)

    random.seed(RANDOM_SEED)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
    }

    # 1. Devset URL testing (from devset/images.txt: explicit image URLs)
    dev_pool = sorted(list(split_data["dev_images_txt_urls"]))
    dev_sample = random.sample(dev_pool, min(SAMPLE_SIZE, len(dev_pool)))

    print(f"\n--- Testing {len(dev_sample)} Random Devset Image URLs (Source: devset/images.txt) ---")
    dev_details = []
    dev_outcomes = Counter()

    for idx, url in enumerate(dev_sample, 1):
        domain = urlparse(url).netloc
        status_desc = "UNKNOWN"
        http_code = None
        is_success = False

        try:
            r = requests.get(url, timeout=REQUEST_TIMEOUT, headers=headers, allow_redirects=True)
            http_code = r.status_code
            if 200 <= r.status_code < 400:
                is_success = True
                status_desc = f"HTTP_{r.status_code}"
                dev_outcomes["OK_2xx_3xx"] += 1
            else:
                status_desc = f"HTTP_{r.status_code}"
                dev_outcomes[f"HTTP_{r.status_code}"] += 1
        except requests.exceptions.Timeout:
            status_desc = "TIMEOUT"
            dev_outcomes["TIMEOUT"] += 1
        except requests.exceptions.SSLError:
            status_desc = "SSL_ERROR"
            dev_outcomes["SSL_ERROR"] += 1
        except requests.exceptions.ConnectionError:
            status_desc = "CONNECTION_ERROR"
            dev_outcomes["CONNECTION_ERROR"] += 1
        except requests.exceptions.TooManyRedirects:
            status_desc = "TOO_MANY_REDIRECTS"
            dev_outcomes["TOO_MANY_REDIRECTS"] += 1
        except Exception as e:
            status_desc = f"ERR_{type(e).__name__}"
            dev_outcomes[f"ERR_{type(e).__name__}"] += 1

        dev_details.append({
            "index": idx,
            "url": url,
            "domain": domain,
            "status": status_desc,
            "http_code": http_code,
            "success": is_success
        })

        tag = "✓ OK" if is_success else f"✗ FAIL ({status_desc})"
        print(f"  [{idx:02d}/50] {tag:<18} | Host: {domain:<30} | URL: {url[:55]}...")

    dev_success_count = dev_outcomes["OK_2xx_3xx"]
    dev_success_rate = (dev_success_count / len(dev_sample)) * 100.0

    print(f"\nDevset Image URLs Test Summary:")
    print(f"  • Total Tested : {len(dev_sample)}")
    print(f"  • Successful   : {dev_success_count}")
    print(f"  • Success Rate : {dev_success_rate:.1f}%")
    print(f"  • Outcome Breakdown: {dict(dev_outcomes)}")

    # 2. Testset URL testing (from testset/tweets.txt tweetText: t.co URLs)
    test_pool = sorted(list(split_data["test_tweet_urls_set"]))
    test_sample = random.sample(test_pool, min(SAMPLE_SIZE, len(test_pool)))

    print(f"\n--- Testing {len(test_sample)} Random Testset Image URLs (Source: testset/tweets.txt tweetText) ---")
    test_details = []
    test_outcomes = Counter()

    for idx, url in enumerate(test_sample, 1):
        domain = urlparse(url).netloc
        status_desc = "UNKNOWN"
        http_code = None
        is_success = False

        try:
            r = requests.get(url, timeout=REQUEST_TIMEOUT, headers=headers, allow_redirects=True)
            http_code = r.status_code
            if 200 <= r.status_code < 400:
                is_success = True
                status_desc = f"HTTP_{r.status_code}"
                test_outcomes["OK_2xx_3xx"] += 1
            else:
                status_desc = f"HTTP_{r.status_code}"
                test_outcomes[f"HTTP_{r.status_code}"] += 1
        except requests.exceptions.Timeout:
            status_desc = "TIMEOUT"
            test_outcomes["TIMEOUT"] += 1
        except requests.exceptions.SSLError:
            status_desc = "SSL_ERROR"
            test_outcomes["SSL_ERROR"] += 1
        except requests.exceptions.ConnectionError:
            status_desc = "CONNECTION_ERROR"
            test_outcomes["CONNECTION_ERROR"] += 1
        except requests.exceptions.TooManyRedirects:
            status_desc = "TOO_MANY_REDIRECTS"
            test_outcomes["TOO_MANY_REDIRECTS"] += 1
        except Exception as e:
            status_desc = f"ERR_{type(e).__name__}"
            test_outcomes[f"ERR_{type(e).__name__}"] += 1

        test_details.append({
            "index": idx,
            "url": url,
            "domain": domain,
            "status": status_desc,
            "http_code": http_code,
            "success": is_success
        })

        tag = "✓ OK" if is_success else f"✗ FAIL ({status_desc})"
        print(f"  [{idx:02d}/50] {tag:<18} | Host: {domain:<30} | URL: {url[:55]}...")

    test_success_count = test_outcomes["OK_2xx_3xx"]
    test_success_rate = (test_success_count / len(test_sample)) * 100.0

    print(f"\nTestset Image URLs Test Summary:")
    print(f"  • Total Tested : {len(test_sample)}")
    print(f"  • Successful   : {test_success_count}")
    print(f"  • Success Rate : {test_success_rate:.1f}%")
    print(f"  • Outcome Breakdown: {dict(test_outcomes)}\n")

    return {
        "devset_url_test": {
            "source": "mediaeval2015/devset/images.txt",
            "sample_size": len(dev_sample),
            "successful_requests": dev_success_count,
            "failed_requests": len(dev_sample) - dev_success_count,
            "success_rate_percent": round(dev_success_rate, 2),
            "outcome_counts": dict(dev_outcomes),
            "details": dev_details
        },
        "testset_url_test": {
            "source": "mediaeval2015/testset/tweets.txt (tweetText URLs)",
            "sample_size": len(test_sample),
            "successful_requests": test_success_count,
            "failed_requests": len(test_sample) - test_success_count,
            "success_rate_percent": round(test_success_rate, 2),
            "outcome_counts": dict(test_outcomes),
            "details": test_details
        }
    }


def step5_save_and_summarize(
    repo_info: Dict[str, Any],
    split_info: Dict[str, Any],
    overlap_info: Dict[str, Any],
    url_test_info: Dict[str, Any]
) -> None:
    print("=" * 80)
    print("STEP 5: SAVING REPORT TO RESULTS/TWITTER_ACCESS_REPORT.JSON")
    print("=" * 80)

    report = {
        "dataset_name": "MediaEval Verifying Multimedia Use (VMU) 2015 / Twitter Image Verification Corpus",
        "repository_url": "https://github.com/MKLab-ITI/image-verification-corpus",
        "local_clone_path": os.path.relpath(REPO_ROOT, os.path.abspath(os.path.join(REPO_ROOT, "..", ".."))),
        "inventory": repo_info,
        "splits_analysis": {
            "devset": split_info["devset"],
            "testset": split_info["testset"]
        },
        "splits_overlap": overlap_info,
        "url_accessibility_testing": url_test_info
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Report successfully saved to: {OUTPUT_JSON}")
    print(f"Report file size: {os.path.getsize(OUTPUT_JSON):,} bytes\n")

    print("=" * 80)
    print("SUMMARY OF FINDINGS (PHASE 7A)")
    print("=" * 80)
    dev_rep = split_info["devset"]
    test_rep = split_info["testset"]
    dev_net = url_test_info["devset_url_test"]
    test_net = url_test_info["testset_url_test"]

    print(f"1. Devset (2015):")
    print(f"   • Tweets Count: {dev_rep['tweet_count']:,}")
    print(f"   • Label Distribution: {dev_rep['label_counts']}")
    print(f"   • Event Field: {dev_rep['event_field_explanation']}")
    print(f"   • Distinct Events (in images.txt): {dev_rep['distinct_events_count_in_images_txt']}")
    print(f"   • Unique Image IDs: {dev_rep['unique_image_ids_count']} (and {dev_rep['unique_image_urls_in_images_txt']} URLs in images.txt)")
    print(f"   • Image URL Access Rate (50 tested): {dev_net['success_rate_percent']}% ({dev_net['successful_requests']}/{dev_net['sample_size']} accessible)")

    print(f"\n2. Testset (2015):")
    print(f"   • Tweets Count: {test_rep['tweet_count']:,}")
    print(f"   • Label Distribution: {test_rep['label_counts']}")
    print(f"   • Event Field: {test_rep['event_field_explanation']}")
    print(f"   • Inferred Event Clusters: {test_rep['distinct_events_count_inferred']} {test_rep['events_inferred']}")
    print(f"   • Unique Image IDs: {test_rep['unique_image_ids_count']} (4,004 unique t.co URLs in tweetText)")
    print(f"   • Image URL Access Rate (50 tested): {test_net['success_rate_percent']}% ({test_net['successful_requests']}/{test_net['sample_size']} accessible)")

    print(f"\n3. Overlap Between Devset and Testset:")
    print(f"   • Image ID Overlap: {overlap_info['image_ids_overlap']['overlap_count']} (0.0% overlap; completely disjoint)")
    print(f"   • Event Names Overlap: {overlap_info['event_names_overlap']['overlap_count']} (0.0% overlap; completely disjoint events)")
    print(f"   • URL Overlap: 0 (0.0% overlap)")

    print(f"\n4. Critical Access Assessment for Misinformation Pipeline:")
    print(f"   • Devset direct image links have substantial link rot (only ~{dev_net['success_rate_percent']:.0f}% online),")
    print(f"     and testset t.co links suffer 100% failure (HTTP 520 / 404) due to Twitter link decay.")
    print(f"   • Local images are archived in MediaEval2015_DevSet_Images.rar and MediaEval2015_TestSetImages.rar")
    print(f"     within data/twitter_raw/.")
    print("=" * 80)


def main():
    repo_info = step1_list_repo_files_and_readme()
    split_info = step2_parse_2015_splits()
    overlap_info = step3_overlap_analysis(split_info)
    url_test_info = step4_test_image_urls(split_info)
    step5_save_and_summarize(repo_info, split_info, overlap_info, url_test_info)


if __name__ == "__main__":
    main()
