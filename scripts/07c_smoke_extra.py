"""
scripts/07c_smoke_extra.py: Extra unit and isolation tests for application components.

All procedures in this script are programmatic unit/smoke tests, NOT benchmark or
empirical evaluation results.

Tests covered:
- TEST 4A: Parliament and Elections 'match' rows analysis (per-seed scores & seeds disagree check).
- TEST 4B: explain() explanation strings for space_mission match and mismatch.
- TEST 4C: Failure isolation (unreadable image, empty text, M4 checkpoints missing, BLIP/CLIP failing).
- TEST 4D: Streamlit AppTest headless execution and limitations box verification.
"""

import os
import sys
import tempfile
from typing import Any, Dict

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from app import run_full_analysis, analyze_classifier, analyze_inconsistency, analyze_evidence
from src.inconsistency import explain


def test_4a_parliament_elections_match():
    print("=" * 80)
    print("[TEST 4A] Unit Smoke Test: Parliament & Elections 'Match' Rows")
    print("Label: Programmatic test execution (NOT an empirical benchmark result)")
    print("=" * 80)

    test_cases = [
        {
            "name": "parliament_match",
            "image": "examples/parliament.jpeg",
            "text": "People gather inside a legislative assembly hall with rows of seats and a central speaker's platform."
        },
        {
            "name": "elections_match",
            "image": "examples/elections.jpg",
            "text": "People are gathered around a table during what appears to be an election-related process, with numbered paper bags and officials recording information."
        }
    ]

    for tc in test_cases:
        print(f"\n--- Testing Case: {tc['name']} ---")
        print(f"Image: {tc['image']}")
        print(f"Text : {tc['text']}")

        res = run_full_analysis(image=tc["image"], text=tc["text"])
        cls_data = res["classifier"]["data"]

        seed_scores = cls_data["seed_scores"]
        model_score = cls_data["model_score"]
        label = cls_data["label"]
        seeds_disagree = cls_data.get("seeds_disagree", False)

        print(f"Per-Seed Scores: Seed 42 = {seed_scores['seed_42']:.4f}, "
              f"Seed 43 = {seed_scores['seed_43']:.4f}, "
              f"Seed 44 = {seed_scores['seed_44']:.4f}")
        print(f"Ensemble Model Score: {model_score:.4f} -> Label: '{label}'")
        print(f"'seeds disagree' flag shown: {seeds_disagree}")
        if seeds_disagree:
            print("  ↳ Disagreement notice: Individual seed scores straddle the 0.5 decision boundary.")
        else:
            print("  ↳ Agreement notice: All seeds concur with respect to the 0.5 decision boundary.")


def test_4b_space_mission_explanations():
    print("\n" + "=" * 80)
    print("[TEST 4B] Unit Smoke Test: explain() Full Explanation Strings")
    print("Label: Programmatic test execution (NOT an empirical benchmark result)")
    print("=" * 80)

    cases = [
        {
            "name": "space_mission_match",
            "image": "examples/space_mission.jpeg",
            "text": "A space shuttle launches from a launch pad with bright flames and smoke."
        },
        {
            "name": "space_mission_mismatch",
            "image": "examples/space_mission.jpeg",
            "text": "A passenger airplane lands on a runway at an international airport."
        }
    ]

    for c in cases:
        print(f"\n{'=' * 30} Explanation for {c['name']} {'=' * 30}")
        out = explain(text=c["text"], image=c["image"])
        print(out["explanation"])
        print(f"{'=' * 75}")


def test_4c_failure_isolation():
    print("\n" + "=" * 80)
    print("[TEST 4C] Unit Smoke Test: Failure Isolation & Non-blocking Behavior")
    print("Label: Programmatic fault-injection test (NOT an empirical benchmark result)")
    print("=" * 80)

    # 1. Unreadable Image
    print("\n[Scenario 1: Unreadable Image (temp text file)]")
    with tempfile.NamedTemporaryFile(suffix=".jpg", mode="w", delete=False) as f:
        f.write("This is invalid raw text data, not a valid image format.")
        corrupt_img = f.name

    try:
        res1 = run_full_analysis(image=corrupt_img, text="A valid news claim text.")
        print(f"  • Inconsistency Succeeded: {res1['inconsistency']['success']}")
        if not res1['inconsistency']['success']:
            print(f"    Message: {res1['inconsistency']['error']}")
        print(f"  • Classifier Succeeded   : {res1['classifier']['success']}")
        if not res1['classifier']['success']:
            print(f"    Message: {res1['classifier']['error']}")
        print(f"  • Evidence Succeeded     : {res1['evidence']['success']}")
        if res1['evidence']['success']:
            print(f"    Message/Status: {res1['evidence']['data']['status']} (Results: {len(res1['evidence']['data']['results'])})")
    finally:
        if os.path.exists(corrupt_img):
            os.remove(corrupt_img)

    # 2. Empty Text
    print("\n[Scenario 2: Empty Text Claim]")
    res2 = run_full_analysis(image="examples/space_mission.jpeg", text="")
    print(f"  • Inconsistency Succeeded: {res2['inconsistency']['success']}")
    if not res2['inconsistency']['success']:
        print(f"    Message: {res2['inconsistency']['error']}")
    print(f"  • Classifier Succeeded   : {res2['classifier']['success']}")
    if not res2['classifier']['success']:
        print(f"    Message: {res2['classifier']['error']}")
    print(f"  • Evidence Succeeded     : {res2['evidence']['success']}")
    if res2['evidence']['success']:
        print(f"    Message/Status: {res2['evidence']['data']['status']} (Results: {len(res2['evidence']['data']['results'])})")

    # 3. M4 Checkpoints Unavailable (Monkeypatch)
    print("\n[Scenario 3: M4 Checkpoints Unavailable (Monkeypatched)]")
    import src.classify
    orig_load_m4 = src.classify.load_m4_models
    orig_m4_cache = src.classify._M4_MODELS
    try:
        src.classify._M4_MODELS = None
        def mock_load_m4(*args, **kwargs):
            raise FileNotFoundError("Mocked: Required M4 checkpoint not found: models/M4_seed42.pt")
        src.classify.load_m4_models = mock_load_m4

        res3 = run_full_analysis(image="examples/space_mission.jpeg", text="A space shuttle launches.")
        print(f"  • Inconsistency Succeeded: {res3['inconsistency']['success']}")
        if res3['inconsistency']['success']:
            print(f"    Signal Level: {res3['inconsistency']['data']['signal_level']}")
        print(f"  • Classifier Succeeded   : {res3['classifier']['success']}")
        if not res3['classifier']['success']:
            print(f"    Message: {res3['classifier']['error']}")
        print(f"  • Evidence Succeeded     : {res3['evidence']['success']}")
        if res3['evidence']['success']:
            print(f"    Message/Status: {res3['evidence']['data']['status']}")
    finally:
        src.classify.load_m4_models = orig_load_m4
        src.classify._M4_MODELS = orig_m4_cache

    # 4. BLIP/CLIP Load Failing (Monkeypatch)
    print("\n[Scenario 4: BLIP/CLIP Pretrained Load Failing (Monkeypatched)]")
    import src.inconsistency
    orig_load_models = src.inconsistency.load_models
    orig_models_cache = src.inconsistency._MODELS
    try:
        src.inconsistency._MODELS = {}
        def mock_load_models():
            raise RuntimeError("Mocked: Simulated CUDA OOM / device error loading pretrained models")
        src.inconsistency.load_models = mock_load_models

        res4 = run_full_analysis(image="examples/space_mission.jpeg", text="A space shuttle launches.")
        print(f"  • Inconsistency Succeeded: {res4['inconsistency']['success']}")
        if not res4['inconsistency']['success']:
            print(f"    Message: {res4['inconsistency']['error']}")
        print(f"  • Classifier Succeeded   : {res4['classifier']['success']}")
        if not res4['classifier']['success']:
            print(f"    Message: {res4['classifier']['error']}")
        print(f"  • Evidence Succeeded     : {res4['evidence']['success']}")
        if res4['evidence']['success']:
            print(f"    Message/Status: {res4['evidence']['data']['status']} (ranking gracefully handled)")
    finally:
        src.inconsistency.load_models = orig_load_models
        src.inconsistency._MODELS = orig_models_cache


def test_4d_streamlit_apptest():
    print("\n" + "=" * 80)
    print("[TEST 4D] Unit Smoke Test: Streamlit AppTest Headless Execution")
    print("Label: Programmatic test execution (NOT an empirical benchmark result)")
    print("=" * 80)

    try:
        from streamlit.testing.v1 import AppTest
        app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app.py"))
        at = AppTest.from_file(app_path)
        at.run()

        exceptions = list(at.exception)
        print(f"AppTest run completed. Exceptions raised: {len(exceptions)}")
        if exceptions:
            for ex in exceptions:
                print(f"  [ERROR] {ex}")
        else:
            print("  ↳ PASS: app.py loaded with no input and raised NO exceptions.")

        # Check Limitations box presence
        info_boxes = list(at.info)
        limitations_present = any("Operational Boundaries & Evaluation Disclaimers" in ib.value or
                                  "Methodological Limitations" in ib.value
                                  for ib in info_boxes)
        print(f"Limitations Box present on initial load: {limitations_present}")
        if limitations_present:
            print("  ↳ PASS: Limitations Box text is present on initial load without input.")
        else:
            print("  ↳ FAIL: Limitations Box text was NOT detected.")

        # Check file upload capability
        file_uploaders = list(at.file_uploader)
        print(f"File uploader elements detected: {len(file_uploaders)}")
        print("Note on file upload in AppTest: streamlit.testing.v1.AppTest does not provide native "
              "browser file upload handling without low-level internal protocol simulation.")

    except Exception as e:
        print(f"AppTest execution error: {type(e).__name__}: {str(e)}")


if __name__ == "__main__":
    test_4a_parliament_elections_match()
    test_4b_space_mission_explanations()
    test_4c_failure_isolation()
    test_4d_streamlit_apptest()
