"""
scripts/07b_wording_audit.py: Comprehensive wording and safety audit.

Audits:
1. Every line in app.py, src/inconsistency.py, src/classify.py, src/evidence.py
   containing (case-insensitive):
   fake, false, debunk, proof, proves, verified, support, contradict, probability,
   accuracy, 65, chance. (Unfiltered)
2. Checks PRESENT/MISSING status with line numbers for each required safety and
   methodological statement.
"""

import os
import re
from typing import Dict, List, Tuple

AUDIT_FILES = [
    "app.py",
    "src/inconsistency.py",
    "src/classify.py",
    "src/evidence.py"
]

KEYWORDS = [
    "fake",
    "false",
    "debunk",
    "proof",
    "proves",
    "verified",
    "support",
    "contradict",
    "probability",
    "accuracy",
    "65",
    "chance"
]

REQUIRED_STATEMENTS = [
    ("warning signal not proof", [
        r"warning\s+signal.*not.*proof",
        r"warning\s+signal\s+not\s+proof"
    ]),
    ("CLIP alone ~65% on test and near chance on some subsets", [
        r"clip\s+alone\s+~?65%?\s+on\s+test\s+and\s+near\s+chance\s+on\s+some\s+subsets",
        r"clip.*65.*test.*chance.*subsets",
        r"clip\s+similarity\s+alone\s+reached\s+about\s+65\.2%.*test.*near\s+chance",
        r"clip.*65.*test.*chance"
    ]),
    ("claim-caption similarity is coarse", [
        r"claim-caption\s+similarity\s+is\s+coarse",
        r"caption-based\s+semantic\s+similarity\s+is\s+coarse"
    ]),
    ("high similarity does not mean true", [
        r"high\s+similarity\s+does\s+not\s+mean\s+true",
        r"high\s+similarity.*not.*guarantee.*true"
    ]),
    ("articles not verified truth", [
        r"articles\s+not\s+verified\s+truth",
        r"articles.*not\s+verified\s+ground\s+truth",
        r"articles\s+are\s+contextual.*not\s+verified\s+truth"
    ]),
    ("the 0.40 threshold is a heuristic, not validated against relevance judgments", [
        r"0\.40\s+threshold\s+is\s+a\s+heuristic,\s+not\s+validated\s+against\s+relevance\s+judgments",
        r"0\.40\s+threshold\s+is\s+a\s+heuristic.*not\s+validated",
        r"0\.40\s+threshold\s+is\s+a\s+heuristic\s+that\s+did\s+not\s+filter\s+results\s+in\s+tests"
    ]),
    ("excerpts unavailable", [
        r"excerpts\s+unavailable",
        r"excerpts\s+are\s+unavailable"
    ]),
    ("accurate captions of generic crowd/room scenes can be flagged", [
        r"accurate\s+captions\s+of\s+generic\s+crowd/room\s+scenes\s+can\s+be\s+flagged",
        r"accurate.*captions.*generic\s+crowd.*room\s+scenes.*flagged"
    ]),
    ("retrieval finds topics, not contradictions", [
        r"retrieval\s+finds\s+topics,\s+not\s+contradictions",
        r"retrieval\s+discovers\s+topically\s+related.*not.*contradictions"
    ]),
    ("social media untested", [
        r"social\s+media\s+untested",
        r"performance\s+on\s+social\s+media.*untested"
    ])
]


def run_keyword_audit() -> Dict[str, List[Tuple[int, str, List[str]]]]:
    print("=" * 80)
    print("WORDING AUDIT: UNFILTERED KEYWORD OCCURRENCES")
    print("Keywords:", ", ".join(KEYWORDS))
    print("Target Files:", ", ".join(AUDIT_FILES))
    print("=" * 80)

    # Build regex pattern for any keyword
    kw_pattern = re.compile(r"(" + "|".join(re.escape(k) for k in KEYWORDS) + r")", re.IGNORECASE)

    results_by_file: Dict[str, List[Tuple[int, str, List[str]]]] = {}

    for fpath in AUDIT_FILES:
        if not os.path.exists(fpath):
            print(f"[ERROR] File not found: {fpath}")
            continue

        results_by_file[fpath] = []
        with open(fpath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for idx, line in enumerate(lines, 1):
            line_clean = line.rstrip("\n")
            matches = list(set(m.lower() for m in kw_pattern.findall(line_clean)))
            if matches:
                results_by_file[fpath].append((idx, line_clean, matches))
                print(f"{fpath}:{idx}: {line_clean}")

    total_matches = sum(len(v) for v in results_by_file.values())
    print("\nTotal matching lines found:", total_matches)
    return results_by_file


def run_statement_audit():
    print("\n" + "=" * 80)
    print("REQUIRED METHODOLOGICAL & SAFETY STATEMENTS AUDIT")
    print("=" * 80)

    # Preload all file contents
    file_contents: Dict[str, List[Tuple[int, str]]] = {}
    for fpath in AUDIT_FILES:
        if os.path.exists(fpath):
            with open(fpath, "r", encoding="utf-8") as f:
                file_contents[fpath] = [(i + 1, l.rstrip("\n")) for i, l in enumerate(f.readlines())]

    all_present = True

    for stmt_name, patterns in REQUIRED_STATEMENTS:
        matches = []
        compiled = [re.compile(p, re.IGNORECASE) for p in patterns]

        for fpath, lines in file_contents.items():
            for line_no, line_text in lines:
                if any(p.search(line_text) for p in compiled):
                    matches.append((fpath, line_no, line_text))

        if matches:
            locs_str = ", ".join(f"{f}:{no}" for f, no, _ in matches)
            print(f"[PRESENT] '{stmt_name}' -> Found at {locs_str}")
            for f, no, text in matches:
                print(f"          ↳ {f}:{no}: {text.strip()}")
        else:
            all_present = False
            print(f"[MISSING] '{stmt_name}'")

    print("\nOverall Required Statements Status:", "ALL PRESENT" if all_present else "SOME MISSING")


if __name__ == "__main__":
    run_keyword_audit()
    run_statement_audit()
