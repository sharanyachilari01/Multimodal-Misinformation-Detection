"""
app.py: Streamlit web application for live multimodal misinformation and out-of-context detection.

Integrates:
1. Multimodal Inconsistency Signal (src.inconsistency.explain)
2. Out-of-Context Classifier Ensemble (src.classify.predict)
3. External Contextual Evidence Retrieval (src.evidence.search_evidence)
4. Explicit Methodological Limitations & Safety Disclaimers
"""

import os
import sys
from typing import Any, Callable, Dict, Optional
from PIL import Image

# Ensure repository root is on sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.inconsistency import explain
from src.classify import predict
from src.evidence import search_evidence

# -----------------------------------------------------------------------------
# Programmatic Analysis Functions (UI-agnostic, for CLI & smoke tests)
# -----------------------------------------------------------------------------

def analyze_inconsistency(image: Any, text: str) -> Dict[str, Any]:
    """
    Executes cross-modal inconsistency reasoning (CLIP + BLIP + MiniLM).
    Returns a dictionary with status and data or error.
    """
    try:
        data = explain(text=text, image=image)
        return {"success": True, "data": data}
    except Exception as e:
        return {"success": False, "error": f"Inconsistency analysis failed: {str(e)}"}


def analyze_classifier(image: Any, text: str) -> Dict[str, Any]:
    """
    Executes M4 interaction classifier ensemble prediction.
    Calculates whether seed scores straddle 0.5 ('seeds disagree').
    """
    try:
        res = predict(text=text, image=image)
        s42 = res["seed_scores"]["seed_42"]
        s43 = res["seed_scores"]["seed_43"]
        s44 = res["seed_scores"]["seed_44"]
        seeds_straddle = (min(s42, s43, s44) < 0.5) and (max(s42, s43, s44) >= 0.5)
        res["seeds_disagree"] = seeds_straddle
        return {"success": True, "data": res}
    except Exception as e:
        return {"success": False, "error": f"Classification failed: {str(e)}"}


def analyze_evidence(
    text: str,
    search_fn: Optional[Callable[..., Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Executes public news evidence retrieval. Accepts custom search_fn for unit testing.
    """
    if search_fn is None:
        search_fn = search_evidence

    try:
        res = search_fn(claim=text)
        return {"success": True, "data": res}
    except Exception as e:
        return {
            "success": False,
            "error": f"Evidence retrieval error: {str(e)}",
            "data": {
                "status": f"retrieval failed: {type(e).__name__}",
                "results": [],
                "note": "Retrieved articles are contextual external sources, not verified truth."
            }
        }


def run_full_analysis(
    image: Any,
    text: str,
    search_fn: Optional[Callable[..., Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Runs all three analysis pipelines with complete isolation.
    Failure in one pipeline does not block or compromise the others.
    """
    inconsistency_res = analyze_inconsistency(image=image, text=text)
    classifier_res = analyze_classifier(image=image, text=text)
    evidence_res = analyze_evidence(text=text, search_fn=search_fn)

    # Cross-reference raw CLIP cosine with M4 score if both succeeded
    raw_clip_cosine = None
    m4_score = None
    if inconsistency_res["success"]:
        raw_clip_cosine = inconsistency_res["data"]["clip_cosine"]
    if classifier_res["success"]:
        m4_score = classifier_res["data"]["model_score"]

    return {
        "inconsistency": inconsistency_res,
        "classifier": classifier_res,
        "evidence": evidence_res,
        "summary": {
            "raw_clip_cosine": raw_clip_cosine,
            "m4_ensemble_score": m4_score
        }
    }


# -----------------------------------------------------------------------------
# Streamlit Web User Interface
# -----------------------------------------------------------------------------

def render_ui():
    import streamlit as st

    st.set_page_config(
        page_title="Multimodal Misinformation Reasoning",
        page_icon="🔍",
        layout="wide"
    )

    st.title("Multimodal Misinformation Reasoning & Out-of-Context Detection")
    st.markdown(
        "Live research prototype for explainable social media misinformation detection, "
        "evaluating visual-semantic inconsistency and contextual evidence retrieval."
    )

    col_input_left, col_input_right = st.columns([1, 1])

    with col_input_left:
        st.subheader("1. Input Image")
        uploaded_file = st.file_uploader(
            "Upload candidate image",
            type=["jpg", "jpeg", "png", "webp"],
            help="Upload an image associated with a social media post or news claim."
        )
        if uploaded_file is not None:
            input_image = Image.open(uploaded_file).convert("RGB")
            st.image(input_image, caption="Uploaded Image", use_container_width=True)
        else:
            input_image = None
            st.info("Please upload an image file (JPG, PNG, or WEBP).")

    with col_input_right:
        st.subheader("2. Input Text Claim")
        claim_text = st.text_area(
            "Claim / Caption text",
            height=180,
            placeholder="Enter the post text, claim, or news caption..."
        )
        analyze_clicked = st.button("🔍 Run Full Multimodal Analysis", type="primary", use_container_width=True)

    if analyze_clicked:
        if input_image is None:
            st.error("Please upload an image before running analysis.")
            return
        if not claim_text.strip():
            st.error("Please enter a non-empty claim or caption text.")
            return

        with st.spinner("Executing live multimodal analysis (CLIP + BLIP + MiniLM + M4 Ensemble + Google News RSS)..."):
            results = run_full_analysis(image=input_image, text=claim_text)

        st.markdown("---")
        st.header("Analysis Results")

        # Top Cross-Modal Summary Banner
        summary = results["summary"]
        if summary["raw_clip_cosine"] is not None and summary["m4_ensemble_score"] is not None:
            st.info(
                f"**Direct Cross-Modal Metric Comparison:** "
                f"Raw CLIP Text-Image Cosine = **{summary['raw_clip_cosine']:.4f}** | "
                f"M4 Interaction Ensemble Score = **{summary['m4_ensemble_score']:.4f}**"
            )

        col_sig, col_cls = st.columns([1, 1])

        # ---------------------------------------------------------------------
        # 1. Inconsistency Signal
        # ---------------------------------------------------------------------
        with col_sig:
            st.subheader("1. Inconsistency Reasoning Signal")
            inc = results["inconsistency"]
            if not inc["success"]:
                st.error(inc["error"])
            else:
                data = inc["data"]
                st.metric("Raw CLIP Cosine Similarity", f"{data['clip_cosine']:.4f}")

                col_p1, col_p2 = st.columns(2)
                col_p1.metric("Percentile (Genuine Pairs)", f"{data['percentile_pristine']:.1f}%")
                col_p2.metric("Percentile (falsified pairs (benchmark))", f"{data['percentile_falsified']:.1f}%")

                sig_level = data["signal_level"]
                if sig_level == "possible mismatch":
                    st.warning(f"**Signal Level:** [{sig_level.upper()}] — Low cross-modal alignment detected.")
                elif sig_level == "weak evidence":
                    st.info(f"**Signal Level:** [{sig_level.upper()}] — Ambiguous cross-modal alignment.")
                else:
                    st.success(f"**Signal Level:** [{sig_level.upper()}] — No strong inconsistency signal.")

                st.markdown(f"**Automated Description (BLIP):** *\"{data['caption']}\"*")
                st.markdown(f"**Claim vs Description Semantic Similarity (MiniLM):** `{data['claim_caption_similarity']:.4f}`")
                st.caption("Low multimodal similarity is a warning signal not proof of falsehood.")
                st.caption("Claim-caption similarity is coarse because automated BLIP captions omit fine-grained entity identifiers.")

        # ---------------------------------------------------------------------
        # 2. Classifier Prediction
        # ---------------------------------------------------------------------
        with col_cls:
            st.subheader("2. Out-of-Context Classifier (M4 Ensemble)")
            cls_out = results["classifier"]
            if not cls_out["success"]:
                st.error(cls_out["error"])
            else:
                cdata = cls_out["data"]
                st.metric("Uncalibrated Model Score", f"{cdata['model_score']:.4f}")
                st.caption("Note: This is an uncalibrated model decision score, not a statistical probability.")

                label_str = cdata["label"]
                if label_str == "likely out-of-context pairing":
                    st.warning(f"**Classification (Threshold 0.5):** **{label_str}**")
                else:
                    st.success(f"**Classification (Threshold 0.5):** **{label_str}**")

                s_scores = cdata["seed_scores"]
                st.markdown(
                    f"**Per-Seed Checkpoint Outputs:** "
                    f"Seed 42 = `{s_scores['seed_42']:.4f}` | "
                    f"Seed 43 = `{s_scores['seed_43']:.4f}` | "
                    f"Seed 44 = `{s_scores['seed_44']:.4f}`"
                )

                if cdata.get("seeds_disagree", False):
                    st.warning("⚠️ **Notice:** Individual model seeds disagree across the 0.5 threshold.")

                st.caption(f"Limitations notice: {cdata['limitations']}")

        # ---------------------------------------------------------------------
        # 3. Evidence Retrieval
        # ---------------------------------------------------------------------
        st.markdown("---")
        st.subheader("3. External News Evidence Retrieval")
        ev = results["evidence"]
        if not ev["success"]:
            st.error(ev["error"])
        else:
            edata = ev["data"]
            st.markdown(f"**Retrieval Status:** `{edata['status']}`")
            st.warning(
                "**Contextual Evidence Notice:** Retrieved articles not verified truth. "
                "The 0.40 threshold is a heuristic, not validated against relevance judgments; excerpts unavailable."
            )

            results_list = edata.get("results", [])
            if not results_list:
                st.write("No candidate news articles retrieved for this query.")
            else:
                for idx, item in enumerate(results_list, 1):
                    sim = item.get("similarity", 0.0)
                    with st.expander(f"[{idx}] {item['title']}", expanded=True):
                        relevance_label = "low relevance" if item.get("is_low_relevance", False) else "topically related"
                        st.markdown(f"**Semantic Relevance:** {relevance_label} (title similarity {sim:.4f})")
                        st.markdown(f"**Publisher:** {item.get('publisher', 'Unknown')} | **Published:** {item.get('published', 'Unknown date')}")
                        st.markdown(f"**Source URL:** [Read Article via Google News]({item.get('url', '#')})")

    # -------------------------------------------------------------------------
    # 4. Methodological Limitations & Safety Framework (Always visible)
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("4. Methodological Limitations & Ethical Framework")
    st.info(
        """
        **Operational Boundaries & Evaluation Disclaimers:**
        1. **Benchmark Domain:** The model was trained on NewsCLIPpings news captions; performance on social media untested.
        2. **CLIP Baseline Performance:** CLIP similarity alone reached about 65.2% accuracy on the NewsCLIPpings test set (M0, validation-selected threshold). On semantics_clip_text_image, M0 pair accuracy was 0.524, near chance.
        3. **Generic Imagery Ambiguity:** Accurate captions of generic crowd/room scenes can be flagged (observed on two of seven correct captions in the v2 hand-made sanity set).
        4. **Truth vs. Consistency:** High similarity does not mean true (authentic imagery can be paired with subtly deceptive claims).
        5. **Retrieval Scope:** External retrieval finds topics, not contradictions or fact verification.
        6. **Inconsistency Warning:** Low multimodal similarity is a warning signal not proof of falsehood.
        7. **Coarse Similarity:** Claim-caption similarity is coarse because automated BLIP captions omit fine-grained entity identifiers.
        8. **Evidence Scope & Heuristics:** Retrieved articles not verified truth; the 0.40 threshold is a heuristic, not validated against relevance judgments; excerpts unavailable from RSS feed summaries.
        9. **Strict Neutrality:** User-submitted content is never characterized as 'fake', 'false', or 'debunked'. Third-party article titles are presented verbatim.
        """
    )


# Launch UI only when executed directly via streamlit
try:
    import streamlit as st
    if st.runtime.exists():
        render_ui()
except Exception:
    pass

if __name__ == "__main__":
    import streamlit as st
    if not st.runtime.exists():
        print("Streamlit app created. Run in browser with:")
        print("  streamlit run app.py")
