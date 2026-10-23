"""
src/classify.py: Live multimodal classifier for out-of-context social media misinformation detection.

Reuses:
- src.inconsistency.clip_similarity: L2-normalized 512-d CLIP image and text embeddings.
- src.models.build_features: Interaction fusion features (M4, 2049-d).
- src.models.MLPClassifier: Baseline MLP architecture.
- models/M4_seed{42,43,44}.pt: Trained ensemble checkpoints.
"""

import os
from typing import Any, Dict, List, Optional
import numpy as np
import torch

from src.models import MLPClassifier, build_features
from src.inconsistency import clip_similarity

_M4_MODELS: Optional[List[MLPClassifier]] = None


def load_m4_models(
    model_dir: str = "models",
    seeds: Optional[List[int]] = None
) -> List[MLPClassifier]:
    """
    Lazy-loads and caches the M4 MLP models for seeds 42, 43, 44.
    """
    global _M4_MODELS
    if _M4_MODELS is not None:
        return _M4_MODELS

    if seeds is None:
        seeds = [42, 43, 44]

    device = torch.device("cpu")  # MLP inference is lightweight; CPU avoids device transfer overhead
    models: List[MLPClassifier] = []
    for s in seeds:
        path = os.path.join(model_dir, f"M4_seed{s}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required M4 checkpoint not found: {path}")
        m = MLPClassifier(in_features=2049, hidden_dim=256, dropout=0.3)
        state_dict = torch.load(path, map_location=device)
        m.load_state_dict(state_dict)
        m.eval()
        models.append(m)

    _M4_MODELS = models
    return _M4_MODELS


def predict(text: str, image: Any) -> Dict[str, Any]:
    """
    Evaluates an image-text pair to predict whether it exhibits an out-of-context pairing.

    Parameters:
    -----------
    text : str
        Claim or caption associated with the image.
    image : Any
        Filepath string (e.g. 'examples/dog_park.jpg') or PIL Image.

    Returns:
    --------
    dict with keys:
        - 'model_score': float, uncalibrated ensemble model score between 0 and 1
        - 'score_description': string explaining that model_score is uncalibrated, not a probability
        - 'label': 'likely out-of-context pairing' (score >= 0.5) or 'no out-of-context signal' (score < 0.5)
        - 'threshold': float (0.5)
        - 'seed_scores': dict mapping seed to individual model sigmoid output
        - 'clip_cosine': float, cosine similarity of raw CLIP image and text embeddings
        - 'limitations': string describing training source and evaluation boundaries
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input text claim must be a non-empty string.")

    # 1. Compute live CLIP embeddings (512-d L2-normalized)
    cosine, img_emb, txt_emb = clip_similarity(image, text)

    assert img_emb.shape == (512,), f"Expected img_emb shape (512,), got {img_emb.shape}"
    assert txt_emb.shape == (512,), f"Expected txt_emb shape (512,), got {txt_emb.shape}"

    # 2. Build M4 interaction features [T, V, T*V, |T-V|, cosine(T, V)] -> (1, 2049)
    T = txt_emb.reshape(1, 512).astype(np.float32)
    V = img_emb.reshape(1, 512).astype(np.float32)
    m4_feat = build_features("M4", T, V)  # shape (1, 2049)

    # 3. Load checkpoints and compute ensemble prediction
    models = load_m4_models()
    x_tensor = torch.from_numpy(m4_feat).to(torch.float32)

    seeds = [42, 43, 44]
    seed_scores: Dict[str, float] = {}
    with torch.no_grad():
        for s, model in zip(seeds, models):
            logit = model(x_tensor)
            prob = torch.sigmoid(logit).item()
            seed_scores[f"seed_{s}"] = float(prob)

    avg_score = float(np.mean(list(seed_scores.values())))

    # 4. Determine label at 0.5 threshold with explicit standard terminology
    threshold = 0.5
    if avg_score >= threshold:
        label = "likely out-of-context pairing"
    else:
        label = "no out-of-context signal"

    limitations = (
        "the model was trained on NewsCLIPpings news captions, labels mean out-of-context "
        "image pairing, performance on social media posts is untested."
    )

    return {
        "model_score": avg_score,
        "score_description": "uncalibrated model score, not a probability",
        "label": label,
        "threshold": threshold,
        "seed_scores": seed_scores,
        "clip_cosine": float(cosine),
        "limitations": limitations
    }
