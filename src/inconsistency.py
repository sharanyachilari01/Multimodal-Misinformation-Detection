import os
import json
import warnings
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
from PIL import Image

_MODELS = {}
_REF_STATS = None

def get_target_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def load_models() -> Dict[str, Any]:
    """
    Lazy-loads and caches the three required pretrained models:
    1. openai/clip-vit-base-patch32
    2. Salesforce/blip-image-captioning-base
    3. sentence-transformers/all-MiniLM-L6-v2
    """
    global _MODELS
    if "loaded" in _MODELS and _MODELS["loaded"]:
        return _MODELS

    device = get_target_device()
    print(f"[inconsistency] Loading models on primary device: {device} ...")

    # 1. CLIP ViT-B/32
    from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
    clip_id = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(clip_id).to(device)
    clip_model.eval()
    clip_processor = CLIPProcessor.from_pretrained(clip_id)
    clip_tokenizer = CLIPTokenizer.from_pretrained(clip_id)

    # 2. BLIP Image Captioning Base
    from transformers import BlipProcessor, BlipForConditionalGeneration
    blip_id = "Salesforce/blip-image-captioning-base"
    blip_processor = BlipProcessor.from_pretrained(blip_id)
    blip_model = BlipForConditionalGeneration.from_pretrained(blip_id)
    
    # Try moving BLIP to device; handle MPS fallback if needed
    blip_device = device
    try:
        blip_model.to(blip_device)
        blip_model.eval()
    except Exception as e:
        print(f"[inconsistency] Notice: Moving BLIP to {device} failed ({e}). Falling back to CPU.")
        blip_device = torch.device("cpu")
        blip_model.to(blip_device)
        blip_model.eval()

    # 3. SentenceTransformer MiniLM-L6-v2
    from sentence_transformers import SentenceTransformer
    minilm_id = "sentence-transformers/all-MiniLM-L6-v2"
    minilm_model = SentenceTransformer(minilm_id, device=str(device))
    minilm_model.eval()

    _MODELS = {
        "clip_model": clip_model,
        "clip_processor": clip_processor,
        "clip_tokenizer": clip_tokenizer,
        "blip_model": blip_model,
        "blip_processor": blip_processor,
        "blip_device": blip_device,
        "minilm_model": minilm_model,
        "device": device,
        "loaded": True
    }
    print("[inconsistency] All models successfully loaded and cached.")
    return _MODELS

def _load_image(image_input: Any) -> Image.Image:
    """Helper to convert string path or PIL Image to RGB PIL Image."""
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    elif isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image file not found: {image_input}")
        try:
            return Image.open(image_input).convert("RGB")
        except Exception as e:
            raise ValueError(f"Unable to read or decode image file '{image_input}': {e}")
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}. Expected filepath or PIL.Image.Image.")

def clip_similarity(image: Any, text: str) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Computes L2-normalized CLIP image and text features, cosine similarity,
    and returns (cosine, image_embedding, text_embedding).
    Truncates text to CLIP's 77-token limit and reports when truncation occurs.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input text cannot be empty or non-string.")

    img = _load_image(image)
    models = load_models()
    device = models["device"]
    
    # Check token count for 77-token limit
    tokenizer = models["clip_tokenizer"]
    token_ids = tokenizer.encode(text, add_special_tokens=True)
    if len(token_ids) > 77:
        print(f"[inconsistency] Notice: Input text exceeded CLIP's 77-token limit ({len(token_ids)} tokens) and was truncated to 77.")

    processor = models["clip_processor"]
    model = models["clip_model"]

    inputs = processor(text=[text], images=img, return_tensors="pt", padding=True, truncation=True, max_length=77)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        img_out = model.get_image_features(pixel_values=inputs["pixel_values"])
        txt_out = model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"))

        img_feat = img_out.pooler_output if hasattr(img_out, "pooler_output") and img_out.pooler_output is not None else img_out
        txt_feat = txt_out.pooler_output if hasattr(txt_out, "pooler_output") and txt_out.pooler_output is not None else txt_out

        img_norm = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_norm = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

        cosine = float((img_norm * txt_norm).sum(dim=-1).item())

    img_emb = img_norm.cpu().numpy()[0].astype(np.float32)
    txt_emb = txt_norm.cpu().numpy()[0].astype(np.float32)
    return cosine, img_emb, txt_emb

def caption_image(image: Any) -> str:
    """
    Generates a descriptive caption using BLIP image captioning base.
    Falls back to CPU if an MPS operation fails.
    """
    img = _load_image(image)
    models = load_models()
    processor = models["blip_processor"]
    model = models["blip_model"]
    blip_device = models["blip_device"]

    inputs = processor(img, return_tensors="pt")

    try:
        inputs_dev = {k: v.to(blip_device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs_dev, max_new_tokens=40)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        print(f"[inconsistency] Notice: BLIP generation failed on device {blip_device} ({type(e).__name__}: {e}). Falling back to CPU.")
        model.to("cpu")
        models["blip_device"] = torch.device("cpu")
        inputs_cpu = {k: v.to("cpu") for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs_cpu, max_new_tokens=40)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption

def claim_caption_similarity(text: str, caption: str) -> float:
    """
    Computes cosine similarity between input claim text and the generated BLIP caption
    using all-MiniLM-L6-v2 sentence embeddings.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Claim text cannot be empty.")
    if not isinstance(caption, str) or not caption.strip():
        return 0.0

    models = load_models()
    minilm = models["minilm_model"]

    emb = minilm.encode([text, caption], convert_to_tensor=True)
    emb_norm = emb / emb.norm(dim=-1, keepdim=True)
    sim = float(torch.sum(emb_norm[0] * emb_norm[1]).item())
    return sim

def clip_reference_stats() -> Dict[str, Any]:
    """
    Loads features/val.npz and models/M0_threshold.json to provide reference validation
    distributions for pristine and falsified CLIP cosines.
    """
    global _REF_STATS
    if _REF_STATS is not None:
        return _REF_STATS

    val_path = "features/val.npz"
    thresh_path = "models/M0_threshold.json"

    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Missing validation features file: {val_path}")
    if not os.path.exists(thresh_path):
        raise FileNotFoundError(f"Missing M0 threshold file: {thresh_path}")

    val_data = np.load(val_path)
    T_val, V_val, y_val = val_data["T"], val_data["V"], val_data["y"]
    cos_val = np.sum(T_val * V_val, axis=1)

    cos_pristine = cos_val[y_val == 0]
    cos_falsified = cos_val[y_val == 1]

    with open(thresh_path, "r", encoding="utf-8") as f:
        thresh_info = json.load(f)
    m0_threshold = float(thresh_info["optimal_threshold"])

    _REF_STATS = {
        "cos_pristine": cos_pristine,
        "cos_falsified": cos_falsified,
        "m0_threshold": m0_threshold,
        "pristine_mean": float(np.mean(cos_pristine)),
        "pristine_std": float(np.std(cos_pristine)),
        "falsified_mean": float(np.mean(cos_falsified)),
        "falsified_std": float(np.std(cos_falsified)),
        "val_min": float(np.min(cos_val)),
        "val_max": float(np.max(cos_val)),
        "val_mean": float(np.mean(cos_val))
    }
    return _REF_STATS

def explain(text: str, image: Any) -> Dict[str, Any]:
    """
    Performs multimodal inconsistency reasoning for a claim text and image.
    Returns:
    - clip_cosine
    - percentile_pristine: percentile among genuine validation pairs
    - percentile_falsified: percentile among falsified validation pairs
    - caption: generated BLIP caption
    - claim_caption_similarity: MiniLM semantic similarity between claim and caption
    - signal_level: 'no strong signal' / 'possible mismatch' / 'weak evidence'
    - explanation: readable justification adhering to non-fabrication & safety rules.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Claim text cannot be empty.")

    # 1. Compute CLIP similarity
    cosine, _, _ = clip_similarity(image, text)

    # 2. Generate BLIP caption
    caption = caption_image(image)

    # 3. Compute Claim-Caption Similarity
    sim_claim_cap = claim_caption_similarity(text, caption)

    # 4. Compare with reference stats
    ref = clip_reference_stats()
    pct_pristine = float(np.mean(ref["cos_pristine"] <= cosine) * 100)
    pct_falsified = float(np.mean(ref["cos_falsified"] <= cosine) * 100)
    threshold = ref["m0_threshold"]

    # 5. Determine Signal Level
    if cosine < threshold:
        if cosine < (ref["pristine_mean"] - 1.5 * ref["pristine_std"]) or sim_claim_cap < 0.25:
            signal_level = "possible mismatch"
        else:
            signal_level = "weak evidence"
    else:
        signal_level = "no strong signal"

    # 6. Formulate Objective, Rigorous Explanation
    explanation = (
        f"Multimodal Inconsistency Assessment:\n"
        f"• CLIP text-image cosine similarity is {cosine:.4f} (Validation M0 decision threshold: {threshold:.4f}).\n"
        f"• This similarity falls at percentile {pct_pristine:.1f} of genuine (pristine) benchmark pairs, "
        f"and percentile {pct_falsified:.1f} of falsified pairs (benchmark).\n"
        f"• Generated image description (BLIP): \"{caption}\".\n"
        f"• Claim vs. image description semantic similarity (MiniLM): {sim_claim_cap:.4f}.\n"
        f"• Assessment Signal: [{signal_level.upper()}].\n"
        f"Important Context & Caveats:\n"
        f"1. Low multimodal similarity serves strictly as a WARNING SIGNAL of potential context inconsistency, NOT definitive proof of falsehood.\n"
        f"2. CLIP similarity alone reached about 65.2% accuracy on the NewsCLIPpings test set (M0, validation-selected threshold). "
        f"On semantics_clip_text_image, M0 pair accuracy was 0.524, near chance.\n"
        f"3. Caption-based semantic similarity is coarse because automated BLIP captions are generic and omit fine-grained entity identifiers.\n"
        f"4. Conversely, a high similarity score does NOT guarantee that the claim is factual or true (e.g., authentic images repurposed in subtly misleading contexts)."
    )

    return {
        "clip_cosine": round(cosine, 4),
        "percentile_pristine": round(pct_pristine, 2),
        "percentile_falsified": round(pct_falsified, 2),
        "caption": caption,
        "claim_caption_similarity": round(sim_claim_cap, 4),
        "signal_level": signal_level,
        "explanation": explanation
    }
