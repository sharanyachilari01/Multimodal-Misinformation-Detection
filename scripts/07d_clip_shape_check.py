"""
scripts/07d_clip_shape_check.py: Verification of live CLIP embedding shapes, norms,
and mathematical equivalence to the manual submodule projection paths.

Tests:
1. Loads live CLIP model and executes clip_similarity on one example pair.
2. Asserts image and text feature shapes are strictly (512,).
3. Asserts L2 norm of both feature vectors is 1.0 (atol=1e-4).
4. Computes manual vision_model -> visual_projection and text_model -> text_projection paths.
5. Verifies equivalence with torch.allclose and reports maximum absolute difference.
"""

import os
import sys
import numpy as np
import torch

# Ensure repository root is on sys.path
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.inconsistency import clip_similarity, load_models, _load_image


def run_clip_shape_check():
    print("=" * 80)
    print("CLIP EMBEDDING SHAPE, NORM & MANUAL PROJECTION VERIFICATION")
    print("Label: Programmatic verification test")
    print("=" * 80)

    sample_image = "examples/space_mission.jpeg"
    sample_text = "A space shuttle launches from a launch pad with bright flames and smoke."

    print(f"Sample Image: {sample_image}")
    print(f"Sample Text : {sample_text}")

    # 1. Compute features via live clip_similarity
    print("\n1. Running live clip_similarity()...")
    cosine, img_emb, txt_emb = clip_similarity(image=sample_image, text=sample_text)

    # 2. Assert Shapes
    print("\n2. Checking Embedding Dimensions:")
    print(f"   Image feature vector shape: {img_emb.shape} (dtype: {img_emb.dtype})")
    print(f"   Text feature vector shape : {txt_emb.shape} (dtype: {txt_emb.dtype})")
    assert img_emb.shape == (512,), f"Expected image feature shape (512,), got {img_emb.shape}"
    assert txt_emb.shape == (512,), f"Expected text feature shape (512,), got {txt_emb.shape}"
    print("   ✓ PASS: Both image and text embedding vectors are exactly 512-dimensional.")

    # 3. Assert L2 Norms
    print("\n3. Checking L2 Normalization:")
    img_l2 = float(np.linalg.norm(img_emb))
    txt_l2 = float(np.linalg.norm(txt_emb))
    print(f"   Image feature L2 norm: {img_l2:.6f}")
    print(f"   Text feature L2 norm : {txt_l2:.6f}")
    assert np.isclose(img_l2, 1.0, atol=1e-4), f"Image L2 norm {img_l2} deviates from 1.0 (atol=1e-4)"
    assert np.isclose(txt_l2, 1.0, atol=1e-4), f"Text L2 norm {txt_l2} deviates from 1.0 (atol=1e-4)"
    print("   ✓ PASS: Both embedding vectors are unit-normalized (L2 norm = 1.0 within 1e-4).")

    # 4. Manual path comparison: vision_model -> visual_projection, text_model -> text_projection
    print("\n4. Comparing Against Manual Submodule Projection Paths:")
    models = load_models()
    clip_model = models["clip_model"]
    clip_processor = models["clip_processor"]
    device = models["device"]

    pil_img = _load_image(sample_image)
    inputs = clip_processor(
        text=[sample_text],
        images=pil_img,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        # Vision manual path
        v_out = clip_model.vision_model(pixel_values=inputs["pixel_values"])
        v_pooled = v_out.pooler_output if hasattr(v_out, "pooler_output") and v_out.pooler_output is not None else v_out[1]
        v_proj = clip_model.visual_projection(v_pooled)
        v_norm = v_proj / v_proj.norm(dim=-1, keepdim=True)

        # Text manual path
        t_out = clip_model.text_model(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"))
        t_pooled = t_out.pooler_output if hasattr(t_out, "pooler_output") and t_out.pooler_output is not None else t_out[1]
        t_proj = clip_model.text_projection(t_pooled)
        t_norm = t_proj / t_proj.norm(dim=-1, keepdim=True)

    img_tensor = torch.from_numpy(img_emb).to(device)
    txt_tensor = torch.from_numpy(txt_emb).to(device)

    max_abs_diff_img = float((img_tensor - v_norm[0]).abs().max().item())
    max_abs_diff_txt = float((txt_tensor - t_norm[0]).abs().max().item())

    print(f"   Manual vision path: vision_model -> visual_projection -> L2 norm")
    print(f"   Max absolute difference (Image): {max_abs_diff_img:.2e}")
    print(f"   Manual text path  : text_model -> text_projection -> L2 norm")
    print(f"   Max absolute difference (Text) : {max_abs_diff_txt:.2e}")

    is_close_img = torch.allclose(img_tensor, v_norm[0], atol=1e-4)
    is_close_txt = torch.allclose(txt_tensor, t_norm[0], atol=1e-4)

    assert is_close_img, f"Image feature mismatch with manual path (diff: {max_abs_diff_img})"
    assert is_close_txt, f"Text feature mismatch with manual path (diff: {max_abs_diff_txt})"
    print("   ✓ PASS: torch.allclose confirmed for both image and text manual projection paths.")

    print(f"\nReported Cosine Similarity: {cosine:.4f}")
    print("=" * 80)
    print("ALL ASSERTIONS PASSED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    run_clip_shape_check()
