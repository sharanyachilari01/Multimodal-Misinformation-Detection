import os, sys, time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
from PIL import Image
from src.inconsistency import caption_image, claim_caption_similarity

df = pd.read_csv("examples/pairs.csv").dropna(how="all")
print("=" * 80); print("LABEL AUDIT (CSV is not modified)"); print("=" * 80)
for img in df.image_path.unique():
    if not os.path.exists(img):
        print(f"MISSING: {img}\n"); continue
    mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(img)))
    size = Image.open(img).size
    cap = caption_image(img)
    match_rows = df[(df.image_path == img) & (df.label == "match")]
    print(f"Image: {img} | modified {mtime} | size {size}")
    print(f"  BLIP caption : {cap}")
    for _, r in match_rows.iterrows():
        sim = claim_caption_similarity(r.text, cap)
        flag = "  <-- FLAG: check this image/caption by eye" if sim < 0.25 else ""
        print(f"  CSV match    : {r.text}")
        print(f"  MiniLM sim   : {sim:.4f}{flag}")
    print()
