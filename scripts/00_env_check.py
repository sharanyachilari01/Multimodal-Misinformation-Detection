#!/usr/bin/env python3
import sys
import torch
import transformers
import gradio
import sklearn

def main():
    print("=== Environment Check ===")
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if hasattr(torch.backends, "mps"):
        print(f"Apple Silicon MPS Available: {torch.backends.mps.is_available()}")
    print(f"Transformers Version: {transformers.__version__}")
    print(f"Gradio Version: {gradio.__version__}")
    print(f"Scikit-learn Version: {sklearn.__version__}")

if __name__ == "__main__":
    main()
