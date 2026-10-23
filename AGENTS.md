# AGENTS.md - Project Guidelines & Rules

## Project Goal
Multimodal misinformation detection with explicit text-image inconsistency reasoning.

## Operational Rules
1. **Inspect Before Modifying**: Inspect existing files before creating or overwriting anything.
2. **No Fabrication**: Never fabricate data, labels, metrics, citations, or successful runs. Metrics must be written to `results/` by a script that was actually executed.
3. **Phase-by-Phase Execution**: Do the current phase only; no advanced models until the primary pipeline works.
4. **Reproducibility & Hyperparameters**: Seeds and hyperparameters come from `config.json`. Use official train/val/test splits. Report class balance.
5. **Model Constraints**: Free pretrained models only; no paid APIs.
6. **Non-blocking Retrieval**: External retrieval must never block the classifier's prediction.
7. **Execution Transparency**: After each script, print what it did and where outputs were saved.
