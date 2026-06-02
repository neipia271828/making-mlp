---
name: fashionmnist-target-improver
description: Analyze the making-mlp FashionMNIST experiment logs in `data/FashionMNIST/logs/FashionMNIST.csv` and per-run `train_logs.csv` files under `data/FashionMNIST/logs/`, compare them against `TARGET.md`, and recommend the next model or training changes needed to reach the target. Use when working in the `making-mlp` repo and the user asks what to try next, how close the current runs are to the target, which architecture or hyperparameters to prioritize, or how to turn the existing logs into a concrete improvement plan.
---

# FashionMNIST Target Improver

## Overview

Analyze the current FashionMNIST summary CSV, inspect the corresponding run logs, read `TARGET.md`, and return evidence-backed next experiments.

Prefer the repository files over memory. Treat `valid_accuracy_last10_avg` as the primary comparison metric unless the user explicitly asks for another metric.

## Workflow

1. Confirm the repo contains:
   - `TARGET.md`
   - `data/FashionMNIST/logs/FashionMNIST.csv`
   - run folders under `data/FashionMNIST/logs/`
2. Run `scripts/summarize_fashionmnist_target.py` from the repo root.
3. Read `references/recommendation-rules.md` if the script output is not enough to decide between close candidates.
4. Inspect the specific `train_logs.csv` files for the top runs when you need to judge whether training still looked improvable at the end.
5. Return a short ranked recommendation list tied to the target in `TARGET.md`.

## Command

Run:

```bash
python3 .codex/skills/fashionmnist-target-improver/scripts/summarize_fashionmnist_target.py \
  --target TARGET.md \
  --summary-csv data/FashionMNIST/logs/FashionMNIST.csv \
  --logs-dir data/FashionMNIST/logs
```

If the user points to different paths, pass those paths instead.

## Response Shape

Return:

1. Target and current best gap
2. Best run summary with timestamp, model, epochs, learning rate, weight decay, scheduler, and validation accuracy
3. Ranked next experiments
4. Risks or caveats

Keep the final recommendation concrete. Prefer exact next runs such as:

- `CNN-v2`, `120 epochs`, `lr=0.003`, `weight_decay=5e-5`, `CosineAnnealingLR`
- `same config + 180 epochs`
- `same config + weaker augmentation`

## Rules

- Use current repo files as the source of truth.
- Prefer `valid_accuracy_last10_avg` over raw final-epoch metrics.
- Mention when a recommendation is inferred from trends rather than directly proven by an existing run.
- If the target is already met, shift from exploration to stabilization and reproducibility advice.
- If the summary CSV or train logs are stale or inconsistent, say so explicitly before recommending changes.
