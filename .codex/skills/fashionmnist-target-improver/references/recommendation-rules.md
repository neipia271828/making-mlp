# Recommendation Rules

Use these rules when the log summary alone does not make the next step obvious.

## Primary metric

- Compare runs with `valid_accuracy_last10_avg` first.
- Use `valid_accuracy_last10_median` as a stability check.
- Use `valid_loss_last10_avg` as a secondary generalization signal.

## Target handling

- Read the threshold from `TARGET.md`.
- Express the gap both in absolute accuracy and percentage points.
- If the gap is under `0.01`, prefer narrow tuning over large architecture changes.

## Architecture preference

- Prefer the best-performing model family unless its gain is negligible.
- If a deeper model consistently beats a shallower one by more than `0.003`, prioritize the deeper model for the next round.

## Hyperparameter preference

- Favor settings that improved at higher epoch counts without obvious collapse.
- Prefer the scheduler setting if it improved the best run for the same model family.
- Prefer lower weight decay only when it did not harm validation loss or validation accuracy.

## Trend reading

When inspecting a top run's `train_logs.csv`:

- If the last 10 validation accuracies still drift upward, recommend more epochs.
- If validation accuracy is flat while validation loss rises, recommend checkpointing, weaker regularization, or lighter augmentation instead of more epochs.
- If train accuracy rises while validation accuracy stalls, mention overfitting risk.

## Output style

- Recommend at most three next experiments unless the user asks for a larger plan.
- Give exact parameter values.
- State when the recommendation is an inference from trends rather than a direct A/B result.
