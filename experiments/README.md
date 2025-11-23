# Experiment 1 — Replication Notes

Utilities for evaluating TRM under two modes: paper-style test-time voting and a single-augmentation (canonical) evaluation. The aim is to quantify the contribution of ensembling and report timing/memory.

## What’s implemented
- `prepare_arc_datasets.py`: builds ARC datasets at multiple augmentation levels, including 0×, 1×, 4×, 16×, 64×, 256×, 576×, 1000×.
- `run_exp1.py`: evaluates a trained TRM checkpoint in two modes:
  - paper-mode (variable ≤1000 votes, as-authors) using `data/arc-aug-1000`
  - single-aug mode (0× augment) using `data/arc-aug-0`
  Captures environment info, timing, peak GPU memory, dataset/checkpoint hashes, and writes per-example logs for bootstrap CIs.
- `ci_utils.py`: computes bootstrap 95% CIs and writes a concise `exp1_table.json`. Also emits a histogram of effective votes to surface the "≤1000" caveat.
- `acc_vs_aug.py`: produces an accuracy-vs-augmentation CSV across the built levels for visualization.

## Methodological caveat
Some ARC puzzles do not support 1000 unique augmentations due to symmetries. The effective ensemble size is ≤1000 and varies per puzzle. We therefore:
- Log `num_votes` per example in `per_example.jsonl` to quantify the variation
- Provide an equalized control level (e.g., 576×) so you can compare single-aug vs a fixed-K ensemble
- Provide an accuracy-vs-augmentation curve for deeper analysis

## Environment capture
- `exp1_report.json` records Python/CUDA versions (from PyTorch), GPU info (if available), timing, peak memory, and simple hashes for datasets/checkpoints. Save `nvidia-smi -q` separately if you need full driver details.

## Quickstart
Assumes a trained TRM checkpoint and GPU PyTorch.

1) Build datasets (run from repo root):
```
python -m experiments.prepare_arc_datasets
```
- Writes `data/arc_exp1_manifest.json` and datasets under `data/arc-aug-*`.

2) Run Experiment 1 (paper vs single-aug):
```
python -m experiments.run_exp1 \
  data_paths='[data/arc-aug-1000]' \
  data_paths_test='[data/arc-aug-0]' \
  load_checkpoint=/ABS/PATH/TO/step_XXXXX \
  checkpoint_path=checkpoints/exp1_arc \
  arch=trm arch.L_cycles=4 arch.H_cycles=3 arch.L_layers=2 \
  global_batch_size=256
```
- If OOM, try `global_batch_size=128`.

3) Optional equalized control (e.g., 576×):
```
python -m experiments.run_exp1 \
  data_paths='[data/arc-aug-576]' \
  data_paths_test='[data/arc-aug-0]' \
  load_checkpoint=/ABS/PATH/TO/step_XXXXX \
  checkpoint_path=checkpoints/exp1_arc_k576 \
  arch=trm
```

4) Optional accuracy vs augmentation curve:
```
python -m experiments.acc_vs_aug \
  load_checkpoint=/ABS/PATH/TO/step_XXXXX \
  checkpoint_path=checkpoints/exp1_arc_curve \
  arch=trm
```

## Outputs to archive
From `checkpoints/exp1_arc/` (and similarly for other runs):
- `exp1_report.json`: environment, commit, dataset dir hash, checkpoint SHA256, latency, memory, commands
- `exp1_table.json`: mean accuracy and 95% CIs for paper vs single-aug, with vote histograms
- `evaluator_ARC_step_0/per_example.jsonl`: per-example correctness, including `num_votes`
- `evaluator_ARC_step_0/submission.json`: selected grids per puzzle
- `step_0_all_preds.0`: raw tensors (optional)

From curve runs:
- `checkpoints/exp1_arc_curve/acc_vs_aug.csv`

## Rationale for adaptations in this folder
- **Voting-size variability (≤1000)**:
  - We record `num_votes` per example in `per_example.jsonl`.
  - We provide an equalized-K dataset level (e.g., 576×) for a fair fixed-ensemble comparison.
- **Determinism and provenance**: We log commit hash, dataset dir hashes, and checkpoint SHA256 in `exp1_report.json` so results are reproducible and attributable.
- **Bootstrap CIs**: We compute 95% CIs from per-example correctness to report uncertainty without retraining.
- **Optional curve**: The accuracy-vs-augmentation CSV helps visualize the marginal benefit of more voting.

## Reproducibility checklist
- Fixed seeds for evaluation
- Logged commit hash, dataset dir hashes, and checkpoint SHA256
- Per-example logs for bootstrap CIs
- Exact commands provided above

## Attribution
These files are replication utilities for evaluating Experiment 1 in the TRM repository. TRM and all associated models/architectures belong to the original authors.
