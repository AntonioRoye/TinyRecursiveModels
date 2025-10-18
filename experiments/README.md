# Experiment 1 — Replication Notes (not a new method)

This folder contains notes and scripts used to replicate and analyze Experiment 1 for TRM (test-time voting vs single-augmentation). The goal here is to measure and document what happens under the two evaluation modes with careful logging; this is not a proposal of new algorithms. All credit for TRM goes to the original authors; these are replication utilities and observations for a narrow evaluation question.

## What’s implemented
- `prepare_arc_datasets.py`: builds ARC datasets at multiple augmentation levels, including 0×, 1×, 4×, 16×, 64×, 256×, 576×, 1000×.
- `run_exp1.py`: evaluates a trained TRM checkpoint in two modes:
  - paper-mode (variable ≤1000 votes, as-authors) using `data/arc-aug-1000`
  - single-aug mode (0× augment) using `data/arc-aug-0`
  Captures environment info, timing, peak GPU memory, dataset/checkpoint hashes, and writes per-example logs for bootstrap CIs.
- `ci_utils.py`: computes bootstrap 95% CIs and writes a concise `exp1_table.json`. Also emits a histogram of effective votes to surface the "≤1000" caveat.
- `acc_vs_aug.py`: produces an accuracy-vs-augmentation CSV across the built levels for visualization.

## Methodological caveat (important)
ARC puzzles cannot always yield 1000 distinct augmentations due to symmetry limits; the effective ensemble size is therefore ≤1000 and varies per puzzle. This is not a bug but a methodological detail that can bias voting strength across puzzles. We therefore:
- Log `num_votes` per example in `per_example.jsonl` to quantify the variation
- Provide an equalized control level (e.g., 576×) so you can compare single-aug vs a fixed-K ensemble
- Provide an accuracy-vs-augmentation curve for deeper analysis

## Environment (example from our EC2 run)
- Instance: g5.xlarge (NVIDIA A10G 24 GB)
- AMI: “Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.8 (Amazon Linux 2023) 20251011”
- OS: Amazon Linux 2023
- Python: 3.9.23 (venv)
- PyTorch: 2.6.0+cu124; CUDA: 12.4; cuDNN: 91002 (torch.backends.cudnn)
- NVIDIA driver (nvidia-smi): 580.95.05; CUDA runtime shown by nvidia-smi: 13.0
- Virtual env: `~/trm-venv`

Capture for Methods:
```
nvidia-smi -q > ~/nvidia-smi-q.txt
python - << 'PY'
import torch, torch.backends.cudnn as c
print({
  'python': '3.9.23',
  'torch': torch.__version__,
  'cuda': torch.version.cuda,
  'cudnn': c.version(),
  'gpu_available': torch.cuda.is_available(),
})
PY
pip freeze > checkpoints/exp1_arc/pip-freeze.txt
```

## Quickstart
Assumes you have a trained TRM checkpoint and have installed dependencies with GPU PyTorch.

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
  arch=trm \
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

From environment:
- Save `nvidia-smi -q` output and Python/PyTorch version prints.

## Rationale for adaptations in this folder
- **Voting-size variability (≤1000)**: We observed some puzzles cannot reach 1000 distinct augments. To keep analysis transparent:
  - We record `num_votes` per example in `per_example.jsonl`.
  - We provide an equalized-K dataset level (e.g., 576×) for a fair fixed-ensemble comparison.
- **Determinism and provenance**: We log commit hash, dataset dir hashes, and checkpoint SHA256 in `exp1_report.json` so results are reproducible and attributable.
- **Bootstrap CIs**: We compute 95% CIs from per-example correctness to report uncertainty without retraining.
- **Optional curve**: The accuracy-vs-augmentation CSV helps visualize the marginal benefit of more voting.

## How to cite caveat in your paper
“While evaluating with 1000 test-time augmentations, several ARC puzzles support fewer unique transformations; the effective ensemble size varies (≤1000). We quantify this with per-example `num_votes` and report both the authors’ variable ensemble and an equalized-K control (e.g., 576×).”

## Reproducibility checklist
- Fixed seeds for evaluation
- Logged commit hash, dataset dir hashes, and checkpoint SHA256
- Per-example logs for bootstrap CIs
- Exact commands provided above

## Attribution
These scripts are replication utilities for evaluating Experiment 1 in the TRM repository. TRM and all associated models/architectures belong to the original authors. This README documents the setup and observations specific to our replication runs and hardware.
