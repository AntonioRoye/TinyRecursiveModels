# Less is More: Recursive Reasoning with Tiny Networks

This is the codebase for the paper: "Less is More: Recursive Reasoning with Tiny Networks". TRM is a recursive reasoning approach that achieves amazing scores of 45% on ARC-AGI-1 and 8% on ARC-AGI-2 using a tiny 7M parameters neural network.

[Paper](https://arxiv.org/abs/2510.04871)

### Reproducibility (concise)

This fork provides a minimal, transparent path to reproduce evaluation and efficiency results using only the repository’s Python modules (no external orchestration required).

Provenance:
- Upstream repo: `SamsungSAILMontreal/TinyRecursiveModels`
- Upstream commit pinned in this fork: `e7b68717f0a6c4cbb4ce6fbef787b14f42083bd9`
- Additions in this fork (non-exhaustive): `modal_trm.py`, `modal_llama.py`, `experiments/run_exp1.py`, `experiments/prepare_arc_datasets.py`, `experiments/quick_checks.py`, `experiments/fetch_hf_checkpoints.py`, `experiments/ci_utils.py`.

Quick start (after environment setup and building datasets):

```bash
# 1) Build ARC datasets (writes data/arc-aug-0 and data/arc-aug-1000 plus a manifest)
python -m experiments.prepare_arc_datasets

# 2) Fetch the public verification checkpoint from HF (path printed on completion)
python -m experiments.fetch_hf_checkpoints --repo_id arcprize/trm_arc_prize_verification --dest checkpoints/hf_trm

# 3) Evaluate TRM in both modes (paper 1000× voting; single-aug 1× canonical)
python -m experiments.run_exp1 \
  data_paths='[data/arc-aug-1000]' \
  data_paths_test='[data/arc-aug-0]' \
  load_checkpoint=/ABS/PATH/TO/step_xxxxx \
  checkpoint_path=checkpoints/exp1_arc \
  arch=trm arch.L_cycles=4 arch.H_cycles=3 arch.L_layers=2

# 4) Optional: accuracy vs augmentation curve (e.g., 0,1,4,...,1000)
python -m experiments.acc_vs_aug \
  load_checkpoint=/ABS/PATH/TO/step_xxxxx \
  checkpoint_path=checkpoints/exp1_arc_curve \
  arch=trm

# 5) Benchmark TRM throughput, latency, and peak VRAM (H100 recommended)
# Option A: via Modal GPU function
modal run modal_trm.py::main --action exp6 --checkpoint /workspace/TinyRecursiveModels/checkpoints/hf_trm/<subdir>/step_xxxxx
# Option B: run your own small timed loop using pretrain.py utilities (see experiments/run_exp1.py for patterns)

# Environment details and metrics (latency/memory) are captured in exp1_report.json. Save `nvidia-smi -q` if you need full driver details.
```

Seeds:
- Default seeds are in Hydra config (see `config/cfg_pretrain.yaml`). Override with `seed=<int>` in the CLI overrides as needed.

Data note (licensing/compliance):
- ARC datasets are copyrighted; do not redistribute raw task files. This repo includes Kaggle JSON references used to build processed arrays. Generate `data/` locally via `python -m experiments.prepare_arc_datasets`.
- The 1000× and 0× datasets for ARC-AGI-1 are placed under `data/arc-aug-1000` and `data/arc-aug-0`. A manifest is written to `data/arc_exp1_manifest.json`.



### How TRM works

<p align="center">
  <img src="https://AlexiaJM.github.io/assets/images/TRM_fig.png" alt="TRM"  style="width: 30%;">
</p>

Tiny Recursion Model (TRM) recursively improves its predicted answer y with a tiny network. It starts with the embedded input question x and initial embedded answer y and latent z. For up to K improvements steps, it tries to improve its answer y. It does so by i) recursively updating n times its latent z given the question x, current answer y, and current latent z (recursive reasoning), and then ii) updating its answer y given the current answer y and current latent z. This recursive process allows the model to progressively improve its answer (potentially addressing any errors from its previous answer) in an extremely parameter-efficient manner while minimizing overfitting.

### Requirements

- Python 3.10 (or similar)
- CUDA 12.1 (or similar)

```bash
pip install --upgrade pip wheel setuptools
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 # install torch build matching your CUDA version
pip install -r requirements.txt # install requirements
pip install --no-cache-dir --no-build-isolation adam-atan2 
wandb login YOUR-LOGIN # login if you want the logger to sync results to your Weights & Biases (https://wandb.ai/)
```

Optional Llama baseline:
- If you plan to run the Llama-3-8B QLoRA baseline, see `modal_llama.py`. It installs `transformers`, `peft`, `accelerate`, `bitsandbytes`, `datasets`, and `unsloth` inside the Modal image.

### Experiment 1 — Paper vs Single-Aug (Modal)

```bash
# Build datasets in the Modal volume
modal run modal_trm.py::main --action prepare_data
# or a smaller pair matching HF v1 checkpoints
modal run modal_trm.py::main --action build_arc1

# (Optional) Fetch verification checkpoint into the Modal volume
modal run modal_trm.py::main --action fetch_ckpt \
  --repo arcprize/trm_arc_prize_verification \
  --dest checkpoints/hf_trm

# Run both modes (paper + single) with your checkpoint path inside the container
modal run modal_trm.py::main --action exp1 \
  --checkpoint /workspace/TinyRecursiveModels/checkpoints/hf_trm/<subdir>/step_XXXXX

# Paper-only or Single-only
modal run modal_trm.py::main --action exp1_paper_only \
  --checkpoint /workspace/TinyRecursiveModels/checkpoints/hf_trm/<subdir>/step_XXXXX
modal run modal_trm.py::main --action exp1_single_only \
  --checkpoint /workspace/TinyRecursiveModels/checkpoints/hf_trm/<subdir>/step_XXXXX

# Quick sanity check over a few batches
modal run modal_trm.py::main --action quick_exp1 \
  --checkpoint /workspace/TinyRecursiveModels/checkpoints/hf_trm/<subdir>/step_XXXXX
```

### Experiment 3 — Llama-3-8B baseline (Modal)

Requires a Modal account and Python client (`pip install modal`). If the HF model is gated, provide a token via a Modal secret named `pipeline-secrets` (key: `HF_TOKEN`) or export `HUGGING_FACE_HUB_TOKEN`/`HF_TOKEN` in your environment.

```bash
# Prepare JSONL data from ARC JSONs baked into the image
modal run modal_llama.py::main --action prepare

# Fine-tune Llama-3-8B-Instruct (QLoRA) on training examples
modal run modal_llama.py::main --action train

# Evaluate exact-match accuracy on ARC-AGI-1 eval set
modal run modal_llama.py::main --action eval
```

Artifacts are saved under the Modal volume mounted at `/workspace/checkpoints` (adapter and merged model).

### Experiment 6 — Efficiency benchmark (Modal)

```bash
modal run modal_trm.py::main --action exp6 --checkpoint /workspace/TinyRecursiveModels/checkpoints/hf_trm/<subdir>/step_XXXXX
```

### Dataset Preparation

```bash
# ARC-AGI-1
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc1concept-aug-1000 \
  --subsets training evaluation concept \
  --test-set-name evaluation

# ARC-AGI-2
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc2concept-aug-1000 \
  --subsets training2 evaluation2 concept \
  --test-set-name evaluation2

## Note: You cannot train on both ARC-AGI-1 and ARC-AGI-2 and evaluate them both because ARC-AGI-2 training data contains some ARC-AGI-1 eval data

# Sudoku-Extreme
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000  --subsample-size 1000 --num-aug 1000  # 1000 examples, 1000 augments

# Maze-Hard
python dataset/build_maze_dataset.py # 1000 examples, 8 augments
```

## Experiments

### ARC-AGI-1 (assuming 4 H-100 GPUs):

```bash
run_name="pretrain_att_arc1concept_4"
torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=trm \
data_paths="[data/arc1concept-aug-1000]" \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=4 \
+run_name=${run_name} ema=True

```

*Runtime:* ~3 days

### ARC-AGI-2 (assuming 4 H-100 GPUs):

```bash
run_name="pretrain_att_arc2concept_4"
torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=trm \
data_paths="[data/arc2concept-aug-1000]" \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=4 \
+run_name=${run_name} ema=True

```

*Runtime:* ~3 days

### Sudoku-Extreme (assuming 1 L40S GPU):

```bash
run_name="pretrain_mlp_t_sudoku"
python pretrain.py \
arch=trm \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
epochs=50000 eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.mlp_t=True arch.pos_encodings=none \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
+run_name=${run_name} ema=True

run_name="pretrain_att_sudoku"
python pretrain.py \
arch=trm \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
epochs=50000 eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
+run_name=${run_name} ema=True
```

*Runtime:* < 36 hours

### Maze-Hard (assuming 4 L40S GPUs):

```bash
run_name="pretrain_att_maze30x30"
torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=trm \
data_paths="[data/maze-30x30-hard-1k]" \
evaluators="[]" \
epochs=50000 eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=4 \
+run_name=${run_name} ema=True
```

*Runtime:* < 24 hours

Outputs are written under `checkpoints/.../exp1_*` (e.g., `exp1_report.json`, ARC evaluator logs in `evaluator_ARC_step_0`) and any Hydra-run output directories configured in your overrides.

## Reference

If you find our work useful, please consider citing:

```bibtex
@misc{roye2025trmrepro,
      title={A Technical Note on the Efficiency and Inductive Bias of Tiny Recursive Models},
      author={Roye-Azar, Antonio},
      year={2025},
      note={Technical report; TRM reproduction and analysis},
}
```

```bibtex
@misc{jolicoeurmartineau2025morerecursivereasoningtiny,
      title={Less is More: Recursive Reasoning with Tiny Networks}, 
      author={Alexia Jolicoeur-Martineau},
      year={2025},
      eprint={2510.04871},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.04871}, 
}
```

and the Hierarchical Reasoning Model (HRM):

```bibtex
@misc{wang2025hierarchicalreasoningmodel,
      title={Hierarchical Reasoning Model}, 
      author={Guan Wang and Jin Li and Yuhao Sun and Xing Chen and Changling Liu and Yue Wu and Meng Lu and Sen Song and Yasin Abbasi Yadkori},
      year={2025},
      eprint={2506.21734},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.21734}, 
}
```

This code is based on the Hierarchical Reasoning Model [code](https://github.com/sapientinc/HRM) and the Hierarchical Reasoning Model Analysis [code](https://github.com/arcprize/hierarchical-reasoning-model-analysis).
