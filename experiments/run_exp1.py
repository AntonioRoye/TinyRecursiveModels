import os
import json
import time
import platform
import subprocess
from typing import Dict, Any, List, Optional
import hashlib

import numpy as np
import torch
import hydra
from omegaconf import DictConfig

# Reuse existing training/eval utilities without running training
from pretrain import (
    load_synced_config,
    create_dataloader,
    create_model,
    TrainState,
    evaluate,
    create_evaluators,
)
from experiments.ci_utils import build_exp1_table
def _make_serializable(obj):
    # Recursively convert numpy/torch scalars and arrays to plain Python types
    try:
        # Handle numpy/torch scalars with .item()
        if hasattr(obj, "item"):
            return obj.item()
    except Exception:
        pass
    try:
        # Handle numpy arrays / torch tensors with .tolist()
        if hasattr(obj, "tolist"):
            return obj.tolist()
    except Exception:
        pass
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    return obj



def _safe_git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        return None
def _sha256_of_file(path: str) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None

def _sha256_of_dir(path: str) -> Optional[str]:
    # Stable directory hash (filenames + file sha256)
    try:
        entries = []
        for root, _, files in os.walk(path):
            for fn in sorted(files):
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, path)
                fhash = _sha256_of_file(full) or ""
                entries.append(rel + ":" + fhash)
        entries.sort()
        h = hashlib.sha256("\n".join(entries).encode("utf-8"))
        return h.hexdigest()
    except Exception:
        return None



def _system_info() -> Dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    device_name = torch.cuda.get_device_name(0) if cuda_available and device_count > 0 else None
    compute_capability = None
    if cuda_available and device_count > 0:
        try:
            major, minor = torch.cuda.get_device_capability(0)
            compute_capability = f"{major}.{minor}"
        except Exception:
            compute_capability = None

    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch_version": torch.__version__,
        "cuda_version": getattr(torch.version, "cuda", None),
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cuda_available": cuda_available,
        "gpu_count": device_count,
        "gpu_name": device_name,
        "gpu_compute_capability": compute_capability,
        "git_commit": _safe_git_commit(),
    }


def _count_eval_examples(dataloader) -> int:
    # Count total examples across all sets by peeking into the dataset's loaded arrays
    dataset = dataloader.dataset
    # Ensure data is loaded
    try:
        dataset._lazy_load_dataset()
    except Exception:
        pass

    total = 0
    try:
        for _set_name, ds in dataset._data.items():  # type: ignore[attr-defined]
            total += len(ds["inputs"])  # numpy array length
    except Exception:
        # Fallback: iterate once (slower, but robust)
        for _set_name, _batch, batch_size in dataloader:
            total += batch_size
    return total


def _count_original_test_pairs(data_paths: List[str]) -> int:
    # Each dataset path contains test_puzzles.json with original (non-augmented) test pairs
    if not len(data_paths):
        return 0
    tp = os.path.join(data_paths[0], "test_puzzles.json")
    try:
        with open(tp, "r") as f:
            test_puzzles = json.load(f)
        return sum(len(p.get("test", [])) for p in test_puzzles.values())
    except Exception:
        return 0


def _ensure_checkpoint_dir(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    os.makedirs(path, exist_ok=True)
    return path


def _run_single_eval(
    *,
    config,
    data_paths: List[str],
    rank: int,
    world_size: int,
    label: str,
    checkpoint_path_suffix: str,
) -> Dict[str, Any]:
    # Clone config to avoid mutating original
    cfg = config.model_copy(deep=True)
    cfg.data_paths = data_paths
    # Ensure saving of raw outputs for auditability
    cfg.eval_save_outputs = [
        "inputs",
        "preds",
        "puzzle_identifiers",
        "q_halt_logits",
    ]

    # Prepare checkpoint dir for this eval label
    base_ckpt = cfg.checkpoint_path
    eval_ckpt_dir = None
    if base_ckpt is not None:
        eval_ckpt_dir = os.path.join(base_ckpt, f"exp1_{checkpoint_path_suffix}")
        _ensure_checkpoint_dir(eval_ckpt_dir)
        cfg.checkpoint_path = eval_ckpt_dir

    # Dataloader (test only)
    eval_loader, eval_metadata = create_dataloader(
        cfg,
        split="test",
        test_set_mode=True,
        epochs_per_iter=1,
        global_batch_size=cfg.global_batch_size,
        rank=rank,
        world_size=world_size,
    )

    # Model (load checkpoint if provided in cfg)
    model, _opts, _lrs = create_model(cfg, eval_metadata, rank=rank, world_size=world_size)
    train_state = TrainState(
        model=model,
        optimizers=[],
        optimizer_lrs=[],
        carry=None,
        step=0,
        total_steps=1,
    )
    train_state.model.eval()

    # Evaluators (ARC) with per-example logging enabled for bootstrap
    if not len(cfg.evaluators):
        cfg.evaluators = [{"name": "arc@ARC", "save_per_example": True}]  # type: ignore[assignment]
    else:
        # inject save_per_example=True into any ARC evaluators
        for e in cfg.evaluators:
            if isinstance(e, dict) and isinstance(e.get("name"), str) and e["name"].startswith("arc@ARC"):
                e["save_per_example"] = True
    evaluator_instances = create_evaluators(cfg, eval_metadata)

    # Count examples (for latency normalization)
    total_examples = _count_eval_examples(eval_loader)
    original_pairs = _count_original_test_pairs(data_paths)

    # Measure performance
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    metrics = evaluate(
        cfg,
        train_state,
        eval_loader,
        eval_metadata,
        evaluators=evaluator_instances,
        rank=rank,
        world_size=world_size,
        cpu_group=None,
    )
    t1 = time.perf_counter()

    elapsed_s = t1 - t0
    peak_mem_bytes = (
        int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
    )

    # Collect evaluator results if ARC evaluator configured
    # Note: When cfg.evaluators includes ARC, evaluate() will append its metrics into the returned dict.
    results = {
        "label": label,
        "data_paths": data_paths,
        "data_dir_hash": _sha256_of_dir(data_paths[0]) if len(data_paths) else None,
        "total_augmented_examples": total_examples,
        "original_test_pairs": original_pairs,
        "augmentations_per_original": (total_examples / original_pairs) if original_pairs > 0 else None,
        "elapsed_seconds": elapsed_s,
        "avg_latency_seconds_per_augmented_example": (elapsed_s / total_examples) if total_examples > 0 else None,
        "avg_time_seconds_per_original_example": (elapsed_s / original_pairs) if original_pairs > 0 else None,
        "peak_gpu_memory_bytes": peak_mem_bytes,
        "metrics": metrics or {},
        "checkpoint_dir": eval_ckpt_dir,
        "checkpoint_file": config.load_checkpoint,
        "checkpoint_sha256": _sha256_of_file(config.load_checkpoint) if config.load_checkpoint else None,
    }
    return results


@hydra.main(config_path="../config", config_name="cfg_pretrain", version_base=None)
def main(hydra_config: DictConfig):
    # Single-GPU default; distributed init not required for g5.xlarge
    rank = 0
    world_size = 1

    # Sync and enrich config
    config = load_synced_config(hydra_config, rank=rank, world_size=world_size)

    # Enforce that a trained checkpoint is provided
    if not config.load_checkpoint:
        raise RuntimeError(
            "load_checkpoint must be provided for evaluation. Example: load_checkpoint=checkpoints/<project>/<run>/step_XXXX"
        )

    # Determinism for reproducibility
    torch.manual_seed(config.seed + rank)
    np.random.seed(config.seed + rank)
    try:
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
    except Exception:
        pass

    # Prepare base checkpoint directory
    _ensure_checkpoint_dir(config.checkpoint_path)

    # System info snapshot
    sysinfo = _system_info()

    # Expect the caller to pass:
    # - data_paths = [paths for 1000× aug dataset]
    # - data_paths_test = [paths for 1× aug dataset]
    paper_paths = config.data_paths
    single_aug_paths = config.data_paths_test if len(config.data_paths_test) else []
    if not len(paper_paths) or not len(single_aug_paths):
        # Provide a clear error with instructions
        raise RuntimeError(
            "Both data_paths (1000×) and data_paths_test (1×) must be provided.\n"
            "Example: data_paths='[data/arc-aug-1000]' data_paths_test='[data/arc-aug-1]'"
        )

    # Ensure ARC evaluator is configured unless explicitly disabled
    if not len(config.evaluators):
        # Default to ARC evaluator with standard pass@Ks; leave aggregated voting to dataset structure
        config.evaluators = [{"name": "arc@ARC"}]  # type: ignore[assignment]

    # Run single-augmentation mode (1×, no voting) FIRST for quick verification
    print("Running Single-Aug Mode (1x)...")
    single = _run_single_eval(
        config=config,
        data_paths=single_aug_paths,
        rank=rank,
        world_size=world_size,
        label="single_aug_1x",
        checkpoint_path_suffix="single_aug_1x",
    )

    # Run paper mode (1000× voting) SECOND
    print("Running Paper Mode (1000x)...")
    paper = _run_single_eval(
        config=config,
        data_paths=paper_paths,
        rank=rank,
        world_size=world_size,
        label="paper_mode_1000x",
        checkpoint_path_suffix="paper_1000x",
    )

    # Combined JSON report with methods snapshot
    report = {
        "experiment": "TRM-Exp1-SecretSauce",
        "environment": sysinfo,
        "seed": config.seed,
        "repo": {
            "git_commit": sysinfo.get("git_commit"),
            "entrypoints": [
                "experiments/prepare_arc_datasets.py",
                "experiments/run_exp1.py",
            ],
        },
        "config_summary": {
            "arch": config.arch.name,
            "loss": config.arch.loss.name,
            "global_batch_size": config.global_batch_size,
            "lr": config.lr,
            "lr_warmup_steps": config.lr_warmup_steps,
            "weight_decay": config.weight_decay,
            "beta1": config.beta1,
            "beta2": config.beta2,
            "puzzle_emb_lr": config.puzzle_emb_lr,
            "puzzle_emb_weight_decay": config.puzzle_emb_weight_decay,
            "ema": config.ema,
            "ema_rate": config.ema_rate,
            "freeze_weights": config.freeze_weights,
        },
        "commands": {
            "prepare_data": "python experiments/prepare_arc_datasets.py",
            "run_paper_mode": "python experiments/run_exp1.py data_paths='[data/arc-aug-1000]' data_paths_test='[data/arc-aug-0]' load_checkpoint=<PATH> checkpoint_path=checkpoints/exp1_arc arch=trm",
        },
        "modes": {
            "paper": paper,
            "single_aug": single,
        },
    }

    # Save report next to checkpoint path if available
    out_dir = config.checkpoint_path or os.getcwd()
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "exp1_report.json")
    with open(out_file, "w") as f:
        json.dump(_make_serializable(report), f, indent=2)

    # Build and save concise CI table
    try:
        paper_dir = paper.get("checkpoint_dir") or out_dir
        single_dir = single.get("checkpoint_dir") or out_dir
        table_path = os.path.join(out_dir, "exp1_table.json")
        build_exp1_table(paper_dir, single_dir, table_path, rng_seed=config.seed)
    except Exception as e:
        print(f"Failed to build CI table: {e}")

    # Print a compact summary for quick inspection
    def _fmt(mode: Dict[str, Any]) -> str:
        acc_keys = [k for k in (mode.get("metrics") or {}).keys() if k.startswith("ARC/")]
        acc = {k: (mode["metrics"][k]) for k in sorted(acc_keys)} if acc_keys else {}
        return json.dumps(
            {
                "label": mode.get("label"),
                "examples": mode.get("total_examples"),
                "avg_latency_s": mode.get("avg_latency_seconds_per_example"),
                "peak_mem_bytes": mode.get("peak_gpu_memory_bytes"),
                "acc": acc,
            },
            indent=2,
        )

    print("\n=== Experiment 1 Summary ===")
    print("Paper mode:")
    print(_fmt(paper))
    print("\nSingle-aug mode:")
    print(_fmt(single))
    print(f"\nFull report written to: {out_file}")


if __name__ == "__main__":
    main()


