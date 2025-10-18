import os
import json
import csv
import time
import hashlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import hydra
from omegaconf import DictConfig

from pretrain import (
    load_synced_config,
    create_dataloader,
    create_model,
    TrainState,
    evaluate,
    create_evaluators,
)


def _sha256_of_file(path: str) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _count_eval_examples(dataloader) -> int:
    dataset = dataloader.dataset
    try:
        dataset._lazy_load_dataset()  # type: ignore[attr-defined]
    except Exception:
        pass
    total = 0
    try:
        for _set_name, ds in dataset._data.items():  # type: ignore[attr-defined]
            total += len(ds["inputs"])  # numpy array length
    except Exception:
        for _set_name, _batch, batch_size in dataloader:
            total += batch_size
    return total


def _count_original_test_pairs(data_paths: List[str]) -> int:
    if not len(data_paths):
        return 0
    tp = os.path.join(data_paths[0], "test_puzzles.json")
    try:
        with open(tp, "r") as f:
            test_puzzles = json.load(f)
        return sum(len(p.get("test", [])) for p in test_puzzles.values())
    except Exception:
        return 0


def _load_per_example(jsonl_path: str) -> np.ndarray:
    accs: List[float] = []
    try:
        with open(jsonl_path, "r") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    accs.append(1.0 if rec.get("top1_correct", False) else 0.0)
                except Exception:
                    continue
    except Exception:
        pass
    return np.array(accs, dtype=np.float64)


def _bootstrap_ci(accs: np.ndarray, num_bootstrap: int = 10000, alpha: float = 0.05, rng_seed: int = 0) -> Tuple[float, float, float]:
    if accs.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(rng_seed)
    n = accs.size
    means = np.empty(num_bootstrap, dtype=np.float64)
    for i in range(num_bootstrap):
        idx = rng.integers(0, n, size=n)
        means[i] = float(np.mean(accs[idx]))
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return float(np.mean(accs)), lo, hi


@hydra.main(config_path="../config", config_name="cfg_pretrain", version_base=None)
def main(hydra_config: DictConfig):
    rank = 0
    world_size = 1

    config = load_synced_config(hydra_config, rank=rank, world_size=world_size)

    if not config.load_checkpoint:
        raise RuntimeError("load_checkpoint must be provided for evaluation.")

    # Determinism
    torch.manual_seed(config.seed + rank)
    np.random.seed(config.seed + rank)
    try:
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
    except Exception:
        pass

    # Read manifest written by prepare_arc_datasets.py
    manifest_path = os.path.join("data", "arc_exp1_manifest.json")
    if not os.path.isfile(manifest_path):
        raise RuntimeError("Run experiments/prepare_arc_datasets.py first to produce arc_exp1_manifest.json")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    level_map: Dict[str, str] = manifest.get("all_levels", {})
    if not level_map:
        raise RuntimeError("Manifest missing all_levels; rebuild datasets.")

    # Output directory
    out_root = config.checkpoint_path or os.getcwd()
    csv_path = os.path.join(out_root, "acc_vs_aug.csv")

    rows: List[List[Any]] = [[
        "augmentations",
        "mean_top1",
        "ci95_lo",
        "ci95_hi",
        "num_examples",
        "elapsed_seconds",
        "avg_time_per_original_example",
        "peak_gpu_memory_bytes",
        "dataset_path",
        "dataset_dir_sha256",
        "checkpoint_sha256",
    ]]

    # Iterate levels in numeric order
    levels_sorted = sorted((int(k), v) for k, v in level_map.items())
    for k, data_path in levels_sorted:
        # Clone and set dataset path
        cfg = config.model_copy(deep=True)
        cfg.data_paths = [data_path]
        cfg.eval_save_outputs = ["inputs", "preds", "puzzle_identifiers", "q_halt_logits"]

        # Dedicated subdir for this level
        level_dir = os.path.join(out_root, f"acc_vs_aug/aug_{k}")
        os.makedirs(level_dir, exist_ok=True)
        cfg.checkpoint_path = level_dir

        # Dataloader
        eval_loader, eval_metadata = create_dataloader(
            cfg,
            split="test",
            test_set_mode=True,
            epochs_per_iter=1,
            global_batch_size=cfg.global_batch_size,
            rank=rank,
            world_size=world_size,
        )

        # Model
        model, _opts, _lrs = create_model(cfg, eval_metadata, rank=rank, world_size=world_size)
        train_state = TrainState(model=model, optimizers=[], optimizer_lrs=[], carry=None, step=0, total_steps=1)
        train_state.model.eval()

        # ARC evaluators with per-example
        if not len(cfg.evaluators):
            cfg.evaluators = [{"name": "arc@ARC", "save_per_example": True}]  # type: ignore[assignment]
        else:
            for e in cfg.evaluators:
                if isinstance(e, dict) and isinstance(e.get("name"), str) and e["name"].startswith("arc@ARC"):
                    e["save_per_example"] = True
        evaluators = create_evaluators(cfg, eval_metadata)

        # Count examples
        total_examples = _count_eval_examples(eval_loader)
        original_pairs = _count_original_test_pairs([data_path])

        # Measure
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        metrics = evaluate(cfg, train_state, eval_loader, eval_metadata, evaluators=evaluators, rank=rank, world_size=world_size, cpu_group=None)
        t1 = time.perf_counter()
        elapsed_s = t1 - t0
        peak_mem = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None

        # Load per-example and compute CI
        jsonl_path = os.path.join(level_dir, "evaluator_ARC_step_0", "per_example.jsonl")
        accs = _load_per_example(jsonl_path)
        mean, lo, hi = _bootstrap_ci(accs, rng_seed=cfg.seed)

        rows.append([
            k,
            mean,
            lo,
            hi,
            int(accs.size),
            elapsed_s,
            (elapsed_s / original_pairs) if original_pairs > 0 else None,
            peak_mem,
            data_path,
            _sha256_of_file(jsonl_path),  # cheap proxy; dataset dirs are large
            _sha256_of_file(cfg.load_checkpoint) if cfg.load_checkpoint else None,
        ])

    # Write CSV
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()


