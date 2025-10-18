import json
import os
from typing import Dict, Any, Tuple

import numpy as np


def _load_per_example(path: str) -> np.ndarray:
    results = []
    with open(path, "r") as f:
        for line in f:
            try:
                rec = json.loads(line)
                results.append(1.0 if rec.get("top1_correct", False) else 0.0)
            except Exception:
                continue
    return np.array(results, dtype=np.float64)


def bootstrap_ci(accs: np.ndarray, num_bootstrap: int = 10000, alpha: float = 0.05, rng_seed: int = 0) -> Tuple[float, float, float]:
    if accs.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(rng_seed)
    means = []
    n = accs.size
    for _ in range(num_bootstrap):
        idx = rng.integers(0, n, size=n)
        means.append(float(np.mean(accs[idx])))
    means = np.array(means)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return float(np.mean(accs)), lo, hi


def build_exp1_table(paper_dir: str, single_dir: str, out_path: str, rng_seed: int = 0) -> Dict[str, Any]:
    paper_jsonl = os.path.join(paper_dir, "evaluator_ARC_step_0", "per_example.jsonl")
    single_jsonl = os.path.join(single_dir, "evaluator_ARC_step_0", "per_example.jsonl")

    paper = _load_per_example(paper_jsonl) if os.path.isfile(paper_jsonl) else np.array([], dtype=np.float64)
    single = _load_per_example(single_jsonl) if os.path.isfile(single_jsonl) else np.array([], dtype=np.float64)

    paper_mean, paper_lo, paper_hi = bootstrap_ci(paper, rng_seed=rng_seed)
    single_mean, single_lo, single_hi = bootstrap_ci(single, rng_seed=rng_seed + 1)

    delta = None
    if np.isfinite(paper_mean) and np.isfinite(single_mean):
        delta = paper_mean - single_mean

    table = {
        "paper_mode": {
            "mean": paper_mean,
            "ci95": [paper_lo, paper_hi],
            "n": int(paper.size),
        },
        "single_aug": {
            "mean": single_mean,
            "ci95": [single_lo, single_hi],
            "n": int(single.size),
        },
        "delta_paper_minus_single": delta,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(table, f, indent=2)

    return table


