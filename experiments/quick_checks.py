import os
import json
from typing import Tuple, Dict, Any

import numpy as np

from argdantic import ArgParser
from pydantic import BaseModel


cli = ArgParser()


class QuickCheckConfig(BaseModel):
    data_dir: str
    split: str = "test"


def _load_test_pairs_count(data_dir: str) -> int:
    """Count original (non-augmented) test input-output pairs."""
    tp = os.path.join(data_dir, "test_puzzles.json")
    with open(tp, "r") as f:
        test_puzzles = json.load(f)
    return sum(len(p.get("test", [])) for p in test_puzzles.values())


def _load_set_arrays(data_dir: str, split: str) -> Dict[str, np.ndarray]:
    """Load minimal arrays for counts without heavy memory usage."""
    # We assume 'all' set for ARC as produced by dataset.build_arc_dataset
    set_name = "all"
    base = os.path.join(data_dir, split)
    arrays = {}
    for name in ("inputs", "puzzle_indices", "group_indices"):
        arrays[name] = np.load(os.path.join(base, f"{set_name}__{name}.npy"), mmap_mode="r")
    return arrays


def compute_aug_stats(data_dir: str, split: str = "test") -> Dict[str, Any]:
    """Compute augmentation statistics from dataset files."""
    arrays = _load_set_arrays(data_dir, split)
    total_examples = int(arrays["inputs"].shape[0])
    original_pairs = _load_test_pairs_count(data_dir)

    # Compute 1× kept examples by selecting only the first variant of each original puzzle group
    group_indices = arrays["group_indices"]  # points to puzzle_id
    puzzle_indices = arrays["puzzle_indices"]  # points to example_id
    kept_examples = 0
    # For each original puzzle group, pick the first puzzle variant (index = group_indices[i])
    for i in range(len(group_indices) - 1):
        first_puzzle_id = int(group_indices[i])
        if first_puzzle_id + 1 < len(puzzle_indices):
            start_ex = int(puzzle_indices[first_puzzle_id])
            end_ex = int(puzzle_indices[first_puzzle_id + 1])
            if end_ex > start_ex:
                kept_examples += (end_ex - start_ex)

    return {
        "data_dir": data_dir,
        "total_augmented_examples": total_examples,
        "original_test_pairs": original_pairs,
        "augmentations_per_original": (total_examples / original_pairs) if original_pairs > 0 else None,
        "single_aug_kept_examples": kept_examples,
        "single_aug_examples_per_original": (kept_examples / original_pairs) if original_pairs > 0 else None,
    }


@cli.command(singleton=True)
def main(config: QuickCheckConfig):
    """
    Quick dataset sanity checks for Exp1 without running the model:
      - Verifies that arc-aug-1000 has ~1000 augmentations per original
      - Simulates the single-aug iterator filter (first variant per group)
    Usage:
      python -m experiments.quick_checks --data_dir data/arc-aug-1000
    """
    stats = compute_aug_stats(config.data_dir, config.split)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()


