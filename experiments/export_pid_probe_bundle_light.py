"""Export a reduced TRM checkpoint without materializing the full ARC dataset.

The script replays the official dataset builder's RNG, puzzle shuffle,
augmentation de-duplication, and identifier assignment exactly. It retains
canonical query examples from the first evaluation puzzles and only the
embedding rows needed by the isolated probe.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dataset.build_arc_dataset import (
    ARCAugmentRetriesFactor,
    arc_grid_to_np,
    aug,
    grid_hash,
    np_grid_to_seq_translational_augment,
)


def _all_pairs(puzzle: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    pairs = []
    for split in ("train", "test"):
        for example in puzzle.get(split, []):
            pairs.append((arc_grid_to_np(example["input"]), arc_grid_to_np(example["output"])))
    return pairs


def _pairs_hash(pairs: list[tuple[np.ndarray, np.ndarray]]) -> str:
    import hashlib

    hashes = [f"{grid_hash(inp)}|{grid_hash(out)}" for inp, out in pairs]
    hashes.sort()
    return hashlib.sha256("|".join(hashes).encode()).hexdigest()


def _variant_names(name: str, puzzle: dict, aug_count: int) -> list[str]:
    pairs = _all_pairs(puzzle)
    names = [name]
    hashes = {_pairs_hash(pairs)}
    for _trial in range(ARCAugmentRetriesFactor * aug_count):
        aug_name, map_grid = aug(name)
        transformed = [(map_grid(inp), map_grid(out)) for inp, out in pairs]
        variant_hash = _pairs_hash(transformed)
        if variant_hash not in hashes:
            hashes.add(variant_hash)
            names.append(aug_name)
        if len(names) >= aug_count + 1:
            break
    return names


def _load_subset(prefix: str, subset: str) -> list[tuple[str, dict]]:
    with open(f"{prefix}_{subset}_challenges.json", "r") as f:
        puzzles = json.load(f)
    solutions_path = Path(f"{prefix}_{subset}_solutions.json")
    if solutions_path.exists():
        with open(solutions_path, "r") as f:
            solutions = json.load(f)
        for puzzle_id, outputs in solutions.items():
            for index, output in enumerate(outputs):
                puzzles[puzzle_id]["test"][index]["output"] = output
    else:
        for puzzle in puzzles.values():
            for example in puzzle["test"]:
                example.setdefault("output", [[0]])
    items = list(puzzles.items())
    np.random.shuffle(items)
    return items


def _find_embedding_key(state: dict[str, torch.Tensor]) -> str:
    keys = [key for key in state if key.endswith("puzzle_emb.weights")]
    if len(keys) != 1:
        raise RuntimeError(f"Expected one puzzle embedding tensor, found {keys}")
    return keys[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-prefix", default="kaggle/combined/arc-agi")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-puzzles", type=int, default=8)
    parser.add_argument("--num-aug", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    next_identifier = 1
    name_to_identifier: dict[str, int] = {}
    fixed_wrong_global_id: int | None = None
    selected_global_ids: list[int] = []
    selected_names: list[str] = []
    selected_inputs: list[np.ndarray] = []
    selected_labels: list[np.ndarray] = []

    for subset in ("training", "evaluation", "concept"):
        items = _load_subset(args.input_prefix, subset)
        for puzzle_name, puzzle in items:
            names = _variant_names(puzzle_name, puzzle, args.num_aug)
            for variant_name in names:
                if variant_name not in name_to_identifier:
                    name_to_identifier[variant_name] = next_identifier
                    next_identifier += 1

            if subset == "training" and fixed_wrong_global_id is None:
                fixed_wrong_global_id = name_to_identifier[puzzle_name]

            if subset == "evaluation" and len(selected_global_ids) < args.num_puzzles:
                selected_global_ids.append(name_to_identifier[puzzle_name])
                selected_names.append(puzzle_name)
                query = puzzle["test"][0]
                inp, label = np_grid_to_seq_translational_augment(
                    arc_grid_to_np(query["input"]),
                    arc_grid_to_np(query["output"]),
                    do_translation=False,
                )
                selected_inputs.append(np.asarray(inp, dtype=np.int32))
                label_array = np.asarray(label, dtype=np.int32)
                label_array[label_array == 0] = -100
                selected_labels.append(label_array)

    if fixed_wrong_global_id is None:
        raise RuntimeError("No training identifier found")
    if len(selected_global_ids) != args.num_puzzles:
        raise RuntimeError(f"Selected only {len(selected_global_ids)} evaluation puzzles")

    retained_global_ids = [0] + selected_global_ids + [fixed_wrong_global_id]
    global_to_local = {gid: lid for lid, gid in enumerate(retained_global_ids)}
    selected_local_ids = np.asarray([global_to_local[x] for x in selected_global_ids], dtype=np.int32)
    fixed_wrong_local_id = global_to_local[fixed_wrong_global_id]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "samples.npz",
        inputs=np.stack(selected_inputs),
        labels=np.stack(selected_labels),
        correct_local_ids=selected_local_ids,
        blank_local_id=np.asarray(0, dtype=np.int32),
        fixed_wrong_local_id=np.asarray(fixed_wrong_local_id, dtype=np.int32),
    )

    print(f"Replayed {next_identifier} identifiers; loading checkpoint with mmap=True", flush=True)
    state = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=True)
    embedding_key = _find_embedding_key(state)
    full_embedding = state[embedding_key]
    if full_embedding.shape[0] != next_identifier:
        raise RuntimeError(
            f"Identifier replay mismatch: checkpoint has {full_embedding.shape[0]} rows, replay produced {next_identifier}"
        )

    reduced_state: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key == embedding_key:
            reduced_state[key] = full_embedding[retained_global_ids].clone()
        else:
            reduced_state[key] = value.clone()
    torch.save(reduced_state, output_dir / "reduced_checkpoint.pt")

    manifest = {
        "source_checkpoint": args.checkpoint,
        "embedding_key": embedding_key,
        "replayed_num_identifiers": next_identifier,
        "num_local_identifiers": len(retained_global_ids),
        "selected_global_ids": selected_global_ids,
        "selected_local_ids": selected_local_ids.tolist(),
        "selected_names": selected_names,
        "fixed_wrong_global_id": fixed_wrong_global_id,
        "fixed_wrong_local_id": fixed_wrong_local_id,
        "retained_global_ids": retained_global_ids,
        "seq_len": int(selected_inputs[0].shape[0]),
        "vocab_size": 12,
        "num_aug": args.num_aug,
        "seed": args.seed,
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2), flush=True)
    print(f"Reduced checkpoint bytes: {(output_dir / 'reduced_checkpoint.pt').stat().st_size}")


if __name__ == "__main__":
    main()
