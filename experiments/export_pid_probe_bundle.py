"""Export a small self-contained bundle for an isolated puzzle-ID probe.

This script expects checkpoint-compatible ARC data to have already been built.
It selects canonical test examples from distinct evaluation puzzles, retains only
needed puzzle-embedding rows, and saves a reduced state_dict plus sample tensors.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from dataset.build_arc_dataset import inverse_aug


def _find_embedding_key(state: dict[str, torch.Tensor]) -> str:
    candidates = [key for key in state if key.endswith("puzzle_emb.weights")]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one puzzle embedding tensor, found {candidates}")
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-puzzles", type=int, default=8)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    test_dir = data_dir / "test"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = np.load(test_dir / "all__inputs.npy", mmap_mode="r")
    labels = np.load(test_dir / "all__labels.npy", mmap_mode="r")
    puzzle_identifiers = np.load(test_dir / "all__puzzle_identifiers.npy")
    puzzle_indices = np.load(test_dir / "all__puzzle_indices.npy")
    group_indices = np.load(test_dir / "all__group_indices.npy")

    with open(data_dir / "identifiers.json", "r") as f:
        identifier_names = json.load(f)
    with open(data_dir / "test_puzzles.json", "r") as f:
        test_puzzles = json.load(f)

    selected_example_indices: list[int] = []
    selected_global_ids: list[int] = []
    selected_names: list[str] = []

    # Each test group is one original ARC puzzle; its first puzzle variant is canonical.
    for group_id in range(min(args.num_puzzles, len(group_indices) - 1)):
        canonical_puzzle_index = int(group_indices[group_id])
        example_index = int(puzzle_indices[canonical_puzzle_index])
        global_id = int(puzzle_identifiers[canonical_puzzle_index])
        selected_example_indices.append(example_index)
        selected_global_ids.append(global_id)
        selected_names.append(identifier_names[global_id])

    evaluation_names = set(test_puzzles)
    fixed_wrong_global_id = None
    for global_id, augmented_name in enumerate(identifier_names):
        if global_id == 0:
            continue
        original_name, _ = inverse_aug(augmented_name)
        if original_name not in evaluation_names:
            fixed_wrong_global_id = global_id
            break
    if fixed_wrong_global_id is None:
        raise RuntimeError("Could not find a non-evaluation trained identifier")

    # Local embedding table: blank, selected correct rows, then one fixed wrong row.
    retained_global_ids = [0] + selected_global_ids + [fixed_wrong_global_id]
    global_to_local = {global_id: local_id for local_id, global_id in enumerate(retained_global_ids)}
    selected_local_ids = np.array([global_to_local[x] for x in selected_global_ids], dtype=np.int32)
    fixed_wrong_local_id = global_to_local[fixed_wrong_global_id]

    sample_inputs = np.asarray(inputs[selected_example_indices], dtype=np.int32)
    sample_labels = np.asarray(labels[selected_example_indices], dtype=np.int32)
    sample_labels[sample_labels == 0] = -100

    np.savez_compressed(
        output_dir / "samples.npz",
        inputs=sample_inputs,
        labels=sample_labels,
        correct_local_ids=selected_local_ids,
        blank_local_id=np.array(0, dtype=np.int32),
        fixed_wrong_local_id=np.array(fixed_wrong_local_id, dtype=np.int32),
    )

    print("Loading checkpoint with mmap=True", flush=True)
    state = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=True)
    embedding_key = _find_embedding_key(state)
    full_embedding = state[embedding_key]
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
        "num_puzzles": len(selected_names),
        "selected_example_indices": selected_example_indices,
        "selected_global_ids": selected_global_ids,
        "selected_local_ids": selected_local_ids.tolist(),
        "selected_names": selected_names,
        "fixed_wrong_global_id": fixed_wrong_global_id,
        "fixed_wrong_local_id": fixed_wrong_local_id,
        "retained_global_ids": retained_global_ids,
        "num_local_identifiers": len(retained_global_ids),
        "seq_len": int(sample_inputs.shape[1]),
        "vocab_size": 12,
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2), flush=True)
    print(f"Reduced checkpoint bytes: {(output_dir / 'reduced_checkpoint.pt').stat().st_size}")


if __name__ == "__main__":
    main()
