"""Fast checkpoint probe for TRM puzzle-identifier dependence.

The ARC evaluator never sees ablated identifiers. This script instead measures
processed test-label accuracy directly while changing only the identifiers
presented to the model. It is intended as a small smoke test, not a replacement
for the full ARC voting evaluation.
"""
from __future__ import annotations

import json
import os
from typing import Dict

import hydra
import torch
from omegaconf import DictConfig

from dataset.build_arc_dataset import inverse_aug
from models.losses import IGNORE_LABEL_ID
from pretrain import TrainState, create_dataloader, create_model, load_synced_config
from experiments.run_expC_puzzle_id import (
    _init_single_process_dist,
    _patch_load_checkpoint,
    _patch_trm_device_alignment,
)


def _evaluation_data_path(config) -> str:
    paths = config.data_paths_test if len(config.data_paths_test) else config.data_paths
    if len(paths) != 1:
        raise ValueError(f"Expected one evaluation data path, received {list(paths)}")
    return str(paths[0])


def _select_non_eval_id(data_path: str, blank_id: int) -> int:
    with open(os.path.join(data_path, "identifiers.json"), "r") as f:
        identifier_map = json.load(f)
    with open(os.path.join(data_path, "test_puzzles.json"), "r") as f:
        evaluation_names = set(json.load(f))

    for identifier, augmented_name in enumerate(identifier_map):
        if identifier == blank_id:
            continue
        original_name, _ = inverse_aug(augmented_name)
        if original_name not in evaluation_names:
            return identifier
    raise RuntimeError("No trained non-evaluation identifier was found")


def _condition_ids(
    routing_ids: torch.Tensor,
    *,
    mode: str,
    blank_id: int,
    fixed_wrong_id: int,
    num_ids: int,
    generator: torch.Generator,
) -> torch.Tensor:
    result = routing_ids.clone()
    valid = routing_ids != blank_id
    if mode == "normal":
        return result
    if mode == "fixed_wrong":
        result[valid] = fixed_wrong_id
        return result
    if mode == "padding_blank":
        result[valid] = blank_id
        return result
    if mode == "random_wrong":
        true_ids = routing_ids[valid]
        # Draw from all nonblank IDs except the true row ID.
        draws = torch.randint(
            1,
            num_ids - 1,
            true_ids.shape,
            generator=generator,
            device=routing_ids.device,
            dtype=routing_ids.dtype,
        )
        draws = draws + (draws >= true_ids).to(draws.dtype)
        result[valid] = draws
        return result
    raise ValueError(mode)


def _run_model(model, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    carry = model.initial_carry(batch)
    while True:
        carry, _loss, _metrics, outputs, finished = model(
            carry=carry,
            batch=batch,
            return_keys={"preds"},
        )
        if finished:
            return outputs["preds"]


def _summarize(
    preds: torch.Tensor,
    labels: torch.Tensor,
    routing_ids: torch.Tensor,
    blank_id: int,
    normal_preds: torch.Tensor | None,
) -> Dict[str, float | int]:
    token_mask = labels != IGNORE_LABEL_ID
    valid_rows = (routing_ids != blank_id) & token_mask.any(dim=-1)
    correct_tokens = (preds == labels) & token_mask
    row_exact = ((preds == labels) | ~token_mask).all(dim=-1) & valid_rows

    token_total = int(token_mask[valid_rows].sum().item())
    token_correct = int(correct_tokens[valid_rows].sum().item())
    result: Dict[str, float | int] = {
        "rows": int(valid_rows.sum().item()),
        "exact_correct": int(row_exact.sum().item()),
        "exact_accuracy": float(row_exact.sum().item() / max(valid_rows.sum().item(), 1)),
        "token_correct": token_correct,
        "token_total": token_total,
        "token_accuracy": float(token_correct / max(token_total, 1)),
    }
    if normal_preds is not None:
        agreement = ((preds == normal_preds) & token_mask)[valid_rows].sum()
        result["agreement_with_normal"] = float(agreement.item() / max(token_total, 1))
        result["rows_identical_to_normal"] = int(
            (((preds == normal_preds) | ~token_mask).all(dim=-1) & valid_rows).sum().item()
        )
    return result


@hydra.main(config_path="../config", config_name="cfg_pretrain", version_base=None)
def main(hydra_config: DictConfig) -> None:
    rank = 0
    world_size = 1
    config = load_synced_config(hydra_config, rank=rank, world_size=world_size)

    _patch_load_checkpoint()
    _patch_trm_device_alignment()
    _init_single_process_dist()

    max_batches = int(os.environ.get("PID_PROBE_MAX_BATCHES", "1"))
    seed = int(os.environ.get("PID_PROBE_SEED", "0"))

    eval_loader, metadata = create_dataloader(
        config,
        split="test",
        test_set_mode=True,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
        rank=rank,
        world_size=world_size,
    )
    model, _opts, _lrs = create_model(config, metadata, rank=rank, world_size=world_size)
    state = TrainState(model=model, optimizers=[], optimizer_lrs=[], carry=None, step=0, total_steps=1)
    state.model.eval()

    data_path = _evaluation_data_path(config)
    fixed_wrong_id = _select_non_eval_id(data_path, metadata.blank_identifier_id)
    rng = torch.Generator(device="cuda")
    rng.manual_seed(seed)

    totals: Dict[str, Dict[str, float]] = {}
    per_batch = []
    modes = ("normal", "fixed_wrong", "padding_blank", "random_wrong")

    with torch.inference_mode():
        for batch_index, (set_name, host_batch, _global_bs) in enumerate(eval_loader):
            if batch_index >= max_batches:
                break
            routing_batch = {key: value.cuda() for key, value in host_batch.items()}
            normal_preds = None
            batch_results = {"batch_index": batch_index, "set_name": set_name, "conditions": {}}

            for mode in modes:
                model_batch = dict(routing_batch)
                model_batch["puzzle_identifiers"] = _condition_ids(
                    routing_batch["puzzle_identifiers"],
                    mode=mode,
                    blank_id=metadata.blank_identifier_id,
                    fixed_wrong_id=fixed_wrong_id,
                    num_ids=metadata.num_puzzle_identifiers,
                    generator=rng,
                )
                preds = _run_model(state.model, model_batch)
                summary = _summarize(
                    preds,
                    routing_batch["labels"],
                    routing_batch["puzzle_identifiers"],
                    metadata.blank_identifier_id,
                    normal_preds,
                )
                batch_results["conditions"][mode] = summary
                if mode == "normal":
                    normal_preds = preds.clone()

                aggregate = totals.setdefault(
                    mode,
                    {
                        "rows": 0.0,
                        "exact_correct": 0.0,
                        "token_correct": 0.0,
                        "token_total": 0.0,
                        "agreement_numerator": 0.0,
                        "agreement_denominator": 0.0,
                    },
                )
                aggregate["rows"] += float(summary["rows"])
                aggregate["exact_correct"] += float(summary["exact_correct"])
                aggregate["token_correct"] += float(summary["token_correct"])
                aggregate["token_total"] += float(summary["token_total"])
                if "agreement_with_normal" in summary:
                    aggregate["agreement_numerator"] += float(summary["agreement_with_normal"]) * float(summary["token_total"])
                    aggregate["agreement_denominator"] += float(summary["token_total"])

                del preds
                torch.cuda.empty_cache()

            per_batch.append(batch_results)
            print(json.dumps(batch_results, indent=2), flush=True)

    aggregate_results = {}
    for mode, values in totals.items():
        aggregate_results[mode] = {
            "rows": int(values["rows"]),
            "exact_correct": int(values["exact_correct"]),
            "exact_accuracy": values["exact_correct"] / max(values["rows"], 1.0),
            "token_accuracy": values["token_correct"] / max(values["token_total"], 1.0),
        }
        if values["agreement_denominator"]:
            aggregate_results[mode]["agreement_with_normal"] = (
                values["agreement_numerator"] / values["agreement_denominator"]
            )

    output = {
        "checkpoint": config.load_checkpoint,
        "data_path": data_path,
        "max_batches": max_batches,
        "global_batch_size": config.global_batch_size,
        "fixed_wrong_id": fixed_wrong_id,
        "num_identifiers": metadata.num_puzzle_identifiers,
        "aggregate": aggregate_results,
        "per_batch": per_batch,
    }
    output_path = os.environ.get("PID_PROBE_OUTPUT", "outputs/pid_probe_results.json")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print("PID_PROBE_FINAL=" + json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
