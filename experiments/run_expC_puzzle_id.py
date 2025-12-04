import os
import json
from typing import Dict, Any, Tuple, Literal

import torch
import torch.distributed as dist
import hydra
from omegaconf import DictConfig

from pretrain import (
	load_synced_config,
	create_dataloader,
	create_model,
	TrainState,
)
from pretrain import evaluate as _unused_original_evaluate  # noqa: F401
from pretrain import create_evaluators as _create_evaluators


def _preflight(config, checkpoint_hint: str, data_paths_hint: list[str]) -> None:
	print("[ExpC] Preflight checklist:")
	print(f"  - Torch: {torch.__version__}, CUDA: {getattr(torch.version, 'cuda', None)}, CUDA available: {torch.cuda.is_available()}")
	if torch.cuda.is_available():
		try:
			print(f"  - GPU: {torch.cuda.get_device_name(0)}")
		except Exception:
			pass
	print(f"  - arch: {config.arch.name}, L_cycles={getattr(config.arch, '__pydantic_extra__', {}).get('L_cycles', 'NA')}, H_cycles={getattr(config.arch, '__pydantic_extra__', {}).get('H_cycles', 'NA')}")
	print(f"  - checkpoint: {checkpoint_hint} (exists={os.path.exists(checkpoint_hint)})")
	for p in data_paths_hint:
		ok = os.path.exists(p) and os.path.exists(os.path.join(p, 'test_puzzles.json'))
		print(f"  - data path: {p} (ok={ok})")


def _patch_load_checkpoint() -> None:
	"""
	Make checkpoint loading more robust to different key prefixes and PyTorch versions.
	Borrows logic from modal_trm.py (exp6).
	"""
	import importlib
	pt = importlib.import_module("pretrain")
	_orig_lc = pt.load_checkpoint

	def _patched_lc(model, config):
		if getattr(config, "load_checkpoint", None) is None:
			return _orig_lc(model, config)
		print(f"[ExpC] Patched load_checkpoint: {config.load_checkpoint}")
		sd = torch.load(config.load_checkpoint, map_location="cuda")
		# Strip possible _orig_mod. prefix used by torch.compile artifacts
		if any(isinstance(k, str) and k.startswith("_orig_mod.") for k in sd.keys()):
			sd = { (k[10:] if isinstance(k, str) and k.startswith("_orig_mod.") else k): v for k, v in sd.items() }
		# Attempt to adapt puzzle embedding shape if it differs
		try:
			exp_shape = model.model.puzzle_emb.weights.shape  # type: ignore[attr-defined]
			key = "model.inner.puzzle_emb.weights"
			if key in sd and getattr(sd[key], "shape", None) != exp_shape:
				pe = sd[key]
				sd[key] = torch.mean(pe, dim=0, keepdim=True).expand(exp_shape).contiguous()
		except Exception:
			pass
		# load non-strict and support assign=True where available
		try:
			model.load_state_dict(sd, strict=False, assign=True)
		except TypeError:
			model.load_state_dict(sd, strict=False)

	pt.load_checkpoint = _patched_lc  # type: ignore[attr-defined]


def _patch_trm_device_alignment() -> None:
	"""
	Patch TRM inner/outer carry helpers to ensure all tensors reside on the module device.
	Matches logic used in modal_trm.py (exp6).
	"""
	import importlib
	try:
		trm_mod = importlib.import_module("models.recursive_reasoning.trm")
	except Exception:
		return
	# 1) Ensure empty_carry allocates on module buffer/device
	if hasattr(trm_mod, "TinyRecursiveReasoningModel_ACTV1_Inner"):
		_orig_empty = trm_mod.TinyRecursiveReasoningModel_ACTV1_Inner.empty_carry
		def _patched_empty(self, batch_size: int):
			try:
				dev = self.H_init.device
			except Exception:
				dev = next(self.parameters()).device  # type: ignore
			return trm_mod.TinyRecursiveReasoningModel_ACTV1InnerCarry(
				z_H=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype, device=dev),
				z_L=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype, device=dev),
			)
		trm_mod.TinyRecursiveReasoningModel_ACTV1_Inner.empty_carry = _patched_empty
		# 2) Ensure reset_carry uses reset_flag and carry on correct device
		_orig_reset = trm_mod.TinyRecursiveReasoningModel_ACTV1_Inner.reset_carry
		def _patched_reset(self, reset_flag, carry):
			try:
				dev = self.H_init.device
			except Exception:
				dev = next(self.parameters()).device  # type: ignore
			reset_flag = reset_flag.to(dev)
			carry = trm_mod.TinyRecursiveReasoningModel_ACTV1InnerCarry(
				z_H=carry.z_H.to(dev),
				z_L=carry.z_L.to(dev),
			)
			return _orig_reset(self, reset_flag, carry)
		trm_mod.TinyRecursiveReasoningModel_ACTV1_Inner.reset_carry = _patched_reset
	# 3) Ensure outer initial_carry has steps/halted on device and inner_carry on device
	if hasattr(trm_mod, "TinyRecursiveReasoningModel_ACTV1"):
		_orig_initc = trm_mod.TinyRecursiveReasoningModel_ACTV1.initial_carry
		def _patched_initc(self, batch):
			out = _orig_initc(self, batch)
			try:
				dev = next(self.parameters()).device
			except Exception:
				dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
			out.steps = out.steps.to(dev)
			out.halted = out.halted.to(dev)
			# Move inner carry tensors
			trm = importlib.import_module("models.recursive_reasoning.trm")
			out.inner_carry = trm.TinyRecursiveReasoningModel_ACTV1InnerCarry(
				z_H=out.inner_carry.z_H.to(dev),
				z_L=out.inner_carry.z_L.to(dev),
			)
			return out
		trm_mod.TinyRecursiveReasoningModel_ACTV1.initial_carry = _patched_initc


def _maybe_override_puzzle_ids(
	puzzle_ids: torch.Tensor,
	blank_identifier_id: int,
	num_puzzle_identifiers: int,
	mode: Literal["normal", "blank", "random"],
	rng: torch.Generator,
) -> torch.Tensor:
	if mode == "normal":
		return puzzle_ids

	ids = puzzle_ids.clone()
	if mode == "blank":
		ids[:] = blank_identifier_id
	elif mode == "random":
		ids[:] = torch.randint(
			low=0,
			high=num_puzzle_identifiers,
			size=ids.shape,
			generator=rng,
			device=ids.device,
		)
	else:
		raise ValueError(f"Unknown puzzle-id mode: {mode}")
	return ids


def _evaluate_with_pid_ablation(
	config,
	mode: Literal["normal", "blank", "random"],
	aggregated_voting: bool,
	rank: int,
	world_size: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
	"""
	Clone of pretrain.evaluate with a small hook to ablate puzzle_identifiers before inference.
	Returns (metrics, debug_info)
	"""
	# Dataloader
	eval_loader, eval_metadata = create_dataloader(
		config,
		split="test",
		test_set_mode=True,
		epochs_per_iter=1,
		global_batch_size=config.global_batch_size,
		rank=rank,
		world_size=world_size,
	)

	# Model
	model, _opts, _lrs = create_model(config, eval_metadata, rank=rank, world_size=world_size)
	train_state = TrainState(
		model=model,
		optimizers=[],
		optimizer_lrs=[],
		carry=None,
		step=0,
		total_steps=1,
	)
	train_state.model.eval()

	# Evaluator (ARC)
	if not len(config.evaluators):
		config.evaluators = [{"name": "arc@ARC", "aggregated_voting": aggregated_voting}]  # type: ignore[assignment]
	else:
		# inject aggregated_voting override if ARC is present
		for e in config.evaluators:
			if isinstance(e, dict) and isinstance(e.get("name"), str) and e["name"].startswith("arc@ARC"):
				e["aggregated_voting"] = aggregated_voting
	evaluators = _create_evaluators(config, eval_metadata)
	for ev in evaluators:
		ev.begin_eval()

	# Required keys from evaluators
	return_keys = set()
	for ev in evaluators:
		for k in getattr(ev, "required_outputs", []):
			return_keys.add(k)

	# RNG for random ablation
	device = "cuda" if torch.cuda.is_available() else "cpu"
	rng = torch.Generator(device=device)
	try:
		seed = int(os.environ.get("EXPC_PID_SEED", "0"))
	except Exception:
		seed = 0
	rng.manual_seed(seed)

	print(f"[ExpC] Starting evaluation | mode={mode} aggregated_voting={aggregated_voting}")
	batch_idx = 0
	with torch.inference_mode():
		for set_name, batch, _global_bs in eval_loader:
			# To device
			batch = {k: v.cuda() for k, v in batch.items()}
			# Override puzzle identifiers BEFORE initial_carry so embeddings use the ablated IDs
			batch["puzzle_identifiers"] = _maybe_override_puzzle_ids(
				batch["puzzle_identifiers"],
				eval_metadata.blank_identifier_id,
				eval_metadata.num_puzzle_identifiers,
				mode=mode,
				rng=rng,
			)
			# Initial carry
			carry = train_state.model.initial_carry(batch)  # type: ignore[attr-defined]

			# Roll until ACT halts
			while True:
				carry, _loss, _metrics, preds, all_finish = train_state.model(  # type: ignore[attr-defined]
					carry=carry, batch=batch, return_keys=return_keys
				)
				if all_finish:
					break

			for ev in evaluators:
				ev.update_batch(batch, preds)  # type: ignore[attr-defined]
			batch_idx += 1
			if batch_idx % 25 == 0:
				print(f"[ExpC] Processed {batch_idx} batches... (latest set={set_name})")

	# Finalize and collect metrics
	metrics = {}
	for ev in evaluators:
		res = ev.result(save_path=None, rank=0, world_size=1, group=None)  # type: ignore[attr-defined]
		if res is not None:
			metrics.update(res)

	debug = {
		"aggregated_voting": aggregated_voting,
		"puzzle_id_mode": mode,
	}
	print(f"[ExpC] Evaluation complete. Batches processed={batch_idx}")
	return metrics, debug


def _init_single_process_dist() -> None:
	# Initialize a single-process GLOO default group so ARC evaluator can call dist.gather_object
	if dist.is_available() and not dist.is_initialized():
		try:
			dist.init_process_group(backend="gloo", init_method="tcp://127.0.0.1:29500", rank=0, world_size=1)
		except Exception:
			# Retry a different port in case of collision
			dist.init_process_group(backend="gloo", init_method="tcp://127.0.0.1:29501", rank=0, world_size=1)


@hydra.main(config_path="../config", config_name="cfg_pretrain", version_base=None)
def main(hydra_config: DictConfig):
	# Single GPU / single process by default
	rank = 0
	world_size = 1

	# Compose config
	config = load_synced_config(hydra_config, rank=rank, world_size=world_size)

	# Robust checkpoint loader
	_patch_load_checkpoint()
	_patch_trm_device_alignment()
	_init_single_process_dist()

	# Required: load_checkpoint must be provided
	if not config.load_checkpoint:
		raise RuntimeError("load_checkpoint=<path> must be provided.")

	# Preflight logs
	_preflight(config, checkpoint_hint=config.load_checkpoint, data_paths_hint=config.data_paths)

	# Optional controls via environment variables (keeps Hydra cfg clean)
	pid_mode = os.environ.get("EXPC_PUZZLE_ID_MODE", "normal").lower()
	if pid_mode not in ("normal", "blank", "random"):
		raise SystemExit(f"Invalid EXPC_PUZZLE_ID_MODE={pid_mode}, expected one of normal|blank|random")
	aggregated_flag = os.environ.get("EXPC_AGGREGATED_VOTING", "1")
	aggregated_voting = aggregated_flag not in ("0", "false", "False")

	metrics, debug = _evaluate_with_pid_ablation(
		config=config,
		mode=pid_mode,  # type: ignore[arg-type]
		aggregated_voting=aggregated_voting,
		rank=rank,
		world_size=world_size,
	)

	# Print compact JSON
	result = {
		"checkpoint": config.load_checkpoint,
		"data_paths": config.data_paths,
		"ARC": {k: v for k, v in metrics.items() if k.startswith("ARC/")},
		**debug,
	}
	print(json.dumps(result, indent=2))
	try:
		if dist.is_initialized():
			dist.destroy_process_group()
	except Exception:
		pass


if __name__ == "__main__":
	main()


