import os
import json
from typing import Dict, Any, List

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
from pretrain import create_evaluators as _create_evaluators


def _patch_load_checkpoint() -> None:
	import importlib
	pt = importlib.import_module("pretrain")
	_orig_lc = pt.load_checkpoint

	def _patched_lc(model, config):
		if getattr(config, "load_checkpoint", None) is None:
			return _orig_lc(model, config)
		print(f"[ExpE] Patched load_checkpoint: {config.load_checkpoint}")
		sd = torch.load(config.load_checkpoint, map_location="cuda")
		if any(isinstance(k, str) and k.startswith("_orig_mod.") for k in sd.keys()):
			sd = { (k[10:] if isinstance(k, str) and k.startswith("_orig_mod.") else k): v for k, v in sd.items() }
		try:
			exp_shape = model.model.puzzle_emb.weights.shape  # type: ignore[attr-defined]
			key = "model.inner.puzzle_emb.weights"
			if key in sd and getattr(sd[key], "shape", None) != exp_shape:
				pe = sd[key]
				sd[key] = torch.mean(pe, dim=0, keepdim=True).expand(exp_shape).contiguous()
		except Exception:
			pass
		try:
			model.load_state_dict(sd, strict=False, assign=True)
		except TypeError:
			model.load_state_dict(sd, strict=False)

	pt.load_checkpoint = _patched_lc  # type: ignore[attr-defined]


def _patch_trm_device_alignment() -> None:
	import importlib
	try:
		trm_mod = importlib.import_module("models.recursive_reasoning.trm")
	except Exception:
		return
	# empty_carry on device
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
		# reset_carry on device
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
	# initial_carry: steps/halted + inner carry on device
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
			trm = importlib.import_module("models.recursive_reasoning.trm")
			out.inner_carry = trm.TinyRecursiveReasoningModel_ACTV1InnerCarry(
				z_H=out.inner_carry.z_H.to(dev),
				z_L=out.inner_carry.z_L.to(dev),
			)
			return out
		trm_mod.TinyRecursiveReasoningModel_ACTV1.initial_carry = _patched_initc


def _preflight(config, checkpoint_hint: str, data_paths_hint: list[str]) -> None:
	print("[ExpE] Preflight checklist:")
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


def _init_single_process_dist() -> None:
	if dist.is_available() and not dist.is_initialized():
		try:
			dist.init_process_group(backend="gloo", init_method="tcp://127.0.0.1:29500", rank=0, world_size=1)
		except Exception:
			dist.init_process_group(backend="gloo", init_method="tcp://127.0.0.1:29501", rank=0, world_size=1)


def _eval_at_step_t(config, t: int, rank: int, world_size: int) -> Dict[str, float]:
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

	# Evaluator (ARC) - per-step evaluation should not aggregate across multiple runs of this function,
	# but does aggregate votes across augmentations within this step (standard behavior).
	if not len(config.evaluators):
		config.evaluators = [{"name": "arc@ARC"}]  # type: ignore[assignment]
	evaluators = _create_evaluators(config, eval_metadata)
	for ev in evaluators:
		ev.begin_eval()

	# Required keys from evaluators
	return_keys = set()
	for ev in evaluators:
		for k in getattr(ev, "required_outputs", []):
			return_keys.add(k)

	print(f"[ExpE] Starting step {t}")
	with torch.inference_mode():
		for _set_name, batch, _global_bs in eval_loader:
			batch = {k: v.cuda() for k, v in batch.items()}
			carry = train_state.model.initial_carry(batch)  # type: ignore[attr-defined]

			# Roll exactly t steps (or until halting earlier)
			step_idx = 0
			preds = {}
			while True:
				carry, _loss, _metrics, preds, all_finish = train_state.model(  # type: ignore[attr-defined]
					carry=carry, batch=batch, return_keys=return_keys
				)
				step_idx += 1
				if all_finish or step_idx >= t:
					break

			for ev in evaluators:
				ev.update_batch(batch, preds)  # type: ignore[attr-defined]

	# Finalize
	metrics = {}
	for ev in evaluators:
		res = ev.result(save_path=None, rank=0, world_size=1, group=None)  # type: ignore[attr-defined]
		if res is not None:
			metrics.update(res)
	print(f"[ExpE] Finished step {t}")
	return metrics


@hydra.main(config_path="../config", config_name="cfg_pretrain", version_base=None)
def main(hydra_config: DictConfig):
	rank = 0
	world_size = 1

	config = load_synced_config(hydra_config, rank=rank, world_size=world_size)
	_patch_load_checkpoint()
	_patch_trm_device_alignment()
	_init_single_process_dist()

	if not config.load_checkpoint:
		raise RuntimeError("load_checkpoint=<path> must be provided.")

	_preflight(config, checkpoint_hint=config.load_checkpoint, data_paths_hint=config.data_paths)

	# Steps to evaluate: use arch.L_cycles unless overridden via env
	try:
		override_steps = int(os.environ.get("EXPE_MAX_STEPS", "0"))
	except Exception:
		override_steps = 0
	L_cycles = int(getattr(config.arch, "__pydantic_extra__", {}).get("L_cycles", 6))  # type: ignore[attr-defined]
	max_steps = override_steps if override_steps > 0 else L_cycles

	pass_at_1_by_step: List[float] = []
	for t in range(1, max_steps + 1):
		metrics = _eval_at_step_t(config, t=t, rank=rank, world_size=world_size)
		p1 = float(metrics.get("ARC/pass@1", 0.0))
		pass_at_1_by_step.append(p1)
		print(json.dumps({"step": t, "ARC/pass@1": p1}, indent=2))

	result = {
		"checkpoint": config.load_checkpoint,
		"data_paths": config.data_paths,
		"steps": max_steps,
		"pass_at_1_per_step": pass_at_1_by_step,
	}
	print(json.dumps(result, indent=2))
	try:
		if dist.is_initialized():
			dist.destroy_process_group()
	except Exception:
		pass


if __name__ == "__main__":
	main()


