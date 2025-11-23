from __future__ import annotations

import os
import subprocess
import json
from typing import Dict, Optional

from modal import App, Image, Volume, gpu

# Resolve local repo dir dynamically so you can run from inside TinyRecursiveModels
LOCAL_TRM_DIR = os.path.dirname(os.path.abspath(__file__))
REMOTE_TRM_DIR = "/workspace/TinyRecursiveModels"

APP_NAME = "trm-experiments"

# Persistent volumes for datasets, outputs, and checkpoints
DATA_VOL = Volume.from_name("trm-data", create_if_missing=True)
OUTPUTS_VOL = Volume.from_name("trm-outputs", create_if_missing=True)
CHECKPOINTS_VOL = Volume.from_name("trm-checkpoints", create_if_missing=True)

# Base image: official PyTorch CUDA runtime so torch+CUDA is preinstalled
# Then install the rest of the repo requirements.
image = (
	Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04")
		.apt_install("git", "build-essential", "python3", "python3-pip", "python3-dev", "python-is-python3")
		.env({
			"CUDA_HOME": "/usr/local/cuda",
			"PIP_INDEX_URL": "https://download.pytorch.org/whl/cu121",
			"PYTHONUNBUFFERED": "1",
		})
		# Install GPU PyTorch first (nn.Buffer available in 2.5+)
		.pip_install("torch", "torchvision", "torchaudio")
		# Then install repo requirements (will be mostly satisfied)
		.pip_install_from_requirements(f"{LOCAL_TRM_DIR}/requirements.txt")
		# Bake only the needed repo content to avoid copying .git/.venv and prevent change-during-build errors.
		.add_local_dir(os.path.join(LOCAL_TRM_DIR, "assets"), os.path.join(REMOTE_TRM_DIR, "assets"))
		.add_local_dir(os.path.join(LOCAL_TRM_DIR, "config"), os.path.join(REMOTE_TRM_DIR, "config"))
		.add_local_dir(os.path.join(LOCAL_TRM_DIR, "dataset"), os.path.join(REMOTE_TRM_DIR, "dataset"))
		.add_local_dir(os.path.join(LOCAL_TRM_DIR, "evaluators"), os.path.join(REMOTE_TRM_DIR, "evaluators"))
		.add_local_dir(os.path.join(LOCAL_TRM_DIR, "experiments"), os.path.join(REMOTE_TRM_DIR, "experiments"))
		.add_local_dir(os.path.join(LOCAL_TRM_DIR, "kaggle"), os.path.join(REMOTE_TRM_DIR, "kaggle"))
		.add_local_dir(os.path.join(LOCAL_TRM_DIR, "models"), os.path.join(REMOTE_TRM_DIR, "models"))
		.add_local_dir(os.path.join(LOCAL_TRM_DIR, "utils"), os.path.join(REMOTE_TRM_DIR, "utils"))
		.add_local_file(os.path.join(LOCAL_TRM_DIR, "pretrain.py"), os.path.join(REMOTE_TRM_DIR, "pretrain.py"))
		.add_local_file(os.path.join(LOCAL_TRM_DIR, "puzzle_dataset.py"), os.path.join(REMOTE_TRM_DIR, "puzzle_dataset.py"))
		.add_local_file(os.path.join(LOCAL_TRM_DIR, "README.md"), os.path.join(REMOTE_TRM_DIR, "README.md"))
)

app = App(APP_NAME)


def _run(
	cmd: list[str],
	env: Optional[Dict[str, str]] = None,
	cwd: Optional[str] = None,
) -> None:
	merged_env = os.environ.copy()
	if env:
		merged_env.update(env)
	subprocess.run(cmd, check=True, cwd=cwd, env=merged_env)


def _default_env() -> Dict[str, str]:
	# Ensure unbuffered logs and module resolution from repo root
	return {
		"PYTHONUNBUFFERED": "1",
		"PYTHONPATH": REMOTE_TRM_DIR,
		"CUDA_DEVICE_ORDER": "PCI_BUS_ID",
		"CUDA_VISIBLE_DEVICES": "0",
	}


@app.function(
	image=image,
	gpu="H100",
	volumes={
		f"{REMOTE_TRM_DIR}/data": DATA_VOL,
		f"{REMOTE_TRM_DIR}/outputs": OUTPUTS_VOL,
		f"{REMOTE_TRM_DIR}/checkpoints": CHECKPOINTS_VOL,
	},
	timeout=60 * 60 * 6,  # 6 hours
)
def prepare_arc_datasets() -> None:
	"""
	Builds ARC datasets for multiple augmentation levels and writes a manifest under data/.
	Outputs are persisted in the `trm-data` volume (mounted to TinyRecursiveModels/data).
	"""
	env = _default_env()
	# Run as a module to keep import paths correct
	_run(
		["python", "-m", "experiments.prepare_arc_datasets"],
		env=env,
		cwd=REMOTE_TRM_DIR,
	)

@app.function(
	image=image,
	volumes={
		f"{REMOTE_TRM_DIR}/data": DATA_VOL,
	},
	timeout=60 * 60,  # 1 hour
)
def build_arc1() -> None:
	"""
	Build ARC-AGI-1 datasets that match HF v1 checkpoints:
	- subsets: training, evaluation, concept
	- test_set_name: evaluation
	Generates data/arc-aug-1000 and data/arc-aug-0
	"""
	env = _default_env()
	os.makedirs(os.path.join(REMOTE_TRM_DIR, "data"), exist_ok=True)
	# 1000× aug
	_run(
		[
			"python",
			"-m",
			"dataset.build_arc_dataset",
			"--input-file-prefix",
			"kaggle/combined/arc-agi",
			"--output-dir",
			"data/arc-aug-1000",
			"--subsets",
			"training",
			"evaluation",
			"concept",
			"--test-set-name",
			"evaluation",
			"--num-aug",
			"1000",
		],
		env=env,
		cwd=REMOTE_TRM_DIR,
	)
	# 0× aug
	_run(
		[
			"python",
			"-m",
			"dataset.build_arc_dataset",
			"--input-file-prefix",
			"kaggle/combined/arc-agi",
			"--output-dir",
			"data/arc-aug-0",
			"--subsets",
			"training",
			"evaluation",
			"concept",
			"--test-set-name",
			"evaluation",
			"--num-aug",
			"0",
		],
		env=env,
		cwd=REMOTE_TRM_DIR,
	)
	# Write manifest for convenience
	manifest = {"paper_mode": "data/arc-aug-1000", "single_aug": "data/arc-aug-0"}
	with open(os.path.join(REMOTE_TRM_DIR, "data", "arc_exp1_manifest.json"), "w") as f:
		json.dump(manifest, f, indent=2)
	print("Built ARC-1 datasets:", json.dumps(manifest, indent=2))


@app.function(
	image=image,
	volumes={
		f"{REMOTE_TRM_DIR}/checkpoints": CHECKPOINTS_VOL,
	},
	timeout=60 * 30,  # 30 minutes
)
def fetch_hf_checkpoints(repo_id: str = "arcprize/trm_arc_prize_verification", dest: str = "checkpoints/hf_trm") -> None:
	"""
	Fetch TRM checkpoints from Hugging Face into the checkpoints volume.
	Prints candidate files and their sha256 digests.
	"""
	env = _default_env()
	_run(
		["python", "-m", "experiments.fetch_hf_checkpoints", "--repo_id", repo_id, "--dest", dest],
		env=env,
		cwd=REMOTE_TRM_DIR,
	)


@app.function(
	image=image,
	gpu="H100",
	volumes={
		f"{REMOTE_TRM_DIR}/data": DATA_VOL,
		f"{REMOTE_TRM_DIR}/outputs": OUTPUTS_VOL,
		f"{REMOTE_TRM_DIR}/checkpoints": CHECKPOINTS_VOL,
	},
	timeout=60 * 60 * 12,  # 12 hours
)
def exp1_secret_sauce(
	load_checkpoint: str,
	paper_aug_level: int = 1000,
	single_aug_level: int = 0,
	arch: str = "trm",
	seed: int = 42,
) -> None:
	"""
	Run Experiment 1 (Secret Sauce): compare 1000× voting vs single-augmentation.

	Args:
		load_checkpoint: Absolute or volume path to the trained checkpoint file/dir to load.
		                  Example: '/workspace/TinyRecursiveModels/checkpoints/<your_run>/step_XXXX'
		paper_aug_level: Augmentation level for paper mode (default 1000).
		single_aug_level: Augmentation level for single-aug mode (default 0).
		arch: Model architecture key from configs (e.g., 'trm', 'hrm', etc.).
		seed: RNG seed used by Hydra config.
	"""
	env = _default_env()

	# Ensure paths exist for outputs/checkpoints (persisted volumes)
	os.makedirs(f"{REMOTE_TRM_DIR}/outputs", exist_ok=True)
	os.makedirs(f"{REMOTE_TRM_DIR}/checkpoints/exp1", exist_ok=True)

	# Hydra overrides: point data and outputs inside mounted volumes
	hydra_overrides = [
		f"arch={arch}",
		"arch.L_cycles=4",
		"arch.H_cycles=3",
		"arch.L_layers=2",
		f"seed={seed}",
		f"+load_checkpoint={load_checkpoint}",
		f"+checkpoint_path=checkpoints/exp1",
		f"data_paths=['data/arc-aug-{paper_aug_level}']",
		f"data_paths_test=['data/arc-aug-{paper_aug_level}']",
		"hydra.run.dir=outputs/exp1/${now:%Y-%m-%d}/${now:%H-%M-%S}",
	]

	# Initialize a single-process default process group for evaluators that call dist.gather_object
	# without changing original repo code. We inject via a small -c wrapper and forward Hydra args.
	pycode = (
		"import sys, runpy, json, numpy as _np\n"
		"import torch.distributed as dist\n"
		"import importlib\n"
		"# Init 1-process default group for evaluators using dist.gather_object\n"
		"dist.init_process_group(backend='gloo', init_method='tcp://127.0.0.1:29500', rank=0, world_size=1)\n"
		"# Ensure test dataloader iterates in-process so iterator patch runs\n"
		"pt = importlib.import_module('pretrain')\n"
		"_orig_cd = pt.create_dataloader\n"
		"from torch.utils.data import DataLoader\n"
		"def _patched_cd(config, split, *args, **kwargs):\n"
		"    import os as _os\n"
		"    if split != 'test' or _os.environ.get('TRM_SINGLE_AUG_ONCE') != '1':\n"
		"        return _orig_cd(config, split, *args, **kwargs)\n"
		"    import puzzle_dataset as pz\n"
		"    rank = kwargs.get('rank', 0)\n"
		"    world_size = kwargs.get('world_size', 1)\n"
		"    test_set_mode = kwargs.get('test_set_mode', True)\n"
		"    epochs_per_iter = kwargs.get('epochs_per_iter', 1)\n"
		"    global_batch_size = kwargs.get('global_batch_size', config.global_batch_size)\n"
		"    ds = pz.PuzzleDataset(pz.PuzzleDatasetConfig(\n"
		"        seed=config.seed,\n"
		"        dataset_paths=config.data_paths_test if len(config.data_paths_test)>0 and split=='test' else config.data_paths,\n"
		"        rank=rank,\n"
		"        num_replicas=world_size,\n"
		"        test_set_mode=test_set_mode,\n"
		"        epochs_per_iter=epochs_per_iter,\n"
		"        global_batch_size=global_batch_size,\n"
		"    ), split=split)\n"
		"    dl = DataLoader(ds, batch_size=None, num_workers=0, pin_memory=False)\n"
		"    return dl, ds.metadata\n"
		"pt.create_dataloader = _patched_cd\n"
		"# Patch single-aug to use same dataset (arc-aug-1000) and feed 1 example per puzzle\n"
		"exp = importlib.import_module('experiments.run_exp1')\n"
		"_orig_rse = exp._run_single_eval\n"
		"def _patched_rse(**kw):\n"
		"    import os\n"
		"    label = kw.get('label','')\n"
		"    if 'single_aug' in label:\n"
		"        os.environ['TRM_SINGLE_AUG_ONCE']='1'\n"
		"        cfg = kw.get('config')\n"
		"        try:\n"
		"            # Disable aggregated voting for single-aug to reflect 1x setting\n"
		"            if not len(cfg.evaluators):\n"
		"                cfg.evaluators = [{'name':'arc@ARC','aggregated_voting': False, 'save_per_example': True}]\n"
		"            else:\n"
		"                for e in cfg.evaluators:\n"
		"                    if isinstance(e, dict) and isinstance(e.get('name'), str) and e['name'].startswith('arc@ARC'):\n"
		"                        e['aggregated_voting'] = False\n"
		"                        e['save_per_example'] = True\n"
		"        except Exception:\n"
		"            pass\n"
		"    else:\n"
		"        os.environ.pop('TRM_SINGLE_AUG_ONCE', None)\n"
		"    return _orig_rse(**kw)\n"
		"exp._run_single_eval = _patched_rse\n"
		"# Monkey-patch test iterator to emit only 1 example per puzzle when TRM_SINGLE_AUG_ONCE=1\n"
		"import numpy as np\n"
		"pd = importlib.import_module('puzzle_dataset')\n"
		"_orig_it = pd.PuzzleDataset._iter_test\n"
		"def _iter_test_once(self):\n"
		"    import os\n"
		"    if os.environ.get('TRM_SINGLE_AUG_ONCE')!='1':\n"
		"        yield from _orig_it(self)\n"
		"        return\n"
		"    # Dense single-aug batching: collect the examples of the first (original) puzzle of each group\n"
		"    all_indices = []\n"
		"    for set_name, dataset in self._data.items():\n"
		"        group_indices = dataset['group_indices']\n"
		"        puzzle_indices = dataset['puzzle_indices']\n"
		"        for i in range(len(group_indices) - 1):\n"
		"            first_puzzle_id = group_indices[i]\n"
		"            if first_puzzle_id < len(puzzle_indices) - 1:\n"
		"                start_ex = puzzle_indices[first_puzzle_id]\n"
		"                end_ex = puzzle_indices[first_puzzle_id + 1]\n"
		"                if end_ex > start_ex:\n"
		"                    all_indices.extend(range(start_ex, end_ex))\n"
		"    if not all_indices:\n"
		"        return\n"
		"    all_indices = _np.array(all_indices, dtype=_np.int64)\n"
		"    batch_size = self.config.global_batch_size\n"
		"    for start in range(0, len(all_indices), batch_size):\n"
		"        end = min(start + batch_size, len(all_indices))\n"
		"        idx = all_indices[start:end]\n"
		"        for set_name, dataset in self._data.items():\n"
		"            if _np.max(idx) < len(dataset['inputs']):\n"
		"                p_idx = _np.searchsorted(dataset['puzzle_indices'], idx, side='right') - 1\n"
		"                batch = self._collate_batch({\n"
		"                    'inputs': dataset['inputs'][idx],\n"
		"                    'labels': dataset['labels'][idx],\n"
		"                    'puzzle_identifiers': dataset['puzzle_identifiers'][p_idx]\n"
		"                })\n"
		"                yield set_name, batch, len(idx)\n"
		"                break\n"
		"pd.PuzzleDataset._iter_test = _iter_test_once\n"
		"# Monkey-patch json to handle numpy/torch types without changing repo code\n"
		"_orig_dump=json.dump\n"
		"_orig_dumps=json.dumps\n"
		"def _safe(o):\n"
		"    try:\n"
		"        import torch as _t\n"
		"    except Exception:\n"
		"        _t=None\n"
		"    if isinstance(o, dict):\n"
		"        return {k:_safe(v) for k,v in o.items()}\n"
		"    if isinstance(o, (list, tuple)):\n"
		"        return [_safe(v) for v in o]\n"
		"    if isinstance(o, (_np.floating,)):\n"
		"        return float(o)\n"
		"    if isinstance(o, (_np.integer,)):\n"
		"        return int(o)\n"
		"    if isinstance(o, _np.ndarray):\n"
		"        return o.tolist()\n"
		"    if _t is not None and isinstance(o, _t.Tensor):\n"
		"        return o.detach().cpu().tolist()\n"
		"    return o\n"
		"json.dump=lambda o, fp, **kw: _orig_dump(_safe(o), fp, **kw)\n"
		"json.dumps=lambda o, **kw: _orig_dumps(_safe(o), **kw)\n"
		"# Forward to the module in the SAME interpreter so patches remain active\n"
		f"sys.argv=['experiments.run_exp1'] + {repr(hydra_overrides)}\n"
		"import experiments.run_exp1\n"
		"experiments.run_exp1.main()\n"
	)
	cmd = ["python", "-c", pycode]
	_run(cmd, env=env, cwd=REMOTE_TRM_DIR)

@app.function(
	image=image,
	gpu="H100",
	volumes={
		f"{REMOTE_TRM_DIR}/data": DATA_VOL,
		f"{REMOTE_TRM_DIR}/outputs": OUTPUTS_VOL,
		f"{REMOTE_TRM_DIR}/checkpoints": CHECKPOINTS_VOL,
	},
	timeout=60 * 60 * 12,  # 12 hours
)
def exp1_paper_only(
	load_checkpoint: str,
	paper_aug_level: int = 1000,
	arch: str = "trm",
	seed: int = 42,
) -> None:
	"""
	Run ONLY paper mode (1000× voting). Single-aug is skipped.
	"""
	env = _default_env()
	os.makedirs(f"{REMOTE_TRM_DIR}/outputs", exist_ok=True)
	os.makedirs(f"{REMOTE_TRM_DIR}/checkpoints/exp1", exist_ok=True)

	hydra_overrides = [
		f"arch={arch}",
		"arch.L_cycles=4",
		"arch.H_cycles=3",
		"arch.L_layers=2",
		f"seed={seed}",
		f"+load_checkpoint={load_checkpoint}",
		f"+checkpoint_path=checkpoints/exp1",
		f"data_paths=['data/arc-aug-{paper_aug_level}']",
		f"data_paths_test=['data/arc-aug-{paper_aug_level}']",
		"hydra.run.dir=outputs/exp1/${now:%Y-%m-%d}/${now:%H-%M-%S}",
	]

	pycode = (
		"import sys, importlib\n"
		"import torch.distributed as dist\n"
		"dist.init_process_group(backend='gloo', init_method='tcp://127.0.0.1:29500', rank=0, world_size=1)\n"
		"exp = importlib.import_module('experiments.run_exp1')\n"
		"_orig_rse = exp._run_single_eval\n"
		"def _paper_only_rse(**kw):\n"
		"    label = kw.get('label','')\n"
		"    if 'single_aug' in label:\n"
		"        return {'label': label, 'metrics': {}, 'skipped': True}\n"
		"    return _orig_rse(**kw)\n"
		"exp._run_single_eval = _paper_only_rse\n"
		f"sys.argv=['experiments.run_exp1'] + {repr(hydra_overrides)}\n"
		"import experiments.run_exp1\n"
		"experiments.run_exp1.main()\n"
	)
	cmd = ["python", "-c", pycode]
	_run(cmd, env=env, cwd=REMOTE_TRM_DIR)

@app.function(
	image=image,
	gpu="H100",
	volumes={
		f"{REMOTE_TRM_DIR}/data": DATA_VOL,
		f"{REMOTE_TRM_DIR}/outputs": OUTPUTS_VOL,
		f"{REMOTE_TRM_DIR}/checkpoints": CHECKPOINTS_VOL,
	},
	timeout=60 * 60 * 12,  # 12 hours
)
def exp1_single_only(
	load_checkpoint: str,
	single_aug_level: int = 0,
	arch: str = "trm",
	seed: int = 42,
) -> None:
	"""
	Run ONLY single-augmentation mode (1×, no voting) using the 1000× dataset with iterator filtering.
	"""
	env = _default_env()
	os.makedirs(f"{REMOTE_TRM_DIR}/outputs", exist_ok=True)
	os.makedirs(f"{REMOTE_TRM_DIR}/checkpoints/exp1", exist_ok=True)

	hydra_overrides = [
		f"arch={arch}",
		"arch.L_cycles=4",
		"arch.H_cycles=3",
		"arch.L_layers=2",
		f"seed={seed}",
		f"+load_checkpoint={load_checkpoint}",
		f"+checkpoint_path=checkpoints/exp1",
		"data_paths=['data/arc-aug-1000']",
		"data_paths_test=['data/arc-aug-1000']",
		"hydra.run.dir=outputs/exp1/${now:%Y-%m-%d}/${now:%H-%M-%S}",
	]

	pycode = (
		"import sys, importlib, numpy as _np\n"
		"import torch.distributed as dist\n"
		"dist.init_process_group(backend='gloo', init_method='tcp://127.0.0.1:29500', rank=0, world_size=1)\n"
		"pt = importlib.import_module('pretrain')\n"
		"_orig_cd = pt.create_dataloader\n"
		"from torch.utils.data import DataLoader\n"
		"def _patched_cd(config, split, *args, **kwargs):\n"
		"    import os as _os\n"
		"    if split != 'test' or _os.environ.get('TRM_SINGLE_AUG_ONCE') != '1':\n"
		"        return _orig_cd(config, split, *args, **kwargs)\n"
		"    import puzzle_dataset as pz\n"
		"    rank = kwargs.get('rank', 0)\n"
		"    world_size = kwargs.get('world_size', 1)\n"
		"    test_set_mode = kwargs.get('test_set_mode', True)\n"
		"    epochs_per_iter = kwargs.get('epochs_per_iter', 1)\n"
		"    global_batch_size = kwargs.get('global_batch_size', config.global_batch_size)\n"
		"    ds = pz.PuzzleDataset(pz.PuzzleDatasetConfig(\n"
		"        seed=config.seed,\n"
		"        dataset_paths=config.data_paths_test if len(config.data_paths_test)>0 and split=='test' else config.data_paths,\n"
		"        rank=rank,\n"
		"        num_replicas=world_size,\n"
	"        test_set_mode=test_set_mode,\n"
		"        epochs_per_iter=epochs_per_iter,\n"
		"        global_batch_size=global_batch_size,\n"
		"    ), split=split)\n"
		"    dl = DataLoader(ds, batch_size=None, num_workers=0, pin_memory=False)\n"
		"    return dl, ds.metadata\n"
		"pt.create_dataloader = _patched_cd\n"
		"pd = importlib.import_module('puzzle_dataset')\n"
		"_orig_it = pd.PuzzleDataset._iter_test\n"
		"def _iter_test_once(self):\n"
		"    import os\n"
		"    if os.environ.get('TRM_SINGLE_AUG_ONCE')!='1':\n"
		"        yield from _orig_it(self)\n"
		"        return\n"
		"    # Build a dense list of indices for the first (original) puzzle in each group\n"
		"    all_indices = []\n"
		"    set_names = []\n"
		"    for set_name, dataset in self._data.items():\n"
		"        group_indices = dataset['group_indices']\n"
		"        puzzle_indices = dataset['puzzle_indices']\n"
		"        for i in range(len(group_indices) - 1):\n"
		"            first_puzzle_id = group_indices[i]\n"
		"            if first_puzzle_id < len(puzzle_indices) - 1:\n"
		"                start_ex = puzzle_indices[first_puzzle_id]\n"
		"                end_ex = puzzle_indices[first_puzzle_id + 1]\n"
		"                if end_ex > start_ex:\n"
		"                    # keep all examples of that single augmentation (usually 1-2 test examples)\n"
		"                    all_indices.extend(range(start_ex, end_ex))\n"
		"                    set_names.append(set_name)\n"
		"    # Pack into dense batches\n"
		"    if not all_indices:\n"
		"        return\n"
		"    all_indices = _np.array(all_indices, dtype=_np.int64)\n"
		"    batch_size = self.config.global_batch_size\n"
		"    for start in range(0, len(all_indices), batch_size):\n"
		"        end = min(start + batch_size, len(all_indices))\n"
		"        idx = all_indices[start:end]\n"
		"        # Use the first set (ARC test is typically a single set named 'all')\n"
		"        # If multiple sets existed, a more elaborate mapping would be needed.\n"
		"        for set_name, dataset in self._data.items():\n"
		"            if _np.max(idx) < len(dataset['inputs']):\n"
		"                p_idx = _np.searchsorted(dataset['puzzle_indices'], idx, side='right') - 1\n"
		"                batch = self._collate_batch({\n"
		"                    'inputs': dataset['inputs'][idx],\n"
		"                    'labels': dataset['labels'][idx],\n"
		"                    'puzzle_identifiers': dataset['puzzle_identifiers'][p_idx]\n"
		"                })\n"
		"                yield set_name, batch, len(idx)\n"
		"                break\n"
		"pd.PuzzleDataset._iter_test = _iter_test_once\n"
		"exp = importlib.import_module('experiments.run_exp1')\n"
		"_orig_rse = exp._run_single_eval\n"
		"def _single_only_rse(**kw):\n"
		"    import os\n"
		"    label = kw.get('label','')\n"
		"    cfg = kw.get('config')\n"
		"    if 'paper_mode' in label:\n"
		"        return {'label': label, 'metrics': {}, 'skipped': True}\n"
		"    os.environ['TRM_SINGLE_AUG_ONCE']='1'\n"
		"    try:\n"
		"        if not len(cfg.evaluators):\n"
		"            cfg.evaluators = [{'name':'arc@ARC','aggregated_voting': False, 'save_per_example': False}]\n"
		"        else:\n"
		"            for e in cfg.evaluators:\n"
		"                if isinstance(e, dict) and isinstance(e.get('name'), str) and e['name'].startswith('arc@ARC'):\n"
		"                    e['aggregated_voting'] = False\n"
		"                    e['save_per_example'] = False\n"
		"    except Exception:\n"
		"        pass\n"
		"    return _orig_rse(**kw)\n"
		"exp._run_single_eval = _single_only_rse\n"
		f"sys.argv=['experiments.run_exp1'] + {repr(hydra_overrides)}\n"
		"import experiments.run_exp1\n"
		"experiments.run_exp1.main()\n"
	)
	cmd = ["python", "-c", pycode]
	_run(cmd, env=env, cwd=REMOTE_TRM_DIR)

@app.function(
	image=image,
	volumes={
		f"{REMOTE_TRM_DIR}/data": DATA_VOL,
	},
	timeout=60 * 5,  # 5 minutes
)
def quick_check_data(data_dir: str = "data/arc-aug-1000") -> None:
	"""
	Run a zero-GPU dataset sanity check on Modal to verify:
	- augmentations_per_original ≈ 1000 for arc-aug-1000
	- single_aug_examples_per_original ≈ 1 if we keep only the first variant per group

	Usage example:
	- modal run modal_trm.py::main --action check_data --data-dir data/arc-aug-1000
	"""
	env = _default_env()
	_run(
		["python", "-m", "experiments.quick_checks", "--data_dir", data_dir],
		env=env,
		cwd=REMOTE_TRM_DIR,
	)

@app.function(
	image=image,
	gpu="H100",
	volumes={
		f"{REMOTE_TRM_DIR}/data": DATA_VOL,
		f"{REMOTE_TRM_DIR}/outputs": OUTPUTS_VOL,
		f"{REMOTE_TRM_DIR}/checkpoints": CHECKPOINTS_VOL,
	},
	timeout=60 * 20,  # 20 minutes
)
def quick_exp1_infer(
	load_checkpoint: str,
	max_batches: int = 5,
	arch: str = "trm",
	seed: int = 42,
) -> None:
	"""
	Fast GPU sanity check over a tiny slice of data to validate Exp1 wiring:
	- Paper mode: 1000× dataset with voting
	- Single-aug mode: same dataset, 1× filtered, no voting
	Runs only `max_batches` per mode and avoids saving large outputs.
	"""
	env = _default_env()
	os.makedirs(f"{REMOTE_TRM_DIR}/outputs", exist_ok=True)
	os.makedirs(f"{REMOTE_TRM_DIR}/checkpoints/quick", exist_ok=True)

	# Smaller batch to reduce GPU memory/time
	hydra_overrides = [
		f"arch={arch}",
		"arch.L_cycles=4",
		"arch.H_cycles=3",
		"arch.L_layers=2",
		f"seed={seed}",
		f"+load_checkpoint={load_checkpoint}",
		f"+checkpoint_path=checkpoints/quick",
		"global_batch_size=128",
		"+eval_save_outputs=[]",
		"hydra.run.dir=outputs/quick/${now:%Y-%m-%d}/${now:%H-%M-%S}",
		# Force both modes to use the 1000× dataset
		"data_paths=['data/arc-aug-1000']",
		"data_paths_test=['data/arc-aug-1000']",
	]

	pycode = (
		"import sys, runpy, json, numpy as _np, os\n"
		"import torch.distributed as dist\n"
		"import importlib\n"
		"dist.init_process_group(backend='gloo', init_method='tcp://127.0.0.1:29500', rank=0, world_size=1)\n"
		"# Limit dataloader to N batches for speed\n"
		"MAXBATCH=" + str(max_batches) + "\n"
		"pt = importlib.import_module('pretrain')\n"
		"_orig_cd = pt.create_dataloader\n"
		"def _limited_cd(config, split, *args, **kwargs):\n"
		"    import itertools\n"
		"    dl, meta = _orig_cd(config, split, *args, **kwargs)\n"
		"    class _LimitedLoader:\n"
		"        def __init__(self, base):\n"
		"            self._base = base\n"
		"            self.dataset = getattr(base, 'dataset', None)\n"
		"        def __iter__(self):\n"
		"            return itertools.islice(iter(self._base), MAXBATCH)\n"
		"    return _LimitedLoader(dl), meta\n"
		"pt.create_dataloader = _limited_cd\n"
		"# Single-aug: disable voting and enable 1× iterator\n"
		"exp = importlib.import_module('experiments.run_exp1')\n"
		"_orig_rse = exp._run_single_eval\n"
		"def _patched_rse(**kw):\n"
		"    label = kw.get('label','')\n"
		"    cfg = kw.get('config')\n"
		"    if 'single_aug' in label:\n"
		"        os.environ['TRM_SINGLE_AUG_ONCE']='1'\n"
		"        try:\n"
		"            if not len(cfg.evaluators):\n"
		"                cfg.evaluators = [{'name':'arc@ARC','aggregated_voting': False, 'save_per_example': False}]\n"
		"            else:\n"
		"                for e in cfg.evaluators:\n"
		"                    if isinstance(e, dict) and isinstance(e.get('name'), str) and e['name'].startswith('arc@ARC'):\n"
		"                        e['aggregated_voting'] = False\n"
		"                        e['save_per_example'] = False\n"
		"        except Exception:\n"
		"            pass\n"
		"    else:\n"
		"        # Ensure voting enabled in paper mode\n"
		"        try:\n"
		"            if not len(cfg.evaluators):\n"
		"                cfg.evaluators = [{'name':'arc@ARC','aggregated_voting': True}]\n"
		"            else:\n"
		"                for e in cfg.evaluators:\n"
		"                    if isinstance(e, dict) and isinstance(e.get('name'), str) and e['name'].startswith('arc@ARC'):\n"
		"                        e.setdefault('aggregated_voting', True)\n"
		"        except Exception:\n"
		"            pass\n"
		"        os.environ.pop('TRM_SINGLE_AUG_ONCE', None)\n"
		"    # Run original eval\n"
		"    result = _orig_rse(**kw)\n"
		"    # Validation printouts\n"
		"    try:\n"
		"        aps = result.get('augmentations_per_original')\n"
		"        data_paths = kw.get('data_paths', cfg.data_paths)\n"
		"        aggregated = None\n"
		"        for e in cfg.evaluators:\n"
		"            if isinstance(e, dict) and isinstance(e.get('name'), str) and e['name'].startswith('arc@ARC'):\n"
		"                aggregated = e.get('aggregated_voting', True)\n"
		"                break\n"
		"        mode = 'single_aug' if 'single_aug' in label else 'paper_mode'\n"
		"        print(f\"[VALIDATION] {mode}: data_paths={data_paths}, aggregated_voting={aggregated}, augmentations_per_original={aps}\")\n"
		"        if mode=='paper_mode':\n"
		"            ok = (aps is not None and aps >= 10) and (aggregated is True)\n"
		"            print(f\"[VALIDATION] paper_mode PASS={ok}\")\n"
		"        else:\n"
		"            ok = (aps is not None and aps <= 1.1) and (aggregated is False)\n"
		"            print(f\"[VALIDATION] single_aug PASS={ok}\")\n"
		"    except Exception as _e:\n"
		"        print(f\"[VALIDATION] error during validation: {_e}\")\n"
		"    return result\n"
		"exp._run_single_eval = _patched_rse\n"
		"# Iterator patch for single-aug (first variant per group)\n"
		"pd = importlib.import_module('puzzle_dataset')\n"
		"_orig_it = pd.PuzzleDataset._iter_test\n"
		"def _iter_test_once(self):\n"
		"    if os.environ.get('TRM_SINGLE_AUG_ONCE')!='1':\n"
		"        yield from _orig_it(self)\n"
		"        return\n"
		"    for set_i, (set_name, dataset) in enumerate(self._data.items()):\n"
		"        group_indices = dataset['group_indices']\n"
		"        puzzle_indices = dataset['puzzle_indices']\n"
		"        keep_indices = []\n"
		"        for i in range(len(group_indices) - 1):\n"
		"            first_puzzle_id = group_indices[i]\n"
		"            if first_puzzle_id < len(puzzle_indices) - 1:\n"
		"                start_ex = puzzle_indices[first_puzzle_id]\n"
		"                end_ex = puzzle_indices[first_puzzle_id + 1]\n"
		"                if end_ex > start_ex:\n"
		"                    keep_indices.extend(range(start_ex, end_ex))\n"
		"        keep_set = set(keep_indices)\n"
		"        total_examples = len(dataset['inputs'])\n"
		"        start_index = 0\n"
		"        while start_index < total_examples:\n"
		"            end_index = min(total_examples, start_index + self.config.global_batch_size)\n"
		"            local_start = start_index + self.config.rank * self.local_batch_size\n"
		"            local_end   = min(start_index + (self.config.rank + 1) * self.local_batch_size, end_index)\n"
		"            batch_indices = _np.arange(local_start, local_end)\n"
		"            valid_mask = [idx in keep_set for idx in batch_indices]\n"
		"            valid_indices = batch_indices[valid_mask]\n"
		"            if len(valid_indices) > 0:\n"
		"                p_idx = _np.searchsorted(dataset['puzzle_indices'], valid_indices, side='right') - 1\n"
		"                batch = self._collate_batch({\n"
		"                    'inputs': dataset['inputs'][valid_indices],\n"
		"                    'labels': dataset['labels'][valid_indices],\n"
		"                    'puzzle_identifiers': dataset['puzzle_identifiers'][p_idx]\n"
		"                })\n"
		"                yield set_name, batch, len(valid_indices)\n"
		"            start_index += self.config.global_batch_size\n"
		"pd.PuzzleDataset._iter_test = _iter_test_once\n"
		"# Run exp1 module with overrides\n"
		"_safe_dump=json.dump; _safe_dumps=json.dumps\n"
		"def _safe(o):\n"
		"    try:\n"
		"        import numpy as _n\n"
		"        import torch as _t\n"
		"    except Exception:\n"
		"        _n=_t=None\n"
		"    if isinstance(o, dict):\n"
		"        return {k:_safe(v) for k,v in o.items()}\n"
		"    if isinstance(o, (list, tuple)):\n"
		"        return [_safe(v) for v in o]\n"
		"    try:\n"
		"        import numpy as _np2\n"
		"        import torch as _t2\n"
		"        from numbers import Number\n"
		"        if hasattr(_np2, 'floating') and isinstance(o, _np2.floating): return float(o)\n"
		"        if hasattr(_np2, 'integer') and isinstance(o, _np2.integer): return int(o)\n"
		"        if hasattr(_np2, 'ndarray') and isinstance(o, _np2.ndarray): return o.tolist()\n"
		"        if hasattr(_t2, 'Tensor') and isinstance(o, _t2.Tensor): return o.detach().cpu().tolist()\n"
		"        if isinstance(o, Number): return o\n"
		"    except Exception:\n"
		"        pass\n"
		"    return o\n"
		"json.dump=lambda o, fp, **kw: _safe_dump(_safe(o), fp, **kw)\n"
		"json.dumps=lambda o, **kw: _safe_dumps(_safe(o), **kw)\n"
		"sys.argv=['run_exp1'] + " + repr(hydra_overrides) + "\n"
		"mod = importlib.import_module('experiments.run_exp1')\n"
		"mod.main()\n"
	)
	cmd = ["python", "-c", pycode]
	_run(cmd, env=env, cwd=REMOTE_TRM_DIR)

@app.function(
	image=image,
	gpu="H100",
	volumes={
		f"{REMOTE_TRM_DIR}/data": DATA_VOL,
		f"{REMOTE_TRM_DIR}/outputs": OUTPUTS_VOL,
		f"{REMOTE_TRM_DIR}/checkpoints": CHECKPOINTS_VOL,
	},
	timeout=60 * 60,  # 1 hour
)
def exp6_efficiency(
	load_checkpoint: str,
	arch: str = "trm",
	seed: int = 42,
	batch_size: int = 32,
) -> None:
	"""
	Experiment 6: Efficiency benchmarking.
	Measures:
	- throughput (samples/sec)
	- latency (ms/sample and ms/batch)
	- peak GPU memory (GB)
	"""
	env = _default_env()
	# Avoid torch.compile/TorchDynamo issues during benchmarking.
	# The repo guards compile with DISABLE_COMPILE; TorchDynamo can also be disabled explicitly.
	env["DISABLE_COMPILE"] = "1"
	env["TORCHDYNAMO_DISABLE"] = "1"
	os.makedirs(f"{REMOTE_TRM_DIR}/outputs", exist_ok=True)

	# Use 1000x dataset to avoid iterator resets across short runs; disable saving to avoid I/O skew.
	hydra_overrides = [
		f"arch={arch}",
		"arch.L_cycles=4",
		"arch.H_cycles=3",
		"arch.L_layers=2",
		f"seed={seed}",
		f"+load_checkpoint={load_checkpoint}",
		f"global_batch_size={batch_size}",
		"data_paths=['data/arc-aug-1000']",
		"data_paths_test=['data/arc-aug-1000']",
		"+eval_save_outputs=[]",
		"hydra.run.dir=outputs/exp6_bench/${now:%Y-%m-%d}/${now:%H-%M-%S}",
	]

	pycode = (
		"import sys, importlib, time, torch, json\n"
		"import numpy as np\n"
		"import torch.distributed as dist\n"
		"# Import pretrain as a module so we can monkey-patch its load_checkpoint\n"
		"pt = importlib.import_module('pretrain')\n"
		"dist.init_process_group(backend='gloo', init_method='tcp://127.0.0.1:29500', rank=0, world_size=1)\n"
		f"sys.argv=['exp6'] + {repr(hydra_overrides)}\n"
		"import hydra\n"
		"@hydra.main(config_path='config', config_name='cfg_pretrain', version_base=None)\n"
		"def run_bench(cfg):\n"
		"    rank = 0; world_size = 1\n"
		"    torch.cuda.set_device(0)\n"
		"    # Patch checkpoint loader: strip _orig_mod. prefix and load non-strict\n"
		"    _orig_lc = pt.load_checkpoint\n"
		"    def _patched_lc(model, config):\n"
		"        if getattr(config, 'load_checkpoint', None) is None:\n"
		"            return _orig_lc(model, config)\n"
		"        print(f\"[Exp6] Patched load_checkpoint: {config.load_checkpoint}\")\n"
		"        sd = torch.load(config.load_checkpoint, map_location='cuda')\n"
		"        if any(k.startswith('_orig_mod.') for k in sd.keys()):\n"
		"            sd = { (k[10:] if k.startswith('_orig_mod.') else k): v for k,v in sd.items() }\n"
		"        try:\n"
		"            exp_shape = model.model.puzzle_emb.weights.shape  # type: ignore\n"
		"            key = 'model.inner.puzzle_emb.weights'\n"
		"            if key in sd and sd[key].shape != exp_shape:\n"
		"                pe = sd[key]\n"
		"                sd[key] = torch.mean(pe, dim=0, keepdim=True).expand(exp_shape).contiguous()\n"
		"        except Exception:\n"
		"            pass\n"
		"        try:\n"
		"            model.load_state_dict(sd, strict=False, assign=True)\n"
		"        except TypeError:\n"
		"            model.load_state_dict(sd, strict=False)\n"
		"    pt.load_checkpoint = _patched_lc\n"
		"    config = pt.load_synced_config(cfg, rank=rank, world_size=world_size)\n"
		"    print(f'[Exp6] Benchmarking with Batch Size: {config.global_batch_size}')\n"
		"    loader, meta = pt.create_dataloader(config, split='test', test_set_mode=True, epochs_per_iter=1, global_batch_size=config.global_batch_size, rank=rank, world_size=world_size)\n"
		"    model, _, _ = pt.create_model(config, meta, rank, world_size)\n"
		"    model.eval()\n"
		"    # Patch TRM to ensure all carry tensors and masks are on the model device\n"
		"    try:\n"
		"        trm_mod = importlib.import_module('models.recursive_reasoning.trm')\n"
		"        # 1) Ensure empty_carry allocates on the module's buffer device (GPU)\n"
		"        if hasattr(trm_mod, 'TinyRecursiveReasoningModel_ACTV1_Inner'):\n"
		"            _orig_empty = trm_mod.TinyRecursiveReasoningModel_ACTV1_Inner.empty_carry\n"
		"            def _patched_empty(self, batch_size):\n"
		"                try:\n"
		"                    dev = self.H_init.device\n"
		"                except Exception:\n"
		"                    dev = next(self.parameters()).device  # type: ignore\n"
		"                return trm_mod.TinyRecursiveReasoningModel_ACTV1InnerCarry(\n"
		"                    z_H=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype, device=dev),\n"
		"                    z_L=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype, device=dev),\n"
		"                )\n"
		"            trm_mod.TinyRecursiveReasoningModel_ACTV1_Inner.empty_carry = _patched_empty\n"
		"            # 2) Ensure reset_carry uses mask on correct device\n"
		"            _orig_reset = trm_mod.TinyRecursiveReasoningModel_ACTV1_Inner.reset_carry\n"
		"            def _patched_reset(self, reset_flag, carry):\n"
		"                try:\n"
		"                    dev = self.H_init.device\n"
		"                except Exception:\n"
		"                    dev = next(self.parameters()).device  # type: ignore\n"
		"                reset_flag = reset_flag.to(dev)\n"
		"                # Also ensure carry tensors live on the same device\n"
		"                carry = trm_mod.TinyRecursiveReasoningModel_ACTV1InnerCarry(\n"
		"                    z_H=carry.z_H.to(dev),\n"
		"                    z_L=carry.z_L.to(dev),\n"
		"                )\n"
		"                return _orig_reset(self, reset_flag, carry)\n"
		"            trm_mod.TinyRecursiveReasoningModel_ACTV1_Inner.reset_carry = _patched_reset\n"
		"        # 3) Ensure outer initial_carry moves steps/halted to model device\n"
		"        if hasattr(trm_mod, 'TinyRecursiveReasoningModel_ACTV1'):\n"
		"            _orig_initc = trm_mod.TinyRecursiveReasoningModel_ACTV1.initial_carry\n"
		"            def _patched_initc(self, batch):\n"
		"                dev = next(self.parameters()).device\n"
		"                out = _orig_initc(self, batch)\n"
		"                out.steps = out.steps.to(dev)\n"
		"                out.halted = out.halted.to(dev)\n"
		"                out.inner_carry = trm_mod.TinyRecursiveReasoningModel_ACTV1InnerCarry(\n"
		"                    z_H=out.inner_carry.z_H.to(dev),\n"
		"                    z_L=out.inner_carry.z_L.to(dev),\n"
		"                )\n"
		"                return out\n"
		"            trm_mod.TinyRecursiveReasoningModel_ACTV1.initial_carry = _patched_initc\n"
		"    except Exception as _e:\n"
		"        print(f'[Exp6] TRM carry/device patches skipped: {_e}')\n"
		"    torch.cuda.reset_peak_memory_stats()\n"
		"    warmup_steps = 10\n"
		"    measure_steps = 50\n"
		"    total_samples = 0\n"
		"    latencies = []\n"
		"    print(f'[Exp6] Starting Warmup ({warmup_steps} steps)...')\n"
		"    with torch.inference_mode():\n"
		"        it = iter(loader)\n"
		"        for _ in range(warmup_steps):\n"
		"            try:\n"
		"                _, batch, bs = next(it)\n"
		"            except StopIteration:\n"
		"                it = iter(loader); _, batch, bs = next(it)\n"
		"            batch = {k: v.cuda() for k, v in batch.items()}\n"
		"            carry = model.initial_carry(batch)\n"
		"            try:\n"
		"                if hasattr(carry, 'halted'):\n"
		"                    carry.halted = carry.halted.to(next(model.parameters()).device)\n"
		"            except Exception:\n"
		"                pass\n"
		"            while True:\n"
		"                carry, _, _, _, done = model(carry=carry, batch=batch, return_keys=[])\n"
		"                if done: break\n"
		"            torch.cuda.synchronize()\n"
		"    print(f'[Exp6] Starting Measurement ({measure_steps} steps)...')\n"
		"    start_event = torch.cuda.Event(enable_timing=True)\n"
		"    end_event = torch.cuda.Event(enable_timing=True)\n"
		"    total_start = time.perf_counter()\n"
		"    with torch.inference_mode():\n"
		"        it = iter(loader)\n"
		"        for _ in range(measure_steps):\n"
		"            try:\n"
		"                _, batch, bs = next(it)\n"
		"            except StopIteration:\n"
		"                it = iter(loader); _, batch, bs = next(it)\n"
		"            batch = {k: v.cuda() for k, v in batch.items()}\n"
		"            start_event.record()\n"
		"            carry = model.initial_carry(batch)\n"
		"            try:\n"
		"                if hasattr(carry, 'halted'):\n"
		"                    carry.halted = carry.halted.to(next(model.parameters()).device)\n"
		"            except Exception:\n"
		"                pass\n"
		"            while True:\n"
		"                carry, _, _, _, done = model(carry=carry, batch=batch, return_keys=[])\n"
		"                if done: break\n"
		"            end_event.record()\n"
		"            torch.cuda.synchronize()\n"
		"            elapsed_ms = start_event.elapsed_time(end_event)\n"
		"            latencies.append(elapsed_ms)\n"
		"            total_samples += bs\n"
		"    total_time = time.perf_counter() - total_start\n"
		"    avg_latency_batch_ms = float(np.mean(latencies)) if latencies else float('nan')\n"
		"    avg_latency_sample_ms = avg_latency_batch_ms / float(config.global_batch_size)\n"
		"    throughput_sps = float(total_samples) / float(total_time) if total_time > 0 else 0.0\n"
		"    peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)\n"
		"    results = {\n"
		"        'batch_size': int(config.global_batch_size),\n"
		"        'steps_measured': int(measure_steps),\n"
		"        'throughput_samples_per_sec': throughput_sps,\n"
		"        'avg_latency_batch_ms': avg_latency_batch_ms,\n"
		"        'avg_latency_sample_ms': avg_latency_sample_ms,\n"
		"        'peak_memory_gb': float(peak_mem_gb),\n"
		"    }\n"
		"    print(json.dumps(results, indent=2))\n"
		"    with open('exp6_results.json', 'w') as f:\n"
		"        json.dump(results, f, indent=2)\n"
		"run_bench()\n"
	)
	cmd = ["python", "-c", pycode]
	_run(cmd, env=env, cwd=REMOTE_TRM_DIR)


@app.local_entrypoint()
def main(
	action: str = "help",
	checkpoint: str = "",
	paper_aug: int = 1000,
	single_aug: int = 0,
	arch: str = "trm",
	seed: int = 42,
	repo: str = "arcprize/trm_arc_prize_verification",
	dest: str = "checkpoints/hf_trm",
	data_dir: str = "data/arc-aug-1000",
) -> None:
	"""
	Local entrypoint for quick testing (run from inside TinyRecursiveModels):
	- modal run modal_trm.py::main --action prepare_data
	- modal run modal_trm.py::main --action exp1 --checkpoint /workspace/TinyRecursiveModels/checkpoints/<your_run>/step_XXXX
	"""
	if action == "prepare_data":
		prepare_arc_datasets.remote()
	elif action == "build_arc1":
		build_arc1.remote()
	elif action == "fetch_ckpt":
		fetch_hf_checkpoints.remote(repo_id=repo, dest=dest)
	elif action == "check_data":
		quick_check_data.remote(data_dir=data_dir)
	elif action == "quick_exp1":
		if not checkpoint:
			raise SystemExit("Please pass --checkpoint=<path_to_checkpoint_in_volume_or_abs_path>")
		quick_exp1_infer.remote(
			load_checkpoint=checkpoint,
			max_batches=5,
			arch=arch,
			seed=seed,
		)
	elif action == "exp1":
		if not checkpoint:
			raise SystemExit("Please pass --checkpoint=<path_to_checkpoint_in_volume_or_abs_path>")
		exp1_secret_sauce.remote(
			load_checkpoint=checkpoint,
			paper_aug_level=paper_aug,
			single_aug_level=single_aug,
			arch=arch,
			seed=seed,
		)
	elif action == "exp1_paper_only":
		if not checkpoint:
			raise SystemExit("Please pass --checkpoint=<path_to_checkpoint_in_volume_or_abs_path>")
		exp1_paper_only.remote(
			load_checkpoint=checkpoint,
			paper_aug_level=paper_aug,
			arch=arch,
			seed=seed,
		)
	elif action == "exp1_single_only":
		if not checkpoint:
			raise SystemExit("Please pass --checkpoint=<path_to_checkpoint_in_volume_or_abs_path>")
		exp1_single_only.remote(
			load_checkpoint=checkpoint,
			single_aug_level=single_aug,
			arch=arch,
			seed=seed,
		)
	elif action == "exp6":
		if not checkpoint:
			raise SystemExit("Please pass --checkpoint=<path_to_checkpoint_in_volume_or_abs_path>")
		exp6_efficiency.remote(
			load_checkpoint=checkpoint,
			arch=arch,
			seed=seed,
			batch_size=32,
		)
	else:
		print("Actions:")
		print("  - prepare_data")
		print("  - build_arc1")
		print("  - fetch_ckpt  (optional: --repo, --dest)")
		print("  - check_data  (optional: --data-dir, default: data/arc-aug-1000)")
		print("  - quick_exp1  (requires --checkpoint) fast GPU sanity of Exp1 with few batches")
		print("  - exp1  (requires --checkpoint)")
		print("  - exp1_paper_only  (requires --checkpoint)")
		print("  - exp1_single_only (requires --checkpoint)")
		print("  - exp6 (requires --checkpoint) efficiency benchmarking")
		print("")
		print("Examples:")
		print("  modal run modal_trm.py::main --action prepare_data")
		print("  modal run modal_trm.py::main --action fetch_ckpt --repo arcprize/trm_arc_prize_verification --dest checkpoints/hf_trm")
		print("  modal run modal_trm.py::main --action check_data --data-dir data/arc-aug-1000")
		print("  modal run modal_trm.py::main --action quick_exp1 --checkpoint /workspace/TinyRecursiveModels/checkpoints/<your_run>/step_XXXX")
		print("  modal run modal_trm.py::main --action exp1 --checkpoint /workspace/TinyRecursiveModels/checkpoints/<your_run>/step_XXXX")
		print("  modal run modal_trm.py::main --action exp1_paper_only --checkpoint /workspace/TinyRecursiveModels/checkpoints/<your_run>/step_XXXX")
		print("  modal run modal_trm.py::main --action exp1_single_only --checkpoint /workspace/TinyRecursiveModels/checkpoints/<your_run>/step_XXXX")
		print("  modal run modal_trm.py::main --action exp6 --checkpoint /workspace/TinyRecursiveModels/checkpoints/hf_trm/arc_v1_public/step_518071")


