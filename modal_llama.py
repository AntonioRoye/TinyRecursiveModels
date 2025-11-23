import os
import json
from modal import App, Image, Volume, Secret

# -----------------------------------------------------------------------------
# Experiment 3 — Fine-tuned Llama-3-8B baseline via Modal + Unsloth (QLoRA)
# - Bakes ARC JSONs from your repo into the image (like modal_trm.py)
# - Persists processed jsonl and checkpoints via Modal Volumes
# - Trains and evaluates a Llama-3-8B-Instruct adapter and reports exact-match accuracy
# -----------------------------------------------------------------------------

APP_NAME = "trm-llama-baseline"
VOL_NAME = "trm-data"
CKPT_VOL = "trm-checkpoints"

# Optional HF token for gated models.
# Use your consolidated Modal secret "pipeline-secrets" (contains HF_TOKEN, OPENAI_API_KEY).
# We'll also pass through your local env variable if set.
secrets_cfg = []
try:
	secrets_cfg.append(Secret.from_name("pipeline-secrets"))
except Exception:
	pass

# Resolve local/remote repo dirs (match style used in modal_trm.py)
LOCAL_TRM_DIR = os.path.dirname(os.path.abspath(__file__))
REMOTE_TRM_DIR = "/workspace/TinyRecursiveModels"

# GPU-ready base with Python + CUDA libs preinstalled
image = (
	Image.from_registry("pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel")
	.apt_install("git", "wget")
	.env({
		"TRANSFORMERS_NO_TORCHVISION": "1",  # disable torchvision usage for text-only training
	})
	.pip_install(
		"unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
		"transformers",
		"trl",
		"peft",
		"accelerate",
		"bitsandbytes",
		"datasets",
		"tqdm",
	)
	# Bake ARC JSONs so we don't rely on Mount (keeps parity with modal_trm.py pattern)
	.add_local_dir(os.path.join(LOCAL_TRM_DIR, "kaggle"), os.path.join(REMOTE_TRM_DIR, "kaggle"))
)

app = App(APP_NAME)

# Persisted volumes for processed data and checkpoints (create if missing)
data_vol = Volume.from_name(VOL_NAME, create_if_missing=True)
ckpt_vol = Volume.from_name(CKPT_VOL, create_if_missing=True)

# Minimal torchvision stub to avoid importing compiled ops for text-only training.
def _install_torchvision_stub():
	try:
		import torchvision  # noqa: F401
		return  # Already present; assume user wants real torchvision
	except Exception:
		pass
	import sys
	import types
	import importlib.machinery as _machinery
	from enum import Enum
	def _mk(name: str) -> types.ModuleType:
		m = types.ModuleType(name)
		m.__spec__ = _machinery.ModuleSpec(name, loader=None)
		return m
	tv = _mk("torchvision")
	tv.__path__ = []
	transforms = _mk("torchvision.transforms")
	class _InterpolationMode(Enum):
		NEAREST = 0
		NEAREST_EXACT = 1
		BILINEAR = 2
		BICUBIC = 3
		BOX = 4
		HAMMING = 5
		LANCZOS = 6
	transforms.InterpolationMode = _InterpolationMode
	v2 = _mk("torchvision.transforms.v2")
	functional = _mk("torchvision.transforms.v2.functional")
	# No-op placeholders to satisfy optional imports
	for _name in ("resize", "to_dtype", "to_pil_image", "to_tensor"):
		setattr(functional, _name, lambda *a, **k: a[0] if a else None)
	v2.functional = functional
	io_mod = _mk("torchvision.io")
	def _unavail(*args, **kwargs):
		raise RuntimeError("torchvision is disabled for this run")
	io_mod.read_video = _unavail
	io_mod.read_image = _unavail
	tv.transforms = transforms
	tv.io = io_mod
	sys.modules["torchvision"] = tv
	sys.modules["torchvision.transforms"] = transforms
	sys.modules["torchvision.transforms.v2"] = v2
	sys.modules["torchvision.transforms.v2.functional"] = functional
	sys.modules["torchvision.io"] = io_mod

# Paths in the remote container
REMOTE_ROOT = "/workspace"
DATA_DIR = f"{REMOTE_ROOT}/data_llama"
RAW_DATA_DIR = f"{REMOTE_TRM_DIR}/kaggle/combined"

# Base model name for Unsloth 4-bit (QLoRA)
BASE_MODEL_NAME = "unsloth/llama-3-8b-Instruct-bnb-4bit"


@app.function(
	image=image,
	volumes={
		DATA_DIR: data_vol,  # write processed JSONL
	},
	timeout=1800,
	secrets=secrets_cfg,
)
def prepare_data():
	"""
	Converts ARC JSON grids into text prompts for Llama-3.
	Llama-3 Instruct format with examples followed by a single test input.
	"""
	import os

	os.makedirs(DATA_DIR, exist_ok=True)

	sets = {
		"train": "arc-agi_training_challenges.json",
		"train_sol": "arc-agi_training_solutions.json",
		"eval": "arc-agi_evaluation_challenges.json",
		"eval_sol": "arc-agi_evaluation_solutions.json",
	}

	def load_json(fname):
		path = os.path.join(RAW_DATA_DIR, fname)
		with open(path, "r") as f:
			return json.load(f)

	print("Loading raw ARC data...")
	train_tasks = load_json(sets["train"])
	train_sols = load_json(sets["train_sol"])
	eval_tasks = load_json(sets["eval"])
	eval_sols = load_json(sets["eval_sol"])

	def fmt_grid(g):
		return str(g).replace(" ", "")

	def to_prompts(tasks, sols=None):
		prompts = []
		for tid, task in tasks.items():
			# 1) Build demonstrations from task["train"]
			demos = ""
			for i, pair in enumerate(task["train"]):
				demos += f"Example {i+1} Input: {fmt_grid(pair['input'])}\n"
				demos += f"Example {i+1} Output: {fmt_grid(pair['output'])}\n\n"

			# 2) For each test pair, create a supervised example if label is known
			for i, pair in enumerate(task["test"]):
				inp = fmt_grid(pair["input"])
				label = None
				if sols and tid in sols:
					label = fmt_grid(sols[tid][i])
				elif "output" in pair:
					label = fmt_grid(pair["output"])

				if label:
					text = (
						f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
						f"You are a puzzle solver. Identify the pattern in the examples and solve the test input.\n\n"
						f"{demos}"
						f"Test Input: {inp}<|eot_id|>"
						f"<|start_header_id|>assistant<|end_header_id|>\n\n"
						f"Test Output: {label}<|eot_id|>"
					)
					prompts.append({"text": text, "input": inp, "output": label})
		return prompts

	print("Formatting training data...")
	train_data = to_prompts(train_tasks, train_sols)
	print(f"Formatting evaluation data (size: {len(eval_tasks)} tasks)...")
	test_data = to_prompts(eval_tasks, eval_sols)

	with open(f"{DATA_DIR}/train.jsonl", "w") as f:
		for item in train_data:
			f.write(json.dumps(item) + "\n")
	with open(f"{DATA_DIR}/test.jsonl", "w") as f:
		for item in test_data:
			f.write(json.dumps(item) + "\n")

	print(f"Saved {len(train_data)} training examples and {len(test_data)} test examples.")


@app.function(
	image=image,
	gpu="H100",  # change to "A100" or "A10G" if needed
	volumes={
		DATA_DIR: data_vol,
		"/workspace/checkpoints": ckpt_vol,
	},
	timeout=7200,
	secrets=secrets_cfg,
)
def finetune_llama(
	base_model: str = BASE_MODEL_NAME,
	epochs: int = 2,
	lr: float = 2e-4,
):
	# Ensure no torchvision imports inside transformers stack
	os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
	_install_torchvision_stub()
	# Normalize HF token env var for huggingface_hub/transformers.
	# If only HF_TOKEN is provided, mirror it to HUGGING_FACE_HUB_TOKEN.
	_hf = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
	if _hf and "HUGGING_FACE_HUB_TOKEN" not in os.environ:
		os.environ["HUGGING_FACE_HUB_TOKEN"] = _hf

	from unsloth import FastLanguageModel
	from trl import SFTTrainer
	from transformers import TrainingArguments
	from datasets import load_dataset
	import torch
	# Hard fail if GPU not present to avoid silent CPU runs
	if not torch.cuda.is_available():
		raise SystemExit("GPU was not allocated. Ensure your Modal env has GPU quota and this app can schedule on H100/A100.")
	from peft import PeftModel

	print(f"Loading model: {base_model}")
	model, tokenizer = FastLanguageModel.from_pretrained(
		model_name=base_model,
		max_seq_length=4096,
		dtype=None,
		load_in_4bit=True,
	)

	# Add LoRA adapters
	model = FastLanguageModel.get_peft_model(
		model,
		r=16,
		target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
		lora_alpha=16,
		lora_dropout=0,
		bias="none",
		use_gradient_checkpointing="unsloth",
	)

	# Load data
	dataset = load_dataset("json", data_files={"train": f"{DATA_DIR}/train.jsonl"}, split="train")
	print(f"Loaded {len(dataset)} training samples.")

	trainer = SFTTrainer(
		model=model,
		tokenizer=tokenizer,
		train_dataset=dataset,
		dataset_text_field="text",
		max_seq_length=4096,
		dataset_num_proc=2,
		packing=False,
		args=TrainingArguments(
			per_device_train_batch_size=2,
			gradient_accumulation_steps=4,
			warmup_steps=5,
			max_steps=0,  # use num_train_epochs
			num_train_epochs=epochs,
			learning_rate=lr,
			fp16=not torch.cuda.is_bf16_supported(),
			bf16=torch.cuda.is_bf16_supported(),
			logging_steps=1,
			optim="adamw_8bit",
			weight_decay=0.01,
			lr_scheduler_type="linear",
			seed=3407,
			output_dir="outputs",
		),
	)

	print("Starting Training...")
	trainer.train()

	# Save artifacts: PEFT adapter (if available) and a merged full model for inference.
	adapter_dir = "/workspace/checkpoints/llama3_arc_adapter"
	merged_dir = "/workspace/checkpoints/llama3_arc_merged"
	os.makedirs(adapter_dir, exist_ok=True)
	os.makedirs(merged_dir, exist_ok=True)
	if isinstance(model, PeftModel):
		print(f"Saving LoRA adapter (PEFT) to {adapter_dir}...")
		model.save_pretrained(adapter_dir)
		tokenizer.save_pretrained(adapter_dir)
	else:
		print("Model is not a PeftModel; skipping PEFT adapter save.")
	# Save merged model for adapter-free inference
	try:
		if isinstance(model, PeftModel) and hasattr(model, "merge_and_unload"):
			print(f"Merging LoRA into base and saving merged model to {merged_dir}...")
			merged_model = model.merge_and_unload()
			merged_model.save_pretrained(merged_dir)
			tokenizer.save_pretrained(merged_dir)
		else:
			print(f"Saving current model state to {merged_dir} (no merge available)...")
			model.save_pretrained(merged_dir)
			tokenizer.save_pretrained(merged_dir)
	except Exception as e:
		print(f"[WARN] Failed to save merged model: {e}")


@app.function(
	image=image,
	gpu="H100",
	volumes={
		DATA_DIR: data_vol,
		"/workspace/checkpoints": ckpt_vol,
	},
	timeout=3600,
	secrets=secrets_cfg,
)
def evaluate_llama(
	base_model: str = BASE_MODEL_NAME,
	adapter_subdir: str = "/workspace/checkpoints/llama3_arc_adapter",
):
	"""
	Runs inference on the test set and computes exact-match accuracy.
	"""
	os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
	_install_torchvision_stub()
	# Normalize HF token env var for huggingface_hub/transformers.
	_hf = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
	if _hf and "HUGGING_FACE_HUB_TOKEN" not in os.environ:
		os.environ["HUGGING_FACE_HUB_TOKEN"] = _hf

	from unsloth import FastLanguageModel
	from peft import PeftModel
	from tqdm import tqdm
	import torch
	if not torch.cuda.is_available():
		raise SystemExit("GPU was not allocated for eval. Ensure GPU scheduling (H100/A100) is enabled in your Modal env.")

	# Load base + PEFT adapter (more robust than merged for 4-bit flows)
	# Long-context settings for evaluation
	MAX_SEQ_EVAL = int(os.environ.get("LLAMA_EVAL_MAX_SEQ", "16384"))
	MAX_NEW_TOKENS = int(os.environ.get("LLAMA_EVAL_MAX_NEW_TOKENS", "128"))

	print(f"Loading base model: {base_model} (max_seq_length={MAX_SEQ_EVAL})")
	model, tokenizer = FastLanguageModel.from_pretrained(
		model_name=base_model,
		max_seq_length=MAX_SEQ_EVAL,
		dtype=None,
		load_in_4bit=True,
	)
	print(f"Loading adapter from {adapter_subdir}")
	model = PeftModel.from_pretrained(model, adapter_subdir)
	FastLanguageModel.for_inference(model)

	# Load test data
	with open(f"{DATA_DIR}/test.jsonl", "r") as f:
		test_data = [json.loads(line) for line in f]
	print(f"Evaluating {len(test_data)} examples...")

	correct = 0
	total = 0

	# Ensure truncation keeps the end of the prompt (the test input + 'Test Output:')
	try:
		tokenizer.truncation_side = "left"
	except Exception:
		pass

	for item in tqdm(test_data):
		# Recreate the prompt up to "Test Output:" so the model must fill only the answer
		prefix = item["text"].split("Test Output:")[0] + "Test Output:"
		target = item["output"].replace(" ", "")

		# Truncate to fit within max context window allowing room for generation
		allowed_input_len = max(256, MAX_SEQ_EVAL - MAX_NEW_TOKENS - 16)
		inputs = tokenizer(
			[prefix],
			return_tensors="pt",
			truncation=True,
			max_length=allowed_input_len,
			padding=False,
		).to("cuda")
		outputs = model.generate(
			**inputs,
			max_new_tokens=MAX_NEW_TOKENS,
			use_cache=True,
			pad_token_id=tokenizer.eos_token_id,
		)
		generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

		try:
			answer_part = generated_text.split("Test Output:")[-1].strip()
			start = answer_part.find("[[")
			end = answer_part.rfind("]]")
			if start != -1 and end != -1:
				prediction = answer_part[start : end + 2]
				prediction = prediction.replace(" ", "")
				if prediction == target:
					correct += 1
		except Exception:
			pass

		total += 1
		if total % 10 == 0:
			print(f"Progress: {correct}/{total} ({correct/total:.2%})")

	accuracy = correct / max(total, 1)
	print("\nFinal Result (Llama-3-8B Fine-Tuned):")
	print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")

	# Save result artifact
	with open("/workspace/checkpoints/exp3_llama_result.json", "w") as f:
		json.dump({"accuracy": accuracy, "correct": correct, "total": total}, f)


@app.local_entrypoint()
def main(action: str = "all"):
	if action in ["prepare", "all"]:
		prepare_data.remote()
	if action in ["train", "all"]:
		finetune_llama.remote()
	if action in ["eval", "all"]:
		evaluate_llama.remote()


