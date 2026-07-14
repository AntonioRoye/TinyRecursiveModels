"""Run the isolated puzzle-ID smoke test on the existing Modal H100 setup."""
from __future__ import annotations

import json
import os
import subprocess

from modal import gpu

from modal_trm import (
    CHECKPOINTS_VOL,
    DATA_VOL,
    OUTPUTS_VOL,
    REMOTE_TRM_DIR,
    _default_env,
    app,
    image,
)


@app.function(
    image=image,
    gpu=gpu.H100(),
    volumes={
        f"{REMOTE_TRM_DIR}/data": DATA_VOL,
        f"{REMOTE_TRM_DIR}/outputs": OUTPUTS_VOL,
        f"{REMOTE_TRM_DIR}/checkpoints": CHECKPOINTS_VOL,
    },
    timeout=60 * 60 * 6,
)
def run_probe(
    checkpoint: str = "/workspace/TinyRecursiveModels/checkpoints/hf_trm/arc_v1_public/step_518071",
    data_dir: str = "data/arc-aug-1000",
    max_batches: int = 1,
    batch_size: int = 64,
) -> dict:
    env = _default_env()
    env.update(
        {
            "DISABLE_COMPILE": "1",
            "TORCHDYNAMO_DISABLE": "1",
            "PID_PROBE_MAX_BATCHES": str(max_batches),
            "PID_PROBE_OUTPUT": "outputs/pid_probe_results.json",
        }
    )
    os.makedirs(os.path.join(REMOTE_TRM_DIR, "outputs"), exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint), exist_ok=True)

    data_abs = os.path.join(REMOTE_TRM_DIR, data_dir)
    if not os.path.exists(os.path.join(data_abs, "test_puzzles.json")):
        print(f"Dataset missing at {data_abs}; building the checkpoint-compatible ARC-1 data.")
        subprocess.run(
            [
                "python",
                "-m",
                "dataset.build_arc_dataset",
                "--input-file-prefix",
                "kaggle/combined/arc-agi",
                "--output-dir",
                data_dir,
                "--subsets",
                "training",
                "evaluation",
                "concept",
                "--test-set-name",
                "evaluation",
                "--num-aug",
                "1000",
            ],
            cwd=REMOTE_TRM_DIR,
            env=env,
            check=True,
        )
        DATA_VOL.commit()

    if not os.path.exists(checkpoint):
        print(f"Checkpoint missing at {checkpoint}; downloading the public ARC-1 checkpoint.")
        download_code = (
            "from huggingface_hub import hf_hub_download; "
            "print(hf_hub_download(" 
            "repo_id='arcprize/trm_arc_prize_verification', "
            "filename='arc_v1_public/step_518071', "
            "local_dir='checkpoints/hf_trm'))"
        )
        subprocess.run(
            ["python", "-c", download_code],
            cwd=REMOTE_TRM_DIR,
            env=env,
            check=True,
        )
        CHECKPOINTS_VOL.commit()

    cmd = [
        "python",
        "-m",
        "experiments.quick_isolated_pid_probe",
        "arch=trm",
        "arch.L_cycles=4",
        "arch.H_cycles=3",
        "arch.L_layers=2",
        f"load_checkpoint={checkpoint}",
        f"data_paths=['{data_dir}']",
        f"data_paths_test=['{data_dir}']",
        f"global_batch_size={batch_size}",
        "+eval_save_outputs=[]",
        "checkpoint_path=checkpoints/pid_probe",
        "hydra.run.dir=outputs/pid_probe_hydra",
    ]
    completed = subprocess.run(
        cmd,
        cwd=REMOTE_TRM_DIR,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(completed.stdout)
    output_path = os.path.join(REMOTE_TRM_DIR, "outputs", "pid_probe_results.json")
    with open(output_path, "r") as f:
        result = json.load(f)
    OUTPUTS_VOL.commit()
    return result


@app.local_entrypoint()
def main(
    checkpoint: str = "/workspace/TinyRecursiveModels/checkpoints/hf_trm/arc_v1_public/step_518071",
    data_dir: str = "data/arc-aug-1000",
    max_batches: int = 1,
    batch_size: int = 64,
) -> None:
    result = run_probe.remote(
        checkpoint=checkpoint,
        data_dir=data_dir,
        max_batches=max_batches,
        batch_size=batch_size,
    )
    print("MODAL_PID_PROBE_RESULT=" + json.dumps(result))
