import os
import argparse
import hashlib
from pathlib import Path

from huggingface_hub import snapshot_download


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Fetch TRM checkpoints from Hugging Face and list candidate weight files.")
    parser.add_argument("--repo_id", default="arcprize/trm_arc_prize_verification", help="HF repo id")
    parser.add_argument("--dest", default="checkpoints/hf_trm", help="Local destination directory")
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    local_dir = snapshot_download(repo_id=args.repo_id, local_dir=args.dest, local_dir_use_symlinks=False)

    print(f"Downloaded to: {local_dir}")

    patterns = ["*.pt", "*.pth", "*step_*", "*.bin"]
    candidates = []
    for pat in patterns:
        candidates.extend(Path(local_dir).rglob(pat))

    if not candidates:
        print("No obvious checkpoint files found. Inspect the directory structure under:")
        print(local_dir)
        return

    print("\nCandidate checkpoint files:")
    for p in sorted(set(map(str, candidates))):
        try:
            digest = sha256_of_file(p)
        except Exception:
            digest = "<unreadable>"
        print(f"- {p} | sha256={digest}")

    print("\nUse one of the above paths as load_checkpoint=... for experiments.run_exp1")


if __name__ == "__main__":
    main()


