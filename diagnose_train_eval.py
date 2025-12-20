"""Train + evaluate with diagnostics (Kaggle-friendly).

Usage (Kaggle notebook cell):
  %cd /kaggle/working/CVDHD
  !python diagnose_train_eval.py \
      --gpu 0 \
      --base-ckpt /kaggle/working/FIFO_final_model.pth \
      --exp-name FDA_diag \
      --train-script main_fda.py \
      --train-extra "--fda-beta 0.01" \
      --num-steps 2100 --num-steps-stop 2100 \
      --steps 200,800,2000 \
      --save-dir /kaggle/working/snapshots/FIFO_model

This script helps answer: "điểm giảm do LR hay do load/save/eval sai?"
It prints:
  - baseline mIoU from base checkpoint
  - whether checkpoint contains fogpass weights
  - exact snapshot dir & checkpoint filenames found
  - eval mIoU at selected steps
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], *, title: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
    if title:
        print("\n" + "=" * 10 + f" {title} " + "=" * 10)
    print("$", " ".join(cmd))
    return subprocess.run(cmd, text=True, capture_output=False, check=check)


def run_capture(cmd: list[str], *, title: str | None = None, check: bool = True) -> tuple[int, str]:
    if title:
        print("\n" + "=" * 10 + f" {title} " + "=" * 10)
    print("$", " ".join(cmd))
    p = subprocess.run(cmd, text=True, capture_output=True)
    out = (p.stdout or "") + "\n" + (p.stderr or "")
    if check and p.returncode != 0:
        print(out)
        raise RuntimeError(f"Command failed (exit={p.returncode}): {' '.join(cmd)}")
    return p.returncode, out


def parse_steps(s: str) -> list[int]:
    if not s.strip():
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def find_ckpt_for_step(exp_name: str, step: int, snapshot_dirs: list[str]) -> str | None:
    # training saves: <snapshot_dir>/<run_name>_FIFO{step}.pth, where run_name = "{file-name}-{MM-DD-HH-MM}"
    patterns = [str(Path(d) / f"*{exp_name}*FIFO{step}.pth") for d in snapshot_dirs]
    matches: list[Path] = []
    for pat in patterns:
        matches.extend(Path(p) for p in glob.glob(pat))
    if not matches:
        return None
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(matches[0])


def find_latest_ckpt(exp_name: str, snapshot_dirs: list[str]) -> str | None:
    patterns = [str(Path(d) / f"*{exp_name}*FIFO*.pth") for d in snapshot_dirs]
    matches: list[Path] = []
    for pat in patterns:
        matches.extend(Path(p) for p in glob.glob(pat))
    if not matches:
        return None
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(matches[0])


def inspect_checkpoint(path: str) -> dict:
    try:
        import torch

        obj = torch.load(path, map_location="cpu", weights_only=False)
        info: dict = {"path": path, "type": type(obj).__name__}
        if isinstance(obj, dict):
            info["keys"] = sorted(list(obj.keys()))
            info["has_state_dict"] = "state_dict" in obj
            info["has_fogpass1"] = "fogpass1_state_dict" in obj
            info["has_fogpass2"] = "fogpass2_state_dict" in obj
        else:
            info["keys"] = None
            info["has_state_dict"] = isinstance(obj, dict)
            info["has_fogpass1"] = False
            info["has_fogpass2"] = False
        return info
    except Exception as e:
        return {"path": path, "error": str(e)}


def eval_ckpt(tag: str, ckpt_path: str, gpu: str) -> dict:
    _, out = run_capture(["python", "evaluate.py", "--file-name", tag, "--restore-from", ckpt_path, "--gpu", str(gpu)], title=f"EVAL {tag}")

    def grab(dataset_header: str) -> float | None:
        idx = out.find(dataset_header)
        if idx < 0:
            return None
        chunk = out[idx : idx + 800]
        m = re.search(r"mIoU:\\s*([0-9]+(?:\\.[0-9]+)?)", chunk)
        return float(m.group(1)) if m else None

    return {
        "FZ": grab("Evaluation on Foggy Zurich"),
        "FDD": grab("Evaluation on Foggy Driving Dense"),
        "FD": grab("Evaluation on Foggy Driving"),
        "Lindau": grab("Evaluation on Cityscapes lindau"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--repo-dir", default="/kaggle/working/CVDHD")
    ap.add_argument("--gpu", default="0")

    ap.add_argument("--base-ckpt", required=True)
    ap.add_argument("--train-script", default="main_fda.py")
    ap.add_argument("--exp-name", required=True)

    ap.add_argument("--num-steps", type=int, default=2100)
    ap.add_argument("--num-steps-stop", type=int, default=2100)
    ap.add_argument("--steps", default="200,800,2000")

    ap.add_argument("--save-dir", default="")
    ap.add_argument("--snapshot-dir", default="./snapshots/FIFO_model")

    ap.add_argument(
        "--train-extra",
        default="",
        help='Extra args passed to trainer as a single string, e.g. "--fda-beta 0.01"',
    )

    # Finetune recipe defaults
    ap.add_argument("--finetune", action="store_true", default=True)
    ap.add_argument("--backbone-lr-mult", default="0.02")
    ap.add_argument("--head-lr-mult", default="0.05")
    ap.add_argument("--freeze-bn", action="store_true", default=True)

    ap.add_argument("--lambda-cl", default="0.0")
    ap.add_argument("--cl-warmup-steps", default="1000")
    ap.add_argument("--cl-temp", default="0.1")
    ap.add_argument("--cl-head-lr", default="1e-3")

    ap.add_argument("--freeze-fogpass-steps", default="2000")
    ap.add_argument("--fpf-lr-mult", default="0.1")

    ap.add_argument("--save-pred-every-early", default="200")
    ap.add_argument("--save-pred-early-until", default="2001")

    ap.add_argument("--amp", default="1")

    args = ap.parse_args()

    repo = Path(args.repo_dir)
    if not repo.exists():
        raise SystemExit(f"Repo not found: {repo}")
    os.chdir(repo)

    # Configure snapshot dirs to search
    snapshot_dirs: list[str] = []
    if args.save_dir:
        snapshot_dirs.append(args.save_dir)
    snapshot_dirs.append(args.snapshot_dir)
    # Kaggle common default if user forgets
    snapshot_dirs.append("/kaggle/working/snapshots/FIFO_model")
    snapshot_dirs = [d for d in dict.fromkeys(snapshot_dirs) if d]

    print("\nSnapshot dirs (search order):")
    for d in snapshot_dirs:
        print(" -", d)

    print("\n[1] Inspect base checkpoint:")
    base_info = inspect_checkpoint(args.base_ckpt)
    print(base_info)

    print("\n[2] Baseline evaluation (before finetune):")
    base_metrics = eval_ckpt(f"{args.exp_name}_BASE", args.base_ckpt, args.gpu)
    print(base_metrics)

    # Train
    common = [
        "--modeltrain",
        "train",
        "--file-name",
        args.exp_name,
        "--restore-from",
        args.base_ckpt,
        "--restore-from-fogpass",
        args.base_ckpt,
        "--num-steps",
        str(args.num_steps),
        "--num-steps-stop",
        str(args.num_steps_stop),
        "--backbone-lr-mult",
        str(args.backbone_lr_mult),
        "--head-lr-mult",
        str(args.head_lr_mult),
        "--lambda-cl",
        str(args.lambda_cl),
        "--cl-warmup-steps",
        str(args.cl_warmup_steps),
        "--cl-temp",
        str(args.cl_temp),
        "--cl-head-lr",
        str(args.cl_head_lr),
        "--freeze-fogpass-steps",
        str(args.freeze_fogpass_steps),
        "--fpf-lr-mult",
        str(args.fpf_lr_mult),
        "--save-pred-every-early",
        str(args.save_pred_every_early),
        "--save-pred-early-until",
        str(args.save_pred_early_until),
        "--amp",
        str(args.amp),
        "--gpu",
        str(args.gpu),
    ]
    if args.finetune:
        common.append("--finetune")
    if args.freeze_bn:
        common.append("--freeze-bn")
    if args.save_dir:
        common += ["--save-dir", args.save_dir]

    extra = args.train_extra.strip().split() if args.train_extra.strip() else []

    run(["python", args.train_script] + common + extra, title=f"TRAIN {args.exp_name} ({args.train_script})")

    # Evaluate snapshots
    steps = parse_steps(args.steps)
    print("\n[3] Evaluate selected steps:")
    for step in steps:
        ckpt = find_ckpt_for_step(args.exp_name, step, snapshot_dirs)
        if not ckpt:
            print(f"[WARN] Missing ckpt for step={step} (exp={args.exp_name})")
            continue
        print(f"\n[CKPT] step={step} -> {ckpt}")
        print("inspect:", inspect_checkpoint(ckpt))
        m = eval_ckpt(f"{args.exp_name}_FIFO{step}", ckpt, args.gpu)
        print(m)

    latest = find_latest_ckpt(args.exp_name, snapshot_dirs)
    if latest:
        print(f"\n[4] Latest checkpoint found: {latest}")
        print("inspect:", inspect_checkpoint(latest))

    # Final (num-steps-stop) ckpt
    final_name = f"{args.exp_name}{args.num_steps_stop}.pth"
    final_paths = [str(Path(d) / final_name) for d in snapshot_dirs]
    final_paths = [p for p in final_paths if os.path.exists(p)]
    if final_paths:
        final_ckpt = final_paths[0]
        print(f"\n[5] Final checkpoint found: {final_ckpt}")
        print("inspect:", inspect_checkpoint(final_ckpt))
        m = eval_ckpt(f"{args.exp_name}_FINAL", final_ckpt, args.gpu)
        print(m)
    else:
        print("\n[5] Final checkpoint not found (this is OK if you rely on FIFO{step} snapshots).")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
