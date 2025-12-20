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


def _fmt(v: float | None) -> str:
    if v is None:
        return "NA"
    return f"{v:.2f}"


def _delta(v: float | None, base: float | None) -> float | None:
    if v is None or base is None:
        return None
    return v - base


def _rank_score(row: dict, *, mode: str) -> tuple[float, float, float, float]:
    """Return a sortable score tuple for a result row.

    Higher is better. Uses deltas (dFZ/dFDD/dFD/dLindau) when present.

    Modes:
      - fdd: prioritize Foggy Driving Dense
      - mean: prioritize mean improvement across datasets
      - min: prioritize worst-case (minimum) improvement across datasets
    """

    deltas: list[float] = []
    for k in ("dFZ", "dFDD", "dFD", "dLindau"):
        v = row.get(k)
        if isinstance(v, float):
            deltas.append(v)

    if not deltas:
        mean_delta = -1e9
        min_delta = -1e9
    else:
        mean_delta = sum(deltas) / len(deltas)
        min_delta = min(deltas)

    d_fdd = row.get("dFDD")
    d_fz = row.get("dFZ")
    d_fdd_v = float(d_fdd) if isinstance(d_fdd, float) else -1e9
    d_fz_v = float(d_fz) if isinstance(d_fz, float) else -1e9

    m = (mode or "fdd").strip().lower()
    if m == "min":
        # Best worst-case; break ties with mean and FDD.
        return (min_delta, mean_delta, d_fdd_v, d_fz_v)
    if m == "mean":
        # Best average; break ties with min and FDD.
        return (mean_delta, min_delta, d_fdd_v, d_fz_v)
    # Default: fdd
    return (d_fdd_v, d_fz_v, mean_delta, min_delta)


def _print_results_table(rows: list[dict]) -> None:
    headers = [
        "exp",
        "FZ",
        "FDD",
        "FD",
        "Lindau",
        "dFZ",
        "dFDD",
        "dFD",
        "dLindau",
    ]

    def cell(r: dict, k: str) -> str:
        v = r.get(k)
        if isinstance(v, float):
            return f"{v:.2f}"
        return str(v) if v is not None else "NA"

    widths = {h: max(len(h), max(len(cell(r, h)) for r in rows)) for h in headers}
    line = " | ".join(h.ljust(widths[h]) for h in headers)
    sep = "-+-".join("-" * widths[h] for h in headers)
    print("\n" + line)
    print(sep)
    for r in rows:
        print(" | ".join(cell(r, h).ljust(widths[h]) for h in headers))


def _replace_or_add_flag(tokens: list[str], flag: str, value: str | None) -> list[str]:
    """Replace occurrences of --flag <val>. If value is None, remove the flag+val.

    This is a simple token-level helper (works for flags that always take a value).
    """
    out: list[str] = []
    i = 0
    while i < len(tokens):
        if tokens[i] == flag and i + 1 < len(tokens):
            # skip existing
            i += 2
            continue
        out.append(tokens[i])
        i += 1
    if value is not None:
        out += [flag, value]
    return out


def parse_base_metrics(s: str) -> dict[str, float] | None:
    """Parse baseline metrics from a string.

    Supported formats:
      - "FZ=48.41,FDD=48.93,FD=50.71,Lindau=64.75"
      - "48.41,48.93,50.71,64.75" (order: FZ,FDD,FD,Lindau)
    """
    if not s or not s.strip():
        return None
    raw = s.strip()

    # Key=val form
    if "=" in raw:
        out: dict[str, float] = {}
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        for p in parts:
            if "=" not in p:
                continue
            k, v = p.split("=", 1)
            k = k.strip()
            v = v.strip()
            if not k:
                continue
            out[k] = float(v)
        # normalize allowed keys
        key_map = {"fz": "FZ", "fdd": "FDD", "fd": "FD", "lindau": "Lindau"}
        norm: dict[str, float] = {}
        for k, v in out.items():
            kk = key_map.get(k.strip().lower())
            if kk:
                norm[kk] = float(v)
        if norm:
            return norm
        return None

    # Positional form
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 4:
        raise ValueError("--base-metrics must have 4 comma-separated values or key=value pairs")
    vals = [float(x) for x in parts]
    return {"FZ": vals[0], "FDD": vals[1], "FD": vals[2], "Lindau": vals[3]}


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


def _mean_delta(row: dict) -> float:
    vals: list[float] = []
    for k in ("dFZ", "dFDD", "dFD", "dLindau"):
        v = row.get(k)
        if isinstance(v, float):
            vals.append(v)
    return sum(vals) / len(vals) if vals else -1e9


def _min_delta(row: dict) -> float:
    vals: list[float] = []
    for k in ("dFZ", "dFDD", "dFD", "dLindau"):
        v = row.get(k)
        if isinstance(v, float):
            vals.append(v)
    return min(vals) if vals else -1e9


def _evaluate_many(
    *,
    tag_prefix: str,
    exp_name_for_ckpt: str,
    steps_to_eval: list[int],
    snapshot_dirs: list[str],
    final_stop: int,
    gpu: str,
    base_metrics: dict,
) -> list[dict]:
    rows: list[dict] = []
    for st in steps_to_eval:
        ckpt = find_ckpt_for_step(exp_name_for_ckpt, st, snapshot_dirs)
        if not ckpt and st == final_stop:
            final_name = f"{exp_name_for_ckpt}{final_stop}.pth"
            final_paths = [str(Path(d) / final_name) for d in snapshot_dirs]
            final_paths = [p for p in final_paths if os.path.exists(p)]
            ckpt = final_paths[0] if final_paths else None
        if not ckpt:
            continue
        m = eval_ckpt(f"{tag_prefix}@{st}", ckpt, gpu)
        rows.append(
            {
                "exp": f"{tag_prefix}@{st}",
                "ckpt": ckpt,
                "FZ": m.get("FZ"),
                "FDD": m.get("FDD"),
                "FD": m.get("FD"),
                "Lindau": m.get("Lindau"),
                "dFZ": _delta(m.get("FZ"), base_metrics.get("FZ")),
                "dFDD": _delta(m.get("FDD"), base_metrics.get("FDD")),
                "dFD": _delta(m.get("FD"), base_metrics.get("FD")),
                "dLindau": _delta(m.get("Lindau"), base_metrics.get("Lindau")),
            }
        )
    return rows


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

    # Robust, line-based parse: find an "Evaluation on ..." header then the next "mIoU:".
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    miou_re = re.compile(r"mIoU:\s*([0-9]+(?:\.[0-9]+)?)", flags=re.IGNORECASE)

    header_aliases: dict[str, list[re.Pattern]] = {
        "FZ": [re.compile(r"^Evaluation\s+on\s+Foggy\s+Zurich\b", flags=re.IGNORECASE)],
        "FDD": [re.compile(r"^Evaluation\s+on\s+Foggy\s+Driving\s+Dense\b", flags=re.IGNORECASE)],
        "FD": [re.compile(r"^Evaluation\s+on\s+Foggy\s+Driving\b", flags=re.IGNORECASE)],
        "Lindau": [
            re.compile(r"^Evaluation\s+on\s+Cityscapes\s+lindau\b", flags=re.IGNORECASE),
            re.compile(r"^Evaluation\s+on\s+Cityscapes\s+lindau\s+40\b", flags=re.IGNORECASE),
        ],
    }

    metrics: dict[str, float | None] = {"FZ": None, "FDD": None, "FD": None, "Lindau": None}
    for i, line in enumerate(lines):
        for key, patterns in header_aliases.items():
            if any(p.search(line) for p in patterns):
                # Search forward a few lines for the first mIoU value
                for j in range(i + 1, min(i + 8, len(lines))):
                    m = miou_re.search(lines[j])
                    if m:
                        metrics[key] = float(m.group(1))
                        break

    # If parsing failed, print a compact snippet to diagnose why (nan mIoU / all skipped / dataset missing).
    if all(v is None for v in metrics.values()):
        tail = "\n".join(lines[-60:])
        print("\n[PARSE-DEBUG] Could not find any mIoU lines in evaluate output.")
        # Common signals
        skip_lines = [ln for ln in lines if ln.lower().startswith("skipping:")]
        if skip_lines:
            print(f"[PARSE-DEBUG] Found {len(skip_lines)} 'Skipping:' lines (likely GT/pred size mismatch for all images).")
            print("[PARSE-DEBUG] Example:")
            print(skip_lines[0])
        missing_lines = [ln for ln in lines if "dataset not available" in ln.lower() or "missing" in ln.lower()]
        if missing_lines:
            print("[PARSE-DEBUG] Dataset/missing-file messages detected.")
            print("[PARSE-DEBUG] Example:")
            print(missing_lines[0])
        miou_lines = [ln for ln in lines if "miou" in ln.lower()]
        if miou_lines:
            print("[PARSE-DEBUG] Lines containing 'mIoU' exist but may be 'nan' or unexpected format.")
            print("[PARSE-DEBUG] Example:")
            print(miou_lines[0])
        print("\n[PARSE-DEBUG] Evaluate output tail (last ~60 lines):")
        print(tail)

    return metrics


def _build_train_cmd(
    *,
    train_script: str,
    exp_name: str,
    base_ckpt: str,
    gpu: str,
    num_steps: int,
    num_steps_stop: int,
    backbone_lr_mult: str,
    head_lr_mult: str,
    lambda_cl: str,
    cl_warmup_steps: str,
    cl_temp: str,
    cl_head_lr: str,
    freeze_fogpass_steps: str,
    fpf_lr_mult: str,
    save_pred_every_early: str,
    save_pred_early_until: str,
    amp: str,
    finetune: bool,
    freeze_bn: bool,
    save_dir: str,
    extra_tokens: list[str],
) -> list[str]:
    cmd = [
        "python",
        train_script,
        "--modeltrain",
        "train",
        "--file-name",
        exp_name,
        "--restore-from",
        base_ckpt,
        "--restore-from-fogpass",
        base_ckpt,
        "--num-steps",
        str(num_steps),
        "--num-steps-stop",
        str(num_steps_stop),
        "--backbone-lr-mult",
        str(backbone_lr_mult),
        "--head-lr-mult",
        str(head_lr_mult),
        "--lambda-cl",
        str(lambda_cl),
        "--cl-warmup-steps",
        str(cl_warmup_steps),
        "--cl-temp",
        str(cl_temp),
        "--cl-head-lr",
        str(cl_head_lr),
        "--freeze-fogpass-steps",
        str(freeze_fogpass_steps),
        "--fpf-lr-mult",
        str(fpf_lr_mult),
        "--save-pred-every-early",
        str(save_pred_every_early),
        "--save-pred-early-until",
        str(save_pred_early_until),
        "--amp",
        str(amp),
        "--gpu",
        str(gpu),
    ]
    if finetune:
        cmd.append("--finetune")
    if freeze_bn:
        cmd.append("--freeze-bn")
    if save_dir:
        cmd += ["--save-dir", save_dir]
    cmd += extra_tokens
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--repo-dir", default="/kaggle/working/CVDHD")
    ap.add_argument("--gpu", default="0")

    ap.add_argument("--base-ckpt", required=True)
    ap.add_argument("--train-script", default="main_fda.py")
    ap.add_argument("--exp-name", required=True)

    ap.add_argument(
        "--base-metrics",
        default="",
        help="Optional: skip baseline evaluation and use these metrics instead. Formats: 'FZ=48.41,FDD=48.93,FD=50.71,Lindau=64.75' or '48.41,48.93,50.71,64.75'",
    )

    ap.add_argument("--num-steps", type=int, default=2100)
    ap.add_argument("--num-steps-stop", type=int, default=2100)
    # Comma-separated list like: 200,800,2000. If provided with no value, means: no step eval.
    ap.add_argument("--steps", nargs="?", const="", default="200,800,2000")

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

    # Safe quick run: tries hard to avoid mIoU drop (good for debugging before long runs)
    ap.add_argument("--safe", action="store_true", help="Run a conservative short finetune and compare to baseline")
    ap.add_argument("--safe-steps", type=int, default=100, help="Train steps for --safe mode")
    ap.add_argument(
        "--safe-fda-beta",
        default="",
        help="Optional: if using main_fda.py in --safe mode, set this to keep FDA enabled (e.g. 0.005). Empty => force beta=0.",
    )

    # Ranking mode for picking the 'best' checkpoint/variant
    ap.add_argument(
        "--rank-mode",
        default="fdd",
        choices=["fdd", "mean", "min"],
        help="How to select best checkpoint/variant: fdd (default), mean (balanced avg delta), min (balanced worst-case delta)",
    )

    # Ablation sweep: runs several short finetunes and compares deltas vs baseline
    ap.add_argument("--sweep", action="store_true", help="Run a small ablation sweep after baseline eval")
    ap.add_argument("--sweep-num-steps", type=int, default=10, help="Train steps per variant when --sweep is set")
    ap.add_argument("--sweep-num-steps-stop", type=int, default=10, help="Stop steps per variant when --sweep is set")

    # Auto-improve: try a small set of recipes and pick best by --rank-mode
    ap.add_argument("--auto-improve", action="store_true", help="Try ProtoCL + Boundary (+ tiny FDA) and pick best checkpoint by --rank-mode")
    ap.add_argument("--auto-steps", type=int, default=200, help="Train steps per recipe in --auto-improve")
    ap.add_argument("--auto-snapshot-every", type=int, default=20, help="Snapshot interval during --auto-improve")
    ap.add_argument("--auto-include-fda", action="store_true", help="Also try tiny FDA betas in --auto-improve")

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

    provided_base = parse_base_metrics(str(getattr(args, "base_metrics", "")))
    if provided_base is not None:
        base_metrics = {"FZ": None, "FDD": None, "FD": None, "Lindau": None}
        base_metrics.update(provided_base)
        print("\n[2] Baseline evaluation: skipped (using --base-metrics)")
        print(base_metrics)
    else:
        print("\n[2] Baseline evaluation (before finetune):")
        base_metrics = eval_ckpt(f"{args.exp_name}_BASE", args.base_ckpt, args.gpu)
        print(base_metrics)

    # Auto-improve mode: run a small set of conservative recipes and pick best by rank-mode.
    if bool(getattr(args, "auto_improve", False)):
        auto_steps = int(getattr(args, "auto_steps", 200))
        auto_stop = auto_steps
        every = int(getattr(args, "auto_snapshot_every", 20))
        every = max(5, every)

        # Use conservative LR to reduce risk of collapse; we rely on checkpoint selection.
        auto_backbone_mult = "0.001"
        auto_head_mult = "0.002"
        auto_snapshot_until = str(auto_steps + 1)

        # Keep BN frozen and keep FogPass frozen for these short runs.
        base_common_tokens = args.train_extra.strip().split() if args.train_extra.strip() else []
        common_kwargs = {
            "gpu": args.gpu,
            "num_steps": auto_steps,
            "num_steps_stop": auto_stop,
            "backbone_lr_mult": auto_backbone_mult,
            "head_lr_mult": auto_head_mult,
            "lambda_cl": str(args.lambda_cl),
            "cl_warmup_steps": str(args.cl_warmup_steps),
            "cl_temp": str(args.cl_temp),
            "cl_head_lr": str(args.cl_head_lr),
            "freeze_fogpass_steps": str(max(2000, auto_steps)),
            "fpf_lr_mult": str(args.fpf_lr_mult),
            "save_pred_every_early": str(every),
            "save_pred_early_until": auto_snapshot_until,
            "amp": str(args.amp),
            "finetune": bool(args.finetune),
            "freeze_bn": True,
            "save_dir": str(args.save_dir),
        }

        # Define recipes (small and fairly safe). These are not guaranteed to improve,
        # but the script will find the best checkpoint among them if any exists.
        recipes: list[dict] = []

        # ProtoCL: start gentle (lower proto-weight first)
        recipes.append(
            {
                "name": "PROTO_w0.01",
                "train_script": "main_proto_cl.py",
                "extra_tokens": base_common_tokens
                + [
                    "--proto-weight",
                    "0.01",
                    "--proto-temp",
                    "0.1",
                    "--proto-momentum",
                    "0.99",
                    "--proto-init-iters",
                    "80",
                ],
            }
        )
        recipes.append(
            {
                "name": "PROTO_w0.05",
                "train_script": "main_proto_cl.py",
                "extra_tokens": base_common_tokens
                + [
                    "--proto-weight",
                    "0.05",
                    "--proto-temp",
                    "0.1",
                    "--proto-momentum",
                    "0.99",
                    "--proto-init-iters",
                    "80",
                ],
            }
        )

        # Boundary: use short warmup for short runs; low weight to avoid hurting fog sets.
        recipes.append(
            {
                "name": "BND_w0.10",
                "train_script": "main_boundary.py",
                "extra_tokens": base_common_tokens
                + [
                    "--boundary-weight",
                    "0.10",
                    "--boundary-warmup-steps",
                    str(min(80, max(20, auto_steps // 3))),
                    "--boundary-lr",
                    "0.0005",
                ],
            }
        )
        recipes.append(
            {
                "name": "BND_w0.25",
                "train_script": "main_boundary.py",
                "extra_tokens": base_common_tokens
                + [
                    "--boundary-weight",
                    "0.25",
                    "--boundary-warmup-steps",
                    str(min(80, max(20, auto_steps // 3))),
                    "--boundary-lr",
                    "0.0005",
                ],
            }
        )

        # Optional: tiny FDA (based on your results, avoid large beta)
        if bool(getattr(args, "auto_include_fda", False)):
            for beta in ("0.0002", "0.0005"):
                recipes.append(
                    {
                        "name": f"FDA_b{beta}",
                        "train_script": "main_fda.py",
                        "extra_tokens": _replace_or_add_flag(list(base_common_tokens), "--fda-beta", beta),
                    }
                )

        # Determine steps to evaluate
        steps_to_eval = list(range(every, auto_steps + 1, every))
        if auto_steps not in steps_to_eval:
            steps_to_eval.append(auto_steps)

        all_rows: list[dict] = []
        best = None
        mode = str(getattr(args, "rank_mode", "fdd"))

        for rec in recipes:
            name = rec["name"]
            train_script = rec["train_script"]
            exp = f"{args.exp_name}_{name}"
            cmd = _build_train_cmd(
                train_script=train_script,
                exp_name=exp,
                base_ckpt=args.base_ckpt,
                extra_tokens=list(rec["extra_tokens"]),
                **common_kwargs,
            )

            run(cmd, title=f"AUTO TRAIN {exp} ({train_script})")

            rows = _evaluate_many(
                tag_prefix=name,
                exp_name_for_ckpt=exp,
                steps_to_eval=steps_to_eval,
                snapshot_dirs=snapshot_dirs,
                final_stop=auto_stop,
                gpu=str(args.gpu),
                base_metrics=base_metrics,
            )

            for r in rows:
                # Add recipe label for easier tables
                r["recipe"] = name
                all_rows.append(r)
                score = _rank_score(r, mode=mode)
                if best is None or score > best[0]:
                    best = (score, r)

        print("\n" + "=" * 10 + " AUTO SUMMARY " + "=" * 10)
        base_row = {
            "exp": "BASE",
            "FZ": base_metrics.get("FZ"),
            "FDD": base_metrics.get("FDD"),
            "FD": base_metrics.get("FD"),
            "Lindau": base_metrics.get("Lindau"),
            "dFZ": 0.0,
            "dFDD": 0.0,
            "dFD": 0.0,
            "dLindau": 0.0,
        }
        # Print a compact table: exp (recipe@step) + metrics
        rows_out = [base_row]
        for r in all_rows:
            rows_out.append(
                {
                    "exp": r.get("exp"),
                    "FZ": r.get("FZ"),
                    "FDD": r.get("FDD"),
                    "FD": r.get("FD"),
                    "Lindau": r.get("Lindau"),
                    "dFZ": r.get("dFZ"),
                    "dFDD": r.get("dFDD"),
                    "dFD": r.get("dFD"),
                    "dLindau": r.get("dLindau"),
                }
            )
        _print_results_table(rows_out)

        if best is None:
            print("\n[AUTO] No checkpoints were evaluated (missing snapshots).")
            return 0

        b = best[1]
        print(
            f"\n[AUTO] Best by {mode}: {b.get('exp')}\n"
            f"  ckpt: {b.get('ckpt')}\n"
            f"  FZ={_fmt(b.get('FZ'))}  FDD={_fmt(b.get('FDD'))}  FD={_fmt(b.get('FD'))}  Lindau={_fmt(b.get('Lindau'))}\n"
            f"  dFZ={_fmt(b.get('dFZ'))}  dFDD={_fmt(b.get('dFDD'))}  dFD={_fmt(b.get('dFD'))}  dLindau={_fmt(b.get('dLindau'))}"
        )

        # Honest status: did we actually improve under the chosen criterion?
        if mode.lower() == "min":
            improved = _min_delta(b) > 0
        elif mode.lower() == "mean":
            improved = _mean_delta(b) > 0
        else:
            improved = isinstance(b.get("dFDD"), float) and float(b.get("dFDD")) > 0

        if improved:
            print("\n[AUTO] Improvement found under selected criterion.")
        else:
            print("\n[AUTO] No improvement found under selected criterion in this search set.")
            print("[AUTO] You can increase --auto-steps or enable --auto-include-fda to expand the search.")

        print("\nDone.")
        return 0

    # Safe short finetune (aim: no drop)
    if bool(getattr(args, "safe", False)):
        safe_steps = int(getattr(args, "safe_steps", 100))
        safe_stop = safe_steps

        # Conservative multipliers to reduce drift.
        safe_backbone_mult = "0.001"
        safe_head_mult = "0.002"

        # Save dense snapshots so we can pick the best checkpoint within these few steps.
        # (Ending checkpoint is often slightly worse even if an earlier one is better.)
        safe_snapshot_every = "20" if safe_steps >= 20 else str(max(1, safe_steps // 2))
        safe_snapshot_until = str(safe_steps + 1)

        safe_extra_tokens = args.train_extra.strip().split() if args.train_extra.strip() else []
        # SAFE defaults to disabling FDA to minimize drift, unless the user explicitly requests a beta.
        if "fda" in args.train_script.lower():
            safe_beta = str(getattr(args, "safe_fda_beta", "") or "").strip()
            if safe_beta:
                safe_extra_tokens = _replace_or_add_flag(safe_extra_tokens, "--fda-beta", safe_beta)
            else:
                safe_extra_tokens = _replace_or_add_flag(safe_extra_tokens, "--fda-beta", "0")

        # Keep BN frozen and fogpass frozen throughout this short run
        safe_cmd = _build_train_cmd(
            train_script=args.train_script,
            exp_name=f"{args.exp_name}_SAFE",
            base_ckpt=args.base_ckpt,
            gpu=args.gpu,
            num_steps=safe_steps,
            num_steps_stop=safe_stop,
            backbone_lr_mult=safe_backbone_mult,
            head_lr_mult=safe_head_mult,
            lambda_cl=str(args.lambda_cl),
            cl_warmup_steps=str(args.cl_warmup_steps),
            cl_temp=str(args.cl_temp),
            cl_head_lr=str(args.cl_head_lr),
            freeze_fogpass_steps=str(max(2000, safe_steps)),
            fpf_lr_mult=str(args.fpf_lr_mult),
            save_pred_every_early=safe_snapshot_every,
            save_pred_early_until=safe_snapshot_until,
            amp=str(args.amp),
            finetune=bool(args.finetune),
            freeze_bn=True,
            save_dir=str(args.save_dir),
            extra_tokens=safe_extra_tokens,
        )

        run(safe_cmd, title=f"SAFE TRAIN {args.exp_name}_SAFE ({safe_steps} steps)")

        # Evaluate multiple checkpoints within safe run (including final)
        safe_rows: list[dict] = []
        safe_steps_to_eval: list[int] = []
        try:
            every = int(safe_snapshot_every)
            if every > 0:
                safe_steps_to_eval = list(range(every, safe_steps + 1, every))
        except Exception:
            safe_steps_to_eval = []
        if safe_steps not in safe_steps_to_eval:
            safe_steps_to_eval.append(safe_steps)

        for st in safe_steps_to_eval:
            # Prefer FIFO snapshots when available; fall back to final checkpoint.
            ckpt = find_ckpt_for_step(f"{args.exp_name}_SAFE", st, snapshot_dirs)
            if not ckpt and st == safe_steps:
                final_name = f"{args.exp_name}_SAFE{safe_stop}.pth"
                final_paths = [str(Path(d) / final_name) for d in snapshot_dirs]
                final_paths = [p for p in final_paths if os.path.exists(p)]
                ckpt = final_paths[0] if final_paths else None
            if not ckpt:
                continue
            m = eval_ckpt(f"{args.exp_name}_SAFE_step{st}", ckpt, args.gpu)
            safe_rows.append(
                {
                    "exp": f"SAFE@{st}",
                    "FZ": m.get("FZ"),
                    "FDD": m.get("FDD"),
                    "FD": m.get("FD"),
                    "Lindau": m.get("Lindau"),
                    "dFZ": _delta(m.get("FZ"), base_metrics.get("FZ")),
                    "dFDD": _delta(m.get("FDD"), base_metrics.get("FDD")),
                    "dFD": _delta(m.get("FD"), base_metrics.get("FD")),
                    "dLindau": _delta(m.get("Lindau"), base_metrics.get("Lindau")),
                }
            )

        # Choose best checkpoint according to rank-mode (fdd/mean/min)
        best = None
        for r in safe_rows:
            score = _rank_score(r, mode=str(getattr(args, "rank_mode", "fdd")))
            if best is None or score > best[0]:
                best = (score, r)
        print("\n" + "=" * 10 + " SAFE SUMMARY " + "=" * 10)
        base_row = {
            "exp": "BASE",
            "FZ": base_metrics.get("FZ"),
            "FDD": base_metrics.get("FDD"),
            "FD": base_metrics.get("FD"),
            "Lindau": base_metrics.get("Lindau"),
            "dFZ": 0.0,
            "dFDD": 0.0,
            "dFD": 0.0,
            "dLindau": 0.0,
        }
        rows_out = [base_row] + safe_rows
        _print_results_table(rows_out)
        if best is not None:
            b = best[1]
            mode = str(getattr(args, "rank_mode", "fdd")).lower()
            print(
                f"\nBest SAFE checkpoint by {mode}: {b['exp']}  "
                f"FZ={_fmt(b.get('FZ'))}  FDD={_fmt(b.get('FDD'))}  FD={_fmt(b.get('FD'))}  Lindau={_fmt(b.get('Lindau'))}"
            )
        print("\nDone.")
        return 0

    # Optional ablation sweep (recommended to diagnose 'why mIoU drops')
    if bool(getattr(args, "sweep", False)):
        base_extra_tokens = args.train_extra.strip().split() if args.train_extra.strip() else []

        # Ensure we have an explicit fda-beta if the user is using main_fda.py.
        # train_config.py defaults may change; keep sweep explicit.
        if "fda" in args.train_script.lower():
            base_extra_tokens = _replace_or_add_flag(base_extra_tokens, "--fda-beta", "0.01")

        variants: list[tuple[str, dict]] = []

        # Variant 1: disable FDA (beta=0 => fda_source_to_target returns original)
        if "fda" in args.train_script.lower():
            variants.append(("FDA_beta0", {"extra_tokens": _replace_or_add_flag(base_extra_tokens, "--fda-beta", "0")}))

        # Variant 2: keep FDA but do NOT freeze BN
        variants.append(("noFreezeBN", {"freeze_bn": False, "extra_tokens": list(base_extra_tokens)}))

        # Variant 3: lower LR multipliers
        variants.append(("lowLR", {"backbone_lr_mult": "0.005", "head_lr_mult": "0.01", "extra_tokens": list(base_extra_tokens)}))

        # Variant 4: do not freeze fogpass
        variants.append(("noFreezeFPF", {"freeze_fogpass_steps": "0", "extra_tokens": list(base_extra_tokens)}))

        rows: list[dict] = []
        rows.append(
            {
                "exp": "BASE",
                "FZ": base_metrics.get("FZ"),
                "FDD": base_metrics.get("FDD"),
                "FD": base_metrics.get("FD"),
                "Lindau": base_metrics.get("Lindau"),
                "dFZ": 0.0,
                "dFDD": 0.0,
                "dFD": 0.0,
                "dLindau": 0.0,
            }
        )

        for suffix, cfg in variants:
            v_exp = f"{args.exp_name}_{suffix}"
            v_num_steps = int(getattr(args, "sweep_num_steps", 10))
            v_num_stop = int(getattr(args, "sweep_num_steps_stop", 10))

            v_cmd = _build_train_cmd(
                train_script=args.train_script,
                exp_name=v_exp,
                base_ckpt=args.base_ckpt,
                gpu=args.gpu,
                num_steps=v_num_steps,
                num_steps_stop=v_num_stop,
                backbone_lr_mult=str(cfg.get("backbone_lr_mult", args.backbone_lr_mult)),
                head_lr_mult=str(cfg.get("head_lr_mult", args.head_lr_mult)),
                lambda_cl=str(args.lambda_cl),
                cl_warmup_steps=str(args.cl_warmup_steps),
                cl_temp=str(args.cl_temp),
                cl_head_lr=str(args.cl_head_lr),
                freeze_fogpass_steps=str(cfg.get("freeze_fogpass_steps", args.freeze_fogpass_steps)),
                fpf_lr_mult=str(args.fpf_lr_mult),
                save_pred_every_early=str(args.save_pred_every_early),
                save_pred_early_until=str(args.save_pred_early_until),
                amp=str(args.amp),
                finetune=bool(args.finetune),
                freeze_bn=bool(cfg.get("freeze_bn", args.freeze_bn)),
                save_dir=str(args.save_dir),
                extra_tokens=list(cfg.get("extra_tokens", base_extra_tokens)),
            )

            run(v_cmd, title=f"SWEEP TRAIN {v_exp} ({suffix})")

            # Evaluate final checkpoint (preferred) else fall back to latest FIFO snapshot
            final_name = f"{v_exp}{v_num_stop}.pth"
            final_paths = [str(Path(d) / final_name) for d in snapshot_dirs]
            final_paths = [p for p in final_paths if os.path.exists(p)]
            ckpt = final_paths[0] if final_paths else find_latest_ckpt(v_exp, snapshot_dirs)
            if not ckpt:
                print(f"[SWEEP-WARN] No checkpoint found for {v_exp}")
                rows.append({"exp": suffix, "FZ": None, "FDD": None, "FD": None, "Lindau": None, "dFZ": None, "dFDD": None, "dFD": None, "dLindau": None})
                continue

            v_metrics = eval_ckpt(f"{v_exp}_FINAL", ckpt, args.gpu)
            rows.append(
                {
                    "exp": suffix,
                    "FZ": v_metrics.get("FZ"),
                    "FDD": v_metrics.get("FDD"),
                    "FD": v_metrics.get("FD"),
                    "Lindau": v_metrics.get("Lindau"),
                    "dFZ": _delta(v_metrics.get("FZ"), base_metrics.get("FZ")),
                    "dFDD": _delta(v_metrics.get("FDD"), base_metrics.get("FDD")),
                    "dFD": _delta(v_metrics.get("FD"), base_metrics.get("FD")),
                    "dLindau": _delta(v_metrics.get("Lindau"), base_metrics.get("Lindau")),
                }
            )

        print("\n" + "=" * 10 + " SWEEP SUMMARY " + "=" * 10)
        _print_results_table(rows)
        # Report best sweep variant by the chosen rank-mode
        best_v = None
        for r in rows:
            if r.get("exp") == "BASE":
                continue
            score = _rank_score(r, mode=str(getattr(args, "rank_mode", "fdd")))
            if best_v is None or score > best_v[0]:
                best_v = (score, r)
        if best_v is not None:
            b = best_v[1]
            mode = str(getattr(args, "rank_mode", "fdd")).lower()
            print(
                f"\nBest SWEEP variant by {mode}: {b['exp']}  "
                f"dFZ={_fmt(b.get('dFZ'))}  dFDD={_fmt(b.get('dFDD'))}  dFD={_fmt(b.get('dFD'))}  dLindau={_fmt(b.get('dLindau'))}"
            )
        print("\nDone.")
        return 0

    # Train (single run mode)
    extra = args.train_extra.strip().split() if args.train_extra.strip() else []
    train_cmd = _build_train_cmd(
        train_script=args.train_script,
        exp_name=args.exp_name,
        base_ckpt=args.base_ckpt,
        gpu=args.gpu,
        num_steps=int(args.num_steps),
        num_steps_stop=int(args.num_steps_stop),
        backbone_lr_mult=str(args.backbone_lr_mult),
        head_lr_mult=str(args.head_lr_mult),
        lambda_cl=str(args.lambda_cl),
        cl_warmup_steps=str(args.cl_warmup_steps),
        cl_temp=str(args.cl_temp),
        cl_head_lr=str(args.cl_head_lr),
        freeze_fogpass_steps=str(args.freeze_fogpass_steps),
        fpf_lr_mult=str(args.fpf_lr_mult),
        save_pred_every_early=str(args.save_pred_every_early),
        save_pred_early_until=str(args.save_pred_early_until),
        amp=str(args.amp),
        finetune=bool(args.finetune),
        freeze_bn=bool(args.freeze_bn),
        save_dir=str(args.save_dir),
        extra_tokens=extra,
    )

    run(train_cmd, title=f"TRAIN {args.exp_name} ({args.train_script})")

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
