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
import shutil
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


def find_all_fifo_ckpts_by_step(exp_name: str, snapshot_dirs: list[str]) -> dict[int, str]:
    """Return a map: step -> checkpoint path for FIFO snapshots.

    If multiple files match the same step (e.g. reruns), keep the most recently modified.
    """

    patterns = [str(Path(d) / f"*{exp_name}*FIFO*.pth") for d in snapshot_dirs]
    matches: list[Path] = []
    for pat in patterns:
        matches.extend(Path(p) for p in glob.glob(pat))

    step_re = re.compile(r"FIFO(\d+)\.pth$", flags=re.IGNORECASE)
    best: dict[int, Path] = {}
    for p in matches:
        m = step_re.search(str(p).replace("\\", "/"))
        if not m:
            continue
        step = int(m.group(1))
        prev = best.get(step)
        if prev is None:
            best[step] = p
        else:
            try:
                if p.stat().st_mtime > prev.stat().st_mtime:
                    best[step] = p
            except OSError:
                # If stat fails, keep the existing one.
                pass

    return {st: str(path) for st, path in best.items()}


def _resolve_steps_to_eval(
    requested_steps: list[int],
    available_ckpts: dict[int, str],
    *,
    steps_mode: str,
    nearest_max_diff: int,
    dedup: bool,
) -> tuple[list[int], list[int], dict[int, int]]:
    """Resolve which steps to evaluate.

    Returns: (steps_to_eval, missing_requested_steps, requested_to_actual_map)
    """

    mode = (steps_mode or "exact").strip().lower()
    avail_steps = sorted(available_ckpts.keys())
    if not avail_steps:
        return ([], requested_steps[:], {})

    if mode == "available":
        return (avail_steps, [], {})

    missing: list[int] = []
    mapping: dict[int, int] = {}
    out: list[int] = []

    def add_step(step: int) -> None:
        if dedup and step in out:
            return
        out.append(step)

    if mode == "nearest":
        for r in requested_steps:
            if r in available_ckpts:
                mapping[r] = r
                add_step(r)
                continue
            # Find nearest available step
            nearest = min(avail_steps, key=lambda s: abs(s - r))
            if abs(nearest - r) <= int(nearest_max_diff):
                mapping[r] = nearest
                add_step(nearest)
            else:
                missing.append(r)
        out.sort()
        return (out, missing, mapping)

    # default: exact
    for r in requested_steps:
        if r in available_ckpts:
            add_step(r)
        else:
            missing.append(r)
    out.sort()
    return (out, missing, {})


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


def _pick_best(rows: list[dict], *, mode: str) -> dict | None:
    best = None
    for r in rows:
        score = _rank_score(r, mode=mode)
        if best is None or score > best[0]:
            best = (score, r)
    return best[1] if best is not None else None


def _copy_best_checkpoint(ckpt_path: str, out_path: str) -> str:
    os.makedirs(str(Path(out_path).parent), exist_ok=True)
    shutil.copy2(ckpt_path, out_path)
    return out_path


def _variant_label(cfg: dict) -> str:
    # Small stable label for printing.
    parts: list[str] = []
    for k in (
        "backbone_lr_mult",
        "head_lr_mult",
        "boundary_weight",
        "boundary_lr",
        "boundary_warmup_steps",
        "freeze_fogpass_steps",
    ):
        if k in cfg:
            parts.append(f"{k}={cfg[k]}")
    return ",".join(parts)


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

    ap.add_argument(
        "--steps-mode",
        default="exact",
        choices=["exact", "nearest", "available"],
        help="How to resolve --steps when snapshots are missing: exact (warn missing), nearest (map to nearest FIFO snapshot), available (ignore --steps and eval all available FIFO steps)",
    )
    ap.add_argument(
        "--steps-nearest-max-diff",
        type=int,
        default=0,
        help="When --steps-mode=nearest, only map requested steps to a snapshot if |requested-available| <= this value. Default 0 (disabled).",
    )
    ap.add_argument(
        "--steps-dedup",
        action="store_true",
        default=True,
        help="Deduplicate resolved steps (useful with --steps-mode=nearest).",
    )

    ap.add_argument("--save-dir", default="")
    ap.add_argument("--snapshot-dir", default="./snapshots/FIFO_model")

    ap.add_argument(
        "--scratch-dir",
        default="",
        help="Optional: store intermediate snapshots here (e.g. /kaggle/temp/snapshots/FIFO_model). Best checkpoints are copied to --save-dir.",
    )

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
        "--safe-snapshot-every",
        type=int,
        default=50,
        help="Snapshot/evaluate interval during --safe (default: 50 to reduce eval time)",
    )
    ap.add_argument(
        "--safe-batch-size",
        type=int,
        default=0,
        help="Optional: override --batch-size during --safe. 0 => auto (Boundary defaults to 1 to avoid OOM).",
    )
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

    # Boundary quick sweep: many short Boundary variants with frequent eval
    ap.add_argument(
        "--boundary-sweep",
        action="store_true",
        help="Run several Boundary hyperparam variants for a short run (default 400 steps) and pick best by --rank-mode",
    )
    ap.add_argument("--boundary-sweep-steps", type=int, default=400, help="Train steps for --boundary-sweep")
    ap.add_argument("--boundary-sweep-snapshot-every", type=int, default=50, help="Snapshot/eval interval for --boundary-sweep")
    ap.add_argument(
        "--boundary-sweep-batch-size",
        type=int,
        default=1,
        help="Batch size for Boundary sweep (default 1 to avoid OOM; repo default is 4)",
    )
    ap.add_argument(
        "--boundary-sweep-num-workers",
        type=int,
        default=2,
        help="num-workers for Boundary sweep (default 2; lower can reduce RAM pressure on Kaggle)",
    )
    ap.add_argument(
        "--boundary-sweep-train-script",
        default="main_boundary.py",
        help="Trainer entrypoint used for --boundary-sweep (default: main_boundary.py)",
    )

    # Auto-improve: try a small set of recipes and pick best by --rank-mode
    ap.add_argument("--auto-improve", action="store_true", help="Try ProtoCL + Boundary (+ tiny FDA) and pick best checkpoint by --rank-mode")
    ap.add_argument("--auto-steps", type=int, default=200, help="Train steps per recipe in --auto-improve")
    ap.add_argument("--auto-snapshot-every", type=int, default=20, help="Snapshot interval during --auto-improve")
    ap.add_argument("--auto-include-fda", action="store_true", help="Also try tiny FDA betas in --auto-improve")

    # Combo: sequentially chain methods and carry forward the best checkpoint
    ap.add_argument("--combo", action="store_true", help="Chain Boundary -> ProtoCL (optional tiny FDA), selecting best checkpoint after each stage")
    ap.add_argument("--combo-boundary-steps", type=int, default=400, help="Steps for Boundary stage in --combo")
    ap.add_argument("--combo-proto-steps", type=int, default=200, help="Steps for ProtoCL stage in --combo")
    ap.add_argument("--combo-fda-steps", type=int, default=0, help="Steps for optional FDA stage in --combo (0 disables)")
    ap.add_argument("--combo-proto-weight", default="0.01", help="ProtoCL stage proto-weight")
    ap.add_argument("--combo-boundary-weight", default="0.10", help="Boundary stage boundary-weight")
    ap.add_argument("--combo-fda-beta", default="0.0002", help="FDA stage beta (very small recommended)")

    args = ap.parse_args()

    repo = Path(args.repo_dir)
    if not repo.exists():
        raise SystemExit(f"Repo not found: {repo}")
    os.chdir(repo)

    scratch_dir = str(getattr(args, "scratch_dir", "") or "").strip()

    # Train-save dir: where main_*.py writes snapshots/checkpoints.
    train_save_dir = scratch_dir if scratch_dir else str(args.save_dir)

    # Persistent dir: where we keep only best checkpoints.
    persist_save_dir = str(args.save_dir) if str(args.save_dir) else ""

    # Configure snapshot dirs to search
    snapshot_dirs: list[str] = []
    if scratch_dir:
        snapshot_dirs.append(scratch_dir)
    if args.save_dir:
        snapshot_dirs.append(args.save_dir)
    snapshot_dirs.append(args.snapshot_dir)
    # Kaggle common defaults
    snapshot_dirs.append("/kaggle/working/snapshots/FIFO_model")
    snapshot_dirs.append("/kaggle/temp/snapshots/FIFO_model")
    snapshot_dirs = [d for d in dict.fromkeys(snapshot_dirs) if d]

    print("\nSnapshot dirs (search order):")
    for d in snapshot_dirs:
        print(" -", d)

    if scratch_dir:
        print(f"\nScratch dir: {scratch_dir} (intermediate snapshots)")
    if persist_save_dir:
        print(f"Persistent save dir: {persist_save_dir} (best checkpoints)")

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

    # Combo mode: sequentially chain methods and carry forward best checkpoint.
    if bool(getattr(args, "combo", False)):
        mode = str(getattr(args, "rank_mode", "min"))
        stage_save_dir = str(train_save_dir) if str(train_save_dir) else str(snapshot_dirs[0])
        best_save_dir = str(persist_save_dir) if str(persist_save_dir) else stage_save_dir
        combo_tag = str(args.exp_name)

        def _run_stage(
            *,
            stage_name: str,
            train_script: str,
            base_ckpt: str,
            steps: int,
            extra_tokens: list[str],
            snapshot_every: int,
        ) -> tuple[str, dict | None, list[dict]]:
            stop = steps
            exp = f"{combo_tag}_{stage_name}"

            cmd = _build_train_cmd(
                train_script=train_script,
                exp_name=exp,
                base_ckpt=base_ckpt,
                gpu=str(args.gpu),
                num_steps=int(steps),
                num_steps_stop=int(stop),
                backbone_lr_mult="0.001",
                head_lr_mult="0.002",
                lambda_cl=str(args.lambda_cl),
                cl_warmup_steps=str(args.cl_warmup_steps),
                cl_temp=str(args.cl_temp),
                cl_head_lr=str(args.cl_head_lr),
                freeze_fogpass_steps=str(max(2000, steps)),
                fpf_lr_mult=str(args.fpf_lr_mult),
                save_pred_every_early=str(snapshot_every),
                save_pred_early_until=str(steps + 1),
                amp=str(args.amp),
                finetune=bool(args.finetune),
                freeze_bn=True,
                save_dir=stage_save_dir,
                extra_tokens=extra_tokens,
            )
            run(cmd, title=f"COMBO TRAIN {exp} ({train_script})")

            steps_to_eval = list(range(snapshot_every, steps + 1, snapshot_every))
            if steps not in steps_to_eval:
                steps_to_eval.append(steps)
            rows = _evaluate_many(
                tag_prefix=f"{stage_name}",
                exp_name_for_ckpt=exp,
                steps_to_eval=steps_to_eval,
                snapshot_dirs=snapshot_dirs,
                final_stop=stop,
                gpu=str(args.gpu),
                base_metrics=base_metrics,
            )
            best_row = _pick_best(rows, mode=mode)
            if best_row is None:
                return base_ckpt, None, rows

            # Copy best checkpoint to a stable filename for the next stage.
            out_best = str(Path(best_save_dir) / f"{combo_tag}_BEST_{stage_name}.pth")
            copied = _copy_best_checkpoint(str(best_row["ckpt"]), out_best)
            return copied, best_row, rows

        print("\n[3] COMBO: Boundary -> ProtoCL" + (" -> FDA" if int(getattr(args, "combo_fda_steps", 0)) > 0 else ""))

        # Stage 1: Boundary (memory heavy => force batch-size=1)
        bnd_steps = int(getattr(args, "combo_boundary_steps", 400))
        bnd_every = 20 if bnd_steps >= 40 else max(5, bnd_steps // 4)
        bnd_tokens = args.train_extra.strip().split() if args.train_extra.strip() else []
        bnd_tokens = _replace_or_add_flag(bnd_tokens, "--batch-size", "1")
        bnd_tokens = _replace_or_add_flag(bnd_tokens, "--num-workers", "2")
        bnd_tokens += [
            "--boundary-weight",
            str(getattr(args, "combo_boundary_weight", "0.10")),
            "--boundary-warmup-steps",
            str(min(80, max(20, bnd_steps // 3))),
            "--boundary-lr",
            "0.0005",
        ]

        current_ckpt = str(args.base_ckpt)
        all_rows: list[dict] = []

        current_ckpt, best_bnd, rows_bnd = _run_stage(
            stage_name="BND",
            train_script="main_boundary.py",
            base_ckpt=current_ckpt,
            steps=bnd_steps,
            extra_tokens=bnd_tokens,
            snapshot_every=bnd_every,
        )
        all_rows += rows_bnd

        # Stage 2: ProtoCL (gentle by default)
        proto_steps = int(getattr(args, "combo_proto_steps", 200))
        proto_every = 20 if proto_steps >= 40 else max(5, proto_steps // 4)
        proto_tokens = args.train_extra.strip().split() if args.train_extra.strip() else []
        proto_tokens += [
            "--proto-weight",
            str(getattr(args, "combo_proto_weight", "0.01")),
            "--proto-temp",
            "0.1",
            "--proto-momentum",
            "0.99",
            "--proto-init-iters",
            "80",
        ]

        current_ckpt, best_proto, rows_proto = _run_stage(
            stage_name="PROTO",
            train_script="main_proto_cl.py",
            base_ckpt=current_ckpt,
            steps=proto_steps,
            extra_tokens=proto_tokens,
            snapshot_every=proto_every,
        )
        all_rows += rows_proto

        # Optional Stage 3: tiny FDA (often risky; keep very small beta)
        fda_steps = int(getattr(args, "combo_fda_steps", 0))
        best_fda = None
        if fda_steps > 0:
            fda_every = 20 if fda_steps >= 40 else max(5, fda_steps // 4)
            fda_tokens = args.train_extra.strip().split() if args.train_extra.strip() else []
            fda_tokens = _replace_or_add_flag(fda_tokens, "--fda-beta", str(getattr(args, "combo_fda_beta", "0.0002")))
            current_ckpt, best_fda, rows_fda = _run_stage(
                stage_name="FDA",
                train_script="main_fda.py",
                base_ckpt=current_ckpt,
                steps=fda_steps,
                extra_tokens=fda_tokens,
                snapshot_every=fda_every,
            )
            all_rows += rows_fda

        # Pick best overall among all evaluated combo checkpoints
        best_overall = _pick_best(all_rows, mode=mode)

        print("\n" + "=" * 10 + " COMBO SUMMARY " + "=" * 10)
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

        if best_overall is not None:
            out_best = str(Path(best_save_dir) / f"{combo_tag}_BEST_OVERALL.pth")
            best_path = _copy_best_checkpoint(str(best_overall["ckpt"]), out_best)
            print(
                f"\n[COMBO] Best overall by {mode}: {best_overall.get('exp')}\n"
                f"  ckpt: {best_path}\n"
                f"  FZ={_fmt(best_overall.get('FZ'))}  FDD={_fmt(best_overall.get('FDD'))}  FD={_fmt(best_overall.get('FD'))}  Lindau={_fmt(best_overall.get('Lindau'))}\n"
                f"  dFZ={_fmt(best_overall.get('dFZ'))}  dFDD={_fmt(best_overall.get('dFDD'))}  dFD={_fmt(best_overall.get('dFD'))}  dLindau={_fmt(best_overall.get('dLindau'))}"
            )
        else:
            print("\n[COMBO] No checkpoints evaluated/found.")

        print("\nDone.")
        return 0

    # Boundary sweep mode: run multiple short boundary variants and rank by --rank-mode
    if bool(getattr(args, "boundary_sweep", False)):
        mode = str(getattr(args, "rank_mode", "min")).lower()
        steps = int(getattr(args, "boundary_sweep_steps", 400))
        snap_every = int(getattr(args, "boundary_sweep_snapshot_every", 50))
        train_script = str(getattr(args, "boundary_sweep_train_script", "main_boundary.py"))
        bsz = int(getattr(args, "boundary_sweep_batch_size", 1))
        nworkers = int(getattr(args, "boundary_sweep_num_workers", 2))

        # Ensure we actually get a FIFO{steps} snapshot (trainer saves snapshots on i_iter).
        # Running steps+1 means the loop reaches i_iter==steps.
        run_steps = steps + 1
        stop_steps = steps + 1

        stage_save_dir = str(train_save_dir) if str(train_save_dir) else str(snapshot_dirs[0])
        best_save_dir = str(persist_save_dir) if str(persist_save_dir) else stage_save_dir
        base_extra_tokens = args.train_extra.strip().split() if args.train_extra.strip() else []

        # A small, opinionated set of boundary variants aimed at stability (avoid big foggy drops).
        # If you want more/other variants later, we can expose this via a config file.
        variants: list[tuple[str, dict]] = [
            (
                "GENTLE_w0.10",
                {
                    "backbone_lr_mult": "0.001",
                    "head_lr_mult": "0.002",
                    "freeze_fogpass_steps": str(max(2000, stop_steps)),
                    "boundary_weight": "0.10",
                    "boundary_lr": "1e-4",
                    "boundary_warmup_steps": "0",
                },
            ),
            (
                "GENTLE_w0.20",
                {
                    "backbone_lr_mult": "0.001",
                    "head_lr_mult": "0.002",
                    "freeze_fogpass_steps": str(max(2000, stop_steps)),
                    "boundary_weight": "0.20",
                    "boundary_lr": "1e-4",
                    "boundary_warmup_steps": "0",
                },
            ),
            (
                "WARMUP100_w0.10",
                {
                    "backbone_lr_mult": "0.001",
                    "head_lr_mult": "0.002",
                    "freeze_fogpass_steps": str(max(2000, stop_steps)),
                    "boundary_weight": "0.10",
                    "boundary_lr": "1e-4",
                    "boundary_warmup_steps": "100",
                },
            ),
            (
                "LOWLR_w0.10",
                {
                    "backbone_lr_mult": "0.001",
                    "head_lr_mult": "0.002",
                    "freeze_fogpass_steps": str(max(2000, stop_steps)),
                    "boundary_weight": "0.10",
                    "boundary_lr": "5e-5",
                    "boundary_warmup_steps": "0",
                },
            ),
            (
                "HEADUP_w0.10",
                {
                    "backbone_lr_mult": "0.001",
                    "head_lr_mult": "0.005",
                    "freeze_fogpass_steps": str(max(2000, stop_steps)),
                    "boundary_weight": "0.10",
                    "boundary_lr": "1e-4",
                    "boundary_warmup_steps": "0",
                },
            ),
            (
                "FPF_AFTER200_w0.10",
                {
                    "backbone_lr_mult": "0.001",
                    "head_lr_mult": "0.002",
                    "freeze_fogpass_steps": "200",
                    "fpf_lr_mult": "0.05",
                    "boundary_weight": "0.10",
                    "boundary_lr": "1e-4",
                    "boundary_warmup_steps": "0",
                },
            ),
        ]

        steps_to_eval = list(range(snap_every, steps + 1, snap_every))
        if steps not in steps_to_eval:
            steps_to_eval.append(steps)

        all_variant_best: list[dict] = []
        overall_best: dict | None = None

        print("\n" + "=" * 10 + f" BOUNDARY SWEEP ({steps} steps, eval every {snap_every}) " + "=" * 10)
        for suffix, cfg in variants:
            v_exp = f"{args.exp_name}_BND_{suffix}"
            # Only pass truly extra flags here; avoid duplicating flags already set by _build_train_cmd.
            cfg_tokens: list[str] = ["--batch-size", str(bsz), "--iter-size", "1", "--num-workers", str(nworkers)]
            if "boundary_weight" in cfg:
                cfg_tokens += ["--boundary-weight", str(cfg["boundary_weight"])]
            if "boundary_lr" in cfg:
                cfg_tokens += ["--boundary-lr", str(cfg["boundary_lr"])]
            if "boundary_warmup_steps" in cfg:
                cfg_tokens += ["--boundary-warmup-steps", str(cfg["boundary_warmup_steps"])]

            v_cmd = _build_train_cmd(
                train_script=train_script,
                exp_name=v_exp,
                base_ckpt=args.base_ckpt,
                gpu=args.gpu,
                num_steps=int(run_steps),
                num_steps_stop=int(stop_steps),
                backbone_lr_mult=str(cfg.get("backbone_lr_mult", args.backbone_lr_mult)),
                head_lr_mult=str(cfg.get("head_lr_mult", args.head_lr_mult)),
                lambda_cl=str(args.lambda_cl),
                cl_warmup_steps=str(args.cl_warmup_steps),
                cl_temp=str(args.cl_temp),
                cl_head_lr=str(args.cl_head_lr),
                freeze_fogpass_steps=str(cfg.get("freeze_fogpass_steps", args.freeze_fogpass_steps)),
                fpf_lr_mult=str(cfg.get("fpf_lr_mult", args.fpf_lr_mult)),
                save_pred_every_early=str(snap_every),
                save_pred_early_until=str(run_steps + 1),
                amp=str(args.amp),
                finetune=bool(args.finetune),
                freeze_bn=True,
                save_dir=str(stage_save_dir),
                extra_tokens=list(base_extra_tokens) + cfg_tokens,
            )

            print("\n" + "-" * 80)
            print(f"[VARIANT] {suffix} :: {_variant_label(cfg)}")
            try:
                run(v_cmd, title=f"BND SWEEP TRAIN {v_exp}")
            except subprocess.CalledProcessError as e:
                # Common on Kaggle if batch size is too large or memory fragments.
                print(f"[WARN] Variant failed: {suffix} (exit={e.returncode}). Skipping to next.")
                continue

            rows = _evaluate_many(
                tag_prefix=suffix,
                exp_name_for_ckpt=v_exp,
                steps_to_eval=steps_to_eval,
                snapshot_dirs=snapshot_dirs,
                final_stop=int(stop_steps),
                gpu=str(args.gpu),
                base_metrics=base_metrics,
            )

            if rows:
                print("\n[TRAJ] Per-step eval:")
                _print_results_table(rows)

            best_row = _pick_best(rows, mode=mode)
            if best_row is None:
                print(f"[WARN] No evaluated checkpoints for variant {suffix}")
                continue

            all_variant_best.append(best_row)

            print(
                f"[BEST@{suffix}] {best_row['exp']}  dFZ={_fmt(best_row.get('dFZ'))}  dFDD={_fmt(best_row.get('dFDD'))}  dFD={_fmt(best_row.get('dFD'))}  dLindau={_fmt(best_row.get('dLindau'))}"
            )

            if overall_best is None or _rank_score(best_row, mode=mode) > _rank_score(overall_best, mode=mode):
                overall_best = best_row

        if all_variant_best:
            print("\n" + "=" * 10 + " BOUNDARY SWEEP BEST PER VARIANT " + "=" * 10)
            _print_results_table(all_variant_best)

        if overall_best is not None:
            print("\n" + "=" * 10 + f" BOUNDARY SWEEP OVERALL BEST ({mode}) " + "=" * 10)
            print(
                f"Winner: {overall_best['exp']}  dFZ={_fmt(overall_best.get('dFZ'))}  dFDD={_fmt(overall_best.get('dFDD'))}  dFD={_fmt(overall_best.get('dFD'))}  dLindau={_fmt(overall_best.get('dLindau'))}"
            )
            if persist_save_dir and overall_best.get("ckpt"):
                out_path = str(Path(best_save_dir) / f"{args.exp_name}_BND_SWEEP_BEST.pth")
                copied = _copy_best_checkpoint(str(overall_best["ckpt"]), out_path)
                print(f"[PERSIST] Copied best checkpoint -> {copied}")

        print("\nDone.")
        return 0

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
            "save_dir": str(train_save_dir),
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

        # If training snapshots were written to scratch, persist only the selected best checkpoint.
        if scratch_dir and persist_save_dir and b.get("ckpt"):
            out_dir = Path(persist_save_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_best = out_dir / f"{args.exp_name}_AUTO_BEST.pth"
            try:
                _copy_best_checkpoint(str(b["ckpt"]), str(out_best))
                print(f"[AUTO] Copied best checkpoint to: {out_best}")
            except Exception as e:
                print(f"[AUTO] Warning: failed to copy best checkpoint: {e}")

        print("\nDone.")
        return 0

    # Safe short finetune (aim: no drop)
    if bool(getattr(args, "safe", False)):
        safe_steps = int(getattr(args, "safe_steps", 100))
        safe_stop = safe_steps

        # Conservative multipliers to reduce drift.
        safe_backbone_mult = "0.001"
        safe_head_mult = "0.002"

        # Save snapshots so we can pick the best checkpoint within these few steps.
        # Default is 50 to reduce evaluation time; override with --safe-snapshot-every.
        safe_every = int(getattr(args, "safe_snapshot_every", 50) or 50)
        safe_every = max(1, min(safe_every, safe_steps))
        safe_snapshot_every = str(safe_every)
        safe_snapshot_until = str(safe_steps + 1)

        safe_extra_tokens = args.train_extra.strip().split() if args.train_extra.strip() else []

        # Memory safety: Boundary variant is heavy at 2048x1024; default to batch-size=1 in SAFE.
        # User can override via --safe-batch-size or by passing --batch-size in --train-extra.
        safe_bs = int(getattr(args, "safe_batch_size", 0) or 0)
        if safe_bs <= 0 and "boundary" in args.train_script.lower():
            safe_bs = 1
        if safe_bs > 0:
            safe_extra_tokens = _replace_or_add_flag(safe_extra_tokens, "--batch-size", str(safe_bs))
            # Also reduce loader workers a bit for stability on Kaggle
            safe_extra_tokens = _replace_or_add_flag(safe_extra_tokens, "--num-workers", "2")
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
            save_dir=str(train_save_dir),
            extra_tokens=safe_extra_tokens,
        )

        run(safe_cmd, title=f"SAFE TRAIN {args.exp_name}_SAFE ({safe_steps} steps)")

        # Evaluate multiple checkpoints within safe run (including final)
        safe_rows: list[dict] = []
        safe_steps_to_eval: list[int] = []
        try:
            every = int(safe_snapshot_every)
            safe_steps_to_eval = list(range(every, safe_steps + 1, every))
        except Exception:
            safe_steps_to_eval = []
        if safe_steps not in safe_steps_to_eval:
            safe_steps_to_eval.append(safe_steps)

        for st in safe_steps_to_eval:
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

            # If training snapshots were written to scratch, persist only the best checkpoint.
            if scratch_dir and persist_save_dir and b.get("ckpt"):
                out_dir = Path(persist_save_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                out_best = out_dir / f"{args.exp_name}_SAFE_BEST.pth"
                try:
                    shutil.copy2(str(b["ckpt"]), str(out_best))
                    print(f"[SAFE] Copied best checkpoint to: {out_best}")
                except Exception as e:
                    print(f"[SAFE] Warning: failed to copy best checkpoint: {e}")
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
                save_dir=str(train_save_dir),
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
        save_dir=str(train_save_dir),
        extra_tokens=extra,
    )

    run(train_cmd, title=f"TRAIN {args.exp_name} ({args.train_script})")

    # Evaluate snapshots
    steps = parse_steps(args.steps)
    print("\n[3] Evaluate selected steps:")

    if steps:
        available = find_all_fifo_ckpts_by_step(args.exp_name, snapshot_dirs)
        avail_steps = sorted(available.keys())
        if not avail_steps:
            print(f"[WARN] No FIFO snapshots found for exp={args.exp_name} in: {snapshot_dirs}")
        else:
            # Print a compact preview so users understand why some steps are missing.
            preview = ",".join(str(s) for s in avail_steps[:20])
            if len(avail_steps) > 20:
                preview += f",...(+{len(avail_steps)-20})"
            print(f"[INFO] Available FIFO steps: {preview}")

            steps_to_eval, missing_steps, mapping = _resolve_steps_to_eval(
                steps,
                available,
                steps_mode=str(args.steps_mode),
                nearest_max_diff=int(args.steps_nearest_max_diff),
                dedup=bool(args.steps_dedup),
            )

            if missing_steps and str(args.steps_mode).lower() == "exact":
                miss_preview = ",".join(str(s) for s in missing_steps[:30])
                if len(missing_steps) > 30:
                    miss_preview += f",...(+{len(missing_steps)-30})"
                print(f"[WARN] Missing ckpt for steps: {miss_preview} (exp={args.exp_name})")
                print("[HINT] Your snapshot interval may not match --steps. Consider --save-pred-every-early 100 or use --steps-mode nearest/available.")

            if mapping and str(args.steps_mode).lower() == "nearest":
                # show up to 20 mappings
                pairs = [f"{r}->{a}" for r, a in sorted(mapping.items())[:20]]
                more = "" if len(mapping) <= 20 else f" ...(+{len(mapping)-20})"
                print(f"[INFO] Nearest-step mapping: {', '.join(pairs)}{more}")

            for step in steps_to_eval:
                ckpt = available.get(step)
                if not ckpt:
                    # Shouldn't happen, but stay robust.
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
