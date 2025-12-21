# Kaggle quick commands (train + eval + balanced checkpoint selection)

This repo includes a helper script `diagnose_train_eval.py` that:
- evaluates a **baseline** checkpoint
- runs a short finetune (SAFE) or an ablation sweep (SWEEP)
- evaluates multiple checkpoints and prints deltas for: **FZ / FDD / FD / Lindau**
- picks the **best checkpoint** using `--rank-mode`:
  - `fdd` (default): prioritize Foggy Driving Dense
  - `mean`: maximize average improvement across all datasets
  - `min`: maximize worst-case improvement (most “even”)

Below are Kaggle notebook cells you can copy/paste.

---

## 0) Set paths

```bash
%cd /kaggle/working/CVDHD

# GPU index
GPU=0

# Base checkpoint (put your path here)
BASE=/kaggle/working/FIFO_final_model.pth

# Persistent outputs (keep only best checkpoints here)
SAVE_DIR=/kaggle/working/snapshots/FIFO_model
mkdir -p $SAVE_DIR

# Scratch space for intermediate snapshots (prevents /kaggle/working 20GB limit)
SCRATCH_DIR=/kaggle/temp/snapshots/FIFO_model
mkdir -p $SCRATCH_DIR
```

---

## 1) Baseline evaluation only

```bash
!python evaluate.py \
  --file-name BASELINE \
  --restore-from $BASE \
  --gpu $GPU
```

---

## 2) SAFE run (recommended first)

SAFE mode is designed to avoid sudden drops and **evaluates multiple checkpoints** (SAFE@20/40/...) then prints the best.

### 2.1 FDA script (balanced selection)

- Even improvement (worst-case): `--rank-mode min`
- Average improvement: `--rank-mode mean`

```bash
!python diagnose_train_eval.py \
  --repo-dir /kaggle/working/CVDHD \
  --gpu $GPU \
  --base-ckpt $BASE \
  --exp-name FDA \
  --train-script main_fda.py \
  --train-extra "--fda-beta 0.01" \
  --scratch-dir $SCRATCH_DIR \
  --save-dir $SAVE_DIR \
  --safe --safe-steps 100 \
  --rank-mode min
```

If you want SAFE mode to actually *use* FDA (instead of forcing beta=0), set `--safe-fda-beta`:

```bash
!python diagnose_train_eval.py \
  --repo-dir /kaggle/working/CVDHD \
  --gpu $GPU \
  --base-ckpt $BASE \
  --exp-name FDA \
  --train-script main_fda.py \
  --scratch-dir $SCRATCH_DIR \
  --save-dir $SAVE_DIR \
  --safe --safe-steps 200 \
  --safe-fda-beta 0.005 \
  --rank-mode min
```

### 2.2 ProtoCL script (balanced selection)

```bash
!python diagnose_train_eval.py \
  --repo-dir /kaggle/working/CVDHD \
  --gpu $GPU \
  --base-ckpt $BASE \
  --exp-name PROTO \
  --train-script main_proto_cl.py \
  --scratch-dir $SCRATCH_DIR \
  --save-dir $SAVE_DIR \
  --safe --safe-steps 100 \
  --rank-mode min
```

### 2.3 Boundary script (balanced selection)

```bash
!python diagnose_train_eval.py \
  --repo-dir /kaggle/working/CVDHD \
  --gpu $GPU \
  --base-ckpt $BASE \
  --exp-name BOUND \
  --train-script main_boundary.py \
  --scratch-dir $SCRATCH_DIR \
  --save-dir $SAVE_DIR \
  --safe --safe-steps 100 \
  --rank-mode min
```

---

## 3) SWEEP run (find what hurts / helps quickly)

This runs several short variants (10 steps by default) and prints deltas vs baseline.

```bash
!python diagnose_train_eval.py \
  --repo-dir /kaggle/working/CVDHD \
  --gpu $GPU \
  --base-ckpt $BASE \
  --exp-name FDA \
  --train-script main_fda.py \
  --train-extra "--fda-beta 0.01" \
  --scratch-dir $SCRATCH_DIR \
  --save-dir $SAVE_DIR \
  --sweep --sweep-num-steps 10 --sweep-num-steps-stop 10 \
  --rank-mode mean
```

---

## 4) Regular run + evaluate specific FIFO steps

Example: train to 2100 steps, save snapshots at 200/800/2000, then evaluate those steps.

```bash
!python diagnose_train_eval.py \
  --repo-dir /kaggle/working/CVDHD \
  --gpu $GPU \
  --base-ckpt $BASE \
  --exp-name FDA_long \
  --train-script main_fda.py \
  --train-extra "--fda-beta 0.01" \
  --num-steps 2100 --num-steps-stop 2100 \
  --steps 200,800,2000 \
  --scratch-dir $SCRATCH_DIR \
  --save-dir $SAVE_DIR
```

Notes:
- If you see `Missing ckpt for step=...`, it usually means the training saved to a different folder.
  Setting `--save-dir $SAVE_DIR` makes train + eval consistent.

---

## 5) COMBO (combine methods automatically)

This runs Boundary first (safe batch size), selects the best checkpoint by `--rank-mode`, then continues with ProtoCL from that checkpoint.
It also writes stable files:
- `$SAVE_DIR/<exp>_BEST_BND.pth`
- `$SAVE_DIR/<exp>_BEST_PROTO.pth`
- `$SAVE_DIR/<exp>_BEST_OVERALL.pth`

```bash
!python diagnose_train_eval.py \
  --repo-dir /kaggle/working/CVDHD \
  --gpu $GPU \
  --base-ckpt $BASE \
  --exp-name COMBO1 \
  --scratch-dir $SCRATCH_DIR \
  --save-dir $SAVE_DIR \
  --rank-mode min \
  --combo \
  --combo-boundary-steps 400 \
  --combo-proto-steps 200 \
  --combo-proto-weight 0.01
```

Optional (often risky): add a tiny FDA stage at the end:

```bash
!python diagnose_train_eval.py \
  --repo-dir /kaggle/working/CVDHD \
  --gpu $GPU \
  --base-ckpt $BASE \
  --exp-name COMBO1 \
  --scratch-dir $SCRATCH_DIR \
  --save-dir $SAVE_DIR \
  --rank-mode min \
  --combo \
  --combo-boundary-steps 400 \
  --combo-proto-steps 200 \
  --combo-proto-weight 0.01 \
  --combo-fda-steps 100 \
  --combo-fda-beta 0.0002

---

## 6) Self-training (pseudo labels) for bigger gains

If you need a larger jump (sometimes ~+1% or more), a common next step is **pseudo-label self-training**:

1) Generate pseudo labels for Foggy Zurich train images using your best checkpoint.
2) Finetune with `main_selftrain.py` (mixing Cityscapes labeled + Foggy Zurich pseudo-labeled) and pick the best checkpoint by `--rank-mode`.

### 6.1 Generate pseudo labels (write to /kaggle/temp)

```bash
# Pick a strong teacher checkpoint (example: best overall from COMBO)
TEACHER=/kaggle/working/snapshots/FIFO_model/COMBO1_BEST_OVERALL.pth

PSEUDO_DIR=/kaggle/temp/pseudo_labels/FZ_LIGHT
mkdir -p $PSEUDO_DIR

!python generate_pseudo_labels.py \
  --restore-from $TEACHER \
  --data-dir-rf /kaggle/input/fifo-dataset \
  --data-list-rf /kaggle/input/fifo-dataset/foggy_zurich/Foggy_Zurich/lists_file_names/RGB_light_filenames.txt \
  --out-dir $PSEUDO_DIR \
  --gpu $GPU \
  --threshold 0.9 \
  --scales 1.0,0.8,0.6
```

### 6.2 Finetune with pseudo labels + select best checkpoint

```bash
!python diagnose_train_eval.py \
  --repo-dir /kaggle/working/CVDHD \
  --gpu $GPU \
  --base-ckpt $TEACHER \
  --exp-name SELFTRN1 \
  --train-script main_selftrain.py \
  --train-extra "--pseudo-label-dir $PSEUDO_DIR --pseudo-weight 0.1" \
  --scratch-dir $SCRATCH_DIR \
  --save-dir $SAVE_DIR \
  --safe --safe-steps 400 \
  --rank-mode mean
```

Notes:
- If training becomes unstable, increase `--threshold` (e.g. 0.95) and/or lower `--pseudo-weight` (e.g. 0.05).
- `--rank-mode mean` usually targets bigger overall gains; `min` targets balanced gains.
```
