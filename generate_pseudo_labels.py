import argparse
import os
import os.path as osp
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn

from model.refinenetlw import rf_lw101

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)


def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if not any(isinstance(k, str) and k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k[len("module.") :]: v for k, v in state_dict.items()}


def _load_state_dict_from_checkpoint(obj):
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    if isinstance(obj, dict):
        # raw state_dict
        return obj
    raise TypeError(f"Unsupported checkpoint type: {type(obj)}")


def _preprocess_bgr(img_pil: Image.Image) -> torch.Tensor:
    arr = np.asarray(img_pil, np.float32)
    arr = arr[:, :, ::-1]  # RGB -> BGR
    arr -= IMG_MEAN
    arr = arr.transpose((2, 0, 1))
    return torch.from_numpy(arr).unsqueeze(0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--restore-from", required=True)
    ap.add_argument("--data-dir-rf", required=True, help="Dataset root, e.g. /kaggle/input/fifo-dataset")
    ap.add_argument("--data-list-rf", required=True, help="List of Foggy Zurich train images")
    ap.add_argument("--out-dir", required=True, help="Where to write pseudo labels")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.9, help="Softmax max-prob threshold; below -> ignore (255)")
    ap.add_argument("--scales", default="1.0,0.8,0.6", help="Comma-separated inference scales")
    ap.add_argument("--amp", type=int, default=1)

    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    ckpt_obj = torch.load(args.restore_from, map_location="cpu", weights_only=False)
    state_dict = _strip_module_prefix(_load_state_dict_from_checkpoint(ckpt_obj))

    model = rf_lw101(num_classes=19)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    scales = [float(x) for x in str(args.scales).split(",") if x.strip()]
    if not scales:
        scales = [1.0]

    with open(args.data_list_rf, "r", encoding="utf-8") as f:
        names = [ln.strip() for ln in f if ln.strip()]

    for idx, name in enumerate(names):
        img_path = osp.join(args.data_dir_rf, f"foggy_zurich/Foggy_Zurich/{name}")
        img = Image.open(img_path).convert("RGB")
        w0, h0 = img.size

        logits_sum = None
        with torch.no_grad():
            for s in scales:
                if s == 1.0:
                    img_s = img
                else:
                    img_s = img.resize((max(1, int(w0 * s)), max(1, int(h0 * s))), resample=Image.BILINEAR)

                x = _preprocess_bgr(img_s).to(device)
                with torch.cuda.amp.autocast(enabled=bool(args.amp)):
                    # model returns multiple outputs; use the segmentation logits (output2)
                    out6, out3, out4, out5, out1, out2 = model(x)
                    up = nn.functional.interpolate(out2, size=(h0, w0), mode="bilinear", align_corners=True)

                if logits_sum is None:
                    logits_sum = up.float()
                else:
                    logits_sum = logits_sum + up.float()

            logits = logits_sum / float(len(scales))
            probs = torch.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)  # (1,H,W)

            pred = pred.squeeze(0).cpu().numpy().astype(np.uint8)
            conf = conf.squeeze(0).cpu().numpy().astype(np.float32)

            if args.threshold > 0:
                pred = pred.astype(np.uint8)
                pred[conf < float(args.threshold)] = 255

        out_path = out_root / name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pred, mode="L").save(str(out_path))

        if (idx + 1) % 50 == 0:
            print(f"[{idx+1}/{len(names)}] wrote {out_path}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
