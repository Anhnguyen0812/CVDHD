import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn

from model.refinenetlw import rf_lw101


def _strip_module_prefix(state_dict: dict) -> dict:
    if not any(isinstance(k, str) and k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k[len("module.") :]: v for k, v in state_dict.items()}


def _load_state_dict_from_checkpoint(obj) -> dict:
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    if isinstance(obj, dict):
        # Some checkpoints are already state_dict
        return obj
    raise TypeError(f"Unsupported checkpoint type: {type(obj)}")


@dataclass
class ConvertArgs:
    in_ckpt: str
    out_ckpt: str
    gpu: int = 0
    num_classes: int = 19
    with_ema: bool = True
    amp: bool = True
    populate_opt_state: bool = True
    optimizer: str = "adamw"  # 'adamw' usually yields larger state than SGD
    lr: float = 1e-4
    momentum: float = 0.9
    weight_decay: float = 1e-5
    dummy_h: int = 64
    dummy_w: int = 128


def _make_optimizer(params, cfg: ConvertArgs) -> torch.optim.Optimizer:
    opt = cfg.optimizer.strip().lower()
    if opt == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    if opt == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")


@torch.no_grad()
def _init_ema(ema_model: nn.Module, model: nn.Module) -> None:
    ema_model.load_state_dict(model.state_dict(), strict=True)
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert a light checkpoint (weights only) into a full training checkpoint")
    ap.add_argument("--in-ckpt", required=True)
    ap.add_argument("--out-ckpt", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--num-classes", type=int, default=19)
    ap.add_argument("--with-ema", action="store_true")
    ap.add_argument("--no-ema", dest="with_ema", action="store_false")
    ap.set_defaults(with_ema=True)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.set_defaults(amp=True)
    ap.add_argument("--populate-opt-state", action="store_true")
    ap.add_argument("--no-populate-opt-state", dest="populate_opt_state", action="store_false")
    ap.set_defaults(populate_opt_state=True)
    ap.add_argument("--optimizer", choices=["sgd", "adamw"], default="adamw")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--dummy-h", type=int, default=64)
    ap.add_argument("--dummy-w", type=int, default=128)

    ns = ap.parse_args()
    cfg = ConvertArgs(
        in_ckpt=ns.in_ckpt,
        out_ckpt=ns.out_ckpt,
        gpu=int(ns.gpu),
        num_classes=int(ns.num_classes),
        with_ema=bool(ns.with_ema),
        amp=bool(ns.amp),
        populate_opt_state=bool(ns.populate_opt_state),
        optimizer=str(ns.optimizer),
        lr=float(ns.lr),
        momentum=float(ns.momentum),
        weight_decay=float(ns.weight_decay),
        dummy_h=int(ns.dummy_h),
        dummy_w=int(ns.dummy_w),
    )

    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")

    in_path = Path(cfg.in_ckpt)
    if not in_path.exists():
        raise SystemExit(f"Input checkpoint not found: {in_path}")

    obj = torch.load(str(in_path), map_location="cpu", weights_only=False)
    state_dict = _strip_module_prefix(_load_state_dict_from_checkpoint(obj))

    model = rf_lw101(num_classes=cfg.num_classes).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.train()

    # Two optimizers to roughly mimic encoder/decoder split (size-wise it's similar).
    enc_opt = _make_optimizer(model.layer0.parameters(), cfg)
    dec_opt = _make_optimizer([p for n, p in model.named_parameters() if not n.startswith("layer0.")], cfg)

    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.amp))

    ema_model = None
    if cfg.with_ema:
        ema_model = rf_lw101(num_classes=cfg.num_classes).to(device)
        _init_ema(ema_model, model)

    train_iter = 0
    if isinstance(obj, dict) and isinstance(obj.get("train_iter"), int):
        train_iter = int(obj["train_iter"])

    if cfg.populate_opt_state:
        # One dummy step to materialize optimizer state tensors (momentum/adam buffers)
        x = torch.randn(1, 3, cfg.dummy_h, cfg.dummy_w, device=device)
        with torch.cuda.amp.autocast(enabled=bool(cfg.amp)):
            *_outs, out2 = model(x)
            loss = out2.mean()
        enc_opt.zero_grad(set_to_none=True)
        dec_opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(enc_opt)
        scaler.step(dec_opt)
        scaler.update()

    out = {
        "state_dict": model.state_dict(),
        "train_iter": train_iter,
        "args": getattr(obj, "get", lambda _k, _d=None: None)("args", None) if isinstance(obj, dict) else None,
        "convert": asdict(cfg),
        "enc_opt": enc_opt.state_dict(),
        "dec_opt": dec_opt.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "ema_state_dict": ema_model.state_dict() if ema_model is not None else None,
    }

    out_path = Path(cfg.out_ckpt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, str(out_path))

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Saved full checkpoint: {out_path} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
