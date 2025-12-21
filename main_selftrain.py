import os
import os.path as osp

import numpy as np
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
from torch.utils import data

from tqdm import tqdm

from model.refinenetlw import rf_lw101
from utils.losses import CrossEntropy2d
from dataset.paired_cityscapes import Pairedcityscapes
from dataset.Foggy_Zurich import foggyzurichDataSet
from configs.train_config import get_arguments
from utils.optimisers import get_optimisers, get_lr_schedulers

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
RESTORE_FROM = "without_pretraining"


def _bgr_mean_tensor(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    m = torch.tensor([float(IMG_MEAN[0]), float(IMG_MEAN[1]), float(IMG_MEAN[2])], device=device, dtype=dtype)
    return m.view(1, 3, 1, 1)


def _strong_augment_bgr(
    x_bgr_mean_sub: torch.Tensor,
    *,
    mean_bgr: torch.Tensor,
    brightness: float,
    contrast: float,
    saturation: float,
    noise_std: float,
    cutout: float,
    blur: bool,
) -> torch.Tensor:
    """Strong photometric + cutout augmentation.

    Expects input in BGR mean-subtracted space (float tensor N,3,H,W).
    Produces output in the same space.
    """

    if x_bgr_mean_sub.dim() != 4 or x_bgr_mean_sub.size(1) != 3:
        raise ValueError(f"Expected (N,3,H,W), got {tuple(x_bgr_mean_sub.shape)}")

    # Convert to [0,1] RGB
    x = (x_bgr_mean_sub + mean_bgr).clamp(0.0, 255.0) / 255.0
    x = x[:, [2, 1, 0], :, :]  # BGR -> RGB

    n, _c, h, w = x.shape

    if brightness and brightness > 0:
        b = (torch.rand((n, 1, 1, 1), device=x.device, dtype=x.dtype) * 2 - 1) * float(brightness)
        x = (x * (1.0 + b)).clamp(0.0, 1.0)

    if contrast and contrast > 0:
        c = (torch.rand((n, 1, 1, 1), device=x.device, dtype=x.dtype) * 2 - 1) * float(contrast)
        mean = x.mean(dim=(2, 3), keepdim=True)
        x = ((x - mean) * (1.0 + c) + mean).clamp(0.0, 1.0)

    if saturation and saturation > 0:
        s = (torch.rand((n, 1, 1, 1), device=x.device, dtype=x.dtype) * 2 - 1) * float(saturation)
        gray = (0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3])
        x = (x * (1.0 + s) + gray * (-s)).clamp(0.0, 1.0)

    if blur and torch.rand(()) < 0.5:
        x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

    if noise_std and noise_std > 0 and torch.rand(()) < 0.5:
        x = (x + torch.randn_like(x) * float(noise_std)).clamp(0.0, 1.0)

    if cutout and cutout > 0 and torch.rand(()) < 0.5:
        cut = int(max(1, float(cutout) * float(min(h, w))))
        for i in range(n):
            cy = int(torch.randint(0, h, (1,), device=x.device).item())
            cx = int(torch.randint(0, w, (1,), device=x.device).item())
            y0 = max(0, cy - cut // 2)
            y1 = min(h, cy + cut // 2)
            x0 = max(0, cx - cut // 2)
            x1 = min(w, cx + cut // 2)
            x[i, :, y0:y1, x0:x1] = 0.0

    # Back to BGR mean-subtracted
    x = x[:, [2, 1, 0], :, :] * 255.0
    x = x - mean_bgr
    return x


@torch.no_grad()
def _ema_update(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    d = float(decay)
    if d < 0.0:
        d = 0.0
    if d > 1.0:
        d = 1.0

    ema_params = dict(ema_model.named_parameters())
    model_params = dict(model.named_parameters())
    for k, p_ema in ema_params.items():
        p = model_params.get(k)
        if p is None:
            continue
        p_ema.data.mul_(d).add_(p.data, alpha=(1.0 - d))

    # Keep buffers synced (BN stats etc.)
    ema_buf = dict(ema_model.named_buffers())
    model_buf = dict(model.named_buffers())
    for k, b_ema in ema_buf.items():
        b = model_buf.get(k)
        if b is None:
            continue
        b_ema.data.copy_(b.data)


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
        return obj
    raise TypeError(f"Unsupported checkpoint type: {type(obj)}")


def _set_batchnorm_eval(m: nn.Module):
    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        m.eval()


def loss_calc(pred, label, device):
    label = label.long().to(device)
    criterion = CrossEntropy2d().to(device)
    return criterion(pred, label)


def setup_optimisers_and_schedulers(args, model):
    enc_lr = 6e-4
    dec_lr = 6e-3
    if bool(getattr(args, "finetune", False)):
        enc_lr *= float(getattr(args, "backbone_lr_mult", 0.1))
        dec_lr *= float(getattr(args, "head_lr_mult", 0.5))

    optimisers = get_optimisers(
        model=model,
        enc_optim_type="sgd",
        enc_lr=enc_lr,
        enc_weight_decay=1e-5,
        enc_momentum=0.9,
        dec_optim_type="sgd",
        dec_lr=dec_lr,
        dec_weight_decay=1e-5,
        dec_momentum=0.9,
    )
    schedulers = get_lr_schedulers(
        enc_optim=optimisers[0],
        dec_optim=optimisers[1],
        enc_lr_gamma=0.5,
        dec_lr_gamma=0.5,
        enc_scheduler_type="multistep",
        dec_scheduler_type="multistep",
        epochs_per_stage=(100, 100, 100),
    )
    return optimisers, schedulers


def fetch_next(loader_iter, loader):
    try:
        return next(loader_iter), loader_iter
    except StopIteration:
        loader_iter = iter(loader)
        return next(loader_iter), loader_iter


def main():
    args = get_arguments()

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    now = datetime.now().strftime("%m-%d-%H-%M")
    run_name = f"{args.file_name}-{now}"

    snapshot_dir = args.save_dir if args.save_dir else args.snapshot_dir
    if not os.path.exists(snapshot_dir):
        os.makedirs(snapshot_dir)

    # Model
    if args.restore_from == RESTORE_FROM:
        model = rf_lw101(num_classes=args.num_classes)
        start_iter = 0
    else:
        ckpt_obj = torch.load(args.restore_from, map_location="cpu", weights_only=False)
        model = rf_lw101(num_classes=args.num_classes)
        state_dict = _strip_module_prefix(_load_state_dict_from_checkpoint(ckpt_obj))
        model.load_state_dict(state_dict, strict=True)
        start_iter = 0

    model.to(device)
    model.train()
    if bool(getattr(args, "freeze_bn", False)):
        model.apply(_set_batchnorm_eval)

    # Datasets
    src_dataset = Pairedcityscapes(
        args.data_dir,
        args.data_dir_cwsf,
        args.data_list,
        args.data_list_cwsf,
        max_iters=args.num_steps * args.iter_size * args.batch_size,
        mean=IMG_MEAN,
        set=args.set,
    )

    use_fixmatch = bool(getattr(args, "fixmatch", False))
    pseudo_dir = str(getattr(args, "pseudo_label_dir", "") or "").strip()
    if use_fixmatch:
        tgt_dataset = foggyzurichDataSet(
            args.data_dir_rf,
            args.data_list_rf,
            max_iters=args.num_steps * args.iter_size * args.batch_size,
            mean=IMG_MEAN,
            set=args.set,
            return_label=False,
        )
    else:
        if not pseudo_dir:
            raise SystemExit("--pseudo-label-dir is required unless --fixmatch is set")
        tgt_dataset = foggyzurichDataSet(
            args.data_dir_rf,
            args.data_list_rf,
            max_iters=args.num_steps * args.iter_size * args.batch_size,
            mean=IMG_MEAN,
            set=args.set,
            pseudo_label_dir=pseudo_dir,
            return_label=True,
        )

    src_loader = data.DataLoader(
        src_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    tgt_loader = data.DataLoader(
        tgt_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    src_iter = iter(src_loader)
    tgt_iter = iter(tgt_loader)

    optimisers, schedulers = setup_optimisers_and_schedulers(args, model=model)
    enc_opt, dec_opt = optimisers[0], optimisers[1]

    scaler = amp.GradScaler(enabled=bool(args.amp))
    pseudo_w = float(getattr(args, "pseudo_weight", 0.1))
    pseudo_every = int(getattr(args, "pseudo_every", 1) or 1)
    if pseudo_every < 1:
        pseudo_every = 1

    ema_decay = float(getattr(args, "ema_decay", 0.99))
    pseudo_thr = float(getattr(args, "pseudo_threshold", 0.95))
    pseudo_thr_start = float(getattr(args, "pseudo_threshold_start", -1.0))
    pseudo_thr_end = float(getattr(args, "pseudo_threshold_end", -1.0))
    pseudo_thr_ramp = int(getattr(args, "pseudo_threshold_ramp", 0) or 0)
    use_thr_schedule = pseudo_thr_ramp > 0 and pseudo_thr_start > 0.0 and pseudo_thr_end > 0.0

    cons_w = float(getattr(args, "consistency_weight", 0.0) or 0.0)
    cons_type = str(getattr(args, "consistency_type", "kl") or "kl").strip().lower()
    cons_temp = float(getattr(args, "consistency_temp", 1.0) or 1.0)
    if cons_temp <= 0:
        cons_temp = 1.0
    strong_brightness = float(getattr(args, "strong_brightness", 0.2))
    strong_contrast = float(getattr(args, "strong_contrast", 0.2))
    strong_saturation = float(getattr(args, "strong_saturation", 0.2))
    strong_noise_std = float(getattr(args, "strong_noise_std", 0.02))
    strong_cutout = float(getattr(args, "strong_cutout", 0.5))
    strong_blur = bool(int(getattr(args, "strong_blur", 1) or 0))

    mean_bgr = _bgr_mean_tensor(device, dtype=torch.float32)

    ema_model = None
    if use_fixmatch:
        ema_model = rf_lw101(num_classes=args.num_classes).to(device)
        ema_model.load_state_dict(model.state_dict(), strict=True)
        ema_model.eval()
        for p in ema_model.parameters():
            p.requires_grad = False

    log_every = int(getattr(args, "log_every", 50) or 0)

    pbar = tqdm(range(start_iter, int(args.num_steps)), desc=f"SELFTRAIN {args.file_name}", unit="iter")
    last_loss = None
    last_src = None
    last_tgt = None

    accum_steps = int(getattr(args, "iter_size", 1) or 1)
    if accum_steps < 1:
        accum_steps = 1

    for i_iter in pbar:
        enc_opt.zero_grad(set_to_none=True)
        dec_opt.zero_grad(set_to_none=True)

        # True gradient accumulation: accum_steps micro-batches per optimizer step.
        sum_loss = 0.0
        sum_src = 0.0
        sum_tgt = 0.0
        apply_pseudo = (i_iter % pseudo_every) == 0

        if use_thr_schedule:
            t = float(i_iter) / float(max(1, pseudo_thr_ramp))
            t = 1.0 if t > 1.0 else (0.0 if t < 0.0 else t)
            current_thr = float(pseudo_thr_start) + (float(pseudo_thr_end) - float(pseudo_thr_start)) * t
        else:
            current_thr = float(pseudo_thr)

        for _sub_i in range(accum_steps):
            # Source labeled batch
            (sf_img, cw_img, src_lbl, size, _sf_name, _cw_name), src_iter = fetch_next(src_iter, src_loader)
            if apply_pseudo:
                if use_fixmatch:
                    (tgt_img, tgt_size, _tgt_name), tgt_iter = fetch_next(tgt_iter, tgt_loader)
                    tgt_lbl = None
                else:
                    (tgt_img, tgt_lbl, tgt_size, _tgt_name), tgt_iter = fetch_next(tgt_iter, tgt_loader)

            with amp.autocast(enabled=bool(args.amp)):
                interp_src = nn.Upsample(size=(int(size[0][0]), int(size[0][1])), mode="bilinear", align_corners=True)
                _out6, _out3, _out4, _out5, _out1, out2 = model(cw_img.to(device))
                pred_src = interp_src(out2)
                loss_src = loss_calc(pred_src, src_lbl.to(device), device)

                if apply_pseudo:
                    if use_fixmatch:
                        assert ema_model is not None
                        x_weak = tgt_img.to(device)
                        x_strong = _strong_augment_bgr(
                            x_weak,
                            mean_bgr=mean_bgr.to(dtype=x_weak.dtype),
                            brightness=strong_brightness,
                            contrast=strong_contrast,
                            saturation=strong_saturation,
                            noise_std=strong_noise_std,
                            cutout=strong_cutout,
                            blur=strong_blur,
                        )

                        # Teacher (EMA) predicts pseudo-labels on weak
                        with torch.no_grad():
                            _t6, _t3, _t4, _t5, _t1, t_out2 = ema_model(x_weak)

                        # Student predicts on strong
                        _s6, _s3, _s4, _s5, _s1, s_out2 = model(x_strong)

                        interp_tgt = nn.Upsample(size=(int(tgt_size[0][0]), int(tgt_size[0][1])), mode="bilinear", align_corners=True)
                        t_logits = interp_tgt(t_out2)
                        s_logits = interp_tgt(s_out2)

                        probs = torch.softmax(t_logits.detach(), dim=1)
                        conf, pseudo = torch.max(probs, dim=1)
                        pseudo = pseudo.long()
                        mask = conf >= float(current_thr)
                        pseudo = pseudo.masked_fill(~mask, 255)

                        loss_tgt = loss_calc(s_logits, pseudo, device)

                        loss_cons = torch.tensor(0.0, device=device)
                        if cons_w > 0.0:
                            # Teacher-student consistency on probabilities (masked by confidence).
                            # This is complementary to hard pseudo-label CE.
                            pt = torch.softmax(t_logits.detach() / float(cons_temp), dim=1)
                            if cons_type == "mse":
                                ps = torch.softmax(s_logits / float(cons_temp), dim=1)
                                per_pix = (ps - pt).pow(2).mean(dim=1)
                            else:
                                log_ps = torch.log_softmax(s_logits / float(cons_temp), dim=1)
                                kl = F.kl_div(log_ps, pt, reduction="none")
                                per_pix = kl.sum(dim=1)
                            m = mask.to(dtype=per_pix.dtype)
                            denom = m.sum().clamp(min=1.0)
                            loss_cons = (per_pix * m).sum() / denom
                            # Standard distillation scaling
                            loss_cons = loss_cons * (float(cons_temp) ** 2)

                        loss = loss_src + (pseudo_w * loss_tgt) + (cons_w * loss_cons)
                    else:
                        interp_tgt = nn.Upsample(size=(int(tgt_size[0][0]), int(tgt_size[0][1])), mode="bilinear", align_corners=True)
                        _out6, _out3, _out4, _out5, _out1, out2 = model(tgt_img.to(device))
                        pred_tgt = interp_tgt(out2)
                        loss_tgt = loss_calc(pred_tgt, tgt_lbl.to(device), device)
                        loss = loss_src + (pseudo_w * loss_tgt)
                else:
                    loss_tgt = torch.tensor(0.0, device=device)
                    loss = loss_src
                loss = loss / float(accum_steps)

            sum_loss += float(loss.detach().cpu().item())
            sum_src += float(loss_src.detach().cpu().item())
            sum_tgt += float(loss_tgt.detach().cpu().item())
            scaler.scale(loss).backward()

        scaler.step(enc_opt)
        scaler.step(dec_opt)
        scaler.update()

        if use_fixmatch and ema_model is not None:
            _ema_update(ema_model, model, decay=ema_decay)

        last_loss = sum_loss
        last_src = sum_src / float(accum_steps)
        last_tgt = sum_tgt / float(accum_steps)

        if last_loss is not None:
            pbar.set_postfix(
                {
                    "loss": f"{last_loss:.3f}",
                    "src": f"{last_src:.3f}",
                    "tgt": f"{last_tgt:.3f}",
                    "pw": f"{pseudo_w:.2f}",
                    "accum": str(accum_steps),
                    "pe": str(pseudo_every),
                    "thr": f"{current_thr:.3f}",
                    "cw": f"{cons_w:.2f}",
                }
            )

        if log_every > 0 and (i_iter % log_every == 0):
            print(
                f"[SELFTRAIN] iter={i_iter} loss={last_loss:.4f} src={last_src:.4f} tgt={last_tgt:.4f} pseudo_w={pseudo_w}",
                flush=True,
            )

        # Snapshot interval
        if int(getattr(args, "save_pred_every_early", 0) or 0) > 0 and int(getattr(args, "save_pred_early_until", 0) or 0) > 0:
            if i_iter < int(getattr(args, "save_pred_early_until")):
                save_every = int(getattr(args, "save_pred_every_early"))
            else:
                save_every = int(args.save_pred_every)
        else:
            save_every = int(args.save_pred_every)

        if i_iter % max(1, save_every) == 0 and i_iter != 0:
            torch.save(
                {"state_dict": model.state_dict(), "train_iter": i_iter, "args": args},
                osp.join(snapshot_dir, run_name) + "_FIFO" + str(i_iter) + ".pth",
            )
            print(f"[SELFTRAIN] snapshot saved: {run_name}_FIFO{i_iter}.pth", flush=True)

        if i_iter >= int(args.num_steps_stop) - 1:
            torch.save(
                {"state_dict": model.state_dict(), "train_iter": i_iter, "args": args},
                osp.join(snapshot_dir, args.file_name + str(args.num_steps_stop) + ".pth"),
            )
            print(f"[SELFTRAIN] final saved: {args.file_name}{args.num_steps_stop}.pth", flush=True)
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
