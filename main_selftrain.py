import os
import os.path as osp

import numpy as np
import random
from datetime import datetime

import torch
import torch.nn as nn
from torch.cuda import amp
from torch.utils import data

from model.refinenetlw import rf_lw101
from utils.losses import CrossEntropy2d
from dataset.paired_cityscapes import Pairedcityscapes
from dataset.Foggy_Zurich import foggyzurichDataSet
from configs.train_config import get_arguments
from utils.optimisers import get_optimisers, get_lr_schedulers

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
RESTORE_FROM = "without_pretraining"


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

    pseudo_dir = str(getattr(args, "pseudo_label_dir", "") or "").strip()
    if not pseudo_dir:
        raise SystemExit("--pseudo-label-dir is required for main_selftrain.py")

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

    for i_iter in range(start_iter, int(args.num_steps)):
        enc_opt.zero_grad(set_to_none=True)
        dec_opt.zero_grad(set_to_none=True)

        # Source labeled batch
        (sf_img, cw_img, src_lbl, size, _sf_name, _cw_name), src_iter = fetch_next(src_iter, src_loader)
        # Target pseudo-labeled batch
        (tgt_img, tgt_lbl, tgt_size, _tgt_name), tgt_iter = fetch_next(tgt_iter, tgt_loader)

        with amp.autocast(enabled=bool(args.amp)):
            interp_src = nn.Upsample(size=(int(size[0][0]), int(size[0][1])), mode="bilinear", align_corners=True)
            out6, out3, out4, out5, out1, out2 = model(cw_img.to(device))
            pred_src = interp_src(out2)
            loss_src = loss_calc(pred_src, src_lbl.to(device), device)

            interp_tgt = nn.Upsample(size=(int(tgt_size[0][0]), int(tgt_size[0][1])), mode="bilinear", align_corners=True)
            out6, out3, out4, out5, out1, out2 = model(tgt_img.to(device))
            pred_tgt = interp_tgt(out2)
            loss_tgt = loss_calc(pred_tgt, tgt_lbl.to(device), device)

            loss = loss_src + (pseudo_w * loss_tgt)
            loss = loss / float(max(1, int(args.iter_size)))

        scaler.scale(loss).backward()
        scaler.step(enc_opt)
        scaler.step(dec_opt)
        scaler.update()

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

        if i_iter >= int(args.num_steps_stop) - 1:
            torch.save(
                {"state_dict": model.state_dict(), "train_iter": i_iter, "args": args},
                osp.join(snapshot_dir, args.file_name + str(args.num_steps_stop) + ".pth"),
            )
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
