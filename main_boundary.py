import os
import os.path as osp
import random
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils import data
from tqdm import tqdm

from configs.train_config import get_arguments
from dataset.Foggy_Zurich import foggyzurichDataSet
from dataset.paired_cityscapes import Pairedcityscapes
from model.fogpassfilter import FogPassFilter_conv1, FogPassFilter_res1
from model.refinenetlw import rf_lw101
from pytorch_metric_learning import losses
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.reducers import MeanReducer
from utils.losses import CrossEntropy2d
from utils.optimisers import get_lr_schedulers, get_optimisers


IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
RESTORE_FROM = "without_pretraining"
RESTORE_FROM_fogpass = "without_pretraining"


def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if not any(isinstance(k, str) and k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k[len("module.") :]: v for k, v in state_dict.items()}


def _load_state_dict_from_checkpoint(obj):
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"], obj
    if isinstance(obj, dict):
        return obj, None
    raise TypeError(f"Unsupported checkpoint type: {type(obj)}")


def load_model_checkpoint(model, ckpt_path, *, strict=True):
    ckpt_obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict, container = _load_state_dict_from_checkpoint(ckpt_obj)
    state_dict = _strip_module_prefix(state_dict)
    incompatible = model.load_state_dict(state_dict, strict=strict)
    return container if container is not None else ckpt_obj, incompatible


def load_fogpass_checkpoint(fpf1, fpf2, ckpt_path, *, strict=True):
    ckpt_obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt_obj, dict):
        raise TypeError("FogPass checkpoint must be a dict containing fogpass state dicts")
    if "fogpass1_state_dict" not in ckpt_obj or "fogpass2_state_dict" not in ckpt_obj:
        raise KeyError("FogPass checkpoint is missing fogpass state dicts")
    sd1 = _strip_module_prefix(ckpt_obj["fogpass1_state_dict"])
    sd2 = _strip_module_prefix(ckpt_obj["fogpass2_state_dict"])
    incompatible1 = fpf1.load_state_dict(sd1, strict=strict)
    incompatible2 = fpf2.load_state_dict(sd2, strict=strict)
    return ckpt_obj, incompatible1, incompatible2


def get_state_dict(module):
    return module.module.state_dict() if isinstance(module, DDP) else module.state_dict()


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


def loss_calc(pred, label, device):
    label = label.long().to(device)
    criterion = CrossEntropy2d().to(device)
    return criterion(pred, label)


def gram_matrix(tensor):
    tensor = tensor.float()
    d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram


def upper_triangular_vector(square_matrix):
    if square_matrix.dim() != 2 or square_matrix.size(0) != square_matrix.size(1):
        raise ValueError(f"Expected square 2D matrix, got {tuple(square_matrix.shape)}")
    n = square_matrix.size(0)
    mask = torch.ones((n, n), device=square_matrix.device, dtype=torch.bool).triu()
    return square_matrix[mask]


def build_loader(dataset, args, distributed, shuffle=True):
    if distributed:
        sampler = data.distributed.DistributedSampler(dataset, shuffle=shuffle, drop_last=True)
        loader = data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    else:
        sampler = None
        loader = data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    return loader, sampler


def fetch_next(loader_iter, loader):
    try:
        return next(loader_iter), loader_iter
    except StopIteration:
        loader_iter = iter(loader)
        return next(loader_iter), loader_iter


def _set_batchnorm_eval(m: nn.Module):
    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        m.eval()


def label_to_edges(label: torch.Tensor, ignore_index: int = 255) -> torch.Tensor:
    """Convert label map (N,H,W) to binary boundary map (N,1,H,W)."""
    if label.dim() != 3:
        raise ValueError(f"Expected (N,H,W), got {tuple(label.shape)}")
    n, h, w = label.shape
    y = label
    valid = y != ignore_index

    # compare with shifted neighbors
    edge = torch.zeros((n, h, w), device=y.device, dtype=torch.bool)
    edge[:, :, 1:] |= (y[:, :, 1:] != y[:, :, :-1]) & valid[:, :, 1:] & valid[:, :, :-1]
    edge[:, :, :-1] |= (y[:, :, :-1] != y[:, :, 1:]) & valid[:, :, :-1] & valid[:, :, 1:]
    edge[:, 1:, :] |= (y[:, 1:, :] != y[:, :-1, :]) & valid[:, 1:, :] & valid[:, :-1, :]
    edge[:, :-1, :] |= (y[:, :-1, :] != y[:, 1:, :]) & valid[:, :-1, :] & valid[:, 1:, :]

    return edge.unsqueeze(1).float()


def boundary_fscore(pred_logits: torch.Tensor, target: torch.Tensor, thresh: float = 0.0, eps: float = 1e-6):
    """Compute boundary F1 on a batch.

    pred_logits: (N,1,H,W)
    target: (N,1,H,W) {0,1}
    """
    pred = (pred_logits > thresh).float()
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return float(precision.detach().cpu()), float(recall.detach().cpu()), float(f1.detach().cpu())


class BoundaryHead(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1, bias=True),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)


def main():
    args = get_arguments()

    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = args.distributed or world_size > 1
    if not is_distributed:
        local_rank = args.gpu

    if is_distributed:
        dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(local_rank)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.random_seed)
        torch.cuda.manual_seed(args.random_seed)

    is_main_process = (not is_distributed) or dist.get_rank() == 0

    now = datetime.now().strftime("%m-%d-%H-%M")
    run_name = f"{args.file_name}-{now}"
    if is_main_process:
        wandb.init(project="FIFO", name=run_name)
        wandb.config.update(vars(args))
    else:
        wandb.init(mode="disabled")

    cudnn.enabled = True
    snapshot_dir = args.save_dir if args.save_dir else args.snapshot_dir
    if is_main_process and not os.path.exists(snapshot_dir):
        os.makedirs(snapshot_dir)

    # Student model
    student = rf_lw101(num_classes=args.num_classes)
    if args.restore_from != RESTORE_FROM:
        load_model_checkpoint(student, args.restore_from, strict=True)
    student.to(device)
    if is_distributed:
        student = DDP(student, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    student.train()

    freeze_bn = bool(getattr(args, "freeze_bn", False)) or bool(getattr(args, "finetune", False))
    if freeze_bn:
        student.apply(_set_batchnorm_eval)

    # Teacher model (fixed) for pseudo labels
    teacher = rf_lw101(num_classes=args.num_classes)
    if args.restore_from != RESTORE_FROM:
        load_model_checkpoint(teacher, args.restore_from, strict=True)
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Boundary head (lazy init because feature channels depend on chosen layer)
    boundary_head = None
    boundary_opt = None
    boundary_lr = float(getattr(args, "boundary_lr", 1e-3))
    boundary_warmup_steps = int(getattr(args, "boundary_warmup_steps", 1000))
    boundary_weight = float(getattr(args, "boundary_weight", 0.5))
    bce = nn.BCEWithLogitsLoss()

    # FogPass
    lr_fpf1 = 5e-4 if args.modeltrain == "train" else 1e-3
    lr_fpf2 = 1e-3

    if getattr(args, "fpf1_lr", None) is not None:
        lr_fpf1 = float(args.fpf1_lr)
    if getattr(args, "fpf2_lr", None) is not None:
        lr_fpf2 = float(args.fpf2_lr)
    lr_fpf_mult = float(getattr(args, "fpf_lr_mult", 1.0))
    lr_fpf1 *= lr_fpf_mult
    lr_fpf2 *= lr_fpf_mult
    FogPassFilter1 = FogPassFilter_conv1(2080).to(device)
    FogPassFilter2 = FogPassFilter_res1(32896).to(device)
    FogPassFilter1_optimizer = torch.optim.Adamax([p for p in FogPassFilter1.parameters() if p.requires_grad], lr=lr_fpf1)
    FogPassFilter2_optimizer = torch.optim.Adamax([p for p in FogPassFilter2.parameters() if p.requires_grad], lr=lr_fpf2)

    if args.restore_from_fogpass != RESTORE_FROM_fogpass:
        load_fogpass_checkpoint(FogPassFilter1, FogPassFilter2, args.restore_from_fogpass, strict=True)

    if is_distributed:
        FogPassFilter1 = DDP(FogPassFilter1, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        FogPassFilter2 = DDP(FogPassFilter2, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    fogpassfilter_loss = losses.ContrastiveLoss(
        pos_margin=0.1, neg_margin=0.1, distance=CosineSimilarity(), reducer=MeanReducer()
    )

    # Data
    cwsf_dataset = Pairedcityscapes(
        args.data_dir,
        args.data_dir_cwsf,
        args.data_list,
        args.data_list_cwsf,
        max_iters=args.num_steps * args.iter_size * args.batch_size,
        mean=IMG_MEAN,
        set=args.set,
    )
    rf_dataset = foggyzurichDataSet(
        args.data_dir_rf,
        args.data_list_rf,
        max_iters=args.num_steps * args.iter_size * args.batch_size,
        mean=IMG_MEAN,
        set=args.set,
    )

    cwsf_pair_loader, cwsf_pair_sampler = build_loader(cwsf_dataset, args, is_distributed, shuffle=True)
    rf_loader, rf_sampler = build_loader(rf_dataset, args, is_distributed, shuffle=True)
    cwsf_pair_loader_fogpass, cwsf_pair_sampler_fogpass = build_loader(cwsf_dataset, args, is_distributed, shuffle=True)
    rf_loader_fogpass, rf_sampler_fogpass = build_loader(rf_dataset, args, is_distributed, shuffle=True)

    rf_loader_iter = iter(rf_loader)
    cwsf_pair_loader_iter = iter(cwsf_pair_loader)
    cwsf_pair_loader_iter_fogpass = iter(cwsf_pair_loader_fogpass)
    rf_loader_iter_fogpass = iter(rf_loader_fogpass)

    optimisers, _schedulers = setup_optimisers_and_schedulers(args, model=student)
    opts = optimisers if isinstance(optimisers, (list, tuple)) else [optimisers]
    kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
    softmax = nn.Softmax(dim=1)
    log_softmax = nn.LogSoftmax(dim=1)

    scaler_seg = amp.GradScaler(enabled=bool(args.amp))
    scaler_fpf = amp.GradScaler(enabled=bool(args.amp))
    scaler_bnd = amp.GradScaler(enabled=bool(args.amp))

    iter_source = tqdm(range(0, args.num_steps)) if is_main_process else range(0, args.num_steps)
    freeze_fogpass_steps = int(getattr(args, "freeze_fogpass_steps", 0))
    for i_iter in iter_source:
        if is_distributed:
            if cwsf_pair_sampler:
                cwsf_pair_sampler.set_epoch(i_iter)
            if rf_sampler:
                rf_sampler.set_epoch(i_iter)
            if cwsf_pair_sampler_fogpass:
                cwsf_pair_sampler_fogpass.set_epoch(i_iter)
            if rf_sampler_fogpass:
                rf_sampler_fogpass.set_epoch(i_iter)

        for opt in opts:
            opt.zero_grad(set_to_none=True)
        if boundary_opt is not None:
            boundary_opt.zero_grad(set_to_none=True)

        for _sub_i in range(args.iter_size):
            # ===== Phase A: train fog-pass filtering modules (same as FIFO) =====
            if i_iter >= freeze_fogpass_steps:
                student.eval()
                for p in student.parameters():
                    p.requires_grad = False
                for p in FogPassFilter1.parameters():
                    p.requires_grad = True
                for p in FogPassFilter2.parameters():
                    p.requires_grad = True

                batch, cwsf_pair_loader_iter_fogpass = fetch_next(cwsf_pair_loader_iter_fogpass, cwsf_pair_loader_fogpass)
                sf_image, cw_image, label, size, _sf_name, _cw_name = batch

                batch_rf, rf_loader_iter_fogpass = fetch_next(rf_loader_iter_fogpass, rf_loader_fogpass)
                rf_img, _rf_size, _rf_name = batch_rf

                with torch.no_grad():
                    with amp.autocast(enabled=bool(args.amp)):
                        img_rf = rf_img.to(device)
                        feature_rf0, feature_rf1, *_rest = student(img_rf)

                        images_sf = sf_image.to(device)
                        feature_sf0, feature_sf1, *_rest2 = student(images_sf)

                        images_cw = cw_image.to(device)
                        feature_cw0, feature_cw1, *_rest3 = student(images_cw)

                fsm_weights = {"layer0": 0.5, "layer1": 0.5}
                sf_features = {"layer0": feature_sf0.detach(), "layer1": feature_sf1.detach()}
                cw_features = {"layer0": feature_cw0.detach(), "layer1": feature_cw1.detach()}
                rf_features = {"layer0": feature_rf0.detach(), "layer1": feature_rf1.detach()}

                total_fpf_loss = 0
                for idx, layer in enumerate(fsm_weights):
                    cw_feature = cw_features[layer]
                    sf_feature = sf_features[layer]
                    rf_feature = rf_features[layer]

                    fogpassfilter = FogPassFilter1 if idx == 0 else FogPassFilter2
                    fogpassfilter_optimizer = FogPassFilter1_optimizer if idx == 0 else FogPassFilter2_optimizer
                    fogpassfilter.train()
                    fogpassfilter_optimizer.zero_grad(set_to_none=True)

                    actual_batch = sf_feature.size(0)
                    fog_factor_list = []
                    fog_factor_labels_list = []
                    for batch_idx in range(actual_batch):
                        sf_g = gram_matrix(sf_feature[batch_idx])
                        cw_g = gram_matrix(cw_feature[batch_idx])
                        rf_g = gram_matrix(rf_feature[batch_idx])

                        vec_sf = upper_triangular_vector(sf_g).requires_grad_()
                        vec_cw = upper_triangular_vector(cw_g).requires_grad_()
                        vec_rf = upper_triangular_vector(rf_g).requires_grad_()

                        fog_factor_list.append(fogpassfilter(vec_sf).unsqueeze(0))
                        fog_factor_list.append(fogpassfilter(vec_cw).unsqueeze(0))
                        fog_factor_list.append(fogpassfilter(vec_rf).unsqueeze(0))
                        fog_factor_labels_list.extend([0, 1, 2])

                    fog_factor_embeddings = torch.cat(fog_factor_list, dim=0)
                    fog_factor_embeddings = fog_factor_embeddings / torch.norm(fog_factor_embeddings, p=2, dim=1).unsqueeze(1).clamp(
                        min=1e-6
                    )
                    fog_factor_labels = torch.LongTensor(fog_factor_labels_list).to(device)
                    fog_pass_filter_loss = fogpassfilter_loss(fog_factor_embeddings, fog_factor_labels)
                    total_fpf_loss = total_fpf_loss + fog_pass_filter_loss

                    if is_main_process:
                        wandb.log({f"layer{idx}/fpf loss": fog_pass_filter_loss}, step=i_iter)
                        wandb.log({f"layer{idx}/total fpf loss": total_fpf_loss}, step=i_iter)

                scaler_fpf.scale(total_fpf_loss).backward()
                scaler_fpf.step(FogPassFilter1_optimizer)
                scaler_fpf.step(FogPassFilter2_optimizer)
                scaler_fpf.update()
                FogPassFilter1_optimizer.zero_grad(set_to_none=True)
                FogPassFilter2_optimizer.zero_grad(set_to_none=True)
            else:
                if is_main_process and i_iter == 0 and freeze_fogpass_steps > 0:
                    print(f"[Freeze] FogPassFilter updates frozen for first {freeze_fogpass_steps} steps")

            # ===== Phase B: boundary head warm-up / finetune =====
            if args.modeltrain == "train":
                batch, cwsf_pair_loader_iter = fetch_next(cwsf_pair_loader_iter, cwsf_pair_loader)
                sf_image, cw_image, label, size, _sf_name, _cw_name = batch
                interp = nn.Upsample(size=(size[0][0], size[0][1]), mode="bilinear")

                batch_rf, rf_loader_iter = fetch_next(rf_loader_iter, rf_loader)
                rf_img, _rf_size, _rf_name = batch_rf
                img_rf = rf_img.to(device)

                # Create pseudo labels for target and convert to edges
                with torch.no_grad():
                    with amp.autocast(enabled=bool(args.amp)):
                        _t0, _t1, _t2, _t3, _t4, t_logits = teacher(img_rf)
                    t_pl = torch.argmax(t_logits, dim=1)
                    edge_target = label_to_edges(t_pl)

                # Choose feature layer for boundary prediction: use out4 (higher-res than out5)
                if i_iter < boundary_warmup_steps:
                    student.eval()
                    for p in student.parameters():
                        p.requires_grad = False
                else:
                    student.train()
                    if freeze_bn:
                        student.apply(_set_batchnorm_eval)
                    for p in student.parameters():
                        p.requires_grad = True
                    for p in FogPassFilter1.parameters():
                        p.requires_grad = False
                    for p in FogPassFilter2.parameters():
                        p.requires_grad = False

                # Forward student on target
                with torch.no_grad() if i_iter < boundary_warmup_steps else torch.enable_grad():
                    with amp.autocast(enabled=bool(args.amp)):
                        out1, out2, out3, out4, out5, out6 = student(img_rf)
                        feat_for_boundary = out4

                if boundary_head is None:
                    boundary_head = BoundaryHead(in_channels=int(feat_for_boundary.shape[1])).to(device)
                    boundary_opt = torch.optim.AdamW(boundary_head.parameters(), lr=boundary_lr, weight_decay=1e-4)

                boundary_head.train()
                with amp.autocast(enabled=bool(args.amp)):
                    bnd_logits = boundary_head(feat_for_boundary.detach() if i_iter < boundary_warmup_steps else feat_for_boundary)
                    bnd_logits_up = F.interpolate(bnd_logits, size=edge_target.shape[-2:], mode="bilinear", align_corners=False)
                    loss_bnd = bce(bnd_logits_up, edge_target)

                # Segmentation (only after warmup)
                if i_iter >= boundary_warmup_steps:
                    if i_iter % 3 == 0:
                        with amp.autocast(enabled=bool(args.amp)):
                            images_sf = sf_image.to(device)
                            feature_sf0, feature_sf1, feature_sf2, feature_sf3, feature_sf4, feature_sf5 = student(images_sf)
                            pred_sf5 = interp(feature_sf5)
                            loss_seg_sf = loss_calc(pred_sf5, label, device)

                            images_cw = cw_image.to(device)
                            feature_cw0, feature_cw1, feature_cw2, feature_cw3, feature_cw4, feature_cw5 = student(images_cw)
                            pred_cw5 = interp(feature_cw5)
                            loss_seg_cw = loss_calc(pred_cw5, label, device)

                            loss_con = kl_loss(log_softmax(feature_sf5), softmax(feature_cw5))
                            fsm_weights = {"layer0": 0.5, "layer1": 0.5}
                            sf_features = {"layer0": feature_sf0, "layer1": feature_sf1}
                            cw_features = {"layer0": feature_cw0, "layer1": feature_cw1}
                    elif i_iter % 3 == 1:
                        with amp.autocast(enabled=bool(args.amp)):
                            images_sf = sf_image.to(device)
                            feature_sf0, feature_sf1, feature_sf2, feature_sf3, feature_sf4, feature_sf5 = student(images_sf)
                            pred_sf5 = interp(feature_sf5)
                            loss_seg_sf = loss_calc(pred_sf5, label, device)
                            loss_seg_cw = 0
                            loss_con = 0

                            feature_rf0, feature_rf1, *_r = student(img_rf)
                            rf_features = {"layer0": feature_rf0, "layer1": feature_rf1}
                            sf_features = {"layer0": feature_sf0, "layer1": feature_sf1}
                            fsm_weights = {"layer0": 0.5, "layer1": 0.5}
                    else:
                        with amp.autocast(enabled=bool(args.amp)):
                            images_cw = cw_image.to(device)
                            feature_cw0, feature_cw1, feature_cw2, feature_cw3, feature_cw4, feature_cw5 = student(images_cw)
                            pred_cw5 = interp(feature_cw5)
                            loss_seg_sf = 0
                            loss_con = 0
                            loss_seg_cw = loss_calc(pred_cw5, label, device)

                            feature_rf0, feature_rf1, *_r = student(img_rf)
                            rf_features = {"layer0": feature_rf0, "layer1": feature_rf1}
                            cw_features = {"layer0": feature_cw0, "layer1": feature_cw1}
                            fsm_weights = {"layer0": 0.5, "layer1": 0.5}

                    # FSM loss
                    loss_fsm = 0
                    for idx, layer in enumerate(fsm_weights):
                        if i_iter % 3 == 0:
                            a_feature = cw_features[layer]
                            b_feature = sf_features[layer]
                        elif i_iter % 3 == 1:
                            a_feature = rf_features[layer]
                            b_feature = sf_features[layer]
                        else:
                            a_feature = rf_features[layer]
                            b_feature = cw_features[layer]

                        na, da, ha, wa = a_feature.size()
                        nb, db, hb, wb = b_feature.size()

                        fogpassfilter = FogPassFilter1 if idx == 0 else FogPassFilter2
                        fogpassfilter.eval()
                        layer_fsm_loss = 0
                        actual_batch_fsm = b_feature.size(0)
                        for batch_idx in range(actual_batch_fsm):
                            b_gram = gram_matrix(b_feature[batch_idx])
                            a_gram = gram_matrix(a_feature[batch_idx])
                            if i_iter % 3 in (1, 2):
                                a_gram = a_gram * (hb * wb) / float(ha * wa)
                            vector_b = upper_triangular_vector(b_gram).requires_grad_()
                            vector_a = upper_triangular_vector(a_gram).requires_grad_()
                            fog_factor_b = fogpassfilter(vector_b)
                            fog_factor_a = fogpassfilter(vector_a)
                            half = int(fog_factor_b.shape[0] / 2)
                            layer_fsm_loss = layer_fsm_loss + (
                                fsm_weights[layer]
                                * torch.mean((fog_factor_b / float(hb * wb) - fog_factor_a / float(ha * wa)) ** 2)
                                / max(1, half)
                                / actual_batch_fsm
                            )
                        loss_fsm = loss_fsm + (layer_fsm_loss / float(actual_batch_fsm))

                    seg_loss = loss_seg_sf + loss_seg_cw + args.lambda_fsm * loss_fsm + args.lambda_con * loss_con
                    total_loss = (seg_loss + boundary_weight * loss_bnd) / float(args.iter_size)
                    scaler_seg.scale(total_loss).backward()
                else:
                    # warm-up: only boundary
                    total_loss = (boundary_weight * loss_bnd) / float(args.iter_size)
                    scaler_seg.scale(total_loss).backward()

                # step
                if boundary_opt is not None:
                    scaler_seg.step(boundary_opt)
                if i_iter >= boundary_warmup_steps:
                    for opt in opts:
                        scaler_seg.step(opt)
                scaler_seg.update()

                if is_main_process:
                    p, r, f1 = boundary_fscore(bnd_logits_up.detach(), edge_target)
                    wandb.log({"boundary_loss": float(loss_bnd.detach().cpu())}, step=i_iter)
                    wandb.log({"boundary_precision": p, "boundary_recall": r, "boundary_f1": f1}, step=i_iter)
                    wandb.log({"boundary_warmup_steps": boundary_warmup_steps}, step=i_iter)
                    wandb.log({"total_loss": float(total_loss.detach().cpu())}, step=i_iter)

        # snapshots
        early_every = int(getattr(args, "save_pred_every_early", 0))
        early_until = int(getattr(args, "save_pred_early_until", 0))
        if early_every > 0 and early_until > 0 and i_iter < early_until:
            save_pred_every = early_every
        else:
            if i_iter < 20000:
                save_pred_every = 2000 if args.modeltrain == "train" else 5000
            else:
                save_pred_every = args.save_pred_every

        if i_iter >= args.num_steps_stop - 1:
            if is_main_process:
                torch.save(get_state_dict(student), osp.join(snapshot_dir, args.file_name + str(args.num_steps_stop) + ".pth"))
            break

        if is_main_process and i_iter % save_pred_every == 0 and i_iter != 0:
            torch.save(
                {
                    "state_dict": get_state_dict(student),
                    "fogpass1_state_dict": get_state_dict(FogPassFilter1),
                    "fogpass2_state_dict": get_state_dict(FogPassFilter2),
                    "boundary_head_state_dict": boundary_head.state_dict() if boundary_head is not None else None,
                    "train_iter": i_iter,
                    "args": args,
                },
                osp.join(snapshot_dir, run_name) + f"_FIFO{i_iter}.pth",
            )


if __name__ == "__main__":
    main()
