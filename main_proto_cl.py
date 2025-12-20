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


def _global_pool_embedding(feat: torch.Tensor) -> torch.Tensor:
    emb = F.adaptive_avg_pool2d(feat.float(), output_size=1).flatten(1)
    return F.normalize(emb, p=2, dim=1, eps=1e-6)


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def init_prototypes_from_labeled_source(
    teacher: nn.Module,
    proj_head: nn.Module,
    loader: data.DataLoader,
    device: torch.device,
    num_classes: int,
    max_iters: int,
    sample_pixels: int,
):
    """Initialize class prototypes by averaging projected features on labeled (SF) samples."""
    sums = None
    counts = torch.zeros((num_classes,), dtype=torch.long)
    it = iter(loader)

    with torch.no_grad():
        for _i in range(max_iters):
            batch, it = fetch_next(it, loader)
            sf_image, _cw_image, label, _size, *_names = batch
            img = sf_image.to(device)
            lab = label.to(device).long()
            with amp.autocast(enabled=False):
                out1, out2, out3, out4, out5, out6 = teacher(img)
                feat = out4.float()  # (N,C,h,w)

            # downsample label to feat size
            lab_ds = F.interpolate(lab.unsqueeze(1).float(), size=feat.shape[-2:], mode="nearest").squeeze(1).long()
            n, c, h, w = feat.shape
            feat_flat = feat.permute(0, 2, 3, 1).reshape(-1, c)
            lab_flat = lab_ds.reshape(-1)
            valid = lab_flat != 255
            feat_flat = feat_flat[valid]
            lab_flat = lab_flat[valid]

            if feat_flat.numel() == 0:
                continue
            if feat_flat.size(0) > sample_pixels:
                idx = torch.randperm(feat_flat.size(0), device=feat_flat.device)[:sample_pixels]
                feat_flat = feat_flat[idx]
                lab_flat = lab_flat[idx]

            emb = F.normalize(proj_head(feat_flat), p=2, dim=1, eps=1e-6)

            if sums is None:
                sums = torch.zeros((num_classes, emb.size(1)), dtype=torch.float32, device="cpu")

            emb_cpu = emb.detach().cpu()
            lab_cpu = lab_flat.detach().cpu()
            for cls in range(num_classes):
                m = lab_cpu == cls
                if m.any():
                    sums[cls] += emb_cpu[m].mean(dim=0)
                    counts[cls] += 1

    if sums is None:
        raise RuntimeError("No valid pixels found to init prototypes")

    prototypes = sums / counts.clamp(min=1).unsqueeze(1)
    prototypes = F.normalize(prototypes, p=2, dim=1, eps=1e-6)
    return prototypes.to(device)


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

    # Teacher model (fixed) to init prototypes
    teacher = rf_lw101(num_classes=args.num_classes)
    if args.restore_from != RESTORE_FROM:
        load_model_checkpoint(teacher, args.restore_from, strict=True)
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Projection head (lazy init)
    proj_head = None
    proj_opt = None
    proj_lr = float(getattr(args, "proto_head_lr", 1e-3))

    # Prototypes configs
    proto_weight = float(getattr(args, "proto_weight", 0.1))
    proto_temp = float(getattr(args, "proto_temp", 0.1))
    proto_momentum = float(getattr(args, "proto_momentum", 0.99))
    proto_init_iters = int(getattr(args, "proto_init_iters", 200))
    proto_sample_pixels = int(getattr(args, "proto_sample_pixels", 4096))

    # FogPass
    lr_fpf1 = 5e-4 if args.modeltrain == "train" else 1e-3
    lr_fpf2 = 1e-3
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
    scaler_proto = amp.GradScaler(enabled=bool(args.amp))

    # Build proj_head + init prototypes once (on each process for simplicity)
    # Determine feature dim by one forward of teacher on one batch
    batch0, _it0 = fetch_next(iter(cwsf_pair_loader), cwsf_pair_loader)
    sf0, _cw0, _lab0, *_rest0 = batch0
    with torch.no_grad():
        out1, out2, out3, out4, out5, out6 = teacher(sf0.to(device))
    feat_dim = int(out4.shape[1])
    proj_head = ProjectionHead(in_dim=feat_dim, hidden_dim=256, out_dim=128).to(device)
    proj_opt = torch.optim.AdamW(proj_head.parameters(), lr=proj_lr, weight_decay=1e-4)

    if is_main_process:
        print(f"[ProtoCL] Initializing prototypes using {proto_init_iters} iters ...")
    prototypes = init_prototypes_from_labeled_source(
        teacher=teacher,
        proj_head=proj_head,
        loader=cwsf_pair_loader,
        device=device,
        num_classes=args.num_classes,
        max_iters=proto_init_iters,
        sample_pixels=proto_sample_pixels,
    )

    iter_source = tqdm(range(0, args.num_steps)) if is_main_process else range(0, args.num_steps)
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
        if proj_opt is not None:
            proj_opt.zero_grad(set_to_none=True)

        for _sub_i in range(args.iter_size):
            # ===== Phase A: fog-pass filtering modules =====
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

            scaler_fpf.scale(total_fpf_loss).backward()
            scaler_fpf.step(FogPassFilter1_optimizer)
            scaler_fpf.step(FogPassFilter2_optimizer)
            scaler_fpf.update()
            FogPassFilter1_optimizer.zero_grad(set_to_none=True)
            FogPassFilter2_optimizer.zero_grad(set_to_none=True)

            # ===== Phase B: segmentation + prototype contrastive =====
            if args.modeltrain == "train":
                student.train()
                for p in student.parameters():
                    p.requires_grad = True
                for p in FogPassFilter1.parameters():
                    p.requires_grad = False
                for p in FogPassFilter2.parameters():
                    p.requires_grad = False

                batch, cwsf_pair_loader_iter = fetch_next(cwsf_pair_loader_iter, cwsf_pair_loader)
                sf_image, cw_image, label, size, _sf_name, _cw_name = batch
                interp = nn.Upsample(size=(size[0][0], size[0][1]), mode="bilinear")

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

                    # Pixel-to-prototype loss (use SF features at out4 resolution)
                    feat = feature_sf4.float()  # (N,1024,h,w)
                    n, c, h, w = feat.shape
                    lab = label.to(device).long()
                    lab_ds = F.interpolate(lab.unsqueeze(1).float(), size=(h, w), mode="nearest").squeeze(1).long()
                    feat_flat = feat.permute(0, 2, 3, 1).reshape(-1, c)
                    lab_flat = lab_ds.reshape(-1)
                    valid = lab_flat != 255
                    feat_flat = feat_flat[valid]
                    lab_flat = lab_flat[valid]

                    if feat_flat.numel() == 0:
                        loss_proto = 0
                    else:
                        if feat_flat.size(0) > proto_sample_pixels:
                            idx = torch.randperm(feat_flat.size(0), device=feat_flat.device)[:proto_sample_pixels]
                            feat_flat = feat_flat[idx]
                            lab_flat = lab_flat[idx]
                        z = F.normalize(proj_head(feat_flat), p=2, dim=1, eps=1e-6)
                        prot = F.normalize(prototypes, p=2, dim=1, eps=1e-6)
                        logits = (z @ prot.t()) / max(1e-6, proto_temp)
                        loss_proto = F.cross_entropy(logits, lab_flat)

                        # EMA update prototypes from this batch
                        with torch.no_grad():
                            for cls in range(args.num_classes):
                                m = lab_flat == cls
                                if m.any():
                                    batch_mean = z[m].mean(dim=0)
                                    prototypes[cls] = (
                                        proto_momentum * prototypes[cls] + (1.0 - proto_momentum) * batch_mean
                                    )
                            prototypes[:] = F.normalize(prototypes, p=2, dim=1, eps=1e-6)

                elif i_iter % 3 == 1:
                    batch_rf, rf_loader_iter = fetch_next(rf_loader_iter, rf_loader)
                    rf_img, _rf_size, _rf_name = batch_rf
                    with amp.autocast(enabled=bool(args.amp)):
                        images_sf = sf_image.to(device)
                        feature_sf0, feature_sf1, feature_sf2, feature_sf3, feature_sf4, feature_sf5 = student(images_sf)
                        pred_sf5 = interp(feature_sf5)
                        loss_seg_sf = loss_calc(pred_sf5, label, device)
                        loss_seg_cw = 0
                        loss_con = 0
                        img_rf = rf_img.to(device)
                        feature_rf0, feature_rf1, *_r = student(img_rf)
                        rf_features = {"layer0": feature_rf0, "layer1": feature_rf1}
                        sf_features = {"layer0": feature_sf0, "layer1": feature_sf1}
                        fsm_weights = {"layer0": 0.5, "layer1": 0.5}
                    loss_proto = 0
                else:
                    batch_rf, rf_loader_iter = fetch_next(rf_loader_iter, rf_loader)
                    rf_img, _rf_size, _rf_name = batch_rf
                    with amp.autocast(enabled=bool(args.amp)):
                        images_cw = cw_image.to(device)
                        feature_cw0, feature_cw1, feature_cw2, feature_cw3, feature_cw4, feature_cw5 = student(images_cw)
                        pred_cw5 = interp(feature_cw5)
                        loss_seg_sf = 0
                        loss_con = 0
                        loss_seg_cw = loss_calc(pred_cw5, label, device)
                        img_rf = rf_img.to(device)
                        feature_rf0, feature_rf1, *_r = student(img_rf)
                        rf_features = {"layer0": feature_rf0, "layer1": feature_rf1}
                        cw_features = {"layer0": feature_cw0, "layer1": feature_cw1}
                        fsm_weights = {"layer0": 0.5, "layer1": 0.5}
                    loss_proto = 0

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

                loss = loss_seg_sf + loss_seg_cw + args.lambda_fsm * loss_fsm + args.lambda_con * loss_con
                if loss_proto != 0:
                    loss = loss + proto_weight * loss_proto
                loss = loss / float(args.iter_size)

                scaler_seg.scale(loss).backward()
                for opt in opts:
                    scaler_seg.step(opt)
                if proj_opt is not None and loss_proto != 0:
                    scaler_seg.step(proj_opt)
                scaler_seg.update()

                if is_main_process:
                    wandb.log({"proto_weight": proto_weight, "proto_temp": proto_temp}, step=i_iter)
                    wandb.log({"proto_loss": float(loss_proto.detach().cpu()) if loss_proto != 0 else 0.0}, step=i_iter)
                    wandb.log({"total_loss": float(loss.detach().cpu())}, step=i_iter)

        # snapshots
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
                    "proj_head_state_dict": proj_head.state_dict() if proj_head is not None else None,
                    "prototypes": prototypes.detach().cpu(),
                    "train_iter": i_iter,
                    "args": args,
                },
                osp.join(snapshot_dir, run_name) + f"_FIFO{i_iter}.pth",
            )


if __name__ == "__main__":
    main()
