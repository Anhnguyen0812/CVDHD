import os
import os.path as osp

import torch
import torch.nn as nn
from torch.utils import data
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import grad 
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda import amp

import numpy as np
import random
import wandb
from tqdm import tqdm
from PIL import Image
from packaging import version
from datetime import datetime

from model.refinenetlw import rf_lw101
from model.fogpassfilter import FogPassFilter_conv1, FogPassFilter_res1
from utils.losses import CrossEntropy2d
from dataset.paired_cityscapes import Pairedcityscapes
from dataset.Foggy_Zurich import foggyzurichDataSet
from configs.train_config import get_arguments
from utils.optimisers import get_optimisers, get_lr_schedulers
from pytorch_metric_learning import losses
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.reducers import MeanReducer

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
RESTORE_FROM = 'without_pretraining'
RESTORE_FROM_fogpass = 'without_pretraining'


def _strip_module_prefix(state_dict):
    """Strip a leading 'module.' prefix that can appear in DDP-saved checkpoints."""
    if not isinstance(state_dict, dict):
        return state_dict
    if not any(isinstance(k, str) and k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k[len("module.") :]: v for k, v in state_dict.items()}


def _load_state_dict_from_checkpoint(obj):
    """Accept either a raw state_dict checkpoint or a dict with a 'state_dict' key."""
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
    if "fogpass1_state_dict" in ckpt_obj and "fogpass2_state_dict" in ckpt_obj:
        sd1 = _strip_module_prefix(ckpt_obj["fogpass1_state_dict"])
        sd2 = _strip_module_prefix(ckpt_obj["fogpass2_state_dict"])
    elif "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
        # Some checkpoints may wrap everything; try common keys.
        raise KeyError(
            "FogPass checkpoint is missing 'fogpass1_state_dict'/'fogpass2_state_dict'. "
            "Please pass a FIFO checkpoint that includes these keys."
        )
    else:
        raise KeyError(
            "FogPass checkpoint is missing 'fogpass1_state_dict'/'fogpass2_state_dict'."
        )

    incompatible1 = fpf1.load_state_dict(sd1, strict=strict)
    incompatible2 = fpf2.load_state_dict(sd2, strict=strict)
    return ckpt_obj, incompatible1, incompatible2

def loss_calc(pred, label, device):
    label = Variable(label.long()).to(device)
    criterion = CrossEntropy2d().to(device)
    return criterion(pred, label)

def gram_matrix(tensor):
    # Force float32 to avoid half/float matmul mismatch under AMP
    tensor = tensor.float()
    d, h, w = tensor.size()
    tensor = tensor.view(d, h*w)
    gram = torch.mm(tensor, tensor.t())
    return gram


def _global_pool_embedding(feat: torch.Tensor) -> torch.Tensor:
    """Convert a feature map (N,C,H,W) to L2-normalized embeddings (N,C)."""
    if feat.dim() != 4:
        raise ValueError(f"Expected feature map of shape (N,C,H,W), got {tuple(feat.shape)}")
    emb = F.adaptive_avg_pool2d(feat.float(), output_size=1).flatten(1)
    return F.normalize(emb, p=2, dim=1, eps=1e-6)


def upper_triangular_vector(square_matrix):
    """Return the upper-triangular (including diagonal) entries as a 1D vector.

    Important: builds the mask on the same device as the matrix to avoid
    CPU/GPU indexing mismatches.
    """
    if square_matrix.dim() != 2 or square_matrix.size(0) != square_matrix.size(1):
        raise ValueError(
            f"Expected a square 2D matrix, got shape {tuple(square_matrix.shape)}"
        )
    n = square_matrix.size(0)
    mask = torch.ones((n, n), device=square_matrix.device, dtype=torch.bool).triu()
    return square_matrix[mask]

def setup_optimisers_and_schedulers(args, model):
    optimisers = get_optimisers(
        model=model,
        enc_optim_type="sgd",
        enc_lr=6e-4,
        enc_weight_decay=1e-5,
        enc_momentum=0.9,
        dec_optim_type="sgd",
        dec_lr=6e-3,
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

def make_list(x):
    """Returns the given input as a list."""
    if isinstance(x, list):
        return x
    elif isinstance(x, tuple):
        return list(x)
    else:
        return [x]


def get_state_dict(module):
    """Return state_dict handling DDP wrapping."""
    return module.module.state_dict() if isinstance(module, DDP) else module.state_dict()


def build_loader(dataset, args, distributed, shuffle=True):
    """Create DataLoader with optional DistributedSampler."""
    if distributed:
        sampler = data.distributed.DistributedSampler(dataset, shuffle=shuffle, drop_last=True)
        loader = data.DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                                 num_workers=args.num_workers, pin_memory=True)
    else:
        sampler = None
        loader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                                 num_workers=args.num_workers, pin_memory=True)
    return loader, sampler


def fetch_next(loader_iter, loader):
    """Get next batch; restart iterator on StopIteration."""
    try:
        return next(loader_iter), loader_iter
    except StopIteration:
        loader_iter = iter(loader)
        return next(loader_iter), loader_iter

def main():
    """Create the model and start the training."""

    args = get_arguments()
    # Auto-detect distributed mode from torchrun env vars
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = args.distributed or world_size > 1
    
    if not is_distributed:
        local_rank = args.gpu

    if is_distributed:
        dist.init_process_group(backend="nccl")

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if device.type == "cuda":
        torch.cuda.set_device(local_rank)
        torch.cuda.manual_seed_all(args.random_seed)
        torch.cuda.manual_seed(args.random_seed)

    is_main_process = (not is_distributed) or dist.get_rank() == 0

    now = datetime.now().strftime('%m-%d-%H-%M')
    run_name = f'{args.file_name}-{now}'
    if is_main_process:
        wandb.init(project='FIFO', name=f'{run_name}')
        wandb.config.update(args)
    else:
        wandb.init(mode='disabled')

    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)

    w_r, h_r = map(int, args.input_size_rf.split(',')) 
    input_size_rf = (w_r, h_r)   

    cudnn.enabled = True
    gpu = local_rank
    snapshot_dir = args.save_dir if args.save_dir else args.snapshot_dir

    start_iter = 0
    model = rf_lw101(num_classes=args.num_classes)
    if args.restore_from != RESTORE_FROM:
        _, incompatible = load_model_checkpoint(model, args.restore_from, strict=True)
        if is_main_process:
            if getattr(incompatible, "missing_keys", None) or getattr(incompatible, "unexpected_keys", None):
                print("[Checkpoint] Loaded with key mismatch:")
                if incompatible.missing_keys:
                    print(f"  missing_keys: {incompatible.missing_keys[:20]}{' ...' if len(incompatible.missing_keys) > 20 else ''}")
                if incompatible.unexpected_keys:
                    print(f"  unexpected_keys: {incompatible.unexpected_keys[:20]}{' ...' if len(incompatible.unexpected_keys) > 20 else ''}")

    model.to(device)
    if is_distributed:
        model = DDP(model, device_ids=[gpu], output_device=gpu, find_unused_parameters=False)
    model.train()

    lr_fpf1 = 1e-3 
    lr_fpf2 = 1e-3

    if args.modeltrain=='train':
        lr_fpf1 = 5e-4

    FogPassFilter1 = FogPassFilter_conv1(2080)
    FogPassFilter1_optimizer = torch.optim.Adamax([p for p in FogPassFilter1.parameters() if p.requires_grad], lr=lr_fpf1)
    FogPassFilter1.to(device)
    FogPassFilter2 = FogPassFilter_res1(32896)
    FogPassFilter2_optimizer = torch.optim.Adamax([p for p in FogPassFilter2.parameters() if p.requires_grad], lr=lr_fpf2)
    FogPassFilter2.to(device)

    if args.restore_from_fogpass != RESTORE_FROM_fogpass:
        _, inc1, inc2 = load_fogpass_checkpoint(FogPassFilter1, FogPassFilter2, args.restore_from_fogpass, strict=True)
        if is_main_process:
            if (getattr(inc1, "missing_keys", None) or getattr(inc1, "unexpected_keys", None) or
                    getattr(inc2, "missing_keys", None) or getattr(inc2, "unexpected_keys", None)):
                print("[FogPass] Loaded with key mismatch (showing first 20 keys each):")
                if inc1.missing_keys:
                    print(f"  fpf1 missing_keys: {inc1.missing_keys[:20]}{' ...' if len(inc1.missing_keys) > 20 else ''}")
                if inc1.unexpected_keys:
                    print(f"  fpf1 unexpected_keys: {inc1.unexpected_keys[:20]}{' ...' if len(inc1.unexpected_keys) > 20 else ''}")
                if inc2.missing_keys:
                    print(f"  fpf2 missing_keys: {inc2.missing_keys[:20]}{' ...' if len(inc2.missing_keys) > 20 else ''}")
                if inc2.unexpected_keys:
                    print(f"  fpf2 unexpected_keys: {inc2.unexpected_keys[:20]}{' ...' if len(inc2.unexpected_keys) > 20 else ''}")

    if is_distributed:
        FogPassFilter1 = DDP(FogPassFilter1, device_ids=[gpu], output_device=gpu, find_unused_parameters=False)
        FogPassFilter2 = DDP(FogPassFilter2, device_ids=[gpu], output_device=gpu, find_unused_parameters=False)

    fogpassfilter_loss = losses.ContrastiveLoss(
        pos_margin=0.1,
        neg_margin=0.1,
        distance=CosineSimilarity(),
        reducer=MeanReducer()
        )

    # Image-level contrastive learning between paired domains (SimCLR-style)
    contrastive_loss_fn = losses.NTXentLoss(temperature=float(getattr(args, "cl_temp", 0.1)))

    cudnn.benchmark = True

    if is_main_process and not os.path.exists(snapshot_dir):
        os.makedirs(snapshot_dir)

    cwsf_dataset = Pairedcityscapes(args.data_dir, args.data_dir_cwsf, args.data_list, args.data_list_cwsf,
                                    max_iters=args.num_steps * args.iter_size * args.batch_size,
                                    mean=IMG_MEAN, set=args.set)
    rf_dataset = foggyzurichDataSet(args.data_dir_rf, args.data_list_rf,
                                    max_iters=args.num_steps * args.iter_size * args.batch_size,
                                    mean=IMG_MEAN, set=args.set)
    cwsf_fogpass_dataset = Pairedcityscapes(args.data_dir, args.data_dir_cwsf, args.data_list, args.data_list_cwsf,
                                           max_iters=args.num_steps * args.iter_size * args.batch_size,
                                           mean=IMG_MEAN, set=args.set)
    rf_fogpass_dataset = foggyzurichDataSet(args.data_dir_rf, args.data_list_rf,
                                           max_iters=args.num_steps * args.iter_size * args.batch_size,
                                           mean=IMG_MEAN, set=args.set)

    cwsf_pair_loader, cwsf_pair_sampler = build_loader(cwsf_dataset, args, is_distributed, shuffle=True)
    rf_loader, rf_sampler = build_loader(rf_dataset, args, is_distributed, shuffle=True)
    cwsf_pair_loader_fogpass, cwsf_pair_sampler_fogpass = build_loader(cwsf_fogpass_dataset, args, is_distributed, shuffle=True)
    rf_loader_fogpass, rf_sampler_fogpass = build_loader(rf_fogpass_dataset, args, is_distributed, shuffle=True)

    rf_loader_iter = iter(rf_loader)
    cwsf_pair_loader_iter = iter(cwsf_pair_loader)
    cwsf_pair_loader_iter_fogpass = iter(cwsf_pair_loader_fogpass)
    rf_loader_iter_fogpass = iter(rf_loader_fogpass)

    optimisers, schedulers = setup_optimisers_and_schedulers(args, model=model)
    opts = make_list(optimisers)
    kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
    m = nn.Softmax(dim=1)
    log_m = nn.LogSoftmax(dim=1)    
    scaler_seg = amp.GradScaler(enabled=bool(args.amp))
    scaler_fpf = amp.GradScaler(enabled=bool(args.amp))

    iter_source = tqdm(range(start_iter, args.num_steps)) if is_main_process else range(start_iter, args.num_steps)
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
        loss_seg_cw_value = 0
        loss_seg_sf_value = 0
        loss_fsm_value = 0
        loss_con_value = 0
        loss_cl_value = 0

        for opt in opts:
            opt.zero_grad()

        for sub_i in range(args.iter_size):
            # train fog-pass filtering module
            # freeze the parameters of segmentation network

            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            for param in FogPassFilter1.parameters():
                param.requires_grad = True
            for param in FogPassFilter2.parameters():
                param.requires_grad = True
  
            batch, cwsf_pair_loader_iter_fogpass = fetch_next(cwsf_pair_loader_iter_fogpass, cwsf_pair_loader_fogpass)
            sf_image, cw_image, label, size, sf_name, cw_name = batch
            interp = nn.Upsample(size=(size[0][0],size[0][1]), mode='bilinear')
            
            batch_rf, rf_loader_iter_fogpass = fetch_next(rf_loader_iter_fogpass, rf_loader_fogpass)
            rf_img,rf_size, rf_name = batch_rf
            # Use torch.no_grad for model forward in fogpass phase since model is frozen
            with torch.no_grad():
                with amp.autocast(enabled=bool(args.amp)):
                    img_rf = rf_img.to(device)
                    feature_rf0, feature_rf1, feature_rf2, feature_rf3, feature_rf4, feature_rf5 = model(img_rf) 

                    images = sf_image.to(device)
                    feature_sf0,feature_sf1,feature_sf2, feature_sf3,feature_sf4,feature_sf5 = model(images)

                    images_cw = cw_image.to(device)
                    feature_cw0, feature_cw1, feature_cw2, feature_cw3, feature_cw4, feature_cw5 = model(images_cw)

            fsm_weights = {'layer0':0.5, 'layer1':0.5}
            sf_features = {'layer0':feature_sf0.detach(), 'layer1':feature_sf1.detach()}                
            cw_features = {'layer0':feature_cw0.detach(), 'layer1':feature_cw1.detach()}
            rf_features = {'layer0':feature_rf0.detach(), 'layer1':feature_rf1.detach()}

            total_fpf_loss = 0

            for idx, layer in enumerate(fsm_weights):
                cw_feature = cw_features[layer]
                sf_feature = sf_features[layer]    
                rf_feature = rf_features[layer]      
                fog_pass_filter_loss = 0 
                
                if idx == 0:
                    fogpassfilter = FogPassFilter1
                    fogpassfilter_optimizer = FogPassFilter1_optimizer
                elif idx == 1:
                    fogpassfilter = FogPassFilter2
                    fogpassfilter_optimizer = FogPassFilter2_optimizer

                fogpassfilter.train()  
                fogpassfilter_optimizer.zero_grad()
                
                # Dynamic batch size handling
                actual_batch = sf_feature.size(0)
                fog_factor_list = []
                fog_factor_labels_list = []
                
                for batch_idx in range(actual_batch):
                    sf_g = gram_matrix(sf_feature[batch_idx])
                    cw_g = gram_matrix(cw_feature[batch_idx])
                    rf_g = gram_matrix(rf_feature[batch_idx])

                    vec_sf = Variable(upper_triangular_vector(sf_g), requires_grad=True)
                    vec_cw = Variable(upper_triangular_vector(cw_g), requires_grad=True)
                    vec_rf = Variable(upper_triangular_vector(rf_g), requires_grad=True)

                    fog_factor_list.append(fogpassfilter(vec_sf).unsqueeze(0))
                    fog_factor_list.append(fogpassfilter(vec_cw).unsqueeze(0))
                    fog_factor_list.append(fogpassfilter(vec_rf).unsqueeze(0))
                    fog_factor_labels_list.extend([0, 1, 2])

                fog_factor_embeddings = torch.cat(fog_factor_list, dim=0)
                fog_factor_embeddings_norm = torch.norm(fog_factor_embeddings, p=2, dim=1).detach()
                fog_factor_embeddings = fog_factor_embeddings / fog_factor_embeddings_norm.unsqueeze(1).clamp(min=1e-6)
                fog_factor_labels = torch.LongTensor(fog_factor_labels_list).to(device)
                fog_pass_filter_loss = fogpassfilter_loss(fog_factor_embeddings, fog_factor_labels)

                total_fpf_loss +=  fog_pass_filter_loss 
              
                if is_main_process:
                    wandb.log({f'layer{idx}/fpf loss': fog_pass_filter_loss}, step=i_iter)
                    wandb.log({f'layer{idx}/total fpf loss': total_fpf_loss}, step=i_iter)

            scaler_fpf.scale(total_fpf_loss).backward(retain_graph=False)

            # Complete fogpass optimizer step BEFORE segmentation phase to avoid inplace op conflicts
            scaler_fpf.step(FogPassFilter1_optimizer)
            scaler_fpf.step(FogPassFilter2_optimizer)
            scaler_fpf.update()
            FogPassFilter1_optimizer.zero_grad()
            FogPassFilter2_optimizer.zero_grad()

            if args.modeltrain=='train':
                # train segmentation network
                # freeze the parameters of fog pass filtering modules

                model.train()
                for param in model.parameters():
                    param.requires_grad = True
                for param in FogPassFilter1.parameters():
                    param.requires_grad = False
                for param in FogPassFilter2.parameters():
                    param.requires_grad = False

                batch, cwsf_pair_loader_iter = fetch_next(cwsf_pair_loader_iter, cwsf_pair_loader)
                sf_image, cw_image, label, size, sf_name, cw_name = batch

                interp = nn.Upsample(size=(size[0][0],size[0][1]), mode='bilinear')

                if i_iter % 3 == 0:
                    with amp.autocast(enabled=bool(args.amp)):
                        images_sf = Variable(sf_image).to(device)
                        feature_sf0,feature_sf1,feature_sf2, feature_sf3,feature_sf4,feature_sf5 = model(images_sf)
                        pred_sf5 = interp(feature_sf5)
                        loss_seg_sf = loss_calc(pred_sf5, label, device)
                        images_cw = Variable(cw_image).to(device)
                        feature_cw0, feature_cw1, feature_cw2, feature_cw3, feature_cw4, feature_cw5 = model(images_cw)
                        pred_cw5 = interp(feature_cw5)
                        feature_cw5_logsoftmax = log_m(feature_cw5)
                        feature_sf5_softmax = m(feature_sf5)
                        feature_sf5_logsoftmax = log_m(feature_sf5)
                        feature_cw5_softmax = m(feature_cw5)
                        loss_con = kl_loss(feature_sf5_logsoftmax, feature_cw5_softmax)
                        loss_seg_cw = loss_calc(pred_cw5, label, device)     
                        fsm_weights = {'layer0':0.5, 'layer1':0.5}
                        sf_features = {'layer0':feature_sf0, 'layer1':feature_sf1}                
                        cw_features = {'layer0':feature_cw0, 'layer1':feature_cw1}

                    # Contrastive loss between SF and CW (paired by index)
                    cl_lambda = float(getattr(args, "lambda_cl", 0.0))
                    if cl_lambda > 0:
                        # Use a higher-level feature map for embeddings
                        emb_a = _global_pool_embedding(feature_sf4)
                        emb_b = _global_pool_embedding(feature_cw4)
                        n = emb_a.size(0)
                        emb = torch.cat([emb_a, emb_b], dim=0)
                        cl_labels = torch.arange(n, device=device, dtype=torch.long).repeat(2)
                        loss_cl = contrastive_loss_fn(emb, cl_labels)
                    else:
                        loss_cl = 0

                if i_iter % 3 == 1:
                    batch_rf, rf_loader_iter = fetch_next(rf_loader_iter, rf_loader)
                    rf_img,rf_size, rf_name = batch_rf
                    with amp.autocast(enabled=bool(args.amp)):
                        images_sf = Variable(sf_image).to(device)
                        feature_sf0,feature_sf1,feature_sf2, feature_sf3,feature_sf4,feature_sf5 = model(images_sf)
                        pred_sf5 = interp(feature_sf5)
                        loss_seg_sf = loss_calc(pred_sf5, label, device)       
                        loss_seg_cw = 0   
                        loss_con = 0
                        img_rf = Variable(rf_img).to(device)
                        feature_rf0, feature_rf1, feature_rf2, feature_rf3, feature_rf4, feature_rf5 = model(img_rf)    
                        rf_features = {'layer0':feature_rf0, 'layer1':feature_rf1}
                        sf_features = {'layer0':feature_sf0, 'layer1':feature_sf1}
                        fsm_weights = {'layer0':0.5, 'layer1':0.5}

                    # Contrastive loss between SF and RF
                    cl_lambda = float(getattr(args, "lambda_cl", 0.0))
                    if cl_lambda > 0:
                        emb_a = _global_pool_embedding(feature_sf4)
                        emb_b = _global_pool_embedding(feature_rf4)
                        n = emb_a.size(0)
                        emb = torch.cat([emb_a, emb_b], dim=0)
                        cl_labels = torch.arange(n, device=device, dtype=torch.long).repeat(2)
                        loss_cl = contrastive_loss_fn(emb, cl_labels)
                    else:
                        loss_cl = 0
                
                if i_iter % 3 == 2:
                    batch_rf, rf_loader_iter = fetch_next(rf_loader_iter, rf_loader)
                    rf_img,rf_size, rf_name = batch_rf
                    with amp.autocast(enabled=bool(args.amp)):
                        images_cw = Variable(cw_image).to(device)
                        feature_cw0, feature_cw1, feature_cw2, feature_cw3, feature_cw4, feature_cw5 = model(images_cw)
                        pred_cw5 = interp(feature_cw5)
                        loss_seg_sf = 0
                        loss_con = 0
                        loss_seg_cw = loss_calc(pred_cw5, label, device)      
                        img_rf = Variable(rf_img).to(device)
                        feature_rf0, feature_rf1, feature_rf2, feature_rf3, feature_rf4, feature_rf5 = model(img_rf)                  
                        rf_features = {'layer0':feature_rf0, 'layer1':feature_rf1}
                        cw_features = {'layer0':feature_cw0, 'layer1':feature_cw1}
                        fsm_weights = {'layer0':0.5, 'layer1':0.5}

                    # Contrastive loss between CW and RF
                    cl_lambda = float(getattr(args, "lambda_cl", 0.0))
                    if cl_lambda > 0:
                        emb_a = _global_pool_embedding(feature_cw4)
                        emb_b = _global_pool_embedding(feature_rf4)
                        n = emb_a.size(0)
                        emb = torch.cat([emb_a, emb_b], dim=0)
                        cl_labels = torch.arange(n, device=device, dtype=torch.long).repeat(2)
                        loss_cl = contrastive_loss_fn(emb, cl_labels)
                    else:
                        loss_cl = 0

                loss_fsm = 0
                fog_pass_filter_loss = 0

                for idx, layer in enumerate(fsm_weights):
                    # fog pass filter loss between different fog conditions a and b
                    if i_iter % 3 == 0:
                        a_feature = cw_features[layer]
                        b_feature = sf_features[layer]    
                    if i_iter % 3 == 1:
                        a_feature = rf_features[layer]
                        b_feature = sf_features[layer]
                    if i_iter % 3 == 2:
                        a_feature = rf_features[layer]
                        b_feature = cw_features[layer]   

                    layer_fsm_loss = 0
                    fog_pass_filter_loss = 0   
                    na,da,ha,wa = a_feature.size()
                    nb,db,hb,wb = b_feature.size()

                    if idx == 0:
                        fogpassfilter = FogPassFilter1
                        fogpassfilter_optimizer = FogPassFilter1_optimizer
                    elif idx == 1:
                        fogpassfilter = FogPassFilter2
                        fogpassfilter_optimizer = FogPassFilter2_optimizer

                    fogpassfilter.eval()

                    actual_batch_fsm = b_feature.size(0)
                    for batch_idx in range(actual_batch_fsm):
                        b_gram = gram_matrix(b_feature[batch_idx])
                        a_gram = gram_matrix(a_feature[batch_idx])

                        if i_iter % 3 == 1 or i_iter % 3 == 2:
                            a_gram = a_gram *(hb*wb)/(ha*wa)

                        vector_b_gram = upper_triangular_vector(b_gram).requires_grad_()
                        vector_a_gram = upper_triangular_vector(a_gram).requires_grad_()

                        fog_factor_b = fogpassfilter(vector_b_gram)
                        fog_factor_a = fogpassfilter(vector_a_gram)
                        half = int(fog_factor_b.shape[0]/2)
                        
                        layer_fsm_loss += fsm_weights[layer]*torch.mean((fog_factor_b/(hb*wb) - fog_factor_a/(ha*wa))**2)/half/ b_feature.size(0)

                    loss_fsm += layer_fsm_loss / float(actual_batch_fsm)

                cl_lambda = float(getattr(args, "lambda_cl", 0.0))
                loss = loss_seg_sf + loss_seg_cw + args.lambda_fsm*loss_fsm + args.lambda_con*loss_con + cl_lambda*loss_cl
                loss = loss / args.iter_size
                scaler_seg.scale(loss).backward()

                if loss_seg_cw != 0:
                    loss_seg_cw_value += loss_seg_cw.data.cpu().numpy() / args.iter_size
                if loss_seg_sf != 0:
                    loss_seg_sf_value += loss_seg_sf.data.cpu().numpy() / args.iter_size
                if loss_fsm != 0:
                    loss_fsm_value += loss_fsm.data.cpu().numpy() / args.iter_size
                if loss_con != 0:
                    loss_con_value += loss_con.data.cpu().numpy() / args.iter_size
                if loss_cl != 0:
                    loss_cl_value += float(loss_cl.detach().cpu().item()) / args.iter_size

            
                if is_main_process:
                    wandb.log({"fsm loss": args.lambda_fsm*loss_fsm_value}, step=i_iter)
                    wandb.log({'SF_loss_seg': loss_seg_sf_value}, step=i_iter)
                    wandb.log({'CW_loss_seg': loss_seg_cw_value}, step=i_iter)
                    wandb.log({'consistency loss':args.lambda_con*loss_con_value}, step=i_iter)
                    wandb.log({'contrastive loss': float(getattr(args, "lambda_cl", 0.0))*loss_cl_value}, step=i_iter)
                    wandb.log({'total_loss': loss}, step=i_iter)           

                for opt in opts:
                    scaler_seg.step(opt)
                scaler_seg.update()

        if i_iter < 20000:
            save_pred_every = 5000
            if args.modeltrain=='train':
                save_pred_every = 2000
        else:
            save_pred_every = args.save_pred_every

        if i_iter >= args.num_steps_stop - 1:
            if is_main_process:
                print('save model ..')
                torch.save(get_state_dict(model), osp.join(snapshot_dir, args.file_name + str(args.num_steps_stop) + '.pth'))
            break
        if is_main_process and args.modeltrain != 'train':
            if i_iter == 5000:
                torch.save({'state_dict':get_state_dict(model),
                'fogpass1_state_dict':get_state_dict(FogPassFilter1),
                'fogpass2_state_dict':get_state_dict(FogPassFilter2),
                'train_iter':i_iter,
                'args':args
                },osp.join(snapshot_dir, run_name)+'_fogpassfilter_'+str(i_iter)+'.pth')

        if is_main_process and i_iter % save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            save_dir = osp.join(f'./result/FIFO_model', args.file_name)
            
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            torch.save({
                'state_dict':get_state_dict(model),
                'fogpass1_state_dict':get_state_dict(FogPassFilter1),
                'fogpass2_state_dict':get_state_dict(FogPassFilter2),
                'train_iter':i_iter,
                'args':args
            },osp.join(snapshot_dir, run_name)+'_FIFO'+str(i_iter)+'.pth')
            
if __name__ == '__main__':
    main()