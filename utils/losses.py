import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable


class CrossEntropy2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3

        n, c, h, w = predict.size()
        n1, h1, w1 = target.size()
        assert n == n1 and h == h1 and w == w1

        # Build valid mask (ignore_label filtered). Use boolean mask to avoid
        # allocating an (N,H,W,C) repeated tensor which can OOM at high res.
        valid_mask = (target != self.ignore_label)
        target_flat = target.view(-1)
        mask_flat = valid_mask.view(-1)
        target_sel = target_flat[mask_flat]
        if target_sel.numel() == 0:
            return predict.sum() * 0.0

        # (N,C,H,W) -> (N*H*W, C)
        predict_flat = predict.permute(0, 2, 3, 1).contiguous().view(-1, c)
        predict_sel = predict_flat[mask_flat]

        # Use modern reduction argument; keep original semantics.
        reduction = "mean" if self.size_average else "sum"
        return F.cross_entropy(predict_sel, target_sel, weight=weight, reduction=reduction)

