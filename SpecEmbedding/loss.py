from typing import Literal

import numpy as np
import numpy.typing as npt
import torch
from torch import nn, device
import torch.nn.functional as F


class TanimotoScoreLoss(nn.Module):
    def __init__(self, score_path: str, device: device, reduction: Literal["mean", "sum"] = "mean") -> None:
        super(TanimotoScoreLoss, self).__init__()
        # lazy read
        self.tanimoto_score: npt.NDArray[np.float32] = np.load(
            score_path, mmap_mode='r'
        )
        self.mse_loss = torch.nn.MSELoss(reduction=reduction)
        self.device = device

    def target_score(self, row: npt.NDArray):
        return np.array(self.tanimoto_score[row][:, row], dtype=np.float32)

    def forward(self, features: torch.Tensor, labels: torch.Tensor = None):
        n_views = features.shape[1]
        tanimoti_score = torch.tensor(
            self.target_score(labels.cpu().numpy())
        ).to(self.device)

        tanimoti_score = tanimoti_score.repeat_interleave(
            n_views, 0).repeat_interleave(n_views, 1)

        normalized_feature = F.normalize(
            features.reshape(-1, features.shape[-1]),
            dim=-1
        )

        pred_cosine_score = torch.matmul(
            normalized_feature, normalized_feature.T
        )

        return self.mse_loss(tanimoti_score, pred_cosine_score)


class SupConLoss(nn.Module):
    def __init__(
        self, device: torch.device,
        temperature=0.2, contrast_mode: Literal["one", "all"] = 'all',
        base_temperature=0.2, reduction: Literal["mean", "sum"] = "mean"
    ):
        super(SupConLoss, self).__init__()
        self.device = device
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.reduction = reduction
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, features: torch.Tensor, labels: torch.Tensor = None, mask: torch.Tensor = None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        features = F.normalize(features, dim=-1)
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError(
                    'Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(self.device)
        else:
            mask = mask.float().to(self.device)

        contrast_count = features.shape[1]
        # [batch, n_views, feature] unbind-> tuple([batch, fearure], n_views length)
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # 使用内积计算相似度分数
        anchor_dot_contrast = torch.div(
            self.leaky_relu(torch.matmul(anchor_feature, contrast_feature.T)),
            self.temperature)

        # for numerical stability
        # 由于后续需要计算 softmax，如果直接计算会导致数值不稳定，因此减去最大值
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1,
                                                         1).to(self.device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        # 对数运算法则 log(a/b) = log(a) - log(b)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size)
        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()
        else:
            raise ValueError(f"Unkown reduction: {self.reduction}")

        return loss


class SupConLossWithTanimotoScore(nn.Module):
    def __init__(
        self, alpha: int, score_path: str, device: device,
        temperature=0.2, contrast_mode: Literal["one", "all"] = 'all',
        base_temperature=0.2, reduction: Literal["mean", "sum"] = "mean"
    ) -> None:
        super(SupConLossWithTanimotoScore, self).__init__()
        self.supcon_loss = SupConLoss(
            device, temperature,
            contrast_mode, base_temperature, reduction
        )
        self.tanimotoscore_loss = TanimotoScoreLoss(
            score_path, device, reduction
        )
        self.alpha = alpha

    def forward(self, features: torch.Tensor, labels: torch.Tensor = None, mask: torch.Tensor = None):
        supcontrastive_loss = self.supcon_loss(features, labels, mask)
        tanimoto_loss = self.tanimotoscore_loss(features, labels)
        return supcontrastive_loss + self.alpha * tanimoto_loss
