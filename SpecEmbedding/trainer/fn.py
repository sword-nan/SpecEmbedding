from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import device

from SpecEmbedding.type import BatchType


def step_train(
    model: nn.Module,
    criterion: nn.Module,
    device: device,
    batch: BatchType,
    custom_fn: Callable[..., torch.Tensor] = None
):
    x, y = batch
    y = y.to(device)
    pred = []
    for item in x:
        item = [d.to(device) for d in item]
        res: torch.Tensor = model(*item).unsqueeze(dim=1)
        pred.append(res)
    pred = torch.cat(pred, dim=1)
    loss: torch.Tensor = criterion(pred, y)
    if custom_fn is not None:
        custom_metric = custom_fn(pred.detach(), y.detach())
        return loss, custom_metric

    return loss


def step_evaluate(
    model: nn.Module,
    criterion: nn.Module,
    device: device,
    batch: BatchType,
    custom_fn: Callable[..., torch.Tensor] = None
):
    x, y = batch
    y = y.to(device)
    pred = []
    for item in x:
        item = [d.to(device) for d in item]
        res: torch.Tensor = model(*item).unsqueeze(dim=1)
        pred.append(res)
    pred = torch.cat(pred, dim=1)
    loss: torch.Tensor = criterion(pred, y)
    if custom_fn is not None:
        custom_metric = custom_fn(pred.detach(), y.detach())
        return loss, custom_metric

    return loss

def top1_accuracy(pred: torch.Tensor, y: torch.Tensor):
    device = pred.device
    replica_count = pred.shape[1]
    batch_size = y.shape[0]

    fearure = torch.cat(
        torch.unbind(pred, dim=1),
        dim=0
    )

    y = y.view(-1, 1)
    mask = torch.eq(y, y.T).float().to(device)
    mask = mask.repeat(replica_count, replica_count)
    slash_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size * replica_count).view(-1, 1).to(device),
        0
    )
    mask = mask * slash_mask
    feature = F.normalize(
        fearure,
        dim=-1
    )
    cosine_score = torch.matmul(
        feature,
        feature.T
    )
    cosine_score = cosine_score * slash_mask
    hit = torch.gather(
        mask,
        dim=1,
        index=torch.argmax(cosine_score, dim=1, keepdim=True)
    ).squeeze()
    
    return hit.sum() / hit.shape[0]
