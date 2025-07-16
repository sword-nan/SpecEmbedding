from typing import TypedDict, Sequence, Callable, Optional, Literal

import torch
from torch import nn
from torch import device
import numpy as np
import numpy.typing as npt


BatchType = Sequence[torch.Tensor]
StepTrain = Callable[[nn.Module, nn.Module, device,
                      BatchType, Optional[Callable[..., int]]], Sequence[torch.Tensor]]
StepVal = Callable[[nn.Module, nn.Module, device,
                    BatchType, Optional[Callable[..., int]]], Sequence[torch.Tensor]]


class Peak(TypedDict):
    mz: str
    intensity: npt.NDArray


class MetaData(TypedDict):
    peaks: Sequence[Peak]
    smiles: str


class TokenSequence(TypedDict):
    mz: npt.NDArray[np.int32]
    intensity: npt.NDArray[np.float32]
    mask: npt.NDArray[np.bool_]
    smiles: str


class OptimizerConfig(TypedDict):
    lr: float
    weight_decay: float


class TokenizerConfig(TypedDict):
    max_len: int
    show_progress_bar: bool


class TrainerConfig(TypedDict):
    n_epoch: int
    early_stop: int
    device: device
    show_progress_bar: bool


class SchedulerConfig(TypedDict):
    warmup_steps: int
    total_steps: int


class DescriptionConfig(TypedDict):
    train: str
    val: str
    end: str


class StepFuncConfig(TypedDict):
    train: StepTrain
    val: StepVal


class DataLoaderConfig(TypedDict):
    batch_size: int
    val_ratio: float


class StorageConfig(TypedDict):
    model: str
    lr: str
    step_loss: str
    loss: str
    custom: str


class CustomMetricConfig(TypedDict):
    name: str
    fn: Callable[..., int]
    destination: Literal["maximum", "minimum"]


class TanimotoLossConfig(TypedDict):
    score_path: str
    device: torch.device
    reduction: Optional[Literal["mean", "sum"]]


class SupConLossConfig(TypedDict):
    device: torch.device
    temperature: Optional[float]
    contrast_mode: Optional[Literal["one", "all"]]
    base_temperature: Optional[float]
    reduction: Optional[Literal["mean", "sum"]]


class SupConLossWithTanimotoScoreConfig(TypedDict):
    alpha: int
    score_path: str
    device: torch.device
    temperature: Optional[float]
    contrast_mode: Optional[Literal["one", "all"]]
    base_temperature: Optional[float]
    reduction: Optional[Literal["mean", "sum"]]


class AugmentationConfig(TypedDict):
    prob: float
    removal_max: float
    removal_intensity: float
    rate_intensity: float


DefaultAugmentationConifg = AugmentationConfig(
    prob=0.5,
    removal_max=0.2,
    removal_intensity=0.3,
    rate_intensity=0.15,
)
