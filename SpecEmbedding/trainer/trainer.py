import math
import random
from typing import Literal, Optional, Union

from tqdm import tqdm
import numpy as np
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from SpecEmbedding.type import (
    StepTrain,
    StepVal,
    TrainerConfig,
    SchedulerConfig,
    DescriptionConfig,
    StepFuncConfig,
    StorageConfig,
    CustomMetricConfig
)

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_device(n: Literal[0, 1]):
    if torch.cuda.is_available():
        return torch.device(f"cuda:{n}")
    return torch.device("cpu")

def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
            optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
            num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
            num_training_steps (:obj:`int`):
            The total number of training steps.
            num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
            last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
            :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    def lr_lambda(current_step):
        # Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # decadence
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi *
                        float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

class Trainer:
    def __init__(
        self,
        model: Module,
        reduction: Literal["mean", "sum"],
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        criterion: Module,
        device: torch.device,
        step_fn: StepFuncConfig,
        desc_config: DescriptionConfig,
        storage_config: StorageConfig,
        trainer_config: TrainerConfig,
        scheduler_config: SchedulerConfig,
        custom_metric_config: Optional[CustomMetricConfig] = None
    ) -> None:
        """
            Trainer class
            used for training and validating the model
            saving the top-1 model based on validation set performance.

            Parameters:
            ---
            -   model: The model to be trained
            -   reduction: Whether the loss function takes the mean or sum
            -   train_loader: Training data
            -   val_loader: Validation data
            -   optimizer: Optimizer
            -   criterion: Loss function
            -   device: Device used for training
            -   step_fn: Mini-batch step function
            -   desc_config: Log statements during training/validation/end
            -   storage_config: Configuration for storage paths
            -   trainer_config: Configuration for training epochs, early stopping, etc.
            -   scheduler_config: Learning rate scheduler configuration
            -   custom_metric_config: Custom metric configuration (used for evaluating validation set metrics instead of using the criterion, with a custom metric function)
        """
        self.model = model
        self.reduction = reduction
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.step_fn = step_fn
        self.desc_config = desc_config
        self.storage_config = storage_config
        self.trainer_config = trainer_config
        self.scheduler_config = scheduler_config
        self.custom_metric_config = custom_metric_config

        self.scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            scheduler_config["warmup_steps"],
            scheduler_config['total_steps']
        )

        self._lr_seq = []
        self._step_loss_seq = []

    @property
    def lrs(self):
        return self._lrs.copy()

    @property
    def step_loss(self):
        return self._step_loss.copy()

    def step_value(self, value: int, size: int, reduction: Literal["mean", "sum"]):
        if reduction == "mean":
            return value * size
        else:
            return value

    def process_epoch(self, epoch: int, dataloader: DataLoader, step_fn: Union[StepTrain, StepVal], is_train: bool = True):
        num = 0
        loss_sum = 0
        custom_metric_sum = 0

        pbar = dataloader
        if self.trainer_config["show_progress_bar"]:
            pbar = tqdm(dataloader, total=len(dataloader))
            if is_train:
                pbar.set_description(
                    f"Train: Epoch [{epoch + 1}/{self.trainer_config['n_epoch']}]")
            else:
                pbar.set_description(
                    f"Validation: Epoch [{epoch + 1}/{self.trainer_config['n_epoch']}]")

        if is_train:
            self.model.train()
        else:
            self.model.eval()

        for batch in pbar:
            size = len(batch[-1])
            num += size
            if self.custom_metric_config is None:
                step_loss = step_fn(
                    self.model, self.criterion,
                    self.device, batch, None
                )
            else:
                step_loss, custom_metric = step_fn(
                    self.model, self.criterion,
                    self.device, batch, self.custom_metric_config["fn"]
                )
                custom_metric_sum += self.step_value(
                    custom_metric, size, self.reduction
                )

            self._step_loss_seq.append(step_loss.item())

            loss_sum += self.step_value(
                step_loss.item(), size, self.reduction
            )

            if is_train:
                self.optimizer.zero_grad()
                step_loss.backward()
                # grad clip
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                self._lr_seq.append(self.optimizer.param_groups[0]['lr'])
                self.scheduler.step()

            if self.trainer_config["show_progress_bar"]:

                if self.custom_metric_config is not None:
                    pbar.set_postfix_str(
                        f"{self.custom_metric_config["name"]}={custom_metric.item():.3f}, loss={step_loss.item(): .4f}")
                else:
                    pbar.set_postfix(loss=f'{step_loss.item(): .4f}')

        return loss_sum / num, custom_metric_sum / num

    def train(self):
        def checkpoint():
            torch.save(self.model.state_dict(), self.storage_config["model"])

        def maximum(x, y):
            return x > y

        def minimum(x, y):
            return x < y

        loss_metrics = {
            'train': [],
            'validation': []
        }

        custom_metrics = {
            'train': [],
            'validation': []
        }

        best_model_loss = torch.inf
        best_custom_metric = torch.inf
        is_save = minimum

        if self.custom_metric_config is not None and self.custom_metric_config['destination'] == "maximum":
            best_custom_metric = 0
            is_save = maximum

        early_stop = 0
        best_epoch = 0
        checkpoint()
        for epoch in range(self.trainer_config["n_epoch"]):
            train_loss, train_custom_metric = self.process_epoch(
                epoch, self.train_loader,
                self.step_fn["train"], True
            )

            if self.custom_metric_config is not None:
                print(
                    self.desc_config["train"].format(
                        epoch + 1,
                        train_loss,
                        self.custom_metric_config["name"],
                        train_custom_metric
                    )
                )
            else:
                print(
                    self.desc_config["train"].format(
                        epoch + 1,
                        train_loss
                    )
                )

            with torch.no_grad():
                val_loss, val_custom_metric = self.process_epoch(
                    epoch, self.val_loader,
                    self.step_fn["val"], False
                )

            if self.custom_metric_config is not None:
                if is_save(val_custom_metric, best_custom_metric):
                    early_stop = 0
                    best_epoch = epoch
                    best_model_loss = val_loss
                    best_custom_metric = val_custom_metric
                    print(
                        self.desc_config["val"].format(
                            epoch + 1,
                            val_loss,
                            self.custom_metric_config["name"],
                            val_custom_metric
                        )
                    )
                    checkpoint()
                else:
                    early_stop += 1
            else:
                if val_loss < best_model_loss:
                    early_stop = 0
                    best_epoch = epoch
                    best_model_loss = val_loss
                    print(
                        self.desc_config["val"].format(
                            epoch + 1,
                            val_loss
                        )
                    )
                    checkpoint()
                else:
                    early_stop += 1

            loss_metrics['train'].append(train_loss)
            loss_metrics['validation'].append(val_loss)
            custom_metrics["train"].append(train_custom_metric)
            custom_metrics["validation"].append(val_custom_metric)
            if early_stop == self.trainer_config["early_stop"]:
                print(f"early stop at epoch {epoch}")
                break

        print(f"best model saved in epoch {best_epoch}")
        if self.custom_metric_config is not None:
            print(
                self.desc_config["end"].format(
                    best_model_loss,
                    self.custom_metric_config["name"],
                    best_custom_metric
                )
            )
        else:
            print(
                self.desc_config["end"].format(
                    best_model_loss,
                )
            )

        print("store train metrics")
        np.save(self.storage_config["loss"], loss_metrics)
        np.save(self.storage_config["custom"], custom_metrics)
        np.save(self.storage_config["lr"], self._lr_seq)
        np.save(self.storage_config["step_loss"], self._step_loss_seq)

class ModelTester:
    def __init__(
        self,
        model: Module,
        device: torch.device,
        show_prgress_bar: bool = True
    ) -> None:
        self.model = model
        self.device = device
        self.show_prgress_bar = show_prgress_bar

    def test(self, dataloader: DataLoader):
        self.model.eval()
        result = []
        with torch.no_grad():
            pbar = dataloader
            if self.show_prgress_bar:
                pbar = tqdm(dataloader, total=len(
                    dataloader), desc="embedding")
            for x in pbar:
                x = [d.to(self.device) for d in x]
                pred: torch.Tensor = self.model(*x)
                result.append(pred.cpu().numpy())
        return np.concatenate(result, axis=0)
