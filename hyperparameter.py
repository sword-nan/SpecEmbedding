from pathlib import Path
from typing import Sequence, Optional, Literal
from argparse import ArgumentParser

import pandas as pd
import numpy as np
import numpy.typing as npt
import optuna
from optuna.trial import Trial
import optuna.importance
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from matchms import Spectrum

from SpecEmbedding.models import SiameseModel
from SpecEmbedding.trainer.trainer import Trainer, ModelTester, set_seed
from SpecEmbedding.trainer.fn import step_train, step_evaluate
from SpecEmbedding.type import (
    AugmentationConfig,
    OptimizerConfig,
    TokenizerConfig,
    DataLoaderConfig,
    StepFuncConfig,
    SchedulerConfig,
    TrainerConfig,
    SupConLossConfig,
    SupConLossWithTanimotoScoreConfig,
    TanimotoLossConfig,
    DescriptionConfig,
    StorageConfig
)
from SpecEmbedding.data.datasets import TrainDataset, TestDataset, TokenSequence
from SpecEmbedding.data.tokenizer import Tokenizer
from SpecEmbedding.utils.clean import get_classified_tokenset
from SpecEmbedding.loss import SupConLossWithTanimotoScore, SupConLoss, TanimotoScoreLoss
from SpecEmbedding.utils.model import embedding, metric
from SpecEmbedding.const import gnps


def objective(
    trial: Trial,
    seed: int,
    device: torch.device,
    model_backbone: Literal["transformer", "mamba"],
    loss_type: Literal["SupConLoss", "TanimotoLoss", "SupConWithTanimotoLoss"],
    data: dict[str, list[TokenSequence]],
    labels: npt.NDArray,
    metric_fn,
    show_progress_bar: Optional[bool] = True,
    is_augment: bool = False
):
    torch.cuda.empty_cache()
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])
    n_layer = trial.suggest_categorical('n_layer', [2, 3, 4, 5, 6])
    n_epoch = trial.suggest_int('n_epoch', 30, 100)
    lr = trial.suggest_float('lr', 1e-6, 1e-3)
    dim_target = trial.suggest_categorical(
        'dim_target', [100, 200, 300, 400, 500, 512])
    weight_decay = trial.suggest_categorical("weight_decay", [0.1, 0.05, 0.01])

    feedward_activation = trial.suggest_categorical(
        "feedward_activation", ["gelu", "selu", "relu"])

    desc = f"model: {model_backbone}, loss_type: {loss_type}, batch_size: {batch_size}, n_layer: {n_layer}, n_epoch: {n_epoch}, lr: {lr}, dim_target: {dim_target}, weight_decay: {weight_decay}, feedward_activation: {feedward_activation}"

    augment_config = None

    if is_augment:
        removal_max = trial.suggest_categorical(
            "removal_max",
            [0.05, 0.1, 0.15, 0.2]
        )
        removal_intensity = trial.suggest_categorical(
            "removal_intensity",
            [0.05, 0.1, 0.2, 0.3]
        )
        rate_intensity = trial.suggest_categorical(
            "rate_intensity",
            [0.05, 0.1, 0.15, 0.2]
        )
        augment_config = AugmentationConfig(
            prob=0.5,
            removal_max=removal_max,
            removal_intensity=removal_intensity,
            rate_intensity=rate_intensity
        )

        desc += f", removal_max: {removal_max}, removal_intensity: {removal_intensity}, rate_intensity: {rate_intensity}"

    # dataset and dataloader
    set_seed(seed)

    dataloader_config = DataLoaderConfig(
        batch_size=batch_size,
        val_ratio=0.2
    )

    length = len(labels)
    val_indices = np.random.choice(
        np.arange(length), int(dataloader_config["val_ratio"] * length), replace=False)
    val_keys = labels[val_indices]
    train_keys = np.delete(labels, val_indices)

    train_dataset = TrainDataset(
        data=data,
        keys=train_keys,
        n_views=2,
        is_augment=is_augment,
        augment_config=augment_config
    )

    val_dataset = TrainDataset(
        data=data,
        keys=val_keys,
        n_views=2,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=dataloader_config["batch_size"],
        shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=dataloader_config["batch_size"],
        shuffle=False
    )
    # model
    if model_backbone == "transformer":
        n_head = trial.suggest_categorical('n_head', [4, 8, 16])
        desc += f", n_head: {n_head}"
        model = SiameseModel(
            512,
            n_head,
            n_layer,
            512,
            dim_target,
            feedward_activation=feedward_activation
        )
    else:
        raise ValueError(f"No such model {model_backbone}")

    model = model.to(device)

    # trainer
    trainer_config = TrainerConfig(
        n_epoch=n_epoch,
        device=device,
        early_stop=20,
        show_progress_bar=show_progress_bar
    )
    # scheduler
    n_step = trainer_config["n_epoch"] * len(train_dataloader)
    scheduler_config = SchedulerConfig(
        warmup_steps=int(n_step * 0.1),
        total_steps=n_step
    )
    # loss
    reduction = "mean"
    tanimoto_score_path = gnps.DIR / gnps.TANOMOTO_SCORE

    if loss_type == "TanimotoLoss":
        loss_config = TanimotoLossConfig(
            score_path=tanimoto_score_path,
            device=device,
            reduction=reduction
        )
        criterion = TanimotoScoreLoss(**loss_config)
    else:
        temperature = trial.suggest_categorical(
            'temperature', [0.5, 0.25, 0.1, 0.07, 0.05, 0.01, 0.005]
        )

        base_temperature = trial.suggest_categorical(
            'base_temperature', [0.5, 0.25, 0.1, 0.07, 0.05, 0.01, 0.005]
        )

        desc += f", temperature: {temperature}, base_temperature: {base_temperature}"

        if loss_type == "SupConLoss":
            loss_config = SupConLossConfig(
                device=device,
                temperature=temperature,
                base_temperature=base_temperature,
                contrast_mode='all',
                reduction=reduction
            )
            criterion = SupConLoss(**loss_config)
        else:
            alpha = trial.suggest_categorical("alpha", [20, 30, 40, 50])
            desc += f", alpha: {alpha}"
            loss_config = SupConLossWithTanimotoScoreConfig(
                alpha=alpha,
                score_path=tanimoto_score_path,
                device=device,
                temperature=temperature,
                contrast_mode="all",
                base_temperature=base_temperature,
                reduction=reduction
            )
            criterion = SupConLossWithTanimotoScore(**loss_config)

    # optimizer
    optimizer_config = OptimizerConfig(
        lr=lr,
        weight_decay=weight_decay
    )
    optimizer = AdamW(model.parameters(), **optimizer_config)

    # desc
    desc_config = DescriptionConfig(
        train="train, epoch {}, loss={:.4f}",
        val="validation, epoch {}, loss={:.4f}",
        end="model train end, best model loss={:.4f}"
    )

    # step func
    stepfunc_config = StepFuncConfig(
        train=step_train,
        val=step_evaluate
    )

    suffix = ""
    if is_augment:
        suffix = "-Augmentation"

    # storage
    model_dir = Path(f"./model/{model_backbone}-{loss_type}{suffix}")

    model_dir.mkdir(parents=True, exist_ok=True)
    storage_config = StorageConfig(
        model=model_dir / f"model.ckpt",
        lr=model_dir / f"lr.npy",
        step_loss=model_dir / f"step_loss.npy",
        loss=model_dir / "epoch_loss.npy",
        custom=model_dir / "custom_metric.npy"
    )

    # model train
    trainer = Trainer(
        model,
        reduction,
        train_dataloader,
        val_dataloader,
        optimizer,
        criterion,
        device,
        stepfunc_config,
        desc_config,
        storage_config,
        trainer_config,
        scheduler_config
    )

    trainer.train()

    # test
    model = SiameseModel(
        512,
        n_head,
        n_layer,
        512,
        dim_target,
        feedward_activation=feedward_activation
    )
    model_state = torch.load(storage_config["model"])
    model.load_state_dict(model_state)
    model = model.to(device)
    print(desc)
    print("calculate the metric")
    metric = metric_fn(model)
    print(metric)
    return metric


if __name__ == '__main__':
    def metric_fn(
        model: SiameseModel, device: torch.device,
        tokenizer: Tokenizer,
        query_spectra: Sequence[Spectrum], ref_spectra: Sequence[Spectrum],
        show_progress_bar: Optional[bool] = True
    ):
        tester = ModelTester(model, device, show_progress_bar)
        query_embedding, query_smiles = embedding(
            tester, tokenizer, 512, query_spectra, show_progress_bar)
        ref_embedding, ref_smiles = embedding(
            tester, tokenizer, 512, ref_spectra, show_progress_bar)
        _, values = metric(
            [1, 5, 10], query_embedding, ref_embedding,
            query_smiles, ref_smiles, show_progress_bar
        )
        return values[0]

    candidate_loss_fn = [
        "SupConLoss",
        "TanimotoLoss",
        "SupConWithTanimotoLoss"
    ]
    candidate_models = ["transformer", "mamba"]
    parser = ArgumentParser()
    parser.add_argument(
        "--loss",
        type=str,
        help="loss type",
        default="SupConLoss"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed",
        default=42
    )
    parser.add_argument(
        "--model",
        type=str,
        help="backbone model",
        default="transformer"
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="if show progress",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="if use data augment"
    )
    args = parser.parse_args()

    if args.loss not in candidate_loss_fn:
        raise ValueError("input loss type not in candidate loss fuctions")
    if args.seed < 0:
        raise ValueError("input random seed must be non negative number")
    if args.model not in candidate_models:
        raise ValueError("input model backbone must be transformer or mamba")

    seed = args.seed
    show_progress_bar = args.progress
    is_augment = args.augment
    query_path = gnps.ORBITRAP_TRAIN_QUERY
    ref_path = gnps.ORBITRAP_TRAIN_REF
    query_spectra = np.load(query_path, allow_pickle=True)
    ref_spectra = np.load(ref_path, allow_pickle=True)
    unique_smiles = np.load(
        gnps.DIR / gnps.UNIQUE_SMILES,
        allow_pickle=True
    )
    print("read raw spectra success")

    tokenizer_config = TokenizerConfig(
        max_len=100,
        show_progress_bar=show_progress_bar
    )
    tokenizer = Tokenizer(**tokenizer_config)
    query_sequences = tokenizer.tokenize_sequence(query_spectra)
    ref_sequences = tokenizer.tokenize_sequence(ref_spectra)
    print("tokenize the query and reference data success")
    data, labels = get_classified_tokenset(
        unique_smiles,
        ref_sequences,
        show_progress_bar
    )
    query_dataset = TestDataset(query_sequences)
    ref_dataset = TestDataset(ref_sequences)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    study_name = 'SpecEmbedding'
    study = optuna.create_study(study_name=study_name, direction="maximize")
    study.optimize(
        lambda trail: objective(
            trail,
            seed,
            device,
            args.model,
            args.loss,
            data,
            labels,
            lambda model: metric_fn(
                model, device, tokenizer,
                query_spectra, ref_spectra, show_progress_bar
            ),
            show_progress_bar,
            is_augment
        ),
        n_trials=100,
        show_progress_bar=show_progress_bar,
    )
    params = study.best_params
    print(params)
    df: pd.DataFrame = study.trials_dataframe(
        attrs=('number', 'value', 'params', 'state')
    )
    print(df)
    df.to_csv(f"{args.model}_{args.loss}_params.tsv", sep='\t')
    importance = optuna.importance.get_param_importances(study)
    for param, importance_value in importance.items():
        print(f"Parameter: {param}, Importance: {importance_value:.4f}")