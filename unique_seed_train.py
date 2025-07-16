from pathlib import Path

import pandas as pd
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

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
    SupConLossWithTanimotoScoreConfig,
    DescriptionConfig,
    StorageConfig
)
from SpecEmbedding.data.datasets import TrainDataset
from SpecEmbedding.data.tokenizer import Tokenizer
from SpecEmbedding.models import SiameseModel
from SpecEmbedding.utils.model import search_with_spectra, search
from SpecEmbedding.utils.clean import get_classified_tokenset
from SpecEmbedding.loss import SupConLossWithTanimotoScore
from SpecEmbedding.const import gnps

spectra_paths = {
    "gnps": {
        "orbitrap": {
            "train": (gnps.ORBITRAP_TRAIN_QUERY, gnps.ORBITRAP_TEST_REF),
            "test": (gnps.ORBITRAP_TEST_QUERY, gnps.ORBITRAP_TEST_REF)
        },
        "qtof": {
            "test": (gnps.QTOF_TEST_QUERY, gnps.QTOF_TEST_REF)
        },
        "other": {
            "test": (gnps.OTHER_TEST_QUERY, gnps.OTHER_TEST_REF)
        }
    }
}
gnps_train_ref = np.load(gnps.ORBITRAP_TRAIN_REF, allow_pickle=True)

show_progress_bar = False
is_augment = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
k_metric = [5, 1, 10]

seeds = [42, 66, 88, 590, 666, 888, 3306, 4402, 6603, 9999]
batch_size = 512
replica_suffix = "-replication-{}"

for i, seed in enumerate(seeds, 1):
    set_seed(seed)
    ref_path = gnps.ORBITRAP_TRAIN_REF
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
    ref_sequences = tokenizer.tokenize_sequence(ref_spectra)
    print("tokenize the query and reference data success")
    data, labels = get_classified_tokenset(
        unique_smiles,
        ref_sequences,
        show_progress_bar
    )
    dataloader_config = DataLoaderConfig(
        batch_size=batch_size,
        val_ratio=0.2
    )
    length = len(labels)
    val_indices = np.random.choice(
        np.arange(length), int(dataloader_config["val_ratio"] * length), replace=False)
    val_keys = labels[val_indices]
    train_keys = np.delete(labels, val_indices)

    augment_config = AugmentationConfig(
        prob=0.5,
        removal_max=0.2,
        removal_intensity=0.3,
        rate_intensity=0.15
    )

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
        n_views=2
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
    model = SiameseModel(
        512,
        16,
        4,
        512,
        512,
        feedward_activation="selu"
    )
    model = model.to(device)
    trainer_config = TrainerConfig(
        n_epoch=50,
        device=device,
        early_stop=20,
        show_progress_bar=show_progress_bar
    )

    reduction = "mean"
    tanimoto_score_path = gnps.DIR / gnps.TANOMOTO_SCORE

    n_step = trainer_config["n_epoch"] * len(train_dataloader)
    scheduler_config = SchedulerConfig(
        warmup_steps=int(n_step * 0.1),
        total_steps=n_step
    )
    loss_config = SupConLossWithTanimotoScoreConfig(
        alpha=30,
        score_path=tanimoto_score_path,
        device=device,
        temperature=0.05,
        contrast_mode="all",
        base_temperature=0.05,
        reduction=reduction
    )
    criterion = SupConLossWithTanimotoScore(**loss_config)
    optimizer_config = OptimizerConfig(
        lr=7.5e-5,
        weight_decay=0.1
    )
    optimizer = AdamW(model.parameters(), **optimizer_config)
    desc_config = DescriptionConfig(
        train="train, epoch {}, loss={:.4f}",
        val="validation, epoch {}, loss={:.4f}",
        end="model train end, best model loss={:.4f}"
    )
    stepfunc_config = StepFuncConfig(
        train=step_train,
        val=step_evaluate
    )

    model_dir = Path(f"./replication_models/replication-seed{seed}")
    model_dir.mkdir(parents=True, exist_ok=True)
    storage_config = StorageConfig(
        model=model_dir / f"model.ckpt",
        lr=model_dir / f"lr.npy",
        step_loss=model_dir / f"step_loss.npy",
        loss=model_dir / "epoch_loss.npy",
        custom=model_dir / "custom_metric.npy"
    )
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
        16,
        4,
        512,
        512,
        feedward_activation="selu"
    )
    model_state = torch.load(storage_config["model"])
    model.load_state_dict(model_state)
    model = model.to(device)
    tester = ModelTester(model, device, show_progress_bar)

    replica_df_seq = []

    for i in range(10):
        df_seq = []
        for db, db_metadata in spectra_paths.items():
            for desc, path_metadata in db_metadata.items():
                for info, paths in path_metadata.items():
                    print("-" * 40, f"{db}-{desc}-{info}", "-" * 40)
                    query_path, ref_path = paths
                    query_path = query_path.with_stem(
                        query_path.stem + replica_suffix.format(i + 1))
                    ref_path = ref_path.with_stem(
                        ref_path.stem + replica_suffix.format(i + 1))
                    if db == "gnps" and desc == "orbitrap":
                        if info == "train":
                            query_path = gnps.ORBITRAP_TRAIN_QUERY

                        ref_spectra = np.load(ref_path, allow_pickle=True)
                        query_spectra = np.load(query_path, allow_pickle=True)
                        ref_spectra = np.hstack((gnps_train_ref, ref_spectra))
                        df = search_with_spectra(
                            f"{db}-{desc}-{info}", tester,
                            k_metric, tokenizer,
                            query_spectra, ref_spectra,
                            512,
                            show_progress_bar
                        )
                    else:
                        df = search(
                            f"{db}-{desc}-{info}", tester,
                            k_metric, tokenizer,
                            query_path, ref_path,
                            512,
                            show_progress_bar
                        )
                    df_seq.append(df)
        df = pd.concat(df_seq, axis=0)
        replica_df_seq.append(df)

    data = []
    indices = replica_df_seq[0].index
    columns = replica_df_seq[0].columns
    for item in replica_df_seq:
        data.append([item.values])

    data = np.concatenate(data, axis=0)
    np.set_printoptions(precision=2, suppress=True)
    print(np.mean(data, axis=0) * 100)
    print(np.std(data, axis=0) * 100)
    mean_df = pd.DataFrame(np.mean(data, axis=0) * 100,
                           index=indices, columns=columns)
    std_df = pd.DataFrame(np.std(data, axis=0) * 100,
                          index=indices, columns=columns)
    mean_df.to_csv(model_dir / "mean.tsv", sep='\t')
    std_df.to_csv(model_dir / "std.tsv", sep='\t')
