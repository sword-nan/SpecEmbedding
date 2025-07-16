from typing import Optional, Literal
from pathlib import Path
from collections.abc import Sequence
 
import torch
import pandas as pd
import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from numba import njit, prange
from matchms import Spectrum
from torch.utils.data import DataLoader

from SpecEmbedding.data.datasets import TestDataset
from SpecEmbedding.data.tokenizer import Tokenizer
from SpecEmbedding.trainer.trainer import ModelTester
from SpecEmbedding.models import SiameseModel
from SpecEmbedding.utils.clean import get_smiles

def embedding(
    tester: ModelTester, tokenizer: Tokenizer,
    batch_size: int,
    spectra: Sequence[Spectrum], show_progress_bar: bool = True
):
    sequences = tokenizer.tokenize_sequence(spectra)
    # print("tokenization success")
    smiles_seq = get_smiles(sequences, show_progress_bar)
    # print("extract smiles success")
    test_dataset = TestDataset(sequences)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size,
        False
    )
    embedding = tester.test(test_dataloader)
    # print("embedding success")
    return embedding, smiles_seq


@njit
def cosine_similarity(A: npt.NDArray, B: npt.NDArray):
    norm_A = np.sqrt(np.sum(A ** 2, axis=1)) + 1e-8
    norm_B = np.sqrt(np.sum(B ** 2, axis=1)) + 1e-8
    normalize_A = A / norm_A[:, np.newaxis]
    normalize_B = B / norm_B[:, np.newaxis]
    scores = np.dot(normalize_A, normalize_B.T)
    return scores


@njit(parallel=True)
def top_k_indices(score, top_k):
    rows, cols = score.shape
    indices = np.empty((rows, top_k), dtype=np.int64)
    for i in prange(rows):
        row = score[i]
        sorted_idx = np.argsort(row)[::-1]
        indices[i] = sorted_idx[:top_k]
    return indices


@njit
def eq(a: npt.NDArray, b: npt.NDArray):
    return np.equal(a, b)


def metric(
    k_metric: list[int],
    query_embedding: npt.NDArray,
    ref_embedding: npt.NDArray,
    query_smiles: npt.NDArray,
    ref_smiles: npt.NDArray,
    show_progress_bar: bool = True,
    batch_size: Optional[int] = None
):
    def init_count():
        count = {
            f"top{k}": 0
            for k in k_metric
        }
        return count

    def calculate_hit_count(start: int, end: int, indices: npt.NDArray):
        for query_index, ref_index in zip(range(start, end), indices):
            q = query_smiles[query_index]
            r = ref_smiles[ref_index]
            for k in k_metric:
                if q in r[:k]:
                    hit_count[f"top{k}"] += 1

    # def calculate_recall_count(start: int, end: int, indices: npt.NDArray):
    #     for query_index, ref_index in zip(range(start, end), indices):
    #         q = query_smiles[query_index]
    #         r = ref_smiles[ref_index]
    #         for k in k_metric:
    #             recall_count[f"top{k}"] += np.sum(r[:k] == q)

    k_metric = sorted(list(set(k_metric)))
    k_max = max(k_metric)
    if batch_size is None:
        batch_size = len(query_embedding)

    # candidate_recall_num = 0
    # pbar = query_smiles
    # if show_progress_bar:
    #     pbar = tqdm(query_smiles, total=len(query_smiles), desc="calculate all the candidates compounds")
    # for q in pbar:
    #     candidate_recall_num += np.sum(ref_smiles == q)
    # print(f"the recall count is {candidate_recall_num}")

    hit_count = init_count()
    # recall_count = init_count()
    start_seq = range(0, len(query_embedding), batch_size)
    end_seq = range(batch_size, len(query_embedding) + batch_size, batch_size)
    pbar = zip(start_seq, end_seq)
    if show_progress_bar:
        pbar = tqdm(pbar, total=len(start_seq),
                    desc="calculate hit and recall count")

    for start, end in pbar:
        score = cosine_similarity(query_embedding[start:end], ref_embedding)
        indices = top_k_indices(score, k_max)
        calculate_hit_count(start, end, indices)
        # calculate_recall_count(start, end, indices)

    for key in hit_count:
        hit_count[key] /= len(query_embedding)
        # recall_count[key] /= candidate_recall_num
    infos = []
    values = []
    for info, value in hit_count.items():
        infos.append(info)
        values.append(value)

    return infos, values

def load_tanimoto_supcon_aug_model(device):
    model_path = "/data1/xp/data/model/transformer-SupConWithTanimotoLoss-Augmentation/model.ckpt"
    model = SiameseModel(
        embedding_dim=512,
        n_head=16,
        n_layer=4,
        dim_feedward=512,
        dim_target=512,
        feedward_activation="selu"
    )
    model_state = torch.load(model_path, device)
    model.load_state_dict(model_state)
    model = model.to(device)
    return model


def load_tanimoto_supcon_model(device):
    model_path = "/data1/xp/code/specEmbedding/SpecEmbedding/model/transformer-SupConWithTanimotoLoss/model.ckpt"
    model = SiameseModel(
        embedding_dim=512,
        n_head=16,
        n_layer=4,
        dim_feedward=512,
        dim_target=512,
        feedward_activation="gelu"
    )
    model_state = torch.load(model_path, device)
    model.load_state_dict(model_state)
    model = model.to(device)
    return model


def load_supcon_model(device):
    model_path = "/data1/xp/code/specEmbedding/SpecEmbedding/model/transformer-SupConLoss/model.ckpt"
    model = SiameseModel(
        embedding_dim=512,
        n_head=16,
        n_layer=4,
        dim_feedward=512,
        dim_target=200,
        feedward_activation="relu"
    )
    model_state = torch.load(model_path, device)
    model.load_state_dict(model_state)
    model = model.to(device)
    return model


def load_tanimoto_model(device):
    model_path = "/data1/xp/code/specEmbedding/SpecEmbedding/model/transformer-TanimotoLoss/model.ckpt"
    model = SiameseModel(
        512,
        16,
        2,
        512,
        500,
        feedward_activation="selu",
    )
    model_state = torch.load(model_path, map_location=device)
    model.load_state_dict(model_state)
    model = model.to(device)
    return model


def load_transformer_model(
    device, 
    loss_type: Literal["TanimotoLoss", "SupConLoss", "SupConWithTanimotoLoss"],
    is_augment: bool=False
):
    if loss_type == "TanimotoLoss":
        model = load_tanimoto_model(device)
    elif loss_type == "SupConLoss":
        model = load_supcon_model(device)

    elif loss_type == "SupConWithTanimotoLoss":
        if not is_augment:
            model = load_tanimoto_supcon_model(device)
        else:
            model = load_tanimoto_supcon_aug_model(device)
    else:
        raise ValueError(f"No such {loss_type} model")

    return model


def search_with_spectra(
    desc: str, tester: ModelTester,
    k_metric: list[int], tokenizer: Tokenizer,
    query_spectra: Sequence[Spectrum], ref_spectra: Sequence[Spectrum],
    loader_batch_size: int,
    show_progress_bar: bool = True,
    batch_size: Optional[int] = None
):
    query_embedding, query_smiles = embedding(
        tester, tokenizer,
        loader_batch_size,
        query_spectra, show_progress_bar
    )
    ref_embedding, ref_smiles = embedding(
        tester, tokenizer,
        loader_batch_size,
        ref_spectra, show_progress_bar
    )
    infos, values = metric(
        k_metric,
        query_embedding,
        ref_embedding,
        query_smiles,
        ref_smiles,
        show_progress_bar,
        batch_size
    )
    df = pd.DataFrame([values], columns=infos, index=[desc])
    return df


def search(
    desc: str, tester: ModelTester,
    k_metric: list[int], tokenizer: Tokenizer,
    query_path: Path, ref_path: Path,
    loader_batch_size: int,
    show_progress_bar: bool = True,
    batch_size: Optional[int] = None
):
    query_spectra = np.load(query_path, allow_pickle=True)
    ref_spectra = np.load(ref_path, allow_pickle=True)
    query_embedding, query_smiles = embedding(
        tester, tokenizer,
        loader_batch_size,
        query_spectra, show_progress_bar
    )
    ref_embedding, ref_smiles = embedding(
        tester, tokenizer,
        loader_batch_size,
        ref_spectra, show_progress_bar
    )
    infos, values = metric(
        k_metric,
        query_embedding,
        ref_embedding,
        query_smiles,
        ref_smiles,
        show_progress_bar,
        batch_size
    )
    df = pd.DataFrame([values], columns=infos, index=[desc])
    return df


def most_similar(
    query_embedding: npt.NDArray,
    ref_embedding: npt.NDArray,
    batch_size: int,
    show_progress_bar: bool = True
):
    start_seq = range(0, len(query_embedding), batch_size)
    end_seq = range(batch_size, len(query_embedding) + batch_size, batch_size)
    pbar = zip(start_seq, end_seq)
    if show_progress_bar:
        pbar = tqdm(pbar, total=len(start_seq),
                    desc="processing")

    scores = []
    most_similar_indices = []

    for start, end in pbar:
        cosine_score = cosine_similarity(query_embedding[start:end], ref_embedding)
        indices = top_k_indices(cosine_score, 1).flatten()
        for i, j in zip(range(cosine_score.shape[0]), indices):
            scores.append(cosine_score[i][j])
            most_similar_indices.append(j)
    return np.array(scores), np.array(most_similar_indices)