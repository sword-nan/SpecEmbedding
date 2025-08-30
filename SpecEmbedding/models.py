import math
from typing import Literal, Union, Iterable, Tuple

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

LAMBDA_MIN = math.pow(10, -3.0)
LAMBDA_MAX = math.pow(10, 3.0)

class MultiFeedForwardModule(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: Union[int, Iterable[int]],
        output_size: int,
        *,
        activation: Literal['relu', 'selu', 'gelu'] = 'relu',
        dropout: float = 0.1,
        dropout_last_layer: bool = True
    ):
        super(MultiFeedForwardModule, self).__init__()
        if activation == 'relu':
            self._activation = nn.ReLU()
        elif activation == 'selu':
            self._activation = nn.SELU()
        elif activation == 'gelu':
            self._activation = nn.GELU()
        else:
            raise ValueError('activation must be relu or selu')

        if not hasattr(hidden_size, '__iter__'):
            if hidden_size is None:
                hidden_size = [output_size]
            else:
                hidden_size = [hidden_size]

        self._layers = []
        layer_dims = [input_size] + hidden_size + [output_size]

        for i in range(1, len(layer_dims) - 1):
            self._layers.append(nn.Linear(layer_dims[i - 1], layer_dims[i]))
            self._layers.append(self._activation)
            self._layers.append(nn.Dropout(dropout))

        self._layers.append(nn.Linear(layer_dims[-2], layer_dims[-1]))

        if dropout_last_layer:
            self._layers.append(nn.Dropout(dropout))
        self._layers = nn.Sequential(*self._layers)

    def forward(self, x):
        return self._layers(x)


class SinusodialMz(nn.Module):
    def __init__(self, embedding_dim: int, *, lambda_params: Tuple[float, float] = (LAMBDA_MIN, LAMBDA_MAX)) -> None:
        super(SinusodialMz, self).__init__()
        self.lambda_min, self.lambda_max = lambda_params
        self.lambda_div_value = self.lambda_max / self.lambda_min
        self.x = torch.arange(0, embedding_dim, 2)
        self.x = (
            2 * math.pi *
            (
                self.lambda_min *
                self.lambda_div_value ** (self.x / (embedding_dim - 2))
            ) ** -1
        )

    def forward(self, mz: torch.Tensor):
        self.x = self.x.to(mz.device)
        x = torch.einsum('bl,d->bld', mz, self.x)
        sin_embedding = torch.sin(x)
        cos_embedding = torch.cos(x)
        b, l, d = sin_embedding.shape
        x = torch.zeros(b, l, 2 * d, dtype=mz.dtype, device=mz.device)
        x[:, :, ::2] = sin_embedding
        x[:, :, 1::2] = cos_embedding
        return x


class SinusodialMzEmbedding(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        *,
        lambda_params: Tuple[float, float] = (LAMBDA_MIN, LAMBDA_MAX),
        feedward_activation: Literal['relu', 'selu', 'gelu'] = 'relu',
        dropout: float = 0.1,
        dropout_last_layer: bool = True
    ):
        super(SinusodialMzEmbedding, self).__init__()
        if embedding_dim % 2 != 0:
            raise ValueError('embedding_dim must be even')
        self.embedding = SinusodialMz(
            embedding_dim, lambda_params=lambda_params)
        self.feedward_layers = MultiFeedForwardModule(
            embedding_dim, embedding_dim, embedding_dim,
            activation=feedward_activation, dropout=dropout, dropout_last_layer=dropout_last_layer
        )

    def forward(self, mz: torch.Tensor):
        x = self.embedding(mz)
        x = self.feedward_layers(x)
        return x


class PeaksEmbedding(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        *,
        lambda_params: Tuple[float, float] = (LAMBDA_MIN, LAMBDA_MAX),
        feedward_activation: Literal['relu', 'selu', 'gelu'] = 'relu',
        dropout: float = 0.1,
        dropout_last_layer: bool = False
    ) -> None:
        super(PeaksEmbedding, self).__init__()
        self.mz_embedding = SinusodialMzEmbedding(
            embedding_dim,
            lambda_params=lambda_params,
            feedward_activation=feedward_activation,
            dropout=dropout,
            dropout_last_layer=dropout_last_layer
        )
        self.intensity_embedding = MultiFeedForwardModule(
            embedding_dim + 1, embedding_dim, embedding_dim,
            activation=feedward_activation,
            dropout=dropout,
            dropout_last_layer=dropout_last_layer
        )

    def forward(self, mz: torch.Tensor, intensity: torch.Tensor):
        mz_tensor = self.mz_embedding(mz)
        intensity_tensor = torch.unsqueeze(intensity, dim=-1)
        x = self.intensity_embedding(
            torch.cat([mz_tensor, intensity_tensor], dim=-1))
        return x


class SiameseModel(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        n_head: int,
        n_layer: int,
        dim_feedward: int,
        dim_target: int,
        *,
        lambda_params: Tuple[float, float] = (LAMBDA_MIN, LAMBDA_MAX),
        feedward_activation: Literal['relu', 'selu', 'gelu'] = 'relu',
        dropout: float = 0.1,
        dropout_last_layer: bool = False,
        norm_first: bool = True
    ) -> None:
        super(SiameseModel, self).__init__()
        if embedding_dim % n_head != 0:
            raise ValueError('embedding must be divisible by n_head')

        self.embedding = PeaksEmbedding(
            embedding_dim,
            lambda_params=lambda_params,
            feedward_activation=feedward_activation,
            dropout=dropout,
            dropout_last_layer=dropout_last_layer
        )

        if feedward_activation == 'selu':
            # transformer encoder activation
            # only gelu or relu
            self.activation = 'gelu'
        else:
            self.activation = feedward_activation

        if feedward_activation == 'relu':
            self._activation = nn.ReLU()
        elif feedward_activation == 'selu':
            self._activation = nn.SELU()
        elif feedward_activation == 'gelu':
            self._activation = nn.GELU()
        else:
            raise ValueError('activation must be relu or selu or gelu')

        encoder_layer = TransformerEncoderLayer(
            embedding_dim,
            n_head,
            dim_feedforward=dim_feedward,
            dropout=dropout,
            activation=self.activation,
            batch_first=True,
            norm_first=norm_first
        )
        self._encoder = TransformerEncoder(
            encoder_layer,
            n_layer,
            enable_nested_tensor=False
        )

        self._decoder = MultiFeedForwardModule(
            embedding_dim,
            dim_feedward,
            dim_target,
            activation=feedward_activation,
            dropout=dropout,
            dropout_last_layer=dropout_last_layer
        )

    def forward(self, mz: torch.Tensor, intensity: torch.Tensor, mask: torch.Tensor):
        x = self.embedding(mz, intensity)
        x = self._encoder(x, src_key_padding_mask=mask)
        # mean pooling or cls position vector
        extened_mask = mask.unsqueeze(-1)
        # [batch, seq_len, hidden_size]
        masked_embeddings: torch.Tensor = x * (~extened_mask)
        # [batch, hidden_size]
        sum_embeddings = masked_embeddings.sum(dim=1)
        # [batch, 1]
        num_valid_tokens = (~mask).sum(dim=1, keepdim=True)
        x = sum_embeddings / num_valid_tokens
        x = self._activation(self._decoder(x))
        return x