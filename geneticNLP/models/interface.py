import torch
import torch.nn as nn
import torch.utils.data as data

from abc import ABC, abstractmethod


class Model(ABC):
    """
    Abstract model interface for neural networks.

    Methods
    ----------
    - __init__(config: dict) -> None
    - forward(self, batch: list) -> list[TT]
    - predict(self, batch: list) -> typle[list[TT],list[int]]
    - loss(self, batch: list) -> nn.CrossEntropyLoss
    - accuracy(self, batch: list)@torch.no_grad() -> float
    - evaluate(self, data_loader: data.DataLoader)@torch.no_grad() -> float
    """

    #
    #
    #  -------- __init__ -----------
    #
    @abstractmethod
    def __init__(config: dict) -> None:
        raise NotImplementedError

    #
    #
    #  -------- forward -----------
    #
    @abstractmethod
    def forward(
        self,
        batch: list,
    ) -> list:
        raise NotImplementedError

    #
    #
    #  -------- predict -----------
    #
    @abstractmethod
    def predict(
        self,
        batch: list,
    ) -> tuple:
        raise NotImplementedError

    #
    #
    #  -------- loss -----------
    #
    @abstractmethod
    def loss(
        self,
        batch: list,
    ) -> nn.CrossEntropyLoss:
        raise NotImplementedError

    #
    #
    #  -------- accuracy -----------
    #
    @abstractmethod
    @torch.no_grad()
    def accuracy(
        self,
        batch: list,
    ) -> float:
        raise NotImplementedError

    #
    #
    #  -------- evalutate -----------
    #
    @abstractmethod
    @torch.no_grad()
    def evaluate(
        self,
        data_loader: data.DataLoader,
    ) -> float:
        raise NotImplementedError
