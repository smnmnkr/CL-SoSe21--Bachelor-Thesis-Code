import torch
import torch.nn as nn
import torch.utils.data as data

from abc import ABC, abstractmethod


class Model(ABC):

    #
    #
    #  -------- __init__ -----------
    #
    @abstractmethod
    def __init__():
        raise NotImplementedError

    #
    #
    #  -------- forward -----------
    #
    @abstractmethod
    def forward(self, batch: list) -> list:
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
    #  -------- accuracy -----------
    #
    @abstractmethod
    @torch.no_grad()
    def accuracy(self, batch: list) -> float:
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
    #  -------- evalutate -----------
    #
    @abstractmethod
    @torch.no_grad()
    def evaluate(
        self,
        data_loader: data.DataLoader,
    ) -> float:
        raise NotImplementedError
