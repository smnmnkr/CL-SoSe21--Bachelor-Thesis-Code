from abc import ABC, abstractmethod

from geneticNLP.utils.types import TT


class Interface(ABC):
    """
    Abstract embedder interface for word vector representations.

    Parameters
    ----------
    - data_path: path to word list or pretrained word vectors
    - dropout: applied dropout on training

    Methods
    ----------
    - forward(self, word: str) -> TT
    - forwards(self, words: list) -> TT
    - load_model(self, data_path: str)
    - embedding_dim(self) -> int
    - embedding_num(self) -> int
    """

    #
    #
    #  -------- __init__ -----------
    #
    @abstractmethod
    def __init__(
        self,
        data_path: str,
        dropout: float,
    ):
        raise NotImplementedError

    #
    #
    #  -------- forward -----------
    #
    @abstractmethod
    def forward(self, word: str) -> TT:
        """Embed single given word."""
        raise NotImplementedError

    #
    #
    #  -------- forwards -----------
    #
    @abstractmethod
    def forwards(self, words: list) -> TT:
        """Embed multiply given words."""
        raise NotImplementedError

    #
    #
    #  -------- load_model -----------
    #
    @abstractmethod
    def load_model(self, data_path: str):
        """Return the loaded model."""
        raise NotImplementedError

    #
    #
    #  -------- embedding_dim -----------
    #
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        raise NotImplementedError

    #
    #
    #  -------- embedding_dim -----------
    #
    @abstractmethod
    def embedding_num(self) -> int:
        """Return the count of the embedded words."""
        raise NotImplementedError
