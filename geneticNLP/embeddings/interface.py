from abc import ABC, abstractmethod

from geneticNLP.utils.types import TT


class Interface(ABC):
    """
    Abstract embedder interface for word vector representations.

    Methods
    ----------
    - forward(self, word: str) -> TT
    - forwards(self, words: list) -> TT
    - load_model(self, *data_path: str) -> Model
    - dimension(self)@property -> int
    - __len__(self) -> int
    """

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

    #  -------- dimension -----------
    #
    @property
    def dimension(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        raise NotImplementedError

    #  -------- __len__ -----------
    #
    @abstractmethod
    def __len__(self) -> int:
        """Return the count of the embedded words."""
        raise NotImplementedError
