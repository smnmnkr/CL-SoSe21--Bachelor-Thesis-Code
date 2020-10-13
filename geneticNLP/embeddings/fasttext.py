import fasttext

import torch
import torch.nn as nn

from geneticNLP.embeddings import Interface
from geneticNLP.utils.types import TT


class FastText(Interface):
    """Module for FastText (binary) word embedding."""

    #
    #
    #  -------- __init__ -----------
    #
    def __init__(
        self,
        data_path,
        dropout: float = 0.0,
    ):

        # save model
        self.model = self.load_model(data_path)

        # save dropout
        self.dropout = nn.Dropout(p=dropout, inplace=False)

    #
    #
    #  -------- forward -----------
    #
    def forward(self, word: str) -> TT:

        emb = torch.tensor(self.model[word], dtype=torch.float)
        return self.dropout(emb)

    #
    #
    #  -------- forwards -----------
    #
    def forwards(self, words: list) -> TT:

        emb: list = []

        for w in words:
            emb.append(self.forward(w))

        return torch.stack(emb)

    #
    #
    #  -------- load_model -----------
    #
    def load_model(self, file_path):
        return fasttext.load_model(file_path)

    #  -------- dimension -----------
    #
    @property
    def dimension(self) -> int:
        return self.model.get_dimension()

    #  -------- __len__ -----------
    #
    def __len__(self) -> int:
        return len(self.model.get_words())
