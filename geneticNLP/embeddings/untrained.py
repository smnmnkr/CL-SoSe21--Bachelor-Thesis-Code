import torch
import torch.nn as nn

from geneticNLP.utils.types import TT


class Untrained(nn.Module):
    """Module for untrained Token Embeddings"""

    def __init__(
        self,
        data: list,
        dimension: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        # save dimension and create lookup table
        self.dimension: int = dimension
        self.lookup: dict = {}

        # fill lookup table with data
        for ix, obj in enumerate(data):
            self.lookup[obj] = ix

        # create padding token
        self.padding_idx: int = len(self.lookup)

        # save model
        self.model = self.load_model()

        # save dropout
        self.dropout = nn.Dropout(p=dropout, inplace=False)

    #
    #
    #  -------- forward -----------
    #
    def forward(self, word: str) -> TT:

        try:
            idx = self.lookup[word]

        except KeyError:
            idx = self.padding_idx

        emb = self.model(torch.tensor(idx, dtype=torch.long))
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
    def load_model(self):
        return nn.Embedding(
            len(self.lookup) + 1,
            self.dimension,
            padding_idx=self.padding_idx,
        )

    #  -------- __len__ -----------
    #
    def __len__(self) -> int:
        return len(self.lookup)
