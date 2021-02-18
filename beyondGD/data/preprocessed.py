import random

import torch.utils.data as data

from beyondGD.data import CONLLU

from beyondGD.encoding import Encoding
from beyondGD.embeddings import FastText


class PreProcessed(data.IterableDataset):
    """"""

    #
    #
    #  -------- __init__ -----------
    #
    def __init__(
        self,
        data_path: str,
        embeddings: FastText,
        encoding: Encoding,
        reduction: float = 0.0,
    ):

        self.embeddings = embeddings
        self.encoding = encoding
        self.reduction = reduction

        # save data
        self.data = list(self.load_data(data_path))

    #
    #
    #  -------- load_data -----------
    #
    def load_data(self, data_path):

        conllu_import: list = CONLLU(data_path)

        for sent in conllu_import:

            if random.uniform(0, 1) < self.reduction:
                continue

            words = map(lambda tok: tok.word, sent)
            poss = map(lambda tok: tok.pos, sent)

            embeds: list = self.embeddings.forward_sent(words)
            encods: list = [
                self.encoding.encode(tag) for tag in poss
            ]

            yield (embeds, encods)

    #
    #
    #  -------- __iter__ -----------
    #
    def __iter__(self):
        yield from self.data

    #
    #
    #  -------- __len__ -----------
    #
    def __len__(self) -> int:
        return len(self.data)
