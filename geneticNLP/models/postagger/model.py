from typing import Iterable, Sequence

import torch.nn as nn
import torch.nn.utils.rnn as rnn

from geneticNLP.neural.nn import MLP, BILSTM
from geneticNLP.utils import get_device, unpad


class POSTagger(nn.Module):
    """POS-Tagger"""

    def __init__(
        self,
        config: dict,
        encoding,
        embedding,
    ):
        super().__init__()

        # store word embedder
        self.enc = encoding
        self.emb = embedding

        # BILSTM to calculate contextualized word embeddings
        self.context = BILSTM(
            in_size=self.emb.dimension,
            out_size=config["lstm"]["hidden_size"],
            depth=config["lstm"]["depth"],
            dropout=config["lstm"]["dropout"],
        )

        # MLP to calculate the POS tags
        self.score = MLP(
            in_size=config["lstm"]["hidden_size"] * 2,
            hid_size=config["score"]["hidden_size"],
            out_size=len(self.enc),
            dropout=config["score"]["dropout"],
        )

    def forward(self, sents: Iterable[Sequence[str]]) -> list:
        """"""
        # CPU/GPU device
        device = get_device()

        # Embed the entire batch
        sents_vec = [self.emb.forwards(sent).to(device) for sent in sents]

        # Contextualize embeddings with bilstm
        sents_vec_context_pack = self.context(sents_vec)

        # Convert packed representation to a padded representation
        padded, length = rnn.pad_packed_sequence(
            sents_vec_context_pack, batch_first=True
        )

        # Calculate the POS-Tag scores
        result = self.score(padded)

        return unpad(result, length)
