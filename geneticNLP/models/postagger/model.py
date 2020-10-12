from typing import Iterable, Sequence

import torch.nn.utils.rnn as rnn

from geneticNLP.neural.nn import MLP, BILSTM
from geneticNLP.utils import get_device


class POSTagger:
    """POS-Tagger"""

    def __init__(
        self,
        config: dict,
        word_emb,
    ):

        # store word embedder
        self.word_emb = word_emb

        # BILSTM to calculate contextualized word embeddings
        self.context = BILSTM(
            in_size=config["lstm"]["in_size"],
            out_size=config["lstm"]["out_size"],
            depth=config["lstm"]["depth"],
            dropout=config["lstm"]["dropout"],
        )

        # MLP to calculate the POS tags
        self.score = MLP(
            in_size=config["lstm"]["out_size"] * 2,
            hid_size=config["score"]["hidden_size"],
            out_size=config["score"]["out_size"],
            dropout=config["score"]["dropout"],
        )

    def forward(self, sents: Iterable[Sequence[str]]) -> list:
        """Apply the biaffine model to the input batch of sentences.

        Argmuments:
        * (Iterable) batch of sentences, where each sentence is a list
            of words (strings)

        Return a list of tuples, one per input sentence.  Each tuple
        is in fact a triple of:
        * Node scores tensor of shape N
        * Head scores tensor of shape N x (N + 1)
        * Arc label scores tensor of shape N
        where N is the sentence length.
        """
        # CPU/GPU device
        device = get_device()

        # Embed the entire batch
        sents_vec = [
            self.word_emb.forwards(sent).to(device) for sent in sents
        ]

        # Contextualize embeddings with bilstm
        sents_vec_context_pack = self.context(sents_vec)

        # Convert packed representation to a padded representation
        padded, length = rnn.pad_packed_sequence(
            sents_vec_context_pack, batch_first=True
        )

        # Calculate the POS-Tag scores
        result = self.score(padded)

        return result
