import torch
import torch.nn as nn


class POSfull(nn.Module):

    #
    #
    #  -------- init -----------
    #
    def __init__(self, config: dict, embedding, encoding):
        super().__init__(config)

        self.embedding = embedding
        self.encoding = encoding

    #
    #
    #  -------- forward -----------
    #
    def forward(self, batch: list) -> list:

        embed_batch: list = self.embedding.forward_batch(batch)

        return super().forward(embed_batch)

    #
    #
    #  -------- predict -----------
    #
    def predict(
        self,
        batch: list,
    ) -> tuple:

        words, poss = list()

        for sent in batch:
            words = map(lambda tok: tok.word, sent)
            poss = map(lambda tok: tok.pos, sent)

        embeds: list = self.embeddings.forward_sent(words)
        encods: list = [self.encoding.encode(tag) for tag in poss]

        predictions: list = self.forward(embeds)

        return predictions, encods
