from .stripped import POSstripped


class POSfull(POSstripped):

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

        word_batch: list = []
        label_batch: list = []

        for sent in batch:
            word_batch.append(map(lambda tok: tok.word, sent))

            label_batch.append(
                [
                    self.encoding.encode(tag)
                    for tag in map(lambda tok: tok.pos, sent)
                ]
            )

        return self.forward(word_batch), label_batch
