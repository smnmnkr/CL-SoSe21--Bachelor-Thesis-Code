import torch
import torch.nn as nn

from geneticNLP.neural.nn import MLP, BILSTM

from geneticNLP.utils import unpad, flatten, get_device


class POSTagger(nn.Module):

    #
    #
    #  -------- init -----------
    #
    def __init__(self, config: dict):
        super().__init__()

        # BILSTM to calculate contextualized word embeddings
        self.context = BILSTM(
            in_size=config["lstm"]["in_size"],
            hid_size=config["lstm"]["hid_size"],
            depth=config["lstm"]["depth"],
            dropout=config["lstm"]["dropout"],
        )

        # MLP to calculate the POS tags
        self.score = MLP(
            in_size=config["lstm"]["hid_size"] * 2,
            hid_size=config["score"]["hid_size"],
            dropout=config["score"]["dropout"],
        )

    #
    #
    #  -------- forward -----------
    #
    def forward(self, embed_batch: list) -> list:

        # Contextualize embeddings with BiLSTM
        pad_context, mask = self.context(embed_batch)

        # Calculate the POS-Tag scores
        pad_scores = self.score(pad_context)

        return unpad(pad_scores, mask)

    #
    #
    #  -------- predict -----------
    #
    def predict(
        self,
        batch: list,
    ) -> tuple:

        embeds: list = []
        encods: list = []

        for sent in batch:
            embeds.append(sent[0])
            encods.append(sent[1])

        predictions: list = self.forward(embeds)

        return predictions, encods

    #
    #
    #  -------- accuracy -----------
    #
    @torch.no_grad()
    def accuracy(self, batch: list) -> float:

        k: float = 0.0
        n: float = 0.0

        predictions, target_ids = self.predict(batch)

        # Process the predictions and compare with the gold labels
        for pred, gold in zip(predictions, target_ids):
            for (p, g) in zip(pred, gold):

                if torch.argmax(p).item() == g:
                    k += 1.0
                n += 1.0

        return k / n

    #
    #
    #  -------- loss -----------
    #
    def loss(
        self,
        batch: list,
    ) -> nn.CrossEntropyLoss:

        predictions, target_ids = self.predict(batch)

        return nn.CrossEntropyLoss()(
            torch.cat(predictions),
            torch.LongTensor(flatten(target_ids)).to(get_device()),
        )

    #
    #
    #  -------- evalutate -----------
    #
    @torch.no_grad()
    def evaluate(self, data_loader) -> float:
        self.eval()

        accuracy_per_batch: list = [
            self.accuracy(batch) for batch in data_loader
        ]

        return sum(accuracy_per_batch) / len(accuracy_per_batch)
