import torch
import torch.nn as nn

from geneticNLP.neural.nn import MLP, BILSTM

from geneticNLP.metric import Metric
from geneticNLP.utils import unpad, flatten, get_device


class POSstripped(nn.Module):

    #
    #
    #  -------- init -----------
    #
    def __init__(self, config: dict):
        super().__init__()

        # save config
        self.config = config
        self.metric = Metric()

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

        embeds, encods = zip(*batch)

        return self.forward(embeds), encods

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
    #  -------- accuracy -----------
    #
    @torch.no_grad()
    def accuracy(
        self,
        batch: list,
        reset: bool = True,
        category: str = None,
    ) -> float:
        self.eval()

        if reset:
            self.metric.reset()

        predictions, target_ids = self.predict(batch)

        # Process the predictions and compare with the gold labels
        for pred, gold in zip(predictions, target_ids):
            for (p, g) in zip(pred, gold):

                p_idx: int = torch.argmax(p).item()

                if p_idx == g:
                    self.metric.add_tp(p_idx)

                if p_idx != g:
                    self.metric.add_fp(p_idx)
                    self.metric.add_fn(g)

        return self.metric.accuracy(class_name=category)

    #
    #
    #  -------- evalutate -----------
    #
    @torch.no_grad()
    def evaluate(
        self,
        data_loader,
        category: str = None,
    ) -> float:
        self.eval()
        self.metric.reset()

        for batch in data_loader:
            _ = self.accuracy(batch, reset=False)

        return self.metric.accuracy(class_name=category)

    #
    #
    #  -------- fitness -----------
    #
    @torch.no_grad()
    def fitness(self, batch: list) -> float:
        self.eval()

        # create batch metric with accuracy function
        _ = self.accuracy(batch)

        for i in range(self.metric.get_classes()):
            pass

        return 0.0
