import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn

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
            in_size=config["lstm"]["input_size"],
            out_size=config["lstm"]["hidden_size"],
            depth=config["lstm"]["depth"],
            dropout=config["lstm"]["dropout"],
        )

        # MLP to calculate the POS tags
        self.score = MLP(
            in_size=config["lstm"]["hidden_size"] * 2,
            hid_size=config["score"]["hidden_size"],
            out_size=config["score"]["output_size"],
            dropout=config["score"]["dropout"],
        )

    #
    #
    #  -------- forward -----------
    #
    def forward(self, embed_batch: list) -> list:

        # Contextualize embeddings with BiLSTM
        sents_vec_context_pack = self.context(embed_batch)

        # Convert packed representation to a padded representation
        padded, length = rnn.pad_packed_sequence(
            sents_vec_context_pack, batch_first=True
        )

        # Calculate the POS-Tag scores
        result = self.score(padded)

        return unpad(result, length)

    #
    #
    #  -------- predict -----------
    #
    def predict(
        self,
        batch: list,
        encoding,
        embedding,
    ) -> tuple:

        to_predict: list = []
        target_ids: list = []

        for sent in batch:
            words = map(lambda tok: tok.word, sent)
            gold_tags = map(lambda tok: tok.pos, sent)

            target_ids_sent: list = []

            for tag in gold_tags:
                target_ids_sent.append(encoding.encode(tag))

            target_ids.append(target_ids_sent)
            to_predict.append(list(words))

        embeddings: list = embedding.forward_batch(to_predict)
        predictions: list = self.forward(embeddings)

        return predictions, target_ids

    #
    #
    #  -------- accuracy -----------
    #
    def accuracy(
        self,
        batch: list,
        encoding,
        embedding,
    ) -> float:

        k: float = 0.0
        n: float = 0.0

        predictions, target_ids = self.predict(batch, encoding, embedding)

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
        encoding,
        embedding,
    ) -> nn.CrossEntropyLoss:

        predictions, target_ids = self.predict(batch, encoding, embedding)

        return nn.CrossEntropyLoss()(
            torch.cat(predictions),
            torch.LongTensor(flatten(target_ids)).to(get_device()),
        )
