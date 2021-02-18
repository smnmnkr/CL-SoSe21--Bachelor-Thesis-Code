from typing import Sequence

import torch.nn as nn
import torch.nn.utils.rnn as rnn

from beyondGD.utils.types import TT


class BILSTM(nn.Module):

    #
    #
    #  -------- __init__ -----------
    #
    def __init__(
        self,
        in_size: int,
        hid_size: int,
        depth: int,
        dropout: float,
    ):
        super().__init__()

        # [LSTM : (Layers -> Dropout)^depth -> Activation]
        self.net = nn.LSTM(
            input_size=in_size,
            hidden_size=hid_size,
            bidirectional=True,
            num_layers=depth,
            dropout=0.0 if (depth == 1) else dropout,
        )
        self.acf = nn.LeakyReLU()

        # custom init weights
        self.init_weights()

    #
    #
    #  -------- init_weights -----------
    #
    def init_weights(self):

        for name, param in self.net.named_parameters():

            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)

            elif "weight_hh" or "bias" in name:
                param.data.uniform_()

    #
    #
    #  -------- forward -----------
    #
    def forward(
        self, batch: Sequence[TT]
    ) -> rnn.PackedSequence:
        """Contextualize the embeddings for each sentence in the batch.

        The method takes on input a list of tensors with shape N x *,
        where N is the dynamic sentence length (i.e. can be different
        for each sentence) and * is any number of trailing dimensions,
        including zero, the same for each sentence.

        Returns a packed sequence.
        """
        self.net.flatten_parameters()

        # Pack sentence vectors as a packed sequence
        packed_batch = rnn.pack_sequence(
            batch, enforce_sorted=False
        )

        # Apply LSTM to the packed sequence of word embeddings
        packed_hidden, _ = self.net(packed_batch)

        # Convert packed representation to a padded representation
        padded_hidden, mask = rnn.pad_packed_sequence(
            packed_hidden, batch_first=True
        )

        # Return the scores
        return (self.acf(padded_hidden), mask)
