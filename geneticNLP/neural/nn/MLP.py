import torch.nn as nn

from geneticNLP.utils.types import TT


class MLP(nn.Module):
    def __init__(
        self,
        in_size: int,
        hid_size: int,
        out_size: int,
        dropout: float,
    ):
        super().__init__()

        # [Linear -> Activation -> Linear -> Dropout]
        self.net = nn.Sequential(
            nn.Linear(in_size, hid_size),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hid_size, out_size),
            nn.Dropout(p=dropout, inplace=True),
        )

    #
    #
    #  -------- forward -----------
    #
    def forward(self, vec: TT) -> TT:
        return self.net(vec)
