import torch.nn as nn

from beyondGD.utils.type import TT


class MLP(nn.Module):

    #
    #
    #  -------- __init__ -----------
    #
    def __init__(
        self,
        in_size: int,
        hid_size: int,
        dropout: float,
    ):
        super().__init__()

        # [Linear -> Dropout -> Activation]
        self.net = nn.Sequential(
            nn.Linear(in_size, hid_size),
            nn.Dropout(p=dropout, inplace=True),
            nn.LeakyReLU(inplace=True),
        )

    #
    #
    #  -------- forward -----------
    #
    def forward(self, vec: TT) -> TT:
        return self.net(vec)
