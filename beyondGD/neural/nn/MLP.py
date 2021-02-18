import torch.nn as nn

from beyondGD.utils.types import TT


class MLP(nn.Module):
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

        # self.init_weights()

    #
    #
    #  -------- init_weights -----------
    #
    def init_weights(self):

        for name, param in self.net.named_parameters():

            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)

            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)

            elif "bias" in name:
                param.data.uniform_()

    #
    #
    #  -------- forward -----------
    #
    def forward(self, vec: TT) -> TT:
        return self.net(vec)
