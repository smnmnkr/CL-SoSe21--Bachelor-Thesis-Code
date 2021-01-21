import argparse

import torch
import random

from beyondGD.tasks import do_train


# make pytorch computations deterministic
# src: https://pytorch.org/docs/stable/notes/randomness.html
random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#
#
#  -------- ARGPARSER: -----------
#
parser = argparse.ArgumentParser(description="beyondGD")

# add model config arg
parser.add_argument(
    "-M",
    dest="model_config",
    required=True,
    help="model config.json file",
    metavar="FILE",
)

# add training config arg
parser.add_argument(
    "-T",
    dest="training_config",
    required=True,
    help="training config.json file",
    metavar="FILE",
)

# add data config arg
parser.add_argument(
    "-D",
    dest="data_config",
    required=True,
    help="data config.json file",
    metavar="FILE",
)

#
#
#  -------- MAIN: -----------
#
if __name__ == "__main__":

    try:

        # get console arguments
        args = parser.parse_args()

        # run task
        do_train(args)

    except KeyboardInterrupt:
        print("[-- Process was interrupted by user, aborting. --]")
        exit()
