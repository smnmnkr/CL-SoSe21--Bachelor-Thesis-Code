import argparse

import torch
import random

from geneticNLP.tasks import do_evolve, do_train, do_hybrid

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
parser = argparse.ArgumentParser(description="geneticNLP")
subparsers = parser.add_subparsers(
    dest="command", help="available commands"
)

#
#
#  -------- EVOLUTION: -----------
#
parser_evolve = subparsers.add_parser(
    "evolve",
    help="use neural evolution",
)

parser_evolve.add_argument(
    "-M",
    dest="model_config",
    required=True,
    help="model config.json file",
    metavar="FILE",
)

parser_evolve.add_argument(
    "-E",
    dest="evolution_config",
    required=True,
    help="evolution config.json file",
    metavar="FILE",
)

parser_evolve.add_argument(
    "-D",
    dest="data_config",
    required=True,
    help="data config.json file",
    metavar="FILE",
)

#
#
#  -------- TRAINING: -----------
#
parser_train = subparsers.add_parser(
    "train",
    help="use gradient based training",
)

parser_train.add_argument(
    "-M",
    dest="model_config",
    required=True,
    help="model config.json file",
    metavar="FILE",
)

parser_train.add_argument(
    "-T",
    dest="training_config",
    required=True,
    help="training config.json file",
    metavar="FILE",
)

parser_train.add_argument(
    "-D",
    dest="data_config",
    required=True,
    help="data config.json file",
    metavar="FILE",
)

#
#
#  -------- HYBRID: -----------
#
parser_hybrid = subparsers.add_parser(
    "hybrid",
    help="use gradient based training",
)

parser_hybrid.add_argument(
    "-M",
    dest="model_config",
    required=True,
    help="model config.json file",
    metavar="FILE",
)

parser_hybrid.add_argument(
    "-H",
    dest="hybrid_config",
    required=True,
    help="hybrid config.json file",
    metavar="FILE",
)

parser_hybrid.add_argument(
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

    # get console arguments
    args = parser.parse_args()

    # choose task
    if args.command == "evolve":
        do_evolve(args)

    if args.command == "train":
        do_train(args)

    if args.command == "hybrid":
        do_hybrid(args)
