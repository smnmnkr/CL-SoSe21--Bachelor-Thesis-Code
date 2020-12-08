import argparse

import torch
import random

from geneticNLP.tasks import (
    do_evolve,
    do_descent,
    do_swarm,
    do_amoeba,
)

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

parser_task: list = [
    #
    #  -------- GRADIENT DESCENT: -----------
    {
        "ref": "parser_descent",
        "command": "descent",
        "description": "use gradient descent training",
        "task": do_descent,
    },
    #
    #  -------- EVOLUTION: -----------
    {
        "ref": "parser_evolve",
        "command": "evolve",
        "description": "use neural evolution",
        "task": do_evolve,
    },
    #
    #  -------- SWARM OPTIMIZE: -----------
    {
        "ref": "parser_swarm",
        "command": "swarm",
        "description": "use particle swarm optimize",
        "task": do_swarm,
    },
    #
    #  -------- AMOEBA: -----------
    {
        "ref": "parser_amoeba",
        "command": "amoeba",
        "description": "use amoeba optimize",
        "task": do_amoeba,
    },
]

# setup each subparser
for task_subparser in parser_task:

    #  add subparser
    task_subparser["ref"] = subparsers.add_parser(
        task_subparser["command"],
        help=task_subparser["description"],
    )

    # add model config arg
    task_subparser["ref"].add_argument(
        "-M",
        dest="model_config",
        required=True,
        help="model config.json file",
        metavar="FILE",
    )

    # add training config arg
    task_subparser["ref"].add_argument(
        "-T",
        dest="training_config",
        required=True,
        help="training config.json file",
        metavar="FILE",
    )

    # add data config arg
    task_subparser["ref"].add_argument(
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

        # choose, run task
        for task_subparser in parser_task:
            if args.command == task_subparser["command"]:
                task_subparser["task"](args)

    except KeyboardInterrupt:
        print("[-- Process was interrupted by user, aborting. --]")
        exit()
