import itertools
from collections import defaultdict

import torch

from geneticNLP.data import batch_loader


class Metric:

    #
    #
    #  -------- __init__ -----------
    #
    def __init__(self) -> None:
        """
        inspired by flairNLP: https://github.com/flairNLP/flair/blob/master/flair/training_utils.py
        """

        #
        self.name = "AVG"

        self._tps = defaultdict(int)
        self._fps = defaultdict(int)
        self._tns = defaultdict(int)
        self._fns = defaultdict(int)

    #
    #
    #  -------- precision -----------
    #
    def precision(self, class_name=None):
        if self.get_tp(class_name) + self.get_fp(class_name) > 0:
            return self.get_tp(class_name) / (
                self.get_tp(class_name) + self.get_fp(class_name)
            )
        return 0.0

    #
    #
    #  -------- recall -----------
    #
    def recall(self, class_name=None):
        if self.get_tp(class_name) + self.get_fn(class_name) > 0:
            return self.get_tp(class_name) / (
                self.get_tp(class_name) + self.get_fn(class_name)
            )
        return 0.0

    #
    #
    #  -------- f_score -----------
    #
    def f_score(self, class_name=None):
        if self.precision(class_name) + self.recall(class_name) > 0:
            return (
                2
                * (self.precision(class_name) * self.recall(class_name))
                / (self.precision(class_name) + self.recall(class_name))
            )
        return 0.0

    #
    #
    #  -------- accuracy -----------
    #
    def accuracy(self, class_name=None):
        if (
            self.get_tp(class_name)
            + self.get_fp(class_name)
            + self.get_fn(class_name)
            + self.get_tn(class_name)
            > 0
        ):
            return (self.get_tp(class_name) + self.get_tn(class_name)) / (
                self.get_tp(class_name)
                + self.get_fp(class_name)
                + self.get_fn(class_name)
                + self.get_tn(class_name)
            )
        return 0.0

    #
    #
    #  -------- get_classes -----------
    #
    def get_classes(self) -> list:

        all_classes = set(
            itertools.chain(
                *[
                    list(keys)
                    for keys in [
                        self._tps.keys(),
                        self._fps.keys(),
                        self._tns.keys(),
                        self._fns.keys(),
                    ]
                ]
            )
        )

        all_classes = [
            class_name
            for class_name in all_classes
            if class_name is not None
        ]

        all_classes.sort()
        return all_classes

    #
    #
    #  -------- __str__ -----------
    #
    def __str__(self):
        all_classes = self.get_classes()
        all_classes = [None] + all_classes
        all_lines = [
            "[--- {:8}\t quantity={:4} \t precision={:.4f} \t recall={:.4f} \t accuracy={:.4f} \t f1-score={:.4f} ---]".format(
                self.name if class_name is None else class_name,
                self.get_tp(class_name)
                + self.get_fp(class_name)
                + self.get_fn(class_name)
                + self.get_tn(class_name),
                self.precision(class_name),
                self.recall(class_name),
                self.accuracy(class_name),
                self.f_score(class_name),
            )
            for class_name in all_classes
        ]
        return "\n".join(all_lines)

    #  -------- reset -----------
    #
    def reset(self):
        self._tps.clear()
        self._fps.clear()
        self._tns.clear()
        self._fns.clear()

    #  -------- add_tp -----------
    #
    def add_tp(self, class_name):
        self._tps[class_name] += 1

    #  -------- add_tn -----------
    #
    def add_tn(self, class_name):
        self._tns[class_name] += 1

    #  -------- add_fp -----------
    #
    def add_fp(self, class_name):
        self._fps[class_name] += 1

    #  -------- add_fn -----------
    #
    def add_fn(self, class_name):
        self._fns[class_name] += 1

    #  -------- _get -----------
    #
    def _get(self, cat: dict, class_name=None):
        if class_name is None:
            return sum(
                [cat[class_name] for class_name in self.get_classes()]
            )
        return cat[class_name]

    #  -------- get_tp -----------
    #
    def get_tp(self, class_name=None):
        return self._get(self._tps, class_name)

    #  -------- get_tn -----------
    #
    def get_tn(self, class_name=None):
        return self._get(self._tns, class_name)

    #  -------- get_fp -----------
    #
    def get_fp(self, class_name=None):
        return self._get(self._fps, class_name)

    #  -------- get_fn -----------
    #
    def get_fn(self, class_name=None):
        return self._get(self._fns, class_name)
