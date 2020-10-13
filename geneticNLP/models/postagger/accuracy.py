from typing import Tuple, Any, Iterable, Callable

import torch

from geneticNLP.data import batch_loader


#
#
#  -------- _accuracy -----------
#
def accuracy(
    model: Callable,
    data_set: Iterable[Tuple[Any, Any]],
    batch_size: int = 32,
) -> float:

    k: float = 0.0
    n: float = 0.0

    # We load the dataset in batches to speed up the calculation
    for batch in batch_loader(
        data_set, batch_size=batch_size, num_workers=0
    ):

        inputs = []
        golds = []

        for sent in batch:
            words = map(lambda tok: tok.word, sent)
            gold_tags = map(lambda tok: tok.pos, sent)

            gold_sent: list = []

            for tag in gold_tags:
                gold_sent.append(model.enc.encode(tag))

            golds.append(gold_sent)
            inputs.append(list(words))

        # Tag all the sentences
        predictions = model.forward(inputs)

        # Process the predictions and compare with the gold labels
        for pred, gold in zip(predictions, golds):
            for (p, g) in zip(pred, gold):

                pid = torch.argmax(p).item()

                if pid == g:
                    k += 1.0
                n += 1.0

    return k / n
