from datetime import datetime

import torch.nn as nn

from torch.optim import Adam
from torch.utils.data import IterableDataset

from geneticNLP.data import batch_loader

#
#
#  -------- train -----------
#
def train(
    model: nn.Module,
    train_set: IterableDataset,
    dev_set: IterableDataset,
    learning_rate: float = 1e-2,
    weight_decay: float = 1e-6,
    gradient_clip: float = 60.0,
    epoch_num: int = 200,
    report_rate: int = 10,
    batch_size: int = 32,
    batch_double: float = 40,
):

    # choose Adam for optimization
    # https://pytorch.org/docs/stable/optim.html#torch.optim.Adam
    optimizer = Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # Perform SGD in a loop
    for t in range(epoch_num):
        time_begin = datetime.now()

        train_loss: float = 0.0

        # handle aptative batch size
        batch_size: int = (
            batch_size if (t + 1) % batch_double != 0 else batch_size * 2
        )

        # create batched loader
        train_loader = batch_loader(
            train_set,
            batch_size=batch_size,
            num_workers=0,
        )

        for batch in train_loader:
            model.train()

            # zero the parameter gradients
            optimizer.zero_grad()

            # compute loss, backward
            loss = model.loss(batch)
            loss.backward()

            # scaling the gradients down, places a limit on the size of the parameter updates
            # https://pytorch.org/docs/stable/nn.html#clip-grad-norm
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            # optimize
            optimizer.step()

            # save for statistics
            train_loss += loss.item()

            # reduce memory usage by deleting loss after calculation
            # https://discuss.pytorch.org/t/calling-loss-backward-reduce-memory-usage/2735
            del loss

        # --- if is reporting epoch
        if (t + 1) % report_rate == 0:

            # create dev loader
            dev_loader = batch_loader(
                dev_set,
                batch_size=batch_size,
                num_workers=0,
            )

            print(
                "[--- @{:02}: \t loss(train)={:2.4f} \t acc(train)={:2.4f} \t acc(dev)={:2.4f} \t time(epoch)={} ---]".format(
                    (t + 1),
                    train_loss / len(train_set),
                    model.evaluate(train_loader),
                    model.evaluate(dev_loader),
                    datetime.now() - time_begin,
                )
            )
