from typing import List

from datetime import datetime

import torch
import torch.nn as nn

from torch.optim import Adam
from torch.utils.data import IterableDataset

from geneticNLP.data import batch_loader


def train(
    model: nn.Module,
    train_set: IterableDataset,
    dev_set: IterableDataset,
    batch_loss: callable,
    accuracy: callable,
    learning_rate: float = 1e-2,
    weight_decay: float = 1e-6,
    clip: float = 60.0,
    epoch_num: int = 60,
    batch_size: int = 16,
    report_rate: int = 10,
):
    """Train the model on the given dataset w.r.t. the batch_loss function.
    The model parameters are updated in-place.

    Args:
        model: the neural model to be trained
        train_set: the dataset to train on
        dev_set: the development dataset (can be None)
        batch_loss: the objective function we want to minimize;
            note that this function must support backpropagation!
        accuracy: accuracy of the model over the given dataset
        learning_rate: hyper-parameter of the SGD method
        decay: learning rate decay applied with scheduler
        clip: gradient clip size, limit learning rate over epoch
        epoch_num: the number of epochs of the training procedure
        batch_size: size of the SGD batches
        report_rate: how often to report the loss/accuracy on train/dev
        shuffle: shuffled data from dataloader
    """

    # choose Adam for optimization
    # https://pytorch.org/docs/stable/optim.html#torch.optim.Adam
    optimizer = Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # create batched loader
    batches = batch_loader(
        train_set,
        batch_size=batch_size,
        num_workers=0,
    )

    # activate gpu usage for the model if possible, else nothing will change
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Perform SGD in a loop
    for t in range(epoch_num):
        model.train()

        time_begin = datetime.now()
        train_loss: float = 0.0

        # We use a PyTorch DataLoader to provide a stream of
        # dataset element batches
        for batch in batches:

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward, backward
            loss = batch_loss(model, batch)
            loss.backward()

            # scaling the gradients down, places a limit on the size of the parameter updates
            # https://pytorch.org/docs/stable/nn.html#clip-grad-norm
            nn.utils.clip_grad_norm_(model.parameters(), clip)

            # optimize
            optimizer.step()

            # save for statistics
            train_loss += loss.item()

            # reduce memory usage by deleting loss after calculation
            # https://discuss.pytorch.org/t/calling-loss-backward-reduce-memory-usage/2735
            del loss

        # reporting (every `report_rate` epochs)
        if (t + 1) % report_rate == 0:
            model.eval()
            with torch.no_grad():

                # dividing by length of train_set making it comparable
                train_loss /= len(train_set)

                # get train acc
                train_acc = accuracy(model, train_set)

                # get dev accurracy if given
                if dev_set:
                    dev_acc = accuracy(model, dev_set)
                else:
                    dev_acc = 0.0

                print(
                    f"@{(t + 1):02}: \t loss(train)={train_loss:2.4f}  \t acc(train)={train_acc:2.4f}  \t acc(dev)={dev_acc:2.4f}  \t time(epoch)={datetime.now() - time_begin}"
                )
