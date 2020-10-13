import torch
import torch.nn as nn

from geneticNLP.utils import get_device


def batch_loss(model, data):
    """Calculate the total loss of the model on the given dataset."""

    # CPU/GPU device
    device = get_device()

    inputs = []
    target_pos_ixs = []

    for sent in data:
        words = map(lambda tok: tok.word, sent)
        gold_tags = map(lambda tok: tok.pos, sent)

        for tag in gold_tags:
            target_pos_ixs.append(model.enc.encode(tag))

        inputs.append(list(words))

    # Calculate the POS tagging-related loss
    return nn.CrossEntropyLoss()(
        torch.cat(model.forward(inputs)),
        torch.LongTensor(target_pos_ixs).to(device),
    )
