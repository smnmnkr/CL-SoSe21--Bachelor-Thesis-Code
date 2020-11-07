# Baseline Gradient Training Results

Results of baseline gradient training on the POS-Tagger.

## Results

Full training logs can be found in `full.txt`.

```
[--- @200:       loss(train)=0.0276      acc(train)=0.9421       acc(dev)=0.7791         time(epoch)=0:00:00.187853 ---]
[--- @400:       loss(train)=0.0186      acc(train)=0.9504       acc(dev)=0.7731         time(epoch)=0:00:00.158674 ---]
[--- @600:       loss(train)=0.0086      acc(train)=0.9607       acc(dev)=0.7728         time(epoch)=0:00:00.139725 ---]
[--- @800:       loss(train)=0.0090      acc(train)=0.9614       acc(dev)=0.7673         time(epoch)=0:00:00.152693 ---]
[--- @1000:      loss(train)=0.0088      acc(train)=0.9633       acc(dev)=0.7753         time(epoch)=0:00:00.152821 ---]

[--- AVG         quantity=4186   precision=0.7732        recall=0.7732   accuracy=0.6302         f1-score=0.7732 ---]
[--- ADJ         quantity= 317   precision=0.5485        recall=0.5045   accuracy=0.3565         f1-score=0.5256 ---]
[--- ADP         quantity= 519   precision=0.9319        recall=0.8689   accuracy=0.8170         f1-score=0.8993 ---]
[--- ADV         quantity= 235   precision=0.3228        recall=0.3984   accuracy=0.2170         f1-score=0.3566 ---]
[--- AUX         quantity= 268   precision=0.8411        recall=0.7692   accuracy=0.6716         f1-score=0.8036 ---]
[--- CCONJ       quantity= 102   precision=0.9200        recall=0.7188   accuracy=0.6765         f1-score=0.8070 ---]
[--- DET         quantity= 451   precision=0.9712        recall=0.9226   accuracy=0.8980         f1-score=0.9463 ---]
[--- INTJ        quantity=   2   precision=0.0000        recall=0.0000   accuracy=0.0000         f1-score=0.0000 ---]
[--- NOUN        quantity= 996   precision=0.7312        recall=0.8778   accuracy=0.6637         f1-score=0.7978 ---]
[--- NUM         quantity=  63   precision=0.9259        recall=0.4098   accuracy=0.3968         f1-score=0.5682 ---]
[--- PART        quantity=  74   precision=0.8000        recall=0.4848   accuracy=0.4324         f1-score=0.6038 ---]
[--- PRON        quantity= 176   precision=0.5563        recall=0.7706   accuracy=0.4773         f1-score=0.6462 ---]
[--- PROPN       quantity= 115   precision=0.6212        recall=0.4556   accuracy=0.3565         f1-score=0.5256 ---]
[--- PUNCT       quantity= 356   precision=0.9522        recall=1.0000   accuracy=0.9522         f1-score=0.9755 ---]
[--- SCONJ       quantity=  53   precision=0.7143        recall=0.0980   accuracy=0.0943         f1-score=0.1724 ---]
[--- VERB        quantity= 453   precision=0.6220        recall=0.6411   accuracy=0.4614         f1-score=0.6314 ---]
[--- X           quantity=   2   precision=0.0000        recall=0.0000   accuracy=0.0000         f1-score=0.0000 ---]
[--- _           quantity=   4   precision=0.0000        recall=0.0000   accuracy=0.0000         f1-score=0.0000 ---]
```

## Config

#### POS-Tagger

```json
{
  "embedding": {
    "size": 32,
    "dropout": 0.0
  },
  "lstm": {
    "hid_size": 16,
    "depth": 1,
    "dropout": 0.5
  },
  "score": {
    "dropout": 0.5
  }
}
```

#### Training

```json
{
  "learning_rate": 5e-2,
  "weight_decay": 1e-6,
  "gradient_clip": 60.0,
  "epoch_num": 1000,
  "report_rate": 50,
  "batch_size": 32,
  "batch_double": 100
}
```

#### Data

```json
{
  "embedding": "./data/cc.en.32.bin",
  "encoding": [
    "ADV",
    "SCONJ",
    "ADP",
    "PRON",
    "PUNCT",
    "AUX",
    "NOUN",
    "PROPN",
    "INTJ",
    "CCONJ",
    "PART",
    "X",
    "NUM",
    "ADJ",
    "SYM",
    "DET",
    "VERB",
    "_"
  ],
  "preprocess": true,
  "train": "./data/en_partut-ud-dev.conllu",
  "dev": "./data/en_partut-ud-test.conllu",
  "test": null
}
```
