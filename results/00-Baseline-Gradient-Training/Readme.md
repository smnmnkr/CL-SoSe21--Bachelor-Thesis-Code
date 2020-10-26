# Baseline Gradient Training Results

Results of baseline gradient training on the POS-Tagger.

## Results

Full training logs can be found in `full.txt`.

```
[--- @200:       loss(train)=0.0191      acc(train)=0.9314       acc(dev)=0.7673         time(epoch)=0:00:00.150925 ---]
[--- @400:       loss(train)=0.0091      acc(train)=0.9519       acc(dev)=0.7658         time(epoch)=0:00:00.130456 ---]
[--- @600:       loss(train)=0.0090      acc(train)=0.9545       acc(dev)=0.7644         time(epoch)=0:00:00.134523 ---]
[--- @800:       loss(train)=0.0089      acc(train)=0.9570       acc(dev)=0.7606         time(epoch)=0:00:00.158880 ---]
[--- @1000:      loss(train)=0.0090      acc(train)=0.9578       acc(dev)=0.7553         time(epoch)=0:00:00.176035 ---]
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
