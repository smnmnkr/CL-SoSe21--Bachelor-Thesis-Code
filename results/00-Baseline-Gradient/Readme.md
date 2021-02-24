# Baseline Gradient Training Results

Results of baseline gradient training on POS-Tagger. Replicate with `make exp0`.

## Results

Full training logs can be found in `full.txt`.

```
[--- @50:        loss(train)=0.0196      acc(train)=0.8809       acc(dev)=0.7863         time(epoch)=0:00:00.201617 ---]
[--- @100:       loss(train)=0.0178      acc(train)=0.9169       acc(dev)=0.7811         time(epoch)=0:00:00.192861 ---]
[--- @150:       loss(train)=0.0171      acc(train)=0.9261       acc(dev)=0.7852         time(epoch)=0:00:00.210749 ---]
[--- @200:       loss(train)=0.0167      acc(train)=0.9395       acc(dev)=0.7885         time(epoch)=0:00:00.192740 ---]
[--- @250:       loss(train)=0.0166      acc(train)=0.9478       acc(dev)=0.7896         time(epoch)=0:00:00.191257 ---]
[--- @300:       loss(train)=0.0161      acc(train)=0.9502       acc(dev)=0.7833         time(epoch)=0:00:00.191886 ---]
[--- @350:       loss(train)=0.0159      acc(train)=0.9502       acc(dev)=0.7929         time(epoch)=0:00:00.199530 ---]
[--- @400:       loss(train)=0.0166      acc(train)=0.9526       acc(dev)=0.7918         time(epoch)=0:00:00.192238 ---]
[--- @450:       loss(train)=0.0166      acc(train)=0.9519       acc(dev)=0.7943         time(epoch)=0:00:00.194015 ---]
[--- @500:       loss(train)=0.0164      acc(train)=0.9567       acc(dev)=0.7954         time(epoch)=0:00:00.192108 ---]

[--- _AVG_       tp:  2751       fp:   661       fn:   661       prec=0.806      rec=0.806       f1=0.806 ---]
[--- ADJ         tp:   124       fp:   100       fn:   100       prec=0.554      rec=0.554       f1=0.554 ---]
[--- ADP         tp:   441       fp:    44       fn:    47       prec=0.909      rec=0.904       f1=0.906 ---]
[--- ADV         tp:    38       fp:    57       fn:    90       prec=0.400      rec=0.297       f1=0.341 ---]
[--- AUX         tp:   221       fp:    37       fn:    13       prec=0.857      rec=0.944       f1=0.898 ---]
[--- CCONJ       tp:    74       fp:     1       fn:    22       prec=0.987      rec=0.771       f1=0.865 ---]
[--- DET         tp:   403       fp:    27       fn:    36       prec=0.937      rec=0.918       f1=0.928 ---]
[--- INTJ        tp:     0       fp:     0       fn:     2       prec=0.000      rec=0.000       f1=0.000 ---]
[--- NOUN        tp:   651       fp:   164       fn:   102       prec=0.799      rec=0.865       f1=0.830 ---]
[--- NUM         tp:    37       fp:     1       fn:    24       prec=0.974      rec=0.607       f1=0.747 ---]
[--- PART        tp:    40       fp:     7       fn:    26       prec=0.851      rec=0.606       f1=0.708 ---]
[--- PRON        tp:    78       fp:    27       fn:    31       prec=0.743      rec=0.716       f1=0.729 ---]
[--- PROPN       tp:    58       fp:    54       fn:    32       prec=0.518      rec=0.644       f1=0.574 ---]
[--- PUNCT       tp:   338       fp:    20       fn:     1       prec=0.944      rec=0.997       f1=0.970 ---]
[--- SCONJ       tp:    17       fp:    23       fn:    34       prec=0.425      rec=0.333       f1=0.374 ---]
[--- VERB        tp:   231       fp:    99       fn:    95       prec=0.700      rec=0.709       f1=0.704 ---]
[--- X           tp:     0       fp:     0       fn:     2       prec=0.000      rec=0.000       f1=0.000 ---]
[--- _           tp:     0       fp:     0       fn:     4       prec=0.000      rec=0.000       f1=0.000 ---]
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
  "tasks": [
    {
      "type": "descent",
      "parameters": {
        "learning_rate": 5e-2,
        "weight_decay": 1e-6,
        "gradient_clip": 60.0,
        "epoch_num": 500,
        "report_rate": 50,
        "batch_size": 96
      }
    }
  ]
}
```

#### Data

```json
{
  "embedding": "./data/cc.en.32.bin",
  "preprocess": true,
  "reduce_train": 0.9,
  "train": "./data/en_partut-ud-train.conllu",
  "dev": "./data/en_partut-ud-dev.conllu",
  "test": "./data/en_partut-ud-test.conllu",
  "load_model": null,
  "save_model": "./results/00-Baseline-Gradient/model"
}
```
