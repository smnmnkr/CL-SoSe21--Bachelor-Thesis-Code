# Baseline Gradient Training Results

Results of baseline gradient training on POS-Tagger. Replicate with `make exp0`.

## Results

Full training logs can be found in `full.txt`.

```
[--- @200:       loss(train)=0.0167      acc(train)=0.9395       acc(dev)=0.7885         time(epoch)=0:00:00.192740 ---]
[--- @400:       loss(train)=0.0166      acc(train)=0.9526       acc(dev)=0.7918         time(epoch)=0:00:00.192238 ---]
[--- @600:       loss(train)=0.0161      acc(train)=0.9604       acc(dev)=0.8032         time(epoch)=0:00:00.234703 ---]
[--- @800:       loss(train)=0.0159      acc(train)=0.9638       acc(dev)=0.8087         time(epoch)=0:00:00.237563 ---]
[--- @1000:      loss(train)=0.0152      acc(train)=0.9662       acc(dev)=0.8145         time(epoch)=0:00:00.231764 ---]
[--- @1200:      loss(train)=0.0156      acc(train)=0.9682       acc(dev)=0.8171         time(epoch)=0:00:00.239337 ---]
[--- @1400:      loss(train)=0.0156      acc(train)=0.9648       acc(dev)=0.8149         time(epoch)=0:00:00.237444 ---]
[--- @1600:      loss(train)=0.0154      acc(train)=0.9718       acc(dev)=0.8142         time(epoch)=0:00:00.235683 ---]
[--- @1800:      loss(train)=0.0151      acc(train)=0.9735       acc(dev)=0.8142         time(epoch)=0:00:00.231299 ---]
[--- @2000:      loss(train)=0.0148      acc(train)=0.9747       acc(dev)=0.8127         time(epoch)=0:00:00.235240 ---]

[--- _AVG_       tp:  2840       fp:   572       fn:   572       prec=0.832      rec=0.832       f1=0.832 ---]
[--- ADJ         tp:   134       fp:   110       fn:    90       prec=0.549      rec=0.598       f1=0.573 ---]
[--- ADP         tp:   454       fp:    46       fn:    34       prec=0.908      rec=0.930       f1=0.919 ---]
[--- ADV         tp:    54       fp:    60       fn:    74       prec=0.474      rec=0.422       f1=0.446 ---]
[--- AUX         tp:   217       fp:    20       fn:    17       prec=0.916      rec=0.927       f1=0.921 ---]
[--- CCONJ       tp:    79       fp:     3       fn:    17       prec=0.963      rec=0.823       f1=0.888 ---]
[--- DET         tp:   414       fp:    11       fn:    25       prec=0.974      rec=0.943       f1=0.958 ---]
[--- INTJ        tp:     0       fp:     0       fn:     2       prec=0.000      rec=0.000       f1=0.000 ---]
[--- NOUN        tp:   659       fp:   127       fn:    94       prec=0.838      rec=0.875       f1=0.856 ---]
[--- NUM         tp:    47       fp:     1       fn:    14       prec=0.979      rec=0.770       f1=0.862 ---]
[--- PART        tp:    55       fp:    10       fn:    11       prec=0.846      rec=0.833       f1=0.840 ---]
[--- PRON        tp:    81       fp:    12       fn:    28       prec=0.871      rec=0.743       f1=0.802 ---]
[--- PROPN       tp:    58       fp:    59       fn:    32       prec=0.496      rec=0.644       f1=0.560 ---]
[--- PUNCT       tp:   339       fp:     9       fn:     0       prec=0.974      rec=1.000       f1=0.987 ---]
[--- SCONJ       tp:    22       fp:     6       fn:    29       prec=0.786      rec=0.431       f1=0.557 ---]
[--- VERB        tp:   227       fp:    98       fn:    99       prec=0.698      rec=0.696       f1=0.697 ---]
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
        "epoch_num": 2000,
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
