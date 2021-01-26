# Baseline Gradient Training Results

Results of baseline gradient training on the POS-Tagger.

## Results

Full training logs can be found in `full.txt`.

```
[--- TIMETRACK || method: setup -- time: 0:00:06.621022 ---]
[--- @50:        loss(train)=0.0476      acc(train)=0.9165       acc(dev)=0.8884         time(epoch)=0:00:02.549536 ---]
[--- @100:       loss(train)=0.0461      acc(train)=0.9266       acc(dev)=0.8928         time(epoch)=0:00:02.540303 ---]
[--- @150:       loss(train)=0.0452      acc(train)=0.9315       acc(dev)=0.8873         time(epoch)=0:00:02.535433 ---]
[--- @200:       loss(train)=0.0224      acc(train)=0.9375       acc(dev)=0.8986         time(epoch)=0:00:01.912087 ---]
[--- @250:       loss(train)=0.0218      acc(train)=0.9394       acc(dev)=0.8935         time(epoch)=0:00:01.910737 ---]
[--- @300:       loss(train)=0.0148      acc(train)=0.9419       acc(dev)=0.8935         time(epoch)=0:00:01.734768 ---]
[--- @350:       loss(train)=0.0145      acc(train)=0.9462       acc(dev)=0.8957         time(epoch)=0:00:01.724351 ---]
[--- @400:       loss(train)=0.0108      acc(train)=0.9458       acc(dev)=0.8997         time(epoch)=0:00:01.631581 ---]
[--- @450:       loss(train)=0.0106      acc(train)=0.9504       acc(dev)=0.9008         time(epoch)=0:00:01.600112 ---]
[--- @500:       loss(train)=0.0091      acc(train)=0.9496       acc(dev)=0.8917         time(epoch)=0:00:01.674434 ---]

[--- _AVG_       tp:  3071       fp:   341       fn:   341       tn:     0       prec=0.900      rec=0.900       acc=0.900       f1=0.900 ---]
[--- ADJ         tp:   132       fp:    29       fn:    92       tn:     0       prec=0.820      rec=0.589       acc=0.589       f1=0.686 ---]
[--- ADP         tp:   464       fp:    27       fn:    24       tn:     0       prec=0.945      rec=0.951       acc=0.951       f1=0.948 ---]
[--- ADV         tp:    82       fp:    24       fn:    46       tn:     0       prec=0.774      rec=0.641       acc=0.641       f1=0.701 ---]
[--- AUX         tp:   224       fp:     3       fn:    10       tn:     0       prec=0.987      rec=0.957       acc=0.957       f1=0.972 ---]
[--- CCONJ       tp:    85       fp:     0       fn:    11       tn:     0       prec=1.000      rec=0.885       acc=0.885       f1=0.939 ---]
[--- DET         tp:   425       fp:     5       fn:    14       tn:     0       prec=0.988      rec=0.968       acc=0.968       f1=0.978 ---]
[--- INTJ        tp:     0       fp:     0       fn:     2       tn:     0       prec=0.000      rec=0.000       acc=0.000       f1=0.000 ---]
[--- NOUN        tp:   710       fp:   107       fn:    43       tn:     0       prec=0.869      rec=0.943       acc=0.943       f1=0.904 ---]
[--- NUM         tp:    57       fp:     1       fn:     4       tn:     0       prec=0.983      rec=0.934       acc=0.934       f1=0.958 ---]
[--- PART        tp:    63       fp:     7       fn:     3       tn:     0       prec=0.900      rec=0.955       acc=0.955       f1=0.926 ---]
[--- PRON        tp:   102       fp:    10       fn:     7       tn:     0       prec=0.911      rec=0.936       acc=0.936       f1=0.923 ---]
[--- PROPN       tp:    75       fp:    22       fn:    15       tn:     0       prec=0.773      rec=0.833       acc=0.833       f1=0.802 ---]
[--- PUNCT       tp:   339       fp:     6       fn:     0       tn:     0       prec=0.983      rec=1.000       acc=1.000       f1=0.991 ---]
[--- SCONJ       tp:    28       fp:     4       fn:    23       tn:     0       prec=0.875      rec=0.549       acc=0.549       f1=0.675 ---]
[--- VERB        tp:   285       fp:    96       fn:    41       tn:     0       prec=0.748      rec=0.874       acc=0.874       f1=0.806 ---]
[--- X           tp:     0       fp:     0       fn:     2       tn:     0       prec=0.000      rec=0.000       acc=0.000       f1=0.000 ---]
[--- _           tp:     0       fp:     0       fn:     4       tn:     0       prec=0.000      rec=0.000       acc=0.000       f1=0.000 ---]
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
  "epoch_num": 500,
  "report_rate": 50,
  "batch_size": 32,
  "batch_double": 100
}
```

#### Data

```json
{
  "embedding": "./data/cc.en.32.bin",
  "preprocess": true,
  "train": "./data/en_partut-ud-train.conllu",
  "dev": "./data/en_partut-ud-dev.conllu",
  "test": "./data/en_partut-ud-test.conllu"
}
```
