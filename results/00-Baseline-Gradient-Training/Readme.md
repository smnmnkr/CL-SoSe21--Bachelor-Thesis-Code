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

[--- _AVG_       tp:  3076       fp:   336       fn:   336       tn:     0       prec=0.902      rec=0.902       acc=0.902       f1=0.902 ---]
[--- ADJ         tp:   166       fp:    61       fn:    58       tn:     0       prec=0.731      rec=0.741       acc=0.741       f1=0.736 ---]
[--- ADP         tp:   464       fp:    19       fn:    24       tn:     0       prec=0.961      rec=0.951       acc=0.951       f1=0.956 ---]
[--- ADV         tp:    73       fp:    23       fn:    55       tn:     0       prec=0.760      rec=0.570       acc=0.570       f1=0.652 ---]
[--- AUX         tp:   227       fp:     7       fn:     7       tn:     0       prec=0.970      rec=0.970       acc=0.970       f1=0.970 ---]
[--- CCONJ       tp:    87       fp:     1       fn:     9       tn:     0       prec=0.989      rec=0.906       acc=0.906       f1=0.946 ---]
[--- DET         tp:   429       fp:     6       fn:    10       tn:     0       prec=0.986      rec=0.977       acc=0.977       f1=0.982 ---]
[--- INTJ        tp:     0       fp:     0       fn:     2       tn:     0       prec=0.000      rec=0.000       acc=0.000       f1=0.000 ---]
[--- NOUN        tp:   687       fp:    73       fn:    66       tn:     0       prec=0.904      rec=0.912       acc=0.912       f1=0.908 ---]
[--- NUM         tp:    57       fp:     2       fn:     4       tn:     0       prec=0.966      rec=0.934       acc=0.934       f1=0.950 ---]
[--- PART        tp:    64       fp:     8       fn:     2       tn:     0       prec=0.889      rec=0.970       acc=0.970       f1=0.928 ---]
[--- PRON        tp:    98       fp:    13       fn:    11       tn:     0       prec=0.883      rec=0.899       acc=0.899       f1=0.891 ---]
[--- PROPN       tp:    74       fp:    33       fn:    16       tn:     0       prec=0.692      rec=0.822       acc=0.822       f1=0.751 ---]
[--- PUNCT       tp:   339       fp:    12       fn:     0       tn:     0       prec=0.966      rec=1.000       acc=1.000       f1=0.983 ---]
[--- SCONJ       tp:    30       fp:     4       fn:    21       tn:     0       prec=0.882      rec=0.588       acc=0.588       f1=0.706 ---]
[--- VERB        tp:   281       fp:    74       fn:    45       tn:     0       prec=0.792      rec=0.862       acc=0.862       f1=0.825 ---]
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
