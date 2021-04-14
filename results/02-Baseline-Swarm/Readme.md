# Baseline Evolution Training Results

Results of baseline PSO training on POS-Tagger. Replicate with `make exp2`.

## Results

Full training logs can be found in `full.txt`.

```java
[--- @200:       acc(train)=0.3711       acc(dev)=0.3772         time(epoch)=0:01:16.913411 ---]
[--- @400:       acc(train)=0.3920       acc(dev)=0.3790         time(epoch)=0:00:58.401214 ---]
[--- @600:       acc(train)=0.4143       acc(dev)=0.4025         time(epoch)=0:00:57.375155 ---]
[--- @800:       acc(train)=0.4236       acc(dev)=0.4135         time(epoch)=0:00:56.876509 ---]
[--- @1000:      acc(train)=0.4296       acc(dev)=0.4176         time(epoch)=0:00:57.304597 ---]
[--- @1200:      acc(train)=0.4287       acc(dev)=0.4231         time(epoch)=0:00:57.432766 ---]
[--- @1400:      acc(train)=0.4355       acc(dev)=0.4238         time(epoch)=0:00:57.279776 ---]
[--- @1600:      acc(train)=0.4394       acc(dev)=0.4300         time(epoch)=0:00:58.803632 ---]
[--- @1800:      acc(train)=0.4450       acc(dev)=0.4319         time(epoch)=0:00:56.846098 ---]
[--- @2000:      acc(train)=0.4462       acc(dev)=0.4341         time(epoch)=0:00:57.277344 ---]

[--- _AVG_       tp:  1602       fp:  1810       fn:  1810       prec=0.470      rec=0.470       f1=0.470 ---]
[--- ADJ         tp:     0       fp:     0       fn:   224       prec=0.000      rec=0.000       f1=0.000 ---]
[--- ADP         tp:   353       fp:   294       fn:   135       prec=0.546      rec=0.723       f1=0.622 ---]
[--- ADV         tp:     0       fp:     0       fn:   128       prec=0.000      rec=0.000       f1=0.000 ---]
[--- AUX         tp:     0       fp:     0       fn:   234       prec=0.000      rec=0.000       f1=0.000 ---]
[--- CCONJ       tp:     0       fp:     0       fn:    96       prec=0.000      rec=0.000       f1=0.000 ---]
[--- DET         tp:   281       fp:   224       fn:   158       prec=0.556      rec=0.640       f1=0.595 ---]
[--- INTJ        tp:     0       fp:     0       fn:     2       prec=0.000      rec=0.000       f1=0.000 ---]
[--- NOUN        tp:   638       fp:  1168       fn:   115       prec=0.353      rec=0.847       f1=0.499 ---]
[--- NUM         tp:     0       fp:     0       fn:    61       prec=0.000      rec=0.000       f1=0.000 ---]
[--- PART        tp:     0       fp:     0       fn:    66       prec=0.000      rec=0.000       f1=0.000 ---]
[--- PRON        tp:     0       fp:     0       fn:   109       prec=0.000      rec=0.000       f1=0.000 ---]
[--- PROPN       tp:     0       fp:     0       fn:    90       prec=0.000      rec=0.000       f1=0.000 ---]
[--- PUNCT       tp:   330       fp:   124       fn:     9       prec=0.727      rec=0.973       f1=0.832 ---]
[--- SCONJ       tp:     0       fp:     0       fn:    51       prec=0.000      rec=0.000       f1=0.000 ---]
[--- VERB        tp:     0       fp:     0       fn:   326       prec=0.000      rec=0.000       f1=0.000 ---]
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
      "type": "swarm",
      "population_size": 400,
      "parameters": {
        "learning_rate": 0.0001,
        "initial_velocity_rate": 0.02,
        "velocity_weight": 1.0,
        "personal_weight": 2.0,
        "global_weight": 2.0,
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
  "save_model": "./results/02-Baseline-Swarm/model"
}
```
