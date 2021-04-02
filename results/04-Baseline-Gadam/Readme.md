# Baseline Gadam Training Results

Results of baseline Gadam training on POS-Tagger. Replicate with `make exp4`.

## Results

Full training logs can be found in `full.txt`.

```
[--- GADAM ---]
[--- @200:       avg(train)=0.7986       best(train)=0.9747      best(dev)=0.8101        time(epoch)=0:00:24.173563 ---]
[--- @400:       avg(train)=0.8010       best(train)=0.9774      best(dev)=0.8120        time(epoch)=0:00:24.162341 ---]
[--- @600:       avg(train)=0.7991       best(train)=0.9837      best(dev)=0.8054        time(epoch)=0:00:23.959327 ---]
[--- @800:       avg(train)=0.8006       best(train)=0.9878      best(dev)=0.8098        time(epoch)=0:00:24.170815 ---]
[--- @1000:      avg(train)=0.8062       best(train)=0.9893      best(dev)=0.8149        time(epoch)=0:00:24.588898 ---]
[--- @1200:      avg(train)=0.8049       best(train)=0.9951      best(dev)=0.8134        time(epoch)=0:00:24.171431 ---]
[--- @1400:      avg(train)=0.7982       best(train)=0.9896      best(dev)=0.8076        time(epoch)=0:00:24.056600 ---]
[--- @1600:      avg(train)=0.8049       best(train)=0.9942      best(dev)=0.8142        time(epoch)=0:00:24.182879 ---]
[--- @1800:      avg(train)=0.8026       best(train)=0.9951      best(dev)=0.8112        time(epoch)=0:00:24.689414 ---]
[--- @2000:      avg(train)=0.8023       best(train)=0.9964      best(dev)=0.8116        time(epoch)=0:00:24.626956 ---]

[--- _AVG_       tp:  2846       fp:   566       fn:   566       prec=0.834      rec=0.834       f1=0.834 ---]
[--- ADJ         tp:   139       fp:   108       fn:    85       prec=0.563      rec=0.621       f1=0.590 ---]
[--- ADP         tp:   447       fp:    50       fn:    41       prec=0.899      rec=0.916       f1=0.908 ---]
[--- ADV         tp:    67       fp:    60       fn:    61       prec=0.528      rec=0.523       f1=0.525 ---]
[--- AUX         tp:   220       fp:    17       fn:    14       prec=0.928      rec=0.940       f1=0.934 ---]
[--- CCONJ       tp:    78       fp:     3       fn:    18       prec=0.963      rec=0.812       f1=0.881 ---]
[--- DET         tp:   409       fp:    12       fn:    30       prec=0.971      rec=0.932       f1=0.951 ---]
[--- INTJ        tp:     0       fp:     0       fn:     2       prec=0.000      rec=0.000       f1=0.000 ---]
[--- NOUN        tp:   648       fp:   107       fn:   105       prec=0.858      rec=0.861       f1=0.859 ---]
[--- NUM         tp:    46       fp:     4       fn:    15       prec=0.920      rec=0.754       f1=0.829 ---]
[--- PART        tp:    49       fp:     6       fn:    17       prec=0.891      rec=0.742       f1=0.810 ---]
[--- PRON        tp:    86       fp:    45       fn:    23       prec=0.656      rec=0.789       f1=0.717 ---]
[--- PROPN       tp:    65       fp:    40       fn:    25       prec=0.619      rec=0.722       f1=0.667 ---]
[--- PUNCT       tp:   339       fp:    18       fn:     0       prec=0.950      rec=1.000       f1=0.974 ---]
[--- SCONJ       tp:    31       fp:    10       fn:    20       prec=0.756      rec=0.608       f1=0.674 ---]
[--- VERB        tp:   222       fp:    86       fn:   104       prec=0.721      rec=0.681       f1=0.700 ---]
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
      "type": "gadam",
      "population_size": 100,
      "parameters": {
        "learning_rate": 5e-2,
        "learning_prob": 1.0,
        "mutation_rate": 0.02,
        "mutation_prob": 0.8,
        "crossover_prob": 0.6,
        "selection_size": 10,
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
  "save_model": "./results/04-Baseline-Gadam/model"
}
```
