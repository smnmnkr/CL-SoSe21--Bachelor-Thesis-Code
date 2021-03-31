# Baseline Gadam Training Results

Results of baseline Gadam training on POS-Tagger. Replicate with `make exp4`.

## Results

Full training logs can be found in `full.txt`.

```
[--- @200:       avg(train)=0.7795       best(train)=0.9354      best(dev)=0.7943        time(epoch)=0:00:42.942030 ---]
[--- @400:       avg(train)=0.7867       best(train)=0.9674      best(dev)=0.7965        time(epoch)=0:00:41.894200 ---]
[--- @600:       avg(train)=0.7932       best(train)=0.9730      best(dev)=0.8054        time(epoch)=0:00:43.787698 ---]
[--- @800:       avg(train)=0.7992       best(train)=0.9791      best(dev)=0.8109        time(epoch)=0:00:42.300379 ---]
[--- @1000:      avg(train)=0.8054       best(train)=0.9825      best(dev)=0.8149        time(epoch)=0:00:43.584458 ---]
[--- @1200:      avg(train)=0.8019       best(train)=0.9832      best(dev)=0.8142        time(epoch)=0:00:41.762770 ---]
[--- @1400:      avg(train)=0.8037       best(train)=0.9861      best(dev)=0.8123        time(epoch)=0:00:41.908117 ---]
[--- @1600:      avg(train)=0.7993       best(train)=0.9837      best(dev)=0.8087        time(epoch)=0:00:42.303976 ---]
[--- @1800:      avg(train)=0.7978       best(train)=0.9883      best(dev)=0.8076        time(epoch)=0:00:44.229272 ---]
[--- @2000:      avg(train)=0.7979       best(train)=0.9886      best(dev)=0.8087        time(epoch)=0:00:42.849146 ---]

[--- EVALUATION ---]
[--- _AVG_       tp:  2812       fp:   600       fn:   600       prec=0.824      rec=0.824       f1=0.824 ---]
[--- ADJ         tp:   124       fp:   108       fn:   100       prec=0.534      rec=0.554       f1=0.544 ---]
[--- ADP         tp:   435       fp:    33       fn:    53       prec=0.929      rec=0.891       f1=0.910 ---]
[--- ADV         tp:    66       fp:    55       fn:    62       prec=0.545      rec=0.516       f1=0.530 ---]
[--- AUX         tp:   218       fp:    19       fn:    16       prec=0.920      rec=0.932       f1=0.926 ---]
[--- CCONJ       tp:    77       fp:     1       fn:    19       prec=0.987      rec=0.802       f1=0.885 ---]
[--- DET         tp:   403       fp:    13       fn:    36       prec=0.969      rec=0.918       f1=0.943 ---]
[--- INTJ        tp:     0       fp:     0       fn:     2       prec=0.000      rec=0.000       f1=0.000 ---]
[--- NOUN        tp:   662       fp:   164       fn:    91       prec=0.801      rec=0.879       f1=0.839 ---]
[--- NUM         tp:    42       fp:     0       fn:    19       prec=1.000      rec=0.689       f1=0.816 ---]
[--- PART        tp:    54       fp:    16       fn:    12       prec=0.771      rec=0.818       f1=0.794 ---]
[--- PRON        tp:    88       fp:    37       fn:    21       prec=0.704      rec=0.807       f1=0.752 ---]
[--- PROPN       tp:    44       fp:    20       fn:    46       prec=0.688      rec=0.489       f1=0.571 ---]
[--- PUNCT       tp:   339       fp:    16       fn:     0       prec=0.955      rec=1.000       f1=0.977 ---]
[--- SCONJ       tp:    22       fp:    13       fn:    29       prec=0.629      rec=0.431       f1=0.512 ---]
[--- VERB        tp:   238       fp:   104       fn:    88       prec=0.696      rec=0.730       f1=0.713 ---]
[--- X           tp:     0       fp:     1       fn:     2       prec=0.000      rec=0.000       f1=0.000 ---]
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
      "population_size": 200,
      "parameters": {
        "learning_rate": 5e-2,
        "learning_prob": 0.8,
        "mutation_rate": 0.02,
        "mutation_prob": 0.6,
        "crossover_prob": 0.6,
        "selection_size": 20,
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
