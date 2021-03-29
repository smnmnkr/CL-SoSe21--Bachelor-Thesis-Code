# Baseline Gadam Training Results

Results of baseline Gadam training on POS-Tagger. Replicate with `make exp4`.

## Results

Full training logs can be found in `full.txt`.

```
[--- @200:       avg(train)=0.9957       best(train)=0.9990      best(dev)=0.7866        time(epoch)=0:00:20.794647 ---]
[--- @400:       avg(train)=0.9968       best(train)=0.9990      best(dev)=0.7921        time(epoch)=0:00:20.248109 ---]
[--- @600:       avg(train)=0.9974       best(train)=0.9998      best(dev)=0.7837        time(epoch)=0:00:20.851611 ---]
[--- @800:       avg(train)=0.9974       best(train)=1.0000      best(dev)=0.7925        time(epoch)=0:00:19.937526 ---]
[--- @1000:      avg(train)=0.9975       best(train)=0.9995      best(dev)=0.7984        time(epoch)=0:00:20.497028 ---]
[--- @1200:      avg(train)=0.9978       best(train)=0.9998      best(dev)=0.7863        time(epoch)=0:00:20.681123 ---]
[--- @1400:      avg(train)=0.9984       best(train)=1.0000      best(dev)=0.7988        time(epoch)=0:00:21.411053 ---]
[--- @1600:      avg(train)=0.9984       best(train)=1.0000      best(dev)=0.7899        time(epoch)=0:00:20.692275 ---]
[--- @1800:      avg(train)=0.9980       best(train)=1.0000      best(dev)=0.7999        time(epoch)=0:00:20.385787 ---]
[--- @2000:      avg(train)=0.9982       best(train)=1.0000      best(dev)=0.7951        time(epoch)=0:00:20.040778 ---]

[--- _AVG_       tp:  2776       fp:   636       fn:   636       prec=0.814      rec=0.814       f1=0.814 ---]
[--- ADJ         tp:   119       fp:   105       fn:   105       prec=0.531      rec=0.531       f1=0.531 ---]
[--- ADP         tp:   448       fp:    50       fn:    40       prec=0.900      rec=0.918       f1=0.909 ---]
[--- ADV         tp:    40       fp:    50       fn:    88       prec=0.444      rec=0.312       f1=0.367 ---]
[--- AUX         tp:   214       fp:    39       fn:    20       prec=0.846      rec=0.915       f1=0.879 ---]
[--- CCONJ       tp:    81       fp:     1       fn:    15       prec=0.988      rec=0.844       f1=0.910 ---]
[--- DET         tp:   415       fp:    30       fn:    24       prec=0.933      rec=0.945       f1=0.939 ---]
[--- INTJ        tp:     0       fp:     0       fn:     2       prec=0.000      rec=0.000       f1=0.000 ---]
[--- NOUN        tp:   640       fp:   140       fn:   113       prec=0.821      rec=0.850       f1=0.835 ---]
[--- NUM         tp:    46       fp:     7       fn:    15       prec=0.868      rec=0.754       f1=0.807 ---]
[--- PART        tp:    53       fp:    22       fn:    13       prec=0.707      rec=0.803       f1=0.752 ---]
[--- PRON        tp:    77       fp:    25       fn:    32       prec=0.755      rec=0.706       f1=0.730 ---]
[--- PROPN       tp:    53       fp:    43       fn:    37       prec=0.552      rec=0.589       f1=0.570 ---]
[--- PUNCT       tp:   339       fp:    13       fn:     0       prec=0.963      rec=1.000       f1=0.981 ---]
[--- SCONJ       tp:    28       fp:    20       fn:    23       prec=0.583      rec=0.549       f1=0.566 ---]
[--- VERB        tp:   223       fp:    91       fn:   103       prec=0.710      rec=0.684       f1=0.697 ---]
[--- X           tp:     0       fp:     0       fn:     2       prec=0.000      rec=0.000       f1=0.000 ---]
[--- _    
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
        "weight_decay": 1e-6,
        "mutation_rate": 0.02,
        "selection_rate": 10,
        "crossover_rate": 0.5,
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
