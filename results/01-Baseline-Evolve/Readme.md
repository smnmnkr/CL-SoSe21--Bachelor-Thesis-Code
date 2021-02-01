# Baseline Evolution Training Results

Results of baseline evolution training on the POS-Tagger.

## Results

Full training logs can be found in `full.txt`.

```
[--- TIMETRACK || method: setup -- time: 0:00:06.621022 ---]
[--- @50:        avg(train)=0.4950       best(train)=0.5026      best(dev)=0.5237        time(epoch)=0:04:06.235168 ---]
[--- @100:       avg(train)=0.5948       best(train)=0.6046      best(dev)=0.6100        time(epoch)=0:04:08.955152 ---]
[--- @150:       avg(train)=0.6220       best(train)=0.6289      best(dev)=0.6291        time(epoch)=0:05:04.837814 ---]
[--- @200:       avg(train)=0.6340       best(train)=0.6375      best(dev)=0.6445        time(epoch)=0:04:06.751611 ---]
[--- @250:       avg(train)=0.6395       best(train)=0.6440      best(dev)=0.6452        time(epoch)=0:04:06.068963 ---]
[--- @300:       avg(train)=0.6409       best(train)=0.6441      best(dev)=0.6397        time(epoch)=0:04:05.093510 ---]
[--- @350:       avg(train)=0.6414       best(train)=0.6450      best(dev)=0.6508        time(epoch)=0:04:06.133698 ---]
[--- @400:       avg(train)=0.6435       best(train)=0.6467      best(dev)=0.6485        time(epoch)=0:04:04.970824 ---]
[--- @450:       avg(train)=0.6520       best(train)=0.6567      best(dev)=0.6427        time(epoch)=0:04:05.139962 ---]
[--- @500:       avg(train)=0.6681       best(train)=0.6725      best(dev)=0.6570        time(epoch)=0:04:04.776638 ---]

[--- _AVG_       tp:  2295       fp:  1117       fn:  1117       tn:     0       prec=0.673      rec=0.673       acc=0.673       f1=0.673 ---]
[--- ADJ         tp:     0       fp:     0       fn:   224       tn:     0       prec=0.000      rec=0.000       acc=0.000       f1=0.000 ---]
[--- ADP         tp:   472       fp:   279       fn:    16       tn:     0       prec=0.628      rec=0.967       acc=0.967       f1=0.762 ---]
[--- ADV         tp:     0       fp:     0       fn:   128       tn:     0       prec=0.000      rec=0.000       acc=0.000       f1=0.000 ---]
[--- AUX         tp:     0       fp:     1       fn:   234       tn:     0       prec=0.000      rec=0.000       acc=0.000       f1=0.000 ---]
[--- CCONJ       tp:     0       fp:     0       fn:    96       tn:     0       prec=0.000      rec=0.000       acc=0.000       f1=0.000 ---]
[--- DET         tp:   420       fp:    48       fn:    19       tn:     0       prec=0.897      rec=0.957       acc=0.957       f1=0.926 ---]
[--- INTJ        tp:     0       fp:     0       fn:     2       tn:     0       prec=0.000      rec=0.000       acc=0.000       f1=0.000 ---]
[--- NOUN        tp:   663       fp:   260       fn:    90       tn:     0       prec=0.718      rec=0.880       acc=0.880       f1=0.791 ---]
[--- NUM         tp:     0       fp:     0       fn:    61       tn:     0       prec=0.000      rec=0.000       acc=0.000       f1=0.000 ---]
[--- PART        tp:     0       fp:     0       fn:    66       tn:     0       prec=0.000      rec=0.000       acc=0.000       f1=0.000 ---]
[--- PRON        tp:    91       fp:    87       fn:    18       tn:     0       prec=0.511      rec=0.835       acc=0.835       f1=0.634 ---]
[--- PROPN       tp:    55       fp:    71       fn:    35       tn:     0       prec=0.437      rec=0.611       acc=0.611       f1=0.509 ---]
[--- PUNCT       tp:   336       fp:    40       fn:     3       tn:     0       prec=0.894      rec=0.991       acc=0.991       f1=0.940 ---]
[--- SCONJ       tp:     0       fp:     0       fn:    51       tn:     0       prec=0.000      rec=0.000       acc=0.000       f1=0.000 ---]
[--- VERB        tp:   258       fp:   331       fn:    68       tn:     0       prec=0.438      rec=0.791       acc=0.791       f1=0.564 ---]
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
  "tasks": [
    {
      "type": "evolve",
      "population_size": 200,
      "parameters": {
        "mutation_rate": 0.02,
        "selection_rate": 10,
        "crossover_rate": 0.5,
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
  "train": "./data/en_partut-ud-train.conllu",
  "dev": "./data/en_partut-ud-dev.conllu",
  "test": "./data/en_partut-ud-test.conllu",
  "load_model": null,
  "save_model": "./.../model"
}
```
