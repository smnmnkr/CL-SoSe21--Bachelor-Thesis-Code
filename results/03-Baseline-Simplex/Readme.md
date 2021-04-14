# Baseline Simplex Training Results

Results of baseline Simplex training on POS-Tagger. Replicate with `make exp3`.

## Results

Full training logs can be found in `full.txt`.

```java
[--- @200:       avg(train)=0.1971       best(train)=0.2073      best(dev)=0.2112        time(epoch)=0:00:56.141741 ---]
[--- @400:       avg(train)=0.2444       best(train)=0.2447      best(dev)=0.2431        time(epoch)=0:00:55.671379 ---]
[--- @600:       avg(train)=0.2612       best(train)=0.2615      best(dev)=0.2512        time(epoch)=0:00:55.916399 ---]
[--- @800:       avg(train)=0.3256       best(train)=0.3259      best(dev)=0.3217        time(epoch)=0:00:55.563603 ---]
[--- @1000:      avg(train)=0.3257       best(train)=0.3266      best(dev)=0.3213        time(epoch)=0:00:56.062957 ---]
[--- @1200:      avg(train)=0.3264       best(train)=0.3266      best(dev)=0.3224        time(epoch)=0:00:55.748726 ---]
[--- @1400:      avg(train)=0.3279       best(train)=0.3281      best(dev)=0.3224        time(epoch)=0:00:55.568578 ---]
[--- @1600:      avg(train)=0.3279       best(train)=0.3281      best(dev)=0.3224        time(epoch)=0:00:55.473780 ---]
[--- @1800:      avg(train)=0.3279       best(train)=0.3281      best(dev)=0.3224        time(epoch)=0:00:55.723392 ---]
[--- @2000:      avg(train)=0.3423       best(train)=0.3429      best(dev)=0.3412        time(epoch)=0:00:55.308003 ---]

[--- _AVG_       tp:  1184       fp:  2228       fn:  2228       prec=0.347      rec=0.347       f1=0.347 ---]
[--- ADJ         tp:     0       fp:     0       fn:   224       prec=0.000      rec=0.000       f1=0.000 ---]
[--- ADP         tp:     0       fp:     0       fn:   488       prec=0.000      rec=0.000       f1=0.000 ---]
[--- ADV         tp:     0       fp:     0       fn:   128       prec=0.000      rec=0.000       f1=0.000 ---]
[--- AUX         tp:     0       fp:     0       fn:   234       prec=0.000      rec=0.000       f1=0.000 ---]
[--- CCONJ       tp:     0       fp:     0       fn:    96       prec=0.000      rec=0.000       f1=0.000 ---]
[--- DET         tp:   390       fp:   775       fn:    49       prec=0.335      rec=0.888       f1=0.486 ---]
[--- INTJ        tp:     0       fp:     0       fn:     2       prec=0.000      rec=0.000       f1=0.000 ---]
[--- NOUN        tp:   400       fp:   329       fn:   353       prec=0.549      rec=0.531       f1=0.540 ---]
[--- NUM         tp:     0       fp:     0       fn:    61       prec=0.000      rec=0.000       f1=0.000 ---]
[--- PART        tp:     0       fp:     0       fn:    66       prec=0.000      rec=0.000       f1=0.000 ---]
[--- PRON        tp:     0       fp:     0       fn:   109       prec=0.000      rec=0.000       f1=0.000 ---]
[--- PROPN       tp:     0       fp:     0       fn:    90       prec=0.000      rec=0.000       f1=0.000 ---]
[--- PUNCT       tp:   339       fp:   157       fn:     0       prec=0.683      rec=1.000       f1=0.812 ---]
[--- SCONJ       tp:     0       fp:     0       fn:    51       prec=0.000      rec=0.000       f1=0.000 ---]
[--- VERB        tp:    55       fp:   965       fn:   271       prec=0.054      rec=0.169       f1=0.082 ---]
[--- X           tp:     0       fp:     2       fn:     2       prec=0.000      rec=0.000       f1=0.000 ---]
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
      "type": "simplex",
      "population_size": 400,
      "parameters": {
        "expansion_rate": 2.0,
        "contraction_rate": 0.5,
        "shrink_rate": 0.02,
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
  "save_model": "./results/03-Baseline-Simplex/model"
}
```
