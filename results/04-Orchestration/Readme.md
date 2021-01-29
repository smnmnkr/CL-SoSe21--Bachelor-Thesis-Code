# Orchestration Experiment

Results of baseline gradient training on the POS-Tagger, chained with the derivative-free algorithms.

## Baseline:

Full training logs can be found in `full.baseline.txt`.

```
[--- @500:       loss(train)=0.0092      acc(train)=0.9345       acc(dev)=0.8832         time(epoch)=0:00:01.844635 ---]

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

## Results

### Evolution

Full training logs can be found in `full.evolve.txt`.

```
[--- @01:        avg(train)=0.9493       best(train)=0.9506      best(dev)=0.8964        time(epoch)=0:01:03.637501 ---]

[--- _AVG_       tp:  3083       fp:   329       fn:   329       tn:     0       prec=0.904      rec=0.904       acc=0.904       f1=0.904 ---]
[--- ADJ         tp:   167       fp:    66       fn:    57       tn:     0       prec=0.717      rec=0.746       acc=0.746       f1=0.731 ---]
[--- ADP         tp:   466       fp:    22       fn:    22       tn:     0       prec=0.955      rec=0.955       acc=0.955       f1=0.955 ---]
[--- ADV         tp:    79       fp:    24       fn:    49       tn:     0       prec=0.767      rec=0.617       acc=0.617       f1=0.684 ---]
[--- AUX         tp:   229       fp:     6       fn:     5       tn:     0       prec=0.974      rec=0.979       acc=0.979       f1=0.977 ---]
[--- CCONJ       tp:    86       fp:     1       fn:    10       tn:     0       prec=0.989      rec=0.896       acc=0.896       f1=0.940 ---]
[--- DET         tp:   429       fp:     7       fn:    10       tn:     0       prec=0.984      rec=0.977       acc=0.977       f1=0.981 ---]
[--- INTJ        tp:     0       fp:     0       fn:     2       tn:     0       prec=0.000      rec=0.000       acc=0.000       f1=0.000 ---]
[--- NOUN        tp:   689       fp:    68       fn:    64       tn:     0       prec=0.910      rec=0.915       acc=0.915       f1=0.913 ---]
[--- NUM         tp:    57       fp:     2       fn:     4       tn:     0       prec=0.966      rec=0.934       acc=0.934       f1=0.950 ---]
[--- PART        tp:    64       fp:     9       fn:     2       tn:     0       prec=0.877      rec=0.970       acc=0.970       f1=0.921 ---]
[--- PRON        tp:    98       fp:    11       fn:    11       tn:     0       prec=0.899      rec=0.899       acc=0.899       f1=0.899 ---]
[--- PROPN       tp:    75       fp:    35       fn:    15       tn:     0       prec=0.682      rec=0.833       acc=0.833       f1=0.750 ---]
[--- PUNCT       tp:   339       fp:    12       fn:     0       tn:     0       prec=0.966      rec=1.000       acc=1.000       f1=0.983 ---]
[--- SCONJ       tp:    31       fp:     4       fn:    20       tn:     0       prec=0.886      rec=0.608       acc=0.608       f1=0.721 ---]
[--- VERB        tp:   274       fp:    62       fn:    52       tn:     0       prec=0.815      rec=0.840       acc=0.840       f1=0.828 ---]
[--- X           tp:     0       fp:     0       fn:     2       tn:     0       prec=0.000      rec=0.000       acc=0.000       f1=0.000 ---]
[--- _           tp:     0       fp:     0       fn:     4       tn:     0       prec=0.000      rec=0.000       acc=0.000       f1=0.000 ---]
```

## Config

### POS-Tagger

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

### Training

```json
{
  "type": "evolve",
  "population_size": 50,
  "parameters": {
    "mutation_rate": 0.02,
    "selection_rate": 5,
    "crossover_rate": 0.5,
    "epoch_num": 1,
    "report_rate": 1,
    "batch_size": 96
  }
}
```

### Data

```json
{
  "embedding": "./data/cc.en.32.bin",
  "preprocess": true,
  "train": "./data/en_partut-ud-train.conllu",
  "dev": "./data/en_partut-ud-dev.conllu",
  "test": "./data/en_partut-ud-test.conllu",
  "test": "./.../model"
}
```
