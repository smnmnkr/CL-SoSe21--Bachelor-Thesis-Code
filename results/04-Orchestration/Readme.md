# Orchestration Experiment (ONGOING)

Results of baseline gradient training on the POS-Tagger, chained with the derivative-free algorithms.
To fully reproduce these results it is necessary to run the experiment `00-Baseline-Gradient-Training` at first using `make exp0`.
Then run `make exp4` to start this experiment.

## Baseline:

Full training logs can be found in `00-Baseline-Gradient-Training`.

```
[--- @500:       loss(train)=0.0164      acc(train)=0.9567       acc(dev)=0.7954         time(epoch)=0:00:00.192108 ---]

[--- _AVG_       tp:  2751       fp:   661       fn:   661       prec=0.806      rec=0.806       f1=0.806 ---]
[--- ADJ         tp:   124       fp:   100       fn:   100       prec=0.554      rec=0.554       f1=0.554 ---]
[--- ADP         tp:   441       fp:    44       fn:    47       prec=0.909      rec=0.904       f1=0.906 ---]
[--- ADV         tp:    38       fp:    57       fn:    90       prec=0.400      rec=0.297       f1=0.341 ---]
[--- AUX         tp:   221       fp:    37       fn:    13       prec=0.857      rec=0.944       f1=0.898 ---]
[--- CCONJ       tp:    74       fp:     1       fn:    22       prec=0.987      rec=0.771       f1=0.865 ---]
[--- DET         tp:   403       fp:    27       fn:    36       prec=0.937      rec=0.918       f1=0.928 ---]
[--- INTJ        tp:     0       fp:     0       fn:     2       prec=0.000      rec=0.000       f1=0.000 ---]
[--- NOUN        tp:   651       fp:   164       fn:   102       prec=0.799      rec=0.865       f1=0.830 ---]
[--- NUM         tp:    37       fp:     1       fn:    24       prec=0.974      rec=0.607       f1=0.747 ---]
[--- PART        tp:    40       fp:     7       fn:    26       prec=0.851      rec=0.606       f1=0.708 ---]
[--- PRON        tp:    78       fp:    27       fn:    31       prec=0.743      rec=0.716       f1=0.729 ---]
[--- PROPN       tp:    58       fp:    54       fn:    32       prec=0.518      rec=0.644       f1=0.574 ---]
[--- PUNCT       tp:   338       fp:    20       fn:     1       prec=0.944      rec=0.997       f1=0.970 ---]
[--- SCONJ       tp:    17       fp:    23       fn:    34       prec=0.425      rec=0.333       f1=0.374 ---]
[--- VERB        tp:   231       fp:    99       fn:    95       prec=0.700      rec=0.709       f1=0.704 ---]
[--- X           tp:     0       fp:     0       fn:     2       prec=0.000      rec=0.000       f1=0.000 ---]
[--- _           tp:     0       fp:     0       fn:     4       prec=0.000      rec=0.000       f1=0.000 ---]
```

## Results

### Evolution

The retrained model improves **0.3%** in accuracy given the test set.

Full training logs can be found in `full.evolve.txt`.

```
[--- EVOLVE ---]
[--- @01:        avg(train)=0.9562       best(train)=0.9577      best(dev)=0.7954        time(epoch)=0:00:23.314061 ---]
[--- @02:        avg(train)=0.9562       best(train)=0.9577      best(dev)=0.7943        time(epoch)=0:00:23.407711 ---]
[--- @03:        avg(train)=0.9563       best(train)=0.9580      best(dev)=0.7988        time(epoch)=0:00:23.295533 ---]
[--- @04:        avg(train)=0.9562       best(train)=0.9577      best(dev)=0.7947        time(epoch)=0:00:23.557220 ---]
[--- @05:        avg(train)=0.9559       best(train)=0.9582      best(dev)=0.7936        time(epoch)=0:00:23.476055 ---]

[--- EVALUATION ---]
[--- _AVG_       tp:  2759       fp:   653       fn:   653       prec=0.809      rec=0.809       f1=0.809 ---]
[--- ADJ         tp:   122       fp:   101       fn:   102       prec=0.547      rec=0.545       f1=0.546 ---]
[--- ADP         tp:   447       fp:    48       fn:    41       prec=0.903      rec=0.916       f1=0.909 ---]
[--- ADV         tp:    39       fp:    50       fn:    89       prec=0.438      rec=0.305       f1=0.359 ---]
[--- AUX         tp:   220       fp:    34       fn:    14       prec=0.866      rec=0.940       f1=0.902 ---]
[--- CCONJ       tp:    75       fp:     2       fn:    21       prec=0.974      rec=0.781       f1=0.867 ---]
[--- DET         tp:   402       fp:    27       fn:    37       prec=0.937      rec=0.916       f1=0.926 ---]
[--- INTJ        tp:     0       fp:     0       fn:     2       prec=0.000      rec=0.000       f1=0.000 ---]
[--- NOUN        tp:   644       fp:   160       fn:   109       prec=0.801      rec=0.855       f1=0.827 ---]
[--- NUM         tp:    38       fp:     1       fn:    23       prec=0.974      rec=0.623       f1=0.760 ---]
[--- PART        tp:    38       fp:     7       fn:    28       prec=0.844      rec=0.576       f1=0.685 ---]
[--- PRON        tp:    86       fp:    30       fn:    23       prec=0.741      rec=0.789       f1=0.764 ---]
[--- PROPN       tp:    60       fp:    64       fn:    30       prec=0.484      rec=0.667       f1=0.561 ---]
[--- PUNCT       tp:   338       fp:    18       fn:     1       prec=0.949      rec=0.997       f1=0.973 ---]
[--- SCONJ       tp:    18       fp:    19       fn:    33       prec=0.486      rec=0.353       f1=0.409 ---]
[--- VERB        tp:   232       fp:    92       fn:    94       prec=0.716      rec=0.712       f1=0.714 ---]
[--- X           tp:     0       fp:     0       fn:     2       prec=0.000      rec=0.000       f1=0.000 ---]
[--- _           tp:     0       fp:     0       fn:     4       prec=0.000      rec=0.000       f1=0.000 ---]
```

### SWARM

The retrained model improves **0.3%** in accuracy given the test set.

Full training logs can be found in `full.swarm.txt`.

```
[--- @01:        acc(train)=0.9563       acc(dev)=0.7969         time(epoch)=0:00:31.920649 ---]
[--- @02:        acc(train)=0.9458       acc(dev)=0.7903         time(epoch)=0:00:25.149925 ---]
[--- @03:        acc(train)=0.9541       acc(dev)=0.7969         time(epoch)=0:00:27.957945 ---]
[--- @04:        acc(train)=0.9502       acc(dev)=0.7962         time(epoch)=0:00:26.445639 ---]
[--- @05:        acc(train)=0.9555       acc(dev)=0.7947         time(epoch)=0:00:24.061831 ---]

[--- _AVG_       tp:  2760       fp:   652       fn:   652       prec=0.809      rec=0.809       f1=0.809 ---]
[--- ADJ         tp:   123       fp:    91       fn:   101       prec=0.575      rec=0.549       f1=0.562 ---]
[--- ADP         tp:   441       fp:    47       fn:    47       prec=0.904      rec=0.904       f1=0.904 ---]
[--- ADV         tp:    38       fp:    53       fn:    90       prec=0.418      rec=0.297       f1=0.347 ---]
[--- AUX         tp:   217       fp:    36       fn:    17       prec=0.858      rec=0.927       f1=0.891 ---]
[--- CCONJ       tp:    74       fp:     2       fn:    22       prec=0.974      rec=0.771       f1=0.860 ---]
[--- DET         tp:   402       fp:    27       fn:    37       prec=0.937      rec=0.916       f1=0.926 ---]
[--- INTJ        tp:     0       fp:     0       fn:     2       prec=0.000      rec=0.000       f1=0.000 ---]
[--- NOUN        tp:   660       fp:   160       fn:    93       prec=0.805      rec=0.876       f1=0.839 ---]
[--- NUM         tp:    38       fp:     2       fn:    23       prec=0.950      rec=0.623       f1=0.752 ---]
[--- PART        tp:    40       fp:     8       fn:    26       prec=0.833      rec=0.606       f1=0.702 ---]
[--- PRON        tp:    83       fp:    32       fn:    26       prec=0.722      rec=0.761       f1=0.741 ---]
[--- PROPN       tp:    59       fp:    56       fn:    31       prec=0.513      rec=0.656       f1=0.576 ---]
[--- PUNCT       tp:   338       fp:    19       fn:     1       prec=0.947      rec=0.997       f1=0.971 ---]
[--- SCONJ       tp:    18       fp:    26       fn:    33       prec=0.409      rec=0.353       f1=0.379 ---]
[--- VERB        tp:   229       fp:    93       fn:    97       prec=0.711      rec=0.702       f1=0.707 ---]
[--- X           tp:     0       fp:     0       fn:     2       prec=0.000      rec=0.000       f1=0.000 ---]
[--- _           tp:     0       fp:     0       fn:     4       prec=0.000      rec=0.000       f1=0.000 ---]
```

### SIMPLEX

The retrained model improves not in accuracy given the test set.

Full training logs can be found in `full.swarm.txt`.

```
[--- @01:        avg(train)=0.9557       best(train)=0.9570      best(dev)=0.7962        time(epoch)=0:00:45.025565 ---]
[--- @02:        avg(train)=0.9569       best(train)=0.9570      best(dev)=0.7962        time(epoch)=0:00:45.734054 ---]
[--- @03:        avg(train)=0.9570       best(train)=0.9570      best(dev)=0.7962        time(epoch)=0:00:48.746822 ---]
[--- @04:        avg(train)=0.9563       best(train)=0.9570      best(dev)=0.7962        time(epoch)=0:00:47.781189 ---]
[--- @05:        avg(train)=0.9567       best(train)=0.9570      best(dev)=0.7962        time(epoch)=0:00:49.066649 ---]

[--- _AVG_       tp:  2751       fp:   661       fn:   661       prec=0.806      rec=0.806       f1=0.806 ---]
[--- ADJ         tp:   124       fp:    99       fn:   100       prec=0.556      rec=0.554       f1=0.555 ---]
[--- ADP         tp:   441       fp:    44       fn:    47       prec=0.909      rec=0.904       f1=0.906 ---]
[--- ADV         tp:    38       fp:    57       fn:    90       prec=0.400      rec=0.297       f1=0.341 ---]
[--- AUX         tp:   221       fp:    37       fn:    13       prec=0.857      rec=0.944       f1=0.898 ---]
[--- CCONJ       tp:    74       fp:     1       fn:    22       prec=0.987      rec=0.771       f1=0.865 ---]
[--- DET         tp:   404       fp:    27       fn:    35       prec=0.937      rec=0.920       f1=0.929 ---]
[--- INTJ        tp:     0       fp:     0       fn:     2       prec=0.000      rec=0.000       f1=0.000 ---]
[--- NOUN        tp:   651       fp:   164       fn:   102       prec=0.799      rec=0.865       f1=0.830 ---]
[--- NUM         tp:    37       fp:     1       fn:    24       prec=0.974      rec=0.607       f1=0.747 ---]
[--- PART        tp:    40       fp:     7       fn:    26       prec=0.851      rec=0.606       f1=0.708 ---]
[--- PRON        tp:    77       fp:    27       fn:    32       prec=0.740      rec=0.706       f1=0.723 ---]
[--- PROPN       tp:    58       fp:    54       fn:    32       prec=0.518      rec=0.644       f1=0.574 ---]
[--- PUNCT       tp:   338       fp:    20       fn:     1       prec=0.944      rec=0.997       f1=0.970 ---]
[--- SCONJ       tp:    17       fp:    23       fn:    34       prec=0.425      rec=0.333       f1=0.374 ---]
[--- VERB        tp:   231       fp:   100       fn:    95       prec=0.698      rec=0.709       f1=0.703 ---]
[--- X           tp:     0       fp:     0       fn:     2       prec=0.000      rec=0.000       f1=0.000 ---]
[--- _           tp:     0       fp:     0       fn:     4       prec=0.000      rec=0.000       f1=0.000 ---]
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
  "tasks": [
    {
      "type": "evolve",
      "population_size": 200,
      "parameters": {
        "mutation_rate": 0.02,
        "selection_rate": 20,
        "crossover_rate": 1,
        "epoch_num": 5,
        "report_rate": 1,
        "batch_size": 96
      }
    }
  ]
}
```

```json
{
  "tasks": [
    {
      "type": "swarm",
      "population_size": 400,
      "parameters": {
        "learning_rate": 1.0,
        "velocity_weight": 1.0,
        "personal_weight": 1.0,
        "global_weight": 1.0,
        "epoch_num": 5,
        "report_rate": 1,
        "batch_size": 96
      }
    }
  ]
}
```

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
        "epoch_num": 5,
        "report_rate": 1,
        "batch_size": 96
      }
    }
  ]
}
```

### Data

```json
{
  "embedding": "./data/cc.en.32.bin",
  "preprocess": true,
  "reduce_train": 0.9,
  "train": "./data/en_partut-ud-train.conllu",
  "dev": "./data/en_partut-ud-dev.conllu",
  "test": "./data/en_partut-ud-test.conllu",
  "load_model": "./results/00-Baseline-Gradient/model",
  "save_model": "./results/04-Orchestration/model"
}
```
