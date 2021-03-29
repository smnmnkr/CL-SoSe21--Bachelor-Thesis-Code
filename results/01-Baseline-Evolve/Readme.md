# Baseline Evolution Training Results

Results of baseline evolution training on POS-Tagger. Replicate with `make exp1`.

## Results

Full training logs can be found in `full.txt`.

```
[--- @200:       avg(train)=0.4989       best(train)=0.5217      best(dev)=0.4991        time(epoch)=0:00:56.180032 ---]
[--- @400:       avg(train)=0.6403       best(train)=0.6622      best(dev)=0.6155        time(epoch)=0:00:55.578147 ---]
[--- @800:       avg(train)=0.6964       best(train)=0.7081      best(dev)=0.6508        time(epoch)=0:00:55.556195 ---]
[--- @1000:      avg(train)=0.7154       best(train)=0.7310      best(dev)=0.6603        time(epoch)=0:00:55.548262 ---]
[--- @1200:      avg(train)=0.7363       best(train)=0.7470      best(dev)=0.6798        time(epoch)=0:00:55.576461 ---]
[--- @1400:      avg(train)=0.7431       best(train)=0.7572      best(dev)=0.6768        time(epoch)=0:00:55.317885 ---]
[--- @1600:      avg(train)=0.7472       best(train)=0.7587      best(dev)=0.6856        time(epoch)=0:00:55.707603 ---]
[--- @1800:      avg(train)=0.7605       best(train)=0.7708      best(dev)=0.6948        time(epoch)=0:00:55.836743 ---]
[--- @2000:      avg(train)=0.7578       best(train)=0.7694      best(dev)=0.7011        time(epoch)=0:00:55.950835 ---]

[--- _AVG_       tp:  2490       fp:   922       fn:   922       prec=0.730      rec=0.730       f1=0.730 ---]
[--- ADJ         tp:   122       fp:   134       fn:   102       prec=0.477      rec=0.545       f1=0.508 ---]
[--- ADP         tp:   451       fp:   134       fn:    37       prec=0.771      rec=0.924       f1=0.841 ---]
[--- ADV         tp:     0       fp:     0       fn:   128       prec=0.000      rec=0.000       f1=0.000 ---]
[--- AUX         tp:   197       fp:   113       fn:    37       prec=0.635      rec=0.842       f1=0.724 ---]
[--- CCONJ       tp:     0       fp:     0       fn:    96       prec=0.000      rec=0.000       f1=0.000 ---]
[--- DET         tp:   400       fp:    56       fn:    39       prec=0.877      rec=0.911       f1=0.894 ---]
[--- INTJ        tp:     0       fp:     0       fn:     2       prec=0.000      rec=0.000       f1=0.000 ---]
[--- NOUN        tp:   629       fp:   149       fn:   124       prec=0.808      rec=0.835       f1=0.822 ---]
[--- NUM         tp:     0       fp:     0       fn:    61       prec=0.000      rec=0.000       f1=0.000 ---]
[--- PART        tp:     0       fp:     0       fn:    66       prec=0.000      rec=0.000       f1=0.000 ---]
[--- PRON        tp:    73       fp:    94       fn:    36       prec=0.437      rec=0.670       f1=0.529 ---]
[--- PROPN       tp:    51       fp:    94       fn:    39       prec=0.352      rec=0.567       f1=0.434 ---]
[--- PUNCT       tp:   338       fp:     7       fn:     1       prec=0.980      rec=0.997       f1=0.988 ---]
[--- SCONJ       tp:     6       fp:     3       fn:    45       prec=0.667      rec=0.118       f1=0.200 ---]
[--- VERB        tp:   223       fp:   138       fn:   103       prec=0.618      rec=0.684       f1=0.649 ---]
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
      "type": "evolve",
      "population_size": 400,
      "parameters": {
        "mutation_rate": 0.02,
        "crossover_prob": 0.5,
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
  "save_model": "./results/01-Baseline-Evolve/model"
}
```
