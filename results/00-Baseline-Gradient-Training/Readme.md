# Baseline Gradient Training Results

Results of baseline gradient training on the POS-Tagger.

## Results

Full training logs can be found in `full.txt`.

```
[--- @50:        loss(train)=0.0569      acc(train)=0.9337       acc(dev)=0.9267         time(epoch)=0:00:00.255271 ---]
[--- @100:       loss(train)=0.0529      acc(train)=0.9383       acc(dev)=0.9316         time(epoch)=0:00:00.253732 ---]
[--- @150:       loss(train)=0.0495      acc(train)=0.9401       acc(dev)=0.9320         time(epoch)=0:00:00.253012 ---]
[--- @200:       loss(train)=0.0278      acc(train)=0.9403       acc(dev)=0.9338         time(epoch)=0:00:00.232329 ---]
[--- @250:       loss(train)=0.0276      acc(train)=0.9419       acc(dev)=0.9335         time(epoch)=0:00:00.231755 ---]
[--- @300:       loss(train)=0.0181      acc(train)=0.9430       acc(dev)=0.9323         time(epoch)=0:00:00.214593 ---]
[--- @350:       loss(train)=0.0179      acc(train)=0.9452       acc(dev)=0.9325         time(epoch)=0:00:00.211256 ---]
[--- @400:       loss(train)=0.0188      acc(train)=0.9442       acc(dev)=0.9320         time(epoch)=0:00:00.213577 ---]
[--- @450:       loss(train)=0.0177      acc(train)=0.9444       acc(dev)=0.9317         time(epoch)=0:00:00.211638 ---]
[--- @500:       loss(train)=0.0088      acc(train)=0.9425       acc(dev)=0.9331         time(epoch)=0:00:00.199701 ---]

[--- _AVG_       tp:  1304       fp:  2108       fn:  2108       tn: 55896       prec=0.382      rec=0.382       acc=0.931       f1=0.382 ---]
[--- ADJ         tp:    52       fp:    54       fn:   172       tn:  3134       prec=0.491      rec=0.232       acc=0.934       f1=0.315 ---]
[--- ADP         tp:   200       fp:    20       fn:   288       tn:  2904       prec=0.909      rec=0.410       acc=0.910       f1=0.565 ---]
[--- ADV         tp:    22       fp:    55       fn:   106       tn:  3229       prec=0.286      rec=0.172       acc=0.953       f1=0.215 ---]
[--- AUX         tp:    83       fp:    25       fn:   151       tn:  3153       prec=0.769      rec=0.355       acc=0.948       f1=0.485 ---]
[--- CCONJ       tp:    30       fp:     6       fn:    66       tn:  3310       prec=0.833      rec=0.312       acc=0.979       f1=0.455 ---]
[--- DET         tp:   192       fp:     4       fn:   247       tn:  2969       prec=0.980      rec=0.437       acc=0.926       f1=0.605 ---]
[--- INTJ        tp:     0       fp:     2       fn:     2       tn:  3408       prec=0.000      rec=0.000       acc=0.999       f1=0.000 ---]
[--- NOUN        tp:   313       fp:   107       fn:   440       tn:  2552       prec=0.745      rec=0.416       acc=0.840       f1=0.534 ---]
[--- NUM         tp:    11       fp:     3       fn:    50       tn:  3348       prec=0.786      rec=0.180       acc=0.984       f1=0.293 ---]
[--- PART        tp:    14       fp:     9       fn:    52       tn:  3337       prec=0.609      rec=0.212       acc=0.982       f1=0.315 ---]
[--- PRON        tp:    42       fp:    28       fn:    67       tn:  3275       prec=0.600      rec=0.385       acc=0.972       f1=0.469 ---]
[--- PROPN       tp:    20       fp:    33       fn:    70       tn:  3289       prec=0.377      rec=0.222       acc=0.970       f1=0.280 ---]
[--- PUNCT       tp:   176       fp:    34       fn:   163       tn:  3039       prec=0.838      rec=0.519       acc=0.942       f1=0.641 ---]
[--- SCONJ       tp:     5       fp:    51       fn:    46       tn:  3310       prec=0.089      rec=0.098       acc=0.972       f1=0.093 ---]
[--- SYM         tp:     0       fp:   127       fn:     0       tn:  3285       prec=0.000      rec=0.000       acc=0.963       f1=0.000 ---]
[--- VERB        tp:   143       fp:   257       fn:   183       tn:  2829       prec=0.357      rec=0.439       acc=0.871       f1=0.394 ---]
[--- X           tp:     0       fp:   442       fn:     2       tn:  2968       prec=0.000      rec=0.000       acc=0.870       f1=0.000 ---]
[--- _           tp:     1       fp:   851       fn:     3       tn:  2557       prec=0.001      rec=0.250       acc=0.750       f1=0.002 ---]
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
  "train": "./data/en_partut-ud-dev.conllu",
  "dev": "./data/en_partut-ud-test.conllu",
  "test": "./data/en_partut-ud-test.conllu"
}
```
