# Baseline Gradient Training Results

Results of baseline gradient training on the POS-Tagger.

## Results

Full training logs can be found in `full.txt`.

```
[--- @05:        avg(train)=0.9060       best(train)=0.9102      best(dev)=0.9098        time(epoch)=0:00:17.265976 ---]
[--- @10:        avg(train)=0.9091       best(train)=0.9142      best(dev)=0.9115        time(epoch)=0:00:17.105425 ---]
[--- @15:        avg(train)=0.9128       best(train)=0.9167      best(dev)=0.9147        time(epoch)=0:00:17.053499 ---]
[--- @20:        avg(train)=0.9145       best(train)=0.9191      best(dev)=0.9174        time(epoch)=0:00:17.017046 ---]
[--- @25:        avg(train)=0.9154       best(train)=0.9195      best(dev)=0.9174        time(epoch)=0:00:17.090610 ---]
[--- @30:        avg(train)=0.9163       best(train)=0.9201      best(dev)=0.9180        time(epoch)=0:00:17.043988 ---]
[--- @35:        avg(train)=0.9168       best(train)=0.9212      best(dev)=0.9206        time(epoch)=0:00:17.038350 ---]
[--- @40:        avg(train)=0.9171       best(train)=0.9221      best(dev)=0.9190        time(epoch)=0:00:17.018941 ---]
[--- @45:        avg(train)=0.9176       best(train)=0.9218      best(dev)=0.9207        time(epoch)=0:00:17.130869 ---]
[--- @50:        avg(train)=0.9178       best(train)=0.9222      best(dev)=0.9199        time(epoch)=0:00:17.019681 ---]

[--- _AVG_       tp:   946       fp:  2466       fn:  2466       tn: 55538       prec=0.277      rec=0.277       acc=0.920       f1=0.277 ---]
[--- ADJ         tp:     3       fp:    45       fn:   221       tn:  3143       prec=0.062      rec=0.013       acc=0.922       f1=0.022 ---]
[--- ADP         tp:   189       fp:   185       fn:   299       tn:  2739       prec=0.505      rec=0.387       acc=0.858       f1=0.439 ---]
[--- ADV         tp:     2       fp:   115       fn:   126       tn:  3169       prec=0.017      rec=0.016       acc=0.929       f1=0.016 ---]
[--- AUX         tp:     0       fp:    31       fn:   234       tn:  3147       prec=0.000      rec=0.000       acc=0.922       f1=0.000 ---]
[--- CCONJ       tp:     0       fp:     0       fn:    96       tn:  3316       prec=0.000      rec=0.000       acc=0.972       f1=0.000 ---]
[--- DET         tp:   163       fp:   480       fn:   276       tn:  2493       prec=0.253      rec=0.371       acc=0.778       f1=0.301 ---]
[--- INTJ        tp:     0       fp:     8       fn:     2       tn:  3402       prec=0.000      rec=0.000       acc=0.997       f1=0.000 ---]
[--- NOUN        tp:   360       fp:   715       fn:   393       tn:  1944       prec=0.335      rec=0.478       acc=0.675       f1=0.394 ---]
[--- NUM         tp:     1       fp:    23       fn:    60       tn:  3328       prec=0.042      rec=0.016       acc=0.976       f1=0.024 ---]
[--- PART        tp:     7       fp:    41       fn:    59       tn:  3305       prec=0.146      rec=0.106       acc=0.971       f1=0.123 ---]
[--- PRON        tp:     0       fp:     0       fn:   109       tn:  3303       prec=0.000      rec=0.000       acc=0.968       f1=0.000 ---]
[--- PROPN       tp:     0       fp:     1       fn:    90       tn:  3321       prec=0.000      rec=0.000       acc=0.973       f1=0.000 ---]
[--- PUNCT       tp:   138       fp:   262       fn:   201       tn:  2811       prec=0.345      rec=0.407       acc=0.864       f1=0.373 ---]
[--- SCONJ       tp:     0       fp:    15       fn:    51       tn:  3346       prec=0.000      rec=0.000       acc=0.981       f1=0.000 ---]
[--- SYM         tp:     0       fp:     2       fn:     0       tn:  3410       prec=0.000      rec=0.000       acc=0.999       f1=0.000 ---]
[--- VERB        tp:    83       fp:   539       fn:   243       tn:  2547       prec=0.133      rec=0.255       acc=0.771       f1=0.175 ---]
[--- X           tp:     0       fp:     2       fn:     2       tn:  3408       prec=0.000      rec=0.000       acc=0.999       f1=0.000 ---]
[--- _           tp:     0       fp:     2       fn:     4       tn:  3406       prec=0.000      rec=0.000       acc=0.998       f1=0.000 ---]
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

#### Evolution

```json
{
  "population_size": 200,
  "selection_rate": 20,
  "crossover_rate": 1.0,
  "epoch_num": 50,
  "report_rate": 5,
  "batch_size": 96
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