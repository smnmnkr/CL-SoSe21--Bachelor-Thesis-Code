# Evolution Experiment on Crossover Rate

Results of the experiment, in which the crossover rate is set to 0.0, 0.5 and 1.0.

## Results

Full training logs can be found in `full.*.txt`.

#### 0.0

```
[--- @200:       avg(train)=0.5110       best(train)=0.5217      best(dev)=0.5334        time(epoch)=0:00:10.398283 ---]
[--- @400:       avg(train)=0.5436       best(train)=0.5619      best(dev)=0.5600        time(epoch)=0:00:10.711796 ---]
[--- @600:       avg(train)=0.5535       best(train)=0.5753      best(dev)=0.5670        time(epoch)=0:00:11.115493 ---]
[--- @800:       avg(train)=0.5725       best(train)=0.5894      best(dev)=0.5670        time(epoch)=0:00:11.525800 ---]
[--- @1000:      avg(train)=0.5795       best(train)=0.5981      best(dev)=0.5749        time(epoch)=0:00:11.314519 ---]
```

#### 0.5

```
[--- @200:       avg(train)=0.5391       best(train)=0.5603      best(dev)=0.5495        time(epoch)=0:00:10.551649 ---]
[--- @400:       avg(train)=0.5894       best(train)=0.6028      best(dev)=0.5961        time(epoch)=0:00:11.004269 ---]
[--- @600:       avg(train)=0.6113       best(train)=0.6265      best(dev)=0.6114        time(epoch)=0:00:11.229853 ---]
[--- @800:       avg(train)=0.6233       best(train)=0.6328      best(dev)=0.6329        time(epoch)=0:00:11.296530 ---]
[--- @1000:      avg(train)=0.6277       best(train)=0.6422      best(dev)=0.6362        time(epoch)=0:00:11.504025 ---]

```

#### 1.0

```
[--- @200:       avg(train)=0.6242       best(train)=0.6454      best(dev)=0.6455        time(epoch)=0:00:11.149978 ---]
[--- @400:       avg(train)=0.6640       best(train)=0.6895      best(dev)=0.6515        time(epoch)=0:00:11.325888 ---]
[--- @600:       avg(train)=0.6844       best(train)=0.7013      best(dev)=0.6635        time(epoch)=0:00:11.458000 ---]
[--- @800:       avg(train)=0.6953       best(train)=0.7139      best(dev)=0.6614        time(epoch)=0:00:11.610022 ---]
[--- @1000:      avg(train)=0.7016       best(train)=0.7132      best(dev)=0.6652        time(epoch)=0:00:11.531929 ---]
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
  "crossover_rate": [0.0, 0.5, 1.0],
  "convergence_min": 0.8,
  "report_rate": 50,
  "batch_size": 96
}
```

#### Data

```json
{
  "embedding": "./data/cc.en.32.bin",
  "encoding": [
    "ADV",
    "SCONJ",
    "ADP",
    "PRON",
    "PUNCT",
    "AUX",
    "NOUN",
    "PROPN",
    "INTJ",
    "CCONJ",
    "PART",
    "X",
    "NUM",
    "ADJ",
    "SYM",
    "DET",
    "VERB",
    "_"
  ],
  "preprocess": true,
  "train": "./data/en_partut-ud-dev.conllu",
  "dev": "./data/en_partut-ud-test.conllu",
  "test": null
}
```
