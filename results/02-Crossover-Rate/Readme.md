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
[--- @200:       avg(train)=0.6190       best(train)=0.6336      best(dev)=0.6261        time(epoch)=0:00:11.141119 ---]
[--- @400:       avg(train)=0.6678       best(train)=0.6824      best(dev)=0.6553        time(epoch)=0:00:11.273738 ---]
[--- @600:       avg(train)=0.6764       best(train)=0.6911      best(dev)=0.6530        time(epoch)=0:00:11.459302 ---]
[--- @800:       avg(train)=0.6865       best(train)=0.6990      best(dev)=0.6520        time(epoch)=0:00:11.398416 ---]
[--- @1000:      avg(train)=0.6867       best(train)=0.6974      best(dev)=0.6550        time(epoch)=0:00:11.540944 ---]
[--- AVG         quantity=4576   precision=0.6589        recall=0.6589   accuracy=0.4913         f1-score=0.6589 ---]
[--- ADJ         quantity= 297   precision=0.6054        recall=0.5000   accuracy=0.3771         f1-score=0.5477 ---]
[--- ADP         quantity= 719   precision=0.6603        recall=0.9201   accuracy=0.6245         f1-score=0.7688 ---]
[--- ADV         quantity= 128   precision=0.0000        recall=0.0000   accuracy=0.0000         f1-score=0.0000 ---]
[--- AUX         quantity= 234   precision=0.0000        recall=0.0000   accuracy=0.0000         f1-score=0.0000 ---]
[--- CCONJ       quantity= 109   precision=0.4583        recall=0.1146   accuracy=0.1009         f1-score=0.1833 ---]
[--- DET         quantity= 500   precision=0.8674        recall=0.9089   accuracy=0.7980         f1-score=0.8877 ---]
[--- INTJ        quantity=   2   precision=0.0000        recall=0.0000   accuracy=0.0000         f1-score=0.0000 ---]
[--- NOUN        quantity=1058   precision=0.6830        recall=0.8725   accuracy=0.6210         f1-score=0.7662 ---]
[--- NUM         quantity=  61   precision=0.0000        recall=0.0000   accuracy=0.0000         f1-score=0.0000 ---]
[--- PART        quantity=  66   precision=0.0000        recall=0.0000   accuracy=0.0000         f1-score=0.0000 ---]
[--- PRON        quantity= 246   precision=0.3687        recall=0.7339   accuracy=0.3252         f1-score=0.4908 ---]
[--- PROPN       quantity=  90   precision=0.0000        recall=0.0000   accuracy=0.0000         f1-score=0.0000 ---]
[--- PUNCT       quantity= 443   precision=0.7631        recall=0.9882   accuracy=0.7562         f1-score=0.8612 ---]
[--- SCONJ       quantity=  58   precision=0.0000        recall=0.0000   accuracy=0.0000         f1-score=0.0000 ---]
[--- SYM         quantity=   1   precision=0.0000        recall=0.0000   accuracy=0.0000         f1-score=0.0000 ---]
[--- VERB        quantity= 557   precision=0.4702        recall=0.6288   accuracy=0.3680         f1-score=0.5381 ---]
[--- X           quantity=   2   precision=0.0000        recall=0.0000   accuracy=0.0000         f1-score=0.0000 ---]
[--- _           quantity=   5   precision=0.0000        recall=0.0000   accuracy=0.0000         f1-score=0.0000 ---]
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
  "preprocess": true,
  "train": "./data/en_partut-ud-dev.conllu",
  "dev": "./data/en_partut-ud-test.conllu",
  "test": null
}
```
