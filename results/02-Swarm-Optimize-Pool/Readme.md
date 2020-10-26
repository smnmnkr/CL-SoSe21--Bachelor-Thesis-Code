# Swarm Experiment (Optimize on Selection v. Generation)

Results of the experiment, in which the swarm queen is optimized through the fittest selection versus the whole generation.

## Results

Full training logs can be found in `full.*.txt`.

#### Selection

```
[--- @200:       swarm(train)=0.2660     queen(train)=0.4567     queen(dev)=0.4864       time(epoch)=0:00:08.586740 ---]
[--- @400:       swarm(train)=0.2799     queen(train)=0.4793     queen(dev)=0.5088       time(epoch)=0:00:08.744133 ---]
[--- @600:       swarm(train)=0.2868     queen(train)=0.4896     queen(dev)=0.5165       time(epoch)=0:00:08.912231 ---]
[--- @800:       swarm(train)=0.2962     queen(train)=0.5034     queen(dev)=0.5261       time(epoch)=0:00:08.909877 ---]
[--- @1000:      swarm(train)=0.2986     queen(train)=0.5134     queen(dev)=0.5353       time(epoch)=0:00:08.833827 ---]
```

#### Generation

```
[--- @200:       swarm(train)=0.2733     queen(train)=0.4845     queen(dev)=0.5009       time(epoch)=0:00:08.103342 ---]
[--- @400:       swarm(train)=0.2894     queen(train)=0.5103     queen(dev)=0.5136       time(epoch)=0:00:08.587583 ---]
[--- @600:       swarm(train)=0.2997     queen(train)=0.5193     queen(dev)=0.5289       time(epoch)=0:00:08.815386 ---]
[--- @800:       swarm(train)=0.3108     queen(train)=0.5276     queen(dev)=0.5346       time(epoch)=0:00:08.706392 ---]
[--- @1000:      swarm(train)=0.3083     queen(train)=0.5295     queen(dev)=0.5427       time(epoch)=0:00:10.275058 ---]
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

#### Hybrid

```json
{
  "noise_std": 0.2,
  "learning_rate": 0.1,
  "convergence_min": 0.8,
  "population_size": 200,
  "selection_rate": 20,
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
