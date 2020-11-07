# Evolution Experiment on Fixed v. Flex Mutation Rate

Results of the experiment, in which the mutation rate is fixed, linear, square root or inverse sigmoid dependent of the model scoring.

## Results

Full training logs can be found in `full.*.txt`.

#### Fixed

```
[--- @200:       avg(train)=0.4969       best(train)=0.5114      best(dev)=0.5334        time(epoch)=0:00:11.653701 ---]
[--- @400:       avg(train)=0.5236       best(train)=0.5390      best(dev)=0.5500        time(epoch)=0:00:11.777872 ---]
[--- @600:       avg(train)=0.5409       best(train)=0.5532      best(dev)=0.5382        time(epoch)=0:00:11.759873 ---]
[--- @800:       avg(train)=0.5502       best(train)=0.5690      best(dev)=0.5566        time(epoch)=0:00:11.595868 ---]
[--- @1000:      avg(train)=0.5588       best(train)=0.5713      best(dev)=0.5611        time(epoch)=0:00:11.438197 ---]
```

#### Linear

```
[--- @200:       avg(train)=0.5391       best(train)=0.5603      best(dev)=0.5495        time(epoch)=0:00:10.551649 ---]
[--- @400:       avg(train)=0.5894       best(train)=0.6028      best(dev)=0.5961        time(epoch)=0:00:11.004269 ---]
[--- @600:       avg(train)=0.6113       best(train)=0.6265      best(dev)=0.6114        time(epoch)=0:00:11.229853 ---]
[--- @800:       avg(train)=0.6233       best(train)=0.6328      best(dev)=0.6329        time(epoch)=0:00:11.296530 ---]
[--- @1000:      avg(train)=0.6277       best(train)=0.6422      best(dev)=0.6362        time(epoch)=0:00:11.504025 ---]
```

#### Square Root

```
[--- @200:       avg(train)=0.5665       best(train)=0.5823      best(dev)=0.5790        time(epoch)=0:00:10.609153 ---]
[--- @400:       avg(train)=0.6054       best(train)=0.6202      best(dev)=0.6092        time(epoch)=0:00:10.745859 ---]
[--- @600:       avg(train)=0.6164       best(train)=0.6344      best(dev)=0.6187        time(epoch)=0:00:10.777898 ---]
[--- @1000:      avg(train)=0.6264       best(train)=0.6407      best(dev)=0.6145        time(epoch)=0:00:10.869540 ---]

```

#### Inverse Sigmoid

```
[--- @200:       avg(train)=0.5527       best(train)=0.5745      best(dev)=0.5562        time(epoch)=0:00:10.864475 ---]
[--- @400:       avg(train)=0.5912       best(train)=0.6028      best(dev)=0.5819        time(epoch)=0:00:11.900924 ---]
[--- @600:       avg(train)=0.6056       best(train)=0.6154      best(dev)=0.6108        time(epoch)=0:00:11.567459 ---]
[--- @800:       avg(train)=0.6196       best(train)=0.6367      best(dev)=0.6114        time(epoch)=0:00:11.475933 ---]
[--- @1000:      avg(train)=0.6322       best(train)=0.6399      best(dev)=0.6166        time(epoch)=0:00:12.592668 ---]

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
  "crossover_rate": 0.5,
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
