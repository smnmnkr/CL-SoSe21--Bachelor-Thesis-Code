# POS-Tagger

## Config:

### Train

```json
{
  "learning_rate": 1e-2,
  "weight_decay": 1e-6,
  "clip": 60.0,
  "epoch_num": 60,
  "batch_size": 16,
  "report_rate": 10,
  "data": {
    "train": "/../universal-dependencies--en-dev.conllu",
    "dev": "/../universal-dependencies--en-test.conllu",
    "test": null
  }
}
```

### Model

```json
{
  "embedding": {
    "type": "fasttext",
    "data": "/../fasttext--cc.en.300.bin",
    "dimension": 300
  },
  "lstm": {
    "hidden_size": 50,
    "depth": 2,
    "dropout": 0.5
  },
  "score": {
    "hidden_size": 50,
    "dropout": 0.5
  }
}
```

```json
{
  "embedding": {
    "type": "untrained",
    "data": "/../universal-dependencies--en-dev.conllu",
    "dimension": 50
  },
  "lstm": {
    "hidden_size": 50,
    "depth": 2,
    "dropout": 0.5
  },
  "score": {
    "hidden_size": 50,
    "dropout": 0.5
  }
}
```

## Results

### Untrained

```
@10:     loss(train)=0.0778      acc(train)=0.9648       acc(dev)=0.8478         time(epoch)=0:00:12.229836
@20:     loss(train)=0.0747      acc(train)=0.9821       acc(dev)=0.8631         time(epoch)=0:00:12.240941
@30:     loss(train)=0.0751      acc(train)=0.9860       acc(dev)=0.8601         time(epoch)=0:00:12.226989
@40:     loss(train)=0.0730      acc(train)=0.9872       acc(dev)=0.8560         time(epoch)=0:00:12.194576
@50:     loss(train)=0.0728      acc(train)=0.9885       acc(dev)=0.8548         time(epoch)=0:00:12.230860
@60:     loss(train)=0.0728      acc(train)=0.9900       acc(dev)=0.8596         time(epoch)=0:00:12.218566
```

### FastText

```
@10:     loss(train)=0.0788      acc(train)=0.9602       acc(dev)=0.9183         time(epoch)=0:00:09.120378
@20:     loss(train)=0.0746      acc(train)=0.9812       acc(dev)=0.9208         time(epoch)=0:00:09.099247
@30:     loss(train)=0.0743      acc(train)=0.9860       acc(dev)=0.9265         time(epoch)=0:00:09.175196
@40:     loss(train)=0.0729      acc(train)=0.9883       acc(dev)=0.9240         time(epoch)=0:00:09.175637
@50:     loss(train)=0.0734      acc(train)=0.9911       acc(dev)=0.9247         time(epoch)=0:00:09.549874
@60:     loss(train)=0.0728      acc(train)=0.9925       acc(dev)=0.9271         time(epoch)=0:00:09.384629
```
