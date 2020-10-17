# POS-Tagger

## Config:

### Train

```json
{
  "mutation_rate": 0.2,
  "population_size": 120,
  "survivor_rate": 10,
  "epoch_num": 800,
  "batch_size": 32,
  "report_rate": 50,
  "data": {
    "train": "/../universal-dependencies--en-dev_reduced.conllu",
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
    "hidden_size": 16,
    "depth": 1,
    "dropout": 0.0
  },
  "score": {
    "hidden_size": 16,
    "dropout": 0.0
  }
}
```

## Results

### Evolution per epoch || generations = 800

```
@050: 	 acc(train)=0.3292 	 acc(dev)=0.2867 	 time(epoch)=0:00:06.883392
@100: 	 acc(train)=0.4097 	 acc(dev)=0.3619 	 time(epoch)=0:00:06.840507
@150: 	 acc(train)=0.4380 	 acc(dev)=0.3651 	 time(epoch)=0:00:06.839555
@200: 	 acc(train)=0.4709 	 acc(dev)=0.3982 	 time(epoch)=0:00:06.809299
@250: 	 acc(train)=0.4900 	 acc(dev)=0.4235 	 time(epoch)=0:00:06.808772
@300: 	 acc(train)=0.5102 	 acc(dev)=0.4384 	 time(epoch)=0:00:06.833845
@350: 	 acc(train)=0.5240 	 acc(dev)=0.4409 	 time(epoch)=0:00:06.854226
@400: 	 acc(train)=0.5400 	 acc(dev)=0.4461 	 time(epoch)=0:00:06.833776
@450: 	 acc(train)=0.5538 	 acc(dev)=0.4594 	 time(epoch)=0:00:06.801663
@500: 	 acc(train)=0.5733 	 acc(dev)=0.4750 	 time(epoch)=0:00:06.846830
@550: 	 acc(train)=0.5851 	 acc(dev)=0.4881 	 time(epoch)=0:00:06.830797
@650: 	 acc(train)=0.6130 	 acc(dev)=0.5084 	 time(epoch)=0:00:06.785823
@700: 	 acc(train)=0.6173 	 acc(dev)=0.5052 	 time(epoch)=0:00:06.897681
@750: 	 acc(train)=0.6221 	 acc(dev)=0.5191 	 time(epoch)=0:00:06.858394
@800: 	 acc(train)=0.6412 	 acc(dev)=0.5283 	 time(epoch)=0:00:06.805732
```

### Evolution per batch (size=32) || 32 \* 800 = 25.600

```
@050: 	 acc(train)=0.4277 	 acc(dev)=0.3761 	 time(epoch)=0:00:10.620822
@100: 	 acc(train)=0.4497 	 acc(dev)=0.3870 	 time(epoch)=0:00:10.627297
@150: 	 acc(train)=0.4590 	 acc(dev)=0.3965 	 time(epoch)=0:00:10.828600
@200: 	 acc(train)=0.4635 	 acc(dev)=0.3916 	 time(epoch)=0:00:10.668341
@250: 	 acc(train)=0.4621 	 acc(dev)=0.3971 	 time(epoch)=0:00:10.718477
@300: 	 acc(train)=0.4647 	 acc(dev)=0.3967 	 time(epoch)=0:00:10.746136
@350: 	 acc(train)=0.4858 	 acc(dev)=0.4209 	 time(epoch)=0:00:10.790273
@400: 	 acc(train)=0.4926 	 acc(dev)=0.4250 	 time(epoch)=0:00:10.780606
@450: 	 acc(train)=0.5067 	 acc(dev)=0.4343 	 time(epoch)=0:00:10.841680
@500: 	 acc(train)=0.5085 	 acc(dev)=0.4449 	 time(epoch)=0:00:11.044431
@550: 	 acc(train)=0.5149 	 acc(dev)=0.4497 	 time(epoch)=0:00:11.035616
@600: 	 acc(train)=0.5118 	 acc(dev)=0.4461 	 time(epoch)=0:00:10.969529
@650: 	 acc(train)=0.5121 	 acc(dev)=0.4479 	 time(epoch)=0:00:11.007132
@700: 	 acc(train)=0.5197 	 acc(dev)=0.4502 	 time(epoch)=0:00:11.181946
@750: 	 acc(train)=0.5234 	 acc(dev)=0.4486 	 time(epoch)=0:00:11.235971
@800: 	 acc(train)=0.5254 	 acc(dev)=0.4465 	 time(epoch)=0:00:11.301081
```
