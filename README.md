# geneticNLP

## Usage [tbc]

```bash
# run demo
make demo
```

## Configuration [tbc]

### POS-Tagger

```json
{
  "embedding": {
    "type": "fasttext/untrained",
    "data": "/path/to/data",
    "dimension": 300
  },
  "lstm": {
    "hidden_size": 50,
    "depth": 2,
    "dropout": 0.5
  },
  "score": {
    "hidden_size": 50,
    "dropout": 0.2
  }
}
```

### Training

```json
{
  "learning_rate": 2e-3,
  "weight_decay": 0.01,
  "clip": 5.0,
  "epoch_num": 60,
  "batch_size": 16,
  "report_rate": 10,
  "data": {
    "train": "/path/to/train",
    "dev": "/path/to/dev",
    "test": "/path/to/test"
  }
}
```

## Testing, Linting, Cleaning

```bash
# test: pytest
make test

# lint: flake8
make lint

# clean: cache/tmp files
make clean
```

## History

- 0.0.1 Release: POS-Tagger preliminary beta

## Roadmap

- Fix: Refractor and Optimize POS-Tagger
- Fix: Refractor and Optimize Gradient Descent Training
- Minor: Create stable experimenting automation
- Minor: Include genetic algorithm training
- Major: Include Dependency Parser
- Major: Include hybrid training approach
