# geneticNLP

## Usage [tbc]

```bash
# run demo evolution
make evolve

# run demo training
make train
```

## Configuration [tbc]

### Model (POS-Tagger)

```json
{
  "lstm": {
    "hidden_size": 16,
    "depth": 1,
    "dropout": 0.5
  },
  "score": {
    "hidden_size": 16,
    "dropout": 0.5
  }
}
```

### Evolution // Training

```json
{
  "mutation_rate": 0.2,
  "population_size": 40,
  "selection_rate": 4,
  "epoch_num": 60,
  "report_rate": 5,
  "batch_size": 32
}
```

```json
{
  "learning_rate": 1e-2,
  "weight_decay": 1e-6,
  "gradient_clip": 60.0,
  "epoch_num": 60,
  "report_rate": 5,
  "batch_size": 32
}
```

### Data

```json
{
  "embedding": "path/to/fasttext-data.bin",
  "encoding": ["LIST", "OF", "POS", "TAGS"],
  "train": "path/to/train.conllu",
  "dev": "path/to/dev.conllu",
  "test": "path/to/test.conllu"
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

- 0.0.1 POS-Tagger preliminary beta
- 0.0.2 Optimized POS-Tagger
- 0.0.3 Optimized Gradient Descent Training
- 0.1.0 Included Genetic Algorithm Training
- 1.0.0 Created stable Experimenting Environment
- 1.0.1 Updated Unittests

## Roadmap

- Major: Include Dependency Parser
- Major: Include hybrid training approach
