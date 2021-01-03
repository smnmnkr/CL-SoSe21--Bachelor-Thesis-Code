# geneticNLP

## Install

```bash
# install dependencies via pip3 (python 3 required)
make install # or:
pip3 install -r requirements.txt

# download ParTUT and FastText [32, EN] data files (optional)
make download
```

## Usage

### Demo

Required: `make download`

```bash
# run demo gradient descent
make descent

# run demo evolution
make evolve

# run demo swarm
make swarm

# run demo amoeba
make amoeba

# run demo orchestra
make orchestra
```

### Custom Configuration

```bash
# run custom training
python3 -m geneticNLP -M ${tagger_config.json} -T ${train_config.json} -D ${data_config.json}
```

## Configuration

### Model (POS-Tagger)

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

### Training

Supports the following optimization algorithms: Gradient Descent, Evolution _(ES)_, Swarm Based Optimiziation _(PSO)_, and the Nelder–Mead method _(Amoeba)_.
It is possible to orchestrate the tasks individually in the training process.

```json
{
  "tasks": [
    {
      "type": "string", // Supports: [descent, evolve, swarm, amoeba]
      "population_size": 200, // Only: [evolve, swarm, amoeba]
      "parameters": {
        "learning_rate": 5e-2, // Only: [descent, evolve, swarm]
        "weight_decay": 1e-6, // Only: [descent]
        "gradient_clip": 60.0, // Only: [descent]
        "batch_double": 100, // Only: [descent]
        "selection_rate": 4, // Only: [evolve]
        "crossover_rate": 1.0, // Only: [evolve]
        "noise_std": 0.75, // Only: [swarm]
        "optimizer": "custom", // Only: [swarm]
        "epoch_num": 5,
        "report_rate": 1,
        "batch_size": 32
      }
    }
  ]
}
```

### Data

```json
{
  "embedding": "path/to/fasttext-data.bin",
  "preprocess": true,
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

- 0.1 POS-Tagger preliminary beta
- 0.2 Optimized POS-Tagger
- 0.3 Optimized Gradient Descent Training
- 0.4 Included Genetic Algorithm Training
- 1.0 Created stable Experimenting Environment
- 2.0 Include swarm training approach
- 2.1 Include advance metrics
- 3.0 Include amoeba training
- 3.1 Reworked tasks interface
- 4.0 Reworked into orchestrated training process
