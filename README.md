# beyond Gradient Descent

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

# run demo simplex
make simplex

# run demo orchestra
make orchestra
```

### Custom Configuration

```bash
# run custom training
python3 -m beyondGD -M ${tagger_config.json} -T ${train_config.json} -D ${data_config.json}
```

## Configuration

### Model (POS-Tagger)

```jsonc
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

Supports the following optimization algorithms: Gradient Descent, Evolution _(ES)_, Swarm Based Optimization _(PSO)_, and the Nelder–Mead method _(Simplex)_.
It is possible to orchestrate the tasks individually in the training process.

```jsonc
{
  "tasks": [
    {
      "type": "string", // Supports: [descent, evolve, swarm, simplex]
      "population_size": 200, // Only: [evolve, swarm, simplex]
      "parameters": {
        // Descent:
        "learning_rate": 5e-2,
        "weight_decay": 1e-6,
        "gradient_clip": 60.0,
        /// Evolve:
        "mutation_rate": 0.02,
        "selection_rate": 10,
        "crossover_rate": 0.5,
        // Simplex:
        "expansion_rate": 2.0,
        "contraction_rate": 0.5,
        "shrink_rate": 0.02,
        // Swarm:
        "learning_rate": 0.02,
        "velocity_weight": 1.0,
        "personal_weight": 0.5,
        "global_weight": 0.75,
        // General:
        "epoch_num": 50,
        "report_rate": 5,
        "batch_size": 32
      }
    }
  ]
}
```

### Data

```jsonc
{
  "embedding": "path/to/fasttext-data.bin",
  "preprocess": true,
  "train": "path/to/train.conllu",
  "dev": "path/to/dev.conllu",
  "test": "path/to/test.conllu",
  "load_model": "path/to/existing_model", // load existing model from .pickle file
  "save_model": "path/to/trained_model" // save model after training as .pickle file
}
```

## Experiments

The experiments located in the results directory, can be reproduced with the following make commands:

```bash
# 00-Baseline-Gradient
make exp0

# 01-Baseline-Evolve
make exp1

# 02-Baseline-Swarm
make exp2

# 03-Baseline-Simplex
make exp3

# 04-Orchestration
make exp4
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
- 3.0 Include simplex training
- 3.1 Reworked tasks interface
- 4.0 Reworked into orchestrated training process
- 4.1 Added model load/save function
- 4.2 Reworked simplex optimization
- 5.0 Added new PSO optimizer, discard old swarm approach
