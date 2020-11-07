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
# run demo training
make train

# run demo evolution
make evolve

# run demo swarm
make swarm
```

### Custom Configuration

```bash
# run demo training
python3 -m geneticNLP train -M ${tagger_config.json} -T ${train_config.json} -D ${data_config.json}

# run demo evolution
python3 -m geneticNLP evolve -M ${tagger_config.json} -E ${evolve_config.json} -D ${data_config.json}

# run demo swarm
python3 -m geneticNLP swarm -M ${tagger_config.json} -S ${swarm_config.json} -D ${data_config.json}
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

### Training, Evolution, Swarm

```json
{
  "learning_rate": 5e-2,
  "weight_decay": 1e-6,
  "gradient_clip": 60.0,
  "epoch_num": 1000,
  "report_rate": 50,
  "batch_size": 32,
  "batch_double": 100
}
```

```json
{
  "population_size": 200,
  "selection_rate": 20,
  "crossover_rate": 1.0,
  "epoch_num": 1000,
  "report_rate": 50,
  "batch_size": 96
}
```

```json
{
  "noise_std": 0.2,
  "learning_rate": 0.1,
  "population_size": 200,
  "selection_rate": 20,
  "crossover_rate": 1.0,
  "epoch_num": 1000,
  "report_rate": 50,
  "batch_size": 96
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

- 0.0.1 POS-Tagger preliminary beta
- 0.0.2 Optimized POS-Tagger
- 0.0.3 Optimized Gradient Descent Training
- 0.1.0 Included Genetic Algorithm Training
- 1.0.0 Created stable Experimenting Environment
- 1.0.1 Updated Unittests
- 2.0.0 Include swarm training approach

## Roadmap

- Major: Include Dependency Parser
