#!/bin/bash

ARGPATH="results/01-Baseline-Evolve/config"

python3 -m beyondGD -M $ARGPATH/model.json -T $ARGPATH/train.json -D $ARGPATH/data.json