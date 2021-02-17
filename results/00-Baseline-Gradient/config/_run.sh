#!/bin/bash

ARGPATH="results/00-Baseline-Gradient/config"

python3 -m beyondGD -M $ARGPATH/model.json -T $ARGPATH/train.json -D $ARGPATH/data.json