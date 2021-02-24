#!/bin/bash

ARGPATH="results/04-Orchestration/config"

python3 -m beyondGD -M $ARGPATH/model.json -T $ARGPATH/train.json -D $ARGPATH/data.json
