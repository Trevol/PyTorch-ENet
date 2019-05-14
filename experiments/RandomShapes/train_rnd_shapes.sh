#!/usr/bin/env bash

ROOT="$(pwd)/../.."

# activate venv
source $ROOT/venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$ROOT
python train_rnd_shapes.py
