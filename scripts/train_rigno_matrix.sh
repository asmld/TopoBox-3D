#!/usr/bin/env bash
set -euo pipefail

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export PYTHONPATH="${PYTHONPATH:-$PWD/src}"

for protocol in A B C D; do
  for degree in 0 1 2; do
    for seed in 0 1 2; do
      python -m topobox3d.experiments.train_rigno \
        --protocol "${protocol}" --degree "${degree}" --seed "${seed}"
    done
  done
done
