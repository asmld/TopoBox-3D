#!/usr/bin/env bash
set -euo pipefail

export DGLBACKEND="${DGLBACKEND:-pytorch}"
export PYTHONPATH="${PYTHONPATH:-$PWD/src}"

epochs="${TOPOBOX_EPOCHS:-300}"
models=(mgn-lite transolver gnot gaot tno)
protocols=(A B C D)
degrees=(0 1 2)
seeds=(0 1 2)

for model in "${models[@]}"; do
  for protocol in "${protocols[@]}"; do
    for degree in "${degrees[@]}"; do
      for seed in "${seeds[@]}"; do
        python -m topobox3d.experiments.train_torch \
          --model "${model}" --protocol "${protocol}" --degree "${degree}" \
          --seed "${seed}" --epochs "${epochs}" --resume --ram-cache
      done
    done
  done
done
