#!/usr/bin/env bash
#SBATCH --job-name=pivot-campaign
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=04:00:00
# Matched-seed roundtrip/direct controls. Run from the repository root.
# Supply your cluster account and partition when submitting.
set -euo pipefail
uv run pytest -q tests/
for SEED in 1 2 3; do
  for VARIANT in price_rt_aux price_direct_iv_aux; do
    uv run python experiments/hyperiv_price_iv_aux/scripts/train.py \
      --config experiments/configs/hyperiv_price_iv_aux.toml \
      --variant "$VARIANT" --seed "$SEED" --epochs 100 \
      --lambda-price 0.1 --lambda-rt 0.1 --tau 1e-6 \
      --equiv-check-every 100 --tag "${VARIANT}_s${SEED}" --no-wandb
  done
done
