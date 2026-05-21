#!/usr/bin/env bash
set -euo pipefail

# Adam sanity check: no consensus.
python src/task_bench_agents_spsa.py \
  --mode adam \
  --task maze12 \
  --out-dir runs/run_maze12_adam_no_consensus_a1 \
  --steps 8000 \
  --hidden 64 \
  --agents 1 \
  --time-steps 8 \
  --consensus mean \
  --batch-size 256 \
  --eval-batches 8 \
  --eval-batch-size 128 \
  --eval-every 250 \
  --adam-lr 0.002 \
  --objective ce

# Adam sanity check: confidence consensus.
python src/task_bench_agents_spsa.py \
  --mode adam \
  --task maze12 \
  --out-dir runs/run_maze12_adam_confidence_consensus_a2 \
  --steps 8000 \
  --hidden 64 \
  --agents 2 \
  --time-steps 8 \
  --consensus confidence \
  --batch-size 256 \
  --eval-batches 8 \
  --eval-batch-size 128 \
  --eval-every 250 \
  --adam-lr 0.002 \
  --objective ce

# SPSA six-point board objective with scheduler.
python src/task_bench_agents_spsa.py \
  --mode spsa \
  --task maze12 \
  --out-dir runs/run_maze12_single_spsa_boardacc_sched \
  --steps 12000 \
  --hidden 64 \
  --agents 1 \
  --time-steps 8 \
  --consensus mean \
  --batch-size 256 \
  --eval-batches 8 \
  --eval-batch-size 128 \
  --eval-every 250 \
  --eps 0.02 \
  --lr 0.004 \
  --max-scalar 5e-5 \
  --spsa-estimator six_point \
  --objective board_acc \
  --scheduler \
  --select-metric board_acc \
  --restore-best-at-end
