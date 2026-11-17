#!/usr/bin/env bash
# Trains HLPD in stages (train/configs/hlpd_staged.json: the reconstruction nets, then the motion nets, then all of them at a lower
# learning rate, see train/stages.py), with train/scripts/train.sh: in a tmux session, on NPROC GPUs (default 4).
# Usage: train/scripts/train_staged.sh [any train/train.py argument, e.g. --resume <run>/best_recon.pt --start-stage motion]
# Environment: as train/scripts/train.sh, but the config is STAGED_CONFIG (default train/configs/hlpd_staged.json) and SESSION defaults
# to train-staged-<date>-<time>.
set -euo pipefail

# NOTE: CONFIG is not taken from the environment, as a shell may have it from elsewhere (e.g. that of a tmux session started by an older
# train.sh, which set CONFIG=train/configs/hlpd.json in the session), which would silently train without stages.
export CONFIG="${STAGED_CONFIG:-train/configs/hlpd_staged.json}"
export SESSION="${SESSION:-train-staged-$(date +%Y%m%d-%H%M%S)}"
exec "$(dirname "${BASH_SOURCE[0]}")/train.sh" "$@"
