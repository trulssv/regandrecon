#!/usr/bin/env bash
# Trains HLPD with DistributedDataParallel on the simulated studies (run data/scripts/simulate.sh first), see train/train.py.
# Usage: train/scripts/train.sh [any train/train.py argument, e.g. --resume <checkpoint> --gif-every 5 --gpus 0,1,2,3]
# Environment: NPROC (number of GPUs/processes, default 4), CONFIG (default train/configs/hlpd.json),
#              SESSION (tmux session name, default train-<date>-<time>), NO_TMUX=1 to run in the current terminal.
# Plots and GIFs go to test/plots/<run_id>/, checkpoints to /media/trulssv/LDDMM/checkpoints/<run_id>/, logs to train/runs/.
#
# The run takes hours, so unless it is already inside tmux (or NO_TMUX=1), the script starts itself in a new tmux session and attaches
# to it. Detach with Ctrl-b d: the run keeps going if the terminal or VS Code disconnects, and a terminal that stops reading the output
# can no longer block it. Reattach with `tmux attach -t <session>` (`tmux ls` lists the sessions). The session stays open after the run
# ends, so its last output can still be read.
set -euo pipefail

NPROC="${NPROC:-4}"
CONFIG="${CONFIG:-train/configs/hlpd.json}"

if [[ -z "${TMUX:-}" && "${NO_TMUX:-0}" != 1 ]]; then
    command -v tmux >/dev/null || { echo "tmux is not installed. Install it, or run with NO_TMUX=1." >&2; exit 1; }
    SESSION="${SESSION:-train-$(date +%Y%m%d-%H%M%S)}"
    if tmux has-session -t "=$SESSION" 2>/dev/null; then
        echo "A tmux session '$SESSION' already exists. Attach with: tmux attach -t $SESSION, or choose another SESSION." >&2
        exit 1
    fi
    script="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
    # NOTE: a new session gets the environment of the tmux server (which may have been started from another shell), so the settings are
    # passed explicitly, to the run only: set in the session (tmux -e), they would also be set in the shell that keeps the session open
    # after the run, and a script started from that shell would take them as its own (e.g. the CONFIG of train.sh in train_staged.sh).
    # The arguments are passed as they are (no shell quoting).
    tmux new-session -d -s "$SESSION" -c "$PWD" \
        env NPROC="$NPROC" CONFIG="$CONFIG" CONDA_ENV="${CONDA_ENV:-regandrecon}" \
        bash -c '"$0" "$@"; status=$?; echo; echo "train.sh exited with status $status."; exec env -u NPROC -u CONFIG -u CONDA_ENV bash' "$script" "$@"
    echo "Started the training in tmux session '$SESSION'. Detach with Ctrl-b d, reattach with: tmux attach -t $SESSION"
    if [[ -t 0 && -t 1 ]]; then
        exec tmux attach -t "=$SESSION"
    fi
    exit 0
fi

source "$(dirname "${BASH_SOURCE[0]}")/../../data/scripts/env.sh"
echo "Training with $CONFIG on $NPROC GPUs."
torchrun --standalone --nproc_per_node="$NPROC" -m train.train --config "$CONFIG" "$@"
