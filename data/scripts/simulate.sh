#!/usr/bin/env bash
# Split, preprocess and simulate projection data for the assessed studies. Run data/scripts/triage.sh first.
# Usage: data/scripts/simulate.sh [--qualities high medium] [--geometry parallel3d|conebeam] [--overwrite]
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
python -m data.pipeline "$@"
