#!/usr/bin/env bash
# Manual quality assessment (triage) of the raw 4D CT studies. Overwrites data/quality_assessment.json, unless --resume is passed.
# Usage: data/scripts/triage.sh [--no-gui-input] [--resume]
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
python -m data.triage "$@"
