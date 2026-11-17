#!/usr/bin/env bash
# Measure the physical extent of all raw studies (data/study_extents.json), used by the study_filter of the pipeline. Rerun when the raw data
# or the object definition (object_hu, min_component_voxels) in data/configs/preprocess.json changes.
# Usage: data/scripts/extents.sh [--workers 16]
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
python -m data.extent "$@"
