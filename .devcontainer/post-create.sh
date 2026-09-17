#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV_PIP=/home/vscode/miniforge3/envs/regandrecon/bin/pip

"$CONDA_ENV_PIP" install --upgrade pip

if [ -f requirements.txt ]; then
    "$CONDA_ENV_PIP" install -r requirements.txt
fi

# Editable install so packages like `operators`, `model`, etc. are importable
# from anywhere (see pyproject.toml) without manually re-running this after
# every container rebuild.
"$CONDA_ENV_PIP" install -e .

npm install -g @anthropic-ai/claude-code