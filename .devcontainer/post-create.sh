#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install --upgrade pip

if [ -f requirements.txt ]; then
    pip install -r requirements.txt
fi

npm install -g @anthropic-ai/claude-code