# Sourced by the data scripts: activates the conda environment and moves to the repository root.
CONDA_ENV="${CONDA_ENV:-regandrecon}"
if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
