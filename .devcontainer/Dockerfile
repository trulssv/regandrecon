FROM mcr.microsoft.com/devcontainers/base:ubuntu-24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

USER vscode
WORKDIR /workspace/RegAndRecon

# Install Miniforge (self-contained conda) for the vscode user
ENV CONDA_DIR=/home/vscode/miniforge3
RUN curl -fsSL -o /tmp/miniforge.sh \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(uname -m).sh" \
    && bash /tmp/miniforge.sh -b -p "$CONDA_DIR" \
    && rm /tmp/miniforge.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# Accept Anaconda's Terms of Service for the "defaults" channel referenced in environment.yml
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

# Recreate the regandrecon project environment from the exported spec
COPY --chown=vscode:vscode environment.yml /tmp/environment.yml
RUN conda env create -n regandrecon -f /tmp/environment.yml \
    && conda clean -afy

# Make the env available in interactive shells by default
RUN conda init bash \
    && echo "conda activate regandrecon" >> ~/.bashrc