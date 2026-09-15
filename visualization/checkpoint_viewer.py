"""
Streamlit-based checkpoint browser for LDDMM training runs.

Usage:
    streamlit run visualization/checkpoint_viewer.py
"""

import os
from pathlib import Path

import streamlit as st

CHECKPOINT_DIR = Path("/mnt/data/LDDMM/checkpoints")

st.set_page_config(page_title="Checkpoint Viewer", layout="wide")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_runs(base: Path) -> list[Path]:
    """Return sorted list of run dirs that contain a losses.png or quality_measures/."""
    candidates = set()
    for p in base.rglob("visualizations/losses.png"):
        candidates.add(p.parent.parent)
    for p in base.rglob("quality_measures"):
        if p.is_dir():
            candidates.add(p.parent)
    return sorted(candidates)


def run_label(run: Path) -> str:
    """Human-readable label: relative path from CHECKPOINT_DIR."""
    try:
        rel = run.relative_to(CHECKPOINT_DIR)
    except ValueError:
        rel = run
    return str(rel)


def epoch_sort_key(path: Path) -> int:
    name = path.name  # e.g. "epoch_80"
    try:
        return int(name.split("_")[-1])
    except ValueError:
        return -1


MODES = ("train", "val")


def strip_mode_prefix(stem: str) -> tuple[str, str]:
    """Return (mode, base_name) by stripping a leading train_/val_ prefix."""
    for m in MODES:
        prefix = f"{m}_"
        if stem.startswith(prefix):
            return m, stem[len(prefix):]
    return "", stem


def gif_group_label(filename: str) -> str:
    """Extract a tidy group name from a GIF filename, ignoring mode prefix."""
    stem = Path(filename).stem
    _, base = strip_mode_prefix(stem)
    for sep in [" dynamic_visualization", " visualization_slices"]:
        if sep in base:
            return base.split(sep)[0]
    return base


# ---------------------------------------------------------------------------
# Sidebar – run selector
# ---------------------------------------------------------------------------

st.sidebar.title("Checkpoint Viewer")

runs = find_runs(CHECKPOINT_DIR)
if not runs:
    st.error(f"No runs found under {CHECKPOINT_DIR}")
    st.stop()

run_labels = [run_label(r) for r in runs]

# Multi-select for comparison; single view uses the first selected
mode = st.sidebar.radio("Mode", ["Single run", "Compare runs"])

if mode == "Single run":
    selected_label = st.sidebar.selectbox(
        "Select run", run_labels, index=len(run_labels) - 1
    )
    selected_runs = [runs[run_labels.index(selected_label)]]
else:
    selected_labels = st.sidebar.multiselect(
        "Select runs to compare", run_labels, default=run_labels[-3:]
    )
    selected_runs = [runs[run_labels.index(l)] for l in selected_labels]

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_loss, tab_qm, tab_vis = st.tabs(
    ["Loss Curves", "Quality Metrics", "Epoch Visualizations"]
)

# ── Loss Curves ─────────────────────────────────────────────────────────────
with tab_loss:
    st.header("Loss Curves")
    if not selected_runs:
        st.info("Select at least one run in the sidebar.")
    else:
        cols = st.columns(max(len(selected_runs), 1))
        for col, run in zip(cols, selected_runs):
            loss_img = run / "visualizations" / "losses.png"
            col.subheader(run_label(run))
            if loss_img.exists():
                col.image(str(loss_img), use_container_width=True)
            else:
                col.warning("losses.png not found")

# ── Quality Metrics ──────────────────────────────────────────────────────────
with tab_qm:
    st.header("Quality Metrics")
    if not selected_runs:
        st.info("Select at least one run in the sidebar.")
    else:
        for run in selected_runs:
            st.subheader(run_label(run))
            qm_dir = run / "quality_measures"
            if not qm_dir.exists():
                st.caption("No quality_measures/ directory found for this run.")
                continue
            pngs = sorted(qm_dir.glob("*.png"))
            if not pngs:
                st.caption("No quality metric PNGs found.")
                continue
            cols = st.columns(min(len(pngs), 4))
            for col, png in zip(cols, pngs):
                col.image(str(png), caption=png.stem, use_container_width=True)

# ── Epoch Visualizations ─────────────────────────────────────────────────────
with tab_vis:
    st.header("Epoch Visualizations")

    if mode != "Single run":
        st.info("Switch to **Single run** mode to browse epoch visualizations.")
    elif not selected_runs:
        st.info("Select a run in the sidebar.")
    else:
        run = selected_runs[0]
        vis_dir = run / "visualizations"

        epoch_dirs = sorted(
            [d for d in vis_dir.iterdir() if d.is_dir() and d.name.startswith("epoch_")],
            key=epoch_sort_key,
        )

        if not epoch_dirs:
            st.warning("No epoch subdirectories found.")
        else:
            epoch_names = [d.name for d in epoch_dirs]
            selected_epoch_name = st.select_slider(
                "Epoch", options=epoch_names, value=epoch_names[-1]
            )
            epoch_dir = vis_dir / selected_epoch_name

            gifs = sorted(epoch_dir.glob("*.gif"))
            if not gifs:
                st.warning("No GIFs found in this epoch directory.")
            else:
                # Detect which modes are present in this epoch dir
                present_modes = sorted(
                    {strip_mode_prefix(g.stem)[0] for g in gifs}
                )
                has_prefixes = any(m in MODES for m in present_modes)

                if has_prefixes:
                    mode_options = [m for m in MODES if m in present_modes]
                    if len(mode_options) > 1:
                        selected_modes = st.multiselect(
                            "Show modes", mode_options, default=mode_options
                        )
                    else:
                        selected_modes = mode_options
                else:
                    selected_modes = [""]

                # Group by base content name (mode-prefix stripped)
                groups: dict[str, dict[str, Path]] = {}
                for gif in gifs:
                    m, base = strip_mode_prefix(gif.stem)
                    group = base
                    for sep in [" dynamic_visualization", " visualization_slices"]:
                        if sep in base:
                            group = base.split(sep)[0]
                            break
                    if m not in selected_modes:
                        continue
                    groups.setdefault(group, {})[m] = gif

                for group_name, mode_gifs in groups.items():
                    st.subheader(group_name.replace("_", " ").title())
                    display_items = [
                        (m, mode_gifs[m]) for m in selected_modes if m in mode_gifs
                    ] or [(m, g) for m, g in mode_gifs.items()]
                    cols = st.columns(max(len(display_items), 1))
                    for col, (m, gif) in zip(cols, display_items):
                        label = gif.stem if not m else f"{m.upper()}: {gif.stem[len(m)+1:]}"
                        col.image(str(gif), caption=label, use_container_width=True)
                        col.download_button(
                            label="Download",
                            data=gif.read_bytes(),
                            file_name=gif.name,
                            mime="image/gif",
                            key=str(gif),  # unique key per file
                        )
