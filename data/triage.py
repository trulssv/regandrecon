"""
Manual quality assessment (triage) of the raw 4D CT studies, the first step of the data pipeline (see data/README.md).
Each study is visualized and classified as high, medium or low quality by the user. The classifications are saved in
data/quality_assessment.json. By default any previous assessment is overwritten; with resume, studies already in it are
skipped and the new classifications are added to it.

Run: data/scripts/triage.sh [--no-gui-input] [--resume]
Or, without a display (e.g. over SSH), open data/triage.ipynb in VS Code.
"""
from pathlib import Path
import argparse
import base64
import io

from data.config import DATA_ROOT, QUALITY_ASSESSMENT_PATH
from data.loaders import Raw4DCTSeries, index_studies, load_study, study_id_from_dir
from data.utils import _to_jsonable, load_json, save_json_atomic
from visualization.dynamic_visualization import DynamicVisualization

QUALITY_LABELS = {"h": "high", "m": "medium", "l": "low"}


def _assessment_entry(series: Raw4DCTSeries, quality: str) -> dict:
    """The quality_assessment.json entry of a study, for a quality key in QUALITY_LABELS."""
    return {
        "quality": QUALITY_LABELS[quality],
        "study_dir": str(series.study_dir),
        "n_time_bins": len(series.data),
        "scan_info": _to_jsonable(series.scan_info),
    }


def _start_assessment(data_root: Path, output_path: Path, resume: bool) -> tuple[dict, list[Path]]:
    """The assessment to add to and the study dirs still to classify. Without resume the assessment starts empty, so
    output_path is overwritten at the first checkpoint; with resume it starts from output_path and assessed studies are skipped."""
    quality_assessment = load_json(output_path) if resume and output_path.exists() else {}
    pending = [d for d in index_studies(data_root) if study_id_from_dir(d) not in quality_assessment]
    if resume:
        print(f"Resuming from {output_path}: {len(quality_assessment)} studies already assessed, {len(pending)} to go.")
    return quality_assessment, pending


def assess_data_quality(use_gui_input: bool = False, data_root: Path = DATA_ROOT, output_path: Path = QUALITY_ASSESSMENT_PATH, resume: bool = False) -> None:
    """
    This function iterates over the dataset and lets the user classify the data quality of each study as high, low or medium by visualizing the data. The user input is validated to ensure that only valid classifications are accepted.
    The classifications are saved in a json file keyed by study id, together with the study directory, number of time bins and scan info. Any previous assessment in output_path is overwritten, unless resume is set, in which case the studies already in it are skipped.
    """

    from matplotlib import pyplot as plt

    quality_assessment, pending = _start_assessment(data_root, output_path, resume)

    print(f"Assessing {len(pending)} studies in data directory: {data_root}")

    for i, study_dir in enumerate(pending):
        series = load_study(study_dir, data_root)
        print(f"Study {i}: {series.study_id}")

        x = [volume.unsqueeze(0) for volume in series.data]  # Add a singleton batch dim; (Dynamic)Visualization expects each time bin as (B, D, H, W).
        scan_info = series.scan_info


        # Visualize the data for the user to assess the data quality

        vis = DynamicVisualization(scan_info, vmin=-1, vmax=0)

        if use_gui_input:
            # Show animation + classification buttons in same window.
            quality = vis.classify_plane_dynamic(x, plane="axial", slice_idx=None, prefix="")
            while quality not in ["h", "l", "m"]:
                print("No valid GUI selection was made. Falling back to terminal input.")
                quality = input("Please classify the data quality of this study as high, low or medium (h/l/m): ")
        else:
            vis.visualize_plane_dynamic(x, plane="axial", slice_idx=None, prefix="")
            quality = input("Please classify the data quality of this study as high, low or medium (h/l/m): ")
            plt.close()  # Close the visualization after the user has made their assessment

            while quality not in ["h", "l", "m"]:
                quality = input("Invalid input. Please classify the data quality of this study as high, low or medium (h/l/m): ")
        quality_assessment[series.study_id] = _assessment_entry(series, quality)

        # Checkpoint after each study to prevent losing progress on later failure.
        save_json_atomic(output_path, quality_assessment)

    # Final write (redundant but explicit)
    save_json_atomic(output_path, quality_assessment)


def _render_frames(series: Raw4DCTSeries, plane: str = "axial") -> list[bytes]:
    """Renders the middle slice of every time bin of a study as PNG bytes, styled like the triage animation."""
    from matplotlib.figure import Figure

    x = [volume.unsqueeze(0) for volume in series.data]  # (Dynamic)Visualization expects each time bin as (B, D, H, W).
    vis = DynamicVisualization(series.scan_info, vmin=-1, vmax=0)
    vis._init_shape(x)

    frames = []
    for t in range(len(x)):
        fig = Figure(figsize=(6, 6))  # Not pyplot, so the figures never show up as inline notebook output.
        vis.visualize_plane(x, ax=fig.subplots(), plane=plane, time_bin=t)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight")
        frames.append(buffer.getvalue())
    return frames


def _to_gif(frames: list[bytes], frame_interval_ms: int) -> bytes:
    """Combines PNG frames into a looping animated GIF."""
    from PIL import Image

    images = [Image.open(io.BytesIO(frame)).convert("RGB") for frame in frames]
    buffer = io.BytesIO()
    images[0].save(buffer, format="GIF", save_all=True, append_images=images[1:], duration=frame_interval_ms, loop=0)
    return buffer.getvalue()


def _img_html(data: bytes, image_format: str, width: int = 500) -> str:
    return f'<img src="data:image/{image_format};base64,{base64.b64encode(data).decode()}" width="{width}">'


def notebook_triage(data_root: Path = DATA_ROOT, output_path: Path = QUALITY_ASSESSMENT_PATH, resume: bool = False, plane: str = "axial", frame_interval_ms: int = 500):
    """
    Notebook version of assess_data_quality, for use without a display (e.g. VS Code over SSH). Returns an ipywidgets
    widget that plays each study and classifies it with High/Medium/Low buttons. As in assess_data_quality, output_path
    is overwritten unless resume is set. Undo removes the last classification of this session and shows that study again.

    Each study plays as an animated GIF that loops in the browser by itself. Moving the time slider pauses the animation
    and shows that time bin; Animate resumes it.
    """
    import ipywidgets as widgets

    quality_assessment, pending = _start_assessment(data_root, output_path, resume)
    history: list[Path] = []  # Study dirs classified in this session, for undo.
    state: dict = {"series": None, "frames": [], "gif": ""}

    # Images are HTML <img> tags rather than widgets.Image, which VS Code fails to load ("Failed to load model class
    # 'ImageModel'"). The animation is a GIF rather than a widgets.Play driving the frames from the kernel, which
    # stalled after a while: the slider kept moving but the image stopped updating.
    image = widgets.HTML()
    animate = widgets.ToggleButton(value=True, description="Animate", icon="play")
    time_slider = widgets.IntSlider(min=0, max=0, description="Time bin")
    status = widgets.HTML()
    buttons = {q: widgets.Button(description=f"{label.capitalize()} ({q})") for q, label in QUALITY_LABELS.items()}
    undo = widgets.Button(description="Undo", icon="undo")

    def _on_slider(change):
        if not state["frames"]:  # The slider is being reset for the next study.
            return
        animate.value = False
        image.value = state["frames"][change["new"]]

    def _on_animate(change):
        if state["frames"]:
            image.value = state["gif"] if change["new"] else state["frames"][time_slider.value]

    time_slider.observe(_on_slider, names="value")
    animate.observe(_on_animate, names="value")

    def _set_enabled(enabled: bool):
        for button in buttons.values():
            button.disabled = not enabled
        undo.disabled = not (enabled and history)

    def _show_next():
        _set_enabled(False)
        state["series"], state["frames"], state["gif"] = None, [], ""
        if not pending:
            status.value = f"<b>Done.</b> {len(quality_assessment)} studies assessed in {output_path}."
            undo.disabled = not history
            return

        study_dir = pending[0]
        status.value = f"Loading {study_id_from_dir(study_dir)} ..."
        series = load_study(study_dir, data_root)
        frames = _render_frames(series, plane=plane)

        time_slider.max = len(frames) - 1
        time_slider.value = 0
        state["series"] = series
        state["frames"] = [_img_html(frame, "png") for frame in frames]
        state["gif"] = _img_html(_to_gif(frames, frame_interval_ms), "gif")
        animate.value = True
        image.value = state["gif"]

        n_assessed = len(quality_assessment)
        status.value = f"<b>{series.study_id}</b> &nbsp; ({n_assessed + 1}/{n_assessed + len(pending)})"
        _set_enabled(True)

    def _classify(quality: str):
        series = state["series"]
        quality_assessment[series.study_id] = _assessment_entry(series, quality)
        save_json_atomic(output_path, quality_assessment)  # Checkpoint after each study, like assess_data_quality.
        history.append(pending.pop(0))
        _show_next()

    def _undo(_button):
        study_dir = history.pop()
        del quality_assessment[study_id_from_dir(study_dir)]
        save_json_atomic(output_path, quality_assessment)
        pending.insert(0, study_dir)
        _show_next()

    for quality, button in buttons.items():
        button.on_click(lambda _button, quality=quality: _classify(quality))
    undo.on_click(_undo)

    _show_next()
    return widgets.VBox([status, image, widgets.HBox([animate, time_slider]), widgets.HBox([*buttons.values(), undo])])


def main():
    parser = argparse.ArgumentParser(description="Assess the quality of the raw 4D CT studies.")
    parser.add_argument(
        "--gui-input",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use GUI button-based data quality classification (--no-gui-input for terminal input).",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Continue from the existing quality assessment, skipping studies already in it, instead of overwriting it.",
    )
    args = parser.parse_args()

    print(f"Data quality mode: {'GUI buttons' if args.gui_input else 'terminal input (h/l/m)'}.")
    assess_data_quality(use_gui_input=args.gui_input, resume=args.resume)



if __name__ == "__main__":
    main()
