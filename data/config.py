import os
from pathlib import Path

# Raw 4D CT data, organized as "patient_*_*/study_*/series_*". Can be overridden with the REGANDRECON_DATA_ROOT environment variable.
DATA_ROOT: Path = Path(os.environ.get("REGANDRECON_DATA_ROOT", "/media/trulssv/LDDMM/processed"))

# Output of the data pipeline, organized as "<geometry>/<quality>/<split>/<study_id>". Can be overridden with the REGANDRECON_SIMULATED_ROOT environment variable.
SIMULATED_ROOT: Path = Path(os.environ.get("REGANDRECON_SIMULATED_ROOT", DATA_ROOT.parent / "simulated"))

DATA_DIR: Path = Path(__file__).resolve().parent
CONFIG_DIR: Path = DATA_DIR / "configs"
QUALITY_ASSESSMENT_PATH: Path = DATA_DIR / "quality_assessment.json"
PREPROCESS_CFG_PATH: Path = CONFIG_DIR / "preprocess.json"

# Default ray transform configs per scanner geometry. Data simulated with a geometry is stored under SIMULATED_ROOT / <geometry>.
RAY_TRAFO_CFG_PATHS: dict[str, Path] = {
    "parallel3d": CONFIG_DIR / "ray_trafo" / "parallel3d.json",
    "conebeam": CONFIG_DIR / "ray_trafo" / "conebeam.json",
}
DEFAULT_GEOMETRY: str = "parallel3d"
DEFAULT_SIMULATED_ROOT: Path = SIMULATED_ROOT / DEFAULT_GEOMETRY

# Studies are skipped if any of these strings occur in the study directory name.
SKIP_STUDIES: set[str] = {
    "16701739", "84231138", "18677610", "63993221",
    "09082784", "04691850", "27484786", "84300604",
}
