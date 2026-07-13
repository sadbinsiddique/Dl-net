from __future__ import annotations

import argparse
import os
from pathlib import Path
from urllib.request import urlretrieve


DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)


def download_model(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(DEFAULT_MODEL_URL, destination)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download the MediaPipe Face Landmarker model used by src/utils/eda.py."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.getenv("XDG_CACHE_HOME") or Path.home() / ".cache") / "dl-net" / "face_landmarker.task",
        help="Where to save the .task model file.",
    )
    args = parser.parse_args()

    model_path = download_model(args.output.expanduser())
    print(f"Downloaded Face Landmarker model to: {model_path}")
    print(f"Set MEDIAPIPE_FACE_LANDMARKER_MODEL={model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())