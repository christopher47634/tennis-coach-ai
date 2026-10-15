import json
import os

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def load_fixture(name: str) -> list:
    """Load a sparse keypoint fixture as a list of 33-landmark frames."""
    with open(os.path.join(FIXTURE_DIR, f"{name}.json"), encoding="utf-8") as f:
        data = json.load(f)
    frames = []
    for sparse in data["frames"]:
        lms = [{"x": 0.0, "y": 0.0, "z": 0.0} for _ in range(33)]
        for idx, (x, y, z) in sparse.items():
            lms[int(idx)] = {"x": x, "y": y, "z": z}
        frames.append(lms)
    return frames
