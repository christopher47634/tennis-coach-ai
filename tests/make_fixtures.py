"""Regenerate the synthetic keypoint fixtures in tests/fixtures/.

The fixtures are deterministic, tiny stand-ins for a pose-extraction result
(no video, no MediaPipe): the 5 reference key poses from app/pipeline.py are
interpolated into a 150-frame serve, with seeded jitter.  Only the landmarks
the scoring code reads are stored.

    python tests/make_fixtures.py          # rewrite serve_*.json
    python tests/make_fixtures.py --golden # also print current outputs, to
                                           # refresh expected_scores.json by hand
"""

import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "app"))

import pipeline  # noqa: E402

USED_LANDMARKS = [11, 12, 13, 14, 15, 16, 20, 23, 24, 25, 26, 27, 28]
# (from key pose, to key pose, frame count); a pose index repeated = hold
SEGMENTS = [(0, 0, 30), (0, 1, 25), (1, 2, 25), (2, 3, 20), (3, 4, 20), (4, 4, 30)]


def build_frames(seed: int, variant: bool) -> list:
    rng = random.Random(seed)
    frames = []
    for a, b, count in SEGMENTS:
        for i in range(count):
            t = 0.0 if a == b else (1 - math.cos(math.pi * i / count)) / 2
            lm = pipeline._lerp_pose(pipeline._KEY_POSES[a], pipeline._KEY_POSES[b], t)
            frame = {}
            for idx in USED_LANDMARKS:
                x, y, z = lm[idx]["x"], lm[idx]["y"], lm[idx]["z"]
                if variant and idx == pipeline.R_KNEE:
                    x += 0.05  # knee pushed out -> unusual knee angle
                if variant and idx == pipeline.R_ELBOW:
                    x -= 0.04
                x += rng.uniform(-0.002, 0.002)
                y += rng.uniform(-0.002, 0.002)
                frame[str(idx)] = [round(x, 4), round(y, 4), round(z, 4)]
            frames.append(frame)
    return frames


def main() -> None:
    out = {"serve_base": (7, False), "serve_variant": (11, True)}
    for name, (seed, variant) in out.items():
        data = {
            "description": f"synthetic {'variant' if variant else 'base'} serve, seed={seed}",
            "fps": 30,
            "frames": build_frames(seed, variant),
        }
        path = os.path.join(HERE, "fixtures", f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, separators=(",", ":"))
        print("wrote", path, os.path.getsize(path), "bytes")
    if "--golden" in sys.argv:
        from helpers import load_fixture
        baselines = pipeline.load_baselines(os.path.join(os.path.dirname(HERE), "data", "expert_baselines.json"))
        for name in out:
            res = pipeline.analyze_landmark_sequence(load_fixture(name), baselines)
            print(name, res["overall_score"], [p["score"] for p in res["phases"]], res["phase_boundaries"])


if __name__ == "__main__":
    main()
