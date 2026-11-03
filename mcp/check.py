"""Self-check for the renderer and metrics. No model or text encoder needed.

Run with the Kimodo venv:  python mcp/check.py /path/to/kimodo
Writes check_sheet.png next to this file so you can look at it.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from kimodo_mcp import metrics, render  # noqa: E402

kimodo_repo = Path(sys.argv[1] if len(sys.argv) > 1 else Path.home() / "kimodo-local/kimodo")
example = kimodo_repo / "kimodo/assets/demo/examples/kimodo-soma-rp/01_single_text_prompt/motion.npz"
motion = dict(np.load(example))  # "runs forward and then leaps over an obstacle", 5 s

m = metrics.analyze(motion)
assert m["frames"] == 150 and m["duration_s"] == 5.0, m
assert 1.5 < m["peaks"]["Head"]["max_height_m"] < 2.2, m["peaks"]["Head"]
assert m["root"]["net_travel_m"] > 2.0, m["root"]  # it runs forward
assert m["lowest_joint_height_m"] > -0.05, m
print(metrics.one_line(m))

for view in render.VIEWS:
    frames = render.pick_frames(m["frames"], 8)
    assert frames[0] == 0 and frames[-1] == 149 and len(frames) == 8
    img = render.contact_sheet(motion, frames, view=view)
    px = np.asarray(img)
    assert img.size == (1200, 600), img.size
    body = (np.abs(px.astype(int) - render.BODY.astype(int)).sum(-1) < 120).mean()
    assert body > 0.01, f"{view}: no body pixels drawn ({body:.4f})"
    if view == "side":
        img.save(Path(__file__).parent / "check_sheet.png")
print("check passed")
