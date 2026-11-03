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

# segments(): sign of twist and shoulder movement on a hand-made pose (facing +z, left = +x).
names = metrics._joint_names(77)
fake = np.zeros((2, 77, 3))
for f in range(2):
    fake[f, names.index("LeftLeg")], fake[f, names.index("RightLeg")] = (0.1, 0.9, 0), (-0.1, 0.9, 0)
    fake[f, names.index("LeftArm")], fake[f, names.index("RightArm")] = (0.2, 1.4, 0), (-0.2, 1.4, 0)
fake[1, names.index("RightArm")] = (-0.2, 1.4, -0.1)  # frame 1: right shoulder pushed 10 cm back
seg = metrics.segments({"posed_joints": fake}, [("hit right", 0, 1)])[0]
assert seg["twist_start_deg"] == 0 and seg["twist_max_deg"] > 10, seg  # positive = right shoulder back
assert abs(seg["right_shoulder_max_back_m"] - 0.1) < 1e-6 and seg["left_shoulder_max_back_m"] == 0, seg

# mirror_pose(): mirroring twice is the identity, and a mirrored pose survives forward kinematics.
import torch  # noqa: E402
from kimodo.skeleton.registry import build_skeleton  # noqa: E402
from kimodo_mcp.server import mirror_pose  # noqa: E402

skel30 = build_skeleton(30)
rots, pos = torch.from_numpy(motion["global_rot_mats"][[30, 76]]), torch.from_numpy(motion["posed_joints"][[30, 76]])
r2, p2 = mirror_pose(*mirror_pose(rots, pos, list(skel30.bone_index)), list(skel30.bone_index))
assert torch.allclose(r2, rots, atol=1e-6) and torch.allclose(p2, pos, atol=1e-6)
rm, pm = mirror_pose(rots, pos, list(skel30.bone_index))
_, fk_pos, _ = skel30.fk(skel30.global_rots_to_local_rots(rm), pm[:, skel30.root_idx])
assert float((fk_pos - pm).abs().max()) < 0.01, "mirrored pose drifts under FK"

# face_heading(): hips end up facing the target, the pose shape and twist are unchanged, FK agrees.
from kimodo_mcp.server import face_heading  # noqa: E402

rh, ph = face_heading(rots, pos, skel30.root_idx, 90.0)
fwd = rh[:, skel30.root_idx, :, 2][:, [0, 2]]
fwd = fwd / fwd.norm(dim=1, keepdim=True)  # horizontal facing (hips also pitch, e.g. when running)
assert torch.allclose(fwd, torch.tensor([[1.0, 0.0]] * 2), atol=1e-4)  # now facing +x
assert torch.allclose(torch.cdist(ph[0], ph[0]), torch.cdist(pos[0], pos[0]), atol=1e-3)  # rigid: same shape (float32)
seg0 = metrics.segments({"posed_joints": pos.numpy()}, [("", 0, 1)])[0]
seg1 = metrics.segments({"posed_joints": ph.numpy()}, [("", 0, 1)])[0]
assert abs(seg0["twist_start_deg"] - seg1["twist_start_deg"]) < 0.5  # twist relative to hips preserved
_, fk_pos, _ = skel30.fk(skel30.global_rots_to_local_rots(rh), ph[:, skel30.root_idx])
assert float((fk_pos - ph).abs().max()) < 0.01, "re-headed pose drifts under FK"

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
