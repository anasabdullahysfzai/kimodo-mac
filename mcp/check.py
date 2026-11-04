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

# hips_heading: only hips and legs turn (here by 40 deg), so torso twist grows by that much; upper body unmoved.
from kimodo_mcp.server import lower_body  # noqa: E402

names30 = list(skel30.bone_index)
fwd0 = rots[:, skel30.root_idx, :, 2]
target = float(torch.rad2deg(torch.atan2(fwd0[0, 0], fwd0[0, 2]))) - 40.0
rl, pl = face_heading(rots[:1], pos[:1], skel30.root_idx, target, lower_body(names30), attach=names30.index("Spine1"))
upper = [i for i in range(len(names30)) if i not in lower_body(names30)]
assert torch.allclose(rl[:, upper], rots[:1, upper])  # upper body keeps its orientation
assert float((pl[:, upper] - pos[:1, upper]).abs().max()) < 0.03  # and moves at most with its attachment
t0 = metrics.segments({"posed_joints": pos[:1].numpy()}, [("", 0, 0)])[0]["twist_start_deg"]
t1 = metrics.segments({"posed_joints": pl.numpy()}, [("", 0, 0)])[0]["twist_start_deg"]
assert abs((t1 - t0) + 40) < 3, (t0, t1)  # hips turned right 40 deg, shoulders stayed -> left shoulder back (-40)
_, fk_pos, _ = skel30.fk(skel30.global_rots_to_local_rots(rl), pl[:, skel30.root_idx])
assert float((fk_pos - pl).abs().max()) < 0.01, "hips-turned pose drifts under FK"

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

# turn_deg: a fixed rotation about the vertical axis, valid even when the hips point at the floor.
rt, pt = face_heading(rots, pos, skel30.root_idx, 0.0, by_deg=90.0)
d0, d1 = (pos - pos[:, :1])[0, 5, [0, 2]], (pt - pt[:, :1])[0, 5, [0, 2]]
assert torch.allclose(torch.stack([d0[1], -d0[0]]), d1, atol=1e-5)  # (x, z) -> (z, -x) is +90 deg toward +x
_, fk_pos, _ = skel30.fk(skel30.global_rots_to_local_rots(rt), pt[:, skel30.root_idx])
assert float((fk_pos - pt).abs().max()) < 0.01, "turned pose drifts under FK"
# mix_upper_body(): legs and hips from one pose, the rest from another; FK then keeps the legs where they were.
from kimodo_mcp.server import mix_upper_body  # noqa: E402

la, lb = skel30.global_rots_to_local_rots(rots[:1]), skel30.global_rots_to_local_rots(rots[1:])
mixed = mix_upper_body(la, lb, names30)
low, up = lower_body(names30), [i for i in range(len(names30)) if i not in lower_body(names30)]
assert torch.equal(mixed[:, low], la[:, low]) and torch.equal(mixed[:, up], lb[:, up])
_, fk_mix, _ = skel30.fk(mixed, pos[:1, skel30.root_idx])
_, fk_a, _ = skel30.fk(la, pos[:1, skel30.root_idx])
assert float((fk_mix[:, low] - fk_a[:, low]).abs().max()) < 1e-5
# edit.py: curves
from kimodo_mcp import edit  # noqa: E402

w = edit.curve_weights(60, "spring", 10, attack_frames=3, decay_frames=20)
assert w[9] == 0 and w[10] == 0 and abs(w[13] - 1) < 1e-9 and abs(w[33]) < 1e-9 and w[34] == 0 and w.min() < 0  # rises, overshoots, ends at 0
w = edit.curve_weights(60, "ease", 10, 30)
assert w[10] == 0 and abs(w[20] - 1) < 1e-9 and abs(w[30]) < 1e-9 and w[31] == 0
w = edit.curve_weights(60, "hold", 10, attack_frames=5)
assert w[10] == 0 and w[15] == 1 and w[59] == 1
w = edit.curve_weights(60, "hold", 10, 50, attack_frames=5, decay_frames=5)
assert w[20] == 1 and w[45] == 1 and abs(w[50]) < 1e-9 and w[51] == 0
w = edit.curve_weights(60, [[10, 0], [20, 1], [30, 0]], 10)
assert abs(w[15] - 0.5) < 1e-9 and w[5] == 0 and w[40] == 0

# edit.py: layers
sk, loc, rt = edit.motion_parts(motion)
nm = list(sk.bone_index)
_, base_pos, _ = sk.fk(loc, rt)
_, zero_pos, _ = sk.fk(edit.apply_layers(loc, rt, sk, [{"joints": "spine", "start": 40}]), rt)
assert float((zero_pos - base_pos).abs().max()) < 1e-3  # zero-strength layer is the identity
L = {"joints": "neck_head", "pitch_deg": 40.0, "start": 40, "curve": "ease", "end": 70}
_, lp, _ = sk.fk(edit.apply_layers(loc, rt, sk, [L]), rt)
moved = (lp - base_pos).norm(dim=-1)
for n in ("LeftFoot", "RightFoot", "LeftHand", "RightHand", "Chest"):
    assert float(moved[:, nm.index(n)].max()) < 1e-4, n  # only the neck and head move
assert float(moved[:40].max()) < 1e-4 and float(moved[71:].max()) < 1e-4  # nothing outside the layer
assert float(moved[55, nm.index("Head")]) > 0.03
# signs, facing +z while the example runs forward (frame 40): + pitch moves the head back
assert float(lp[55, nm.index("Head"), 2] - base_pos[55, nm.index("Head"), 2]) < -0.02
tw = {"joints": "spine", "twist_deg": 30.0, "start": 40, "curve": "ease", "end": 70}
_, tp, _ = sk.fk(edit.apply_layers(loc, rt, sk, [tw]), rt)
assert float(tp[55, nm.index("RightArm"), 2] - base_pos[55, nm.index("RightArm"), 2]) < -0.02  # + twist: right shoulder back
assert float(tp[55, nm.index("LeftArm"), 2] - base_pos[55, nm.index("LeftArm"), 2]) > 0.02
sd = {"joints": "spine", "side_deg": 30.0, "start": 40, "curve": "ease", "end": 70}
_, sp, _ = sk.fk(edit.apply_layers(loc, rt, sk, [sd]), rt)
assert float(sp[55, nm.index("Head"), 0] - base_pos[55, nm.index("Head"), 0]) < -0.02  # + side: toward the right (-x)
assert edit.preset_layers("impact_front", 40, 2.0)[0]["pitch_deg"] == 76
assert edit.foot_drift(base_pos.numpy(), base_pos.numpy(), nm) == 0

# edit.py: splice keeps outside frames exactly and has no jump at the seams
other = sk.global_rots_to_local_rots(torch.from_numpy(motion["global_rot_mats"][::-1].copy()))
other_root = rt.flip(0) + torch.tensor([0.3, 0.0, 0.2])
sl, sr = edit.splice_range(loc, rt, other, other_root, 60, 100, 6)
assert torch.equal(sl[:60], loc[:60]) and torch.equal(sl[101:], loc[101:]) and torch.equal(sr[:60], rt[:60]) and torch.equal(sr[101:], rt[101:])
assert torch.equal(sl[66:95], other[66:95]) and torch.equal(sr[66:95], other_root[66:95])
_, sp, _ = sk.fk(sl, sr)
full_jump = float((other_root[60] - rt[59]).norm())
assert float((sr[60] - rt[59]).norm()) < 0.35 * full_jump  # first new frame is mostly source
step = (sp[1:] - sp[:-1]).norm(dim=-1).amax(1)
_, op, _ = sk.fk(other, other_root)
_, sj, _ = sk.fk(edit.splice_range(loc, rt, other, other_root, 60, 100, 1)[0], edit.splice_range(loc, rt, other, other_root, 60, 100, 1)[1])
assert float(step[58:68].max()) < float((sj[1:] - sj[:-1]).norm(dim=-1).amax(1)[58:68].max())  # blending beats a 1-frame cut
# edit.py: ground_lift covers every frame that needs it, never pops, and stays out of frames far from the sink
low = np.zeros(100); low[40:50] = -0.1
g = edit.ground_lift(low, smooth_frames=8)
assert (g[40:50] >= 0.1 - 1e-9).all() and g[:20].max() == 0 and g[70:].max() == 0
assert np.abs(np.diff(g)).max() < 0.1 / 8  # a gradual ease, not a jump
assert edit.ground_lift(low, start=60).max() == 0  # nothing to fix after frame 60
print("check passed")
