"""Self-check for the renderer and metrics. No model or text encoder needed.

Run with the Kimodo venv:  python mcp/check.py /path/to/kimodo
Writes check_sheet.png next to this file so you can look at it.
"""

import json
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
# edit.py: gaussian_smooth reduces jerk, leaves constant motion alone, and honours its joint list
sm_l, sm_r = edit.gaussian_smooth(loc, rt, 3.0)
_, sm_p, _ = sk.fk(sm_l, sm_r)
jerk = lambda q: float((q[2:] - 2 * q[1:-1] + q[:-2]).norm(dim=-1).mean())
assert jerk(sm_p) < 0.6 * jerk(base_pos)
still_l = loc[:1].repeat(30, 1, 1, 1); still_r = rt[:1].repeat(30, 1)
ss_l, ss_r = edit.gaussian_smooth(still_l, still_r, 3.0)
assert float((ss_l - still_l).abs().max()) < 1e-4 and float((ss_r - still_r).abs().max()) < 1e-5
arm = edit.joint_indices("right_arm", nm); others = [i for i in range(len(nm)) if i not in arm]
sa_l, sa_r = edit.gaussian_smooth(loc, rt, 3.0, arm, smooth_root=False)
assert torch.equal(sa_l[:, others], loc[:, others]) and torch.equal(sa_r, rt) and not torch.equal(sa_l[:, arm], loc[:, arm])
# metrics.audit: each flaw is found on a constructed motion, and a clean one raises none of them
def fake_motion(T=150):
    q = np.zeros((T, 77, 3))
    q[:, nm77.index("LeftLeg")], q[:, nm77.index("RightLeg")] = (0.1, 0.9, 0), (-0.1, 0.9, 0)
    q[:, nm77.index("Head")] = (0, 1.5, 0)
    q[:, nm77.index("LeftFoot")], q[:, nm77.index("RightFoot")] = (0.1, 0.5, 0), (-0.1, 0.5, 0)  # not planted
    for i in range(T):  # slow steady hand drift
        q[i, nm77.index("LeftHand")], q[i, nm77.index("RightHand")] = (0.3, 1.0, 0.3 + i * 0.001), (-0.3, 1.0, 0.3 + i * 0.001)
        q[i, nm77.index("Hips")] = (0, 0.9, 0)
    for i in range(T):  # every joint drifts just enough to not count as an idle tail
        q[i] += (0, 0, i * 0.002)
    return q


nm77 = metrics._joint_names(77)
clean = metrics.audit({"posed_joints": fake_motion()})
assert clean["counts"] == {}, clean["findings"]
q = fake_motion(); q[50:70, nm77.index("Head"), 2] -= np.linspace(0, 0.4, 20); q[70:90, nm77.index("Head"), 2] -= np.linspace(0.4, 0, 20)
assert metrics.audit({"posed_joints": q})["counts"].get("reversal") == 1
q = fake_motion()
for h in ("LeftHand", "RightHand"):
    q[40:46, nm77.index(h), 1] += np.linspace(0, 0.5, 6); q[46:60, nm77.index(h), 1] += 0.5; q[60:66, nm77.index(h), 1] += np.linspace(0.5, 1.0, 6); q[66:, nm77.index(h), 1] += 1.0
assert metrics.audit({"posed_joints": q})["counts"].get("freeze") == 1
assert metrics.audit({"posed_joints": fake_motion()}, np.full(150, -0.05))["counts"]["floor_sink"] == 1
q = fake_motion()[:150]; q[100:] = q[100]
assert metrics.audit({"posed_joints": q})["counts"].get("idle_tail") == 1

# metrics.match_pose finds a frame again, whatever the facing and position; metrics.score reads nested values
mp = metrics.match_pose(motion, motion, 76)
assert mp["best_frame"] == 76 and mp["error_m"] < 1e-6
turned = {"posed_joints": ph.numpy()}  # frames 30 and 76, re-faced earlier
assert metrics.match_pose(motion, turned, 1)["best_frame"] == 76
sc = metrics.score({"a": {"b": 1.0}, "audit": {"counts": {}}}, [{"metric": "a.b", "op": "<", "value": 2}, {"metric": "audit.counts.freeze", "op": "==", "value": 0}, {"metric": "a.c", "op": ">", "value": 0}])
assert sc["passed"] == 2 and sc["total"] == 3

# edit.py: two_bone_ik reaches the target, keeps limb lengths and hand orientation, and leaves the rest alone
sk77 = build_skeleton(77); nm_77 = list(sk77.bone_index)
loc77 = sk77.to_SOMASkeleton77(loc) if hasattr(sk77, "to_SOMASkeleton77") else None
loc77 = sk.to_SOMASkeleton77(loc)
g0, p0, _ = sk77.fk(loc77, rt)
T = len(loc77)
for joint in edit.IK_CHAINS:
    h = nm_77.index(joint)
    tgt = p0[:, h] + torch.tensor([0.05, 0.08, -0.06])
    w = np.zeros(T); w[60:90] = 1.0
    ik, missed = edit.two_bone_ik(loc77, rt, sk77, joint, tgt, w)
    g1, p1, _ = sk77.fk(ik, rt)
    reached = [f for f in range(60, 90) if f not in missed]
    assert len(reached) > 20, (joint, len(missed))
    assert float((p1[reached, h] - tgt[reached]).norm(dim=-1).max()) < 5e-3, joint  # at the target
    assert float((g1[reached, h] - g0[reached, h]).abs().max()) < 1e-3, joint  # orientation kept
    a_, b_, c_ = (nm_77.index(n) for n in edit.IK_CHAINS[joint])
    assert abs(float((p1[70, b_] - p1[70, a_]).norm() - (p0[70, b_] - p0[70, a_]).norm())) < 1e-4  # bones keep length
    assert float((p1[:60] - p0[:60]).abs().max()) < 1e-4 and float((p1[90:] - p0[90:]).abs().max()) < 1e-4  # outside unchanged
    other = "RightFoot" if joint != "RightFoot" else "LeftFoot"
    assert float((p1[:, nm_77.index(other)] - p0[:, nm_77.index(other)]).abs().max()) < 1e-4  # other limbs unchanged
far = p0[:, nm_77.index("RightHand")] + torch.tensor([5.0, 0.0, 0.0])
w = np.zeros(T); w[70] = 1.0
_, missed = edit.two_bone_ik(loc77, rt, sk77, "RightHand", far, w)
assert missed == [70]  # out of reach is reported

# server.safe_name keeps saved animation names inside the saved folder
from kimodo_mcp.server import safe_name  # noqa: E402

assert safe_name("kneel-death_back2") == "kneel-death_back2"
for bad in ("../evil", "a/b", "", ".hidden", "a b"):
    try:
        safe_name(bad)
        raise AssertionError(bad)
    except ValueError:
        pass

# edit.retime: speed 2 halves a range's length, ends stay put, and poses stay on the source's path
rl, rr, tau = edit.retime(loc, rt, [{"start": 40, "end": 100, "speed": 2.0}], ease_frames=0)
assert abs(len(rl) - (150 - 30)) <= 1 and torch.allclose(rr[0], rt[0]) and float((rr[-1] - rt[-1]).norm()) < 1e-3
assert abs(tau[100] - tau[40] - 30) < 1 and abs(tau[39] - 39) < 1
same_l, same_r, _ = edit.retime(loc, rt, [{"start": 40, "end": 100, "speed": 1.0}])
assert len(same_l) == 150 and float((same_r - rt).abs().max()) < 1e-4
eased = edit.retime(loc, rt, [{"start": 40, "end": 100, "speed": 3.0}], ease_frames=8)[1]
abrupt = edit.retime(loc, rt, [{"start": 40, "end": 100, "speed": 3.0}], ease_frames=0)[1]
accel = lambda r: float((r[2:] - 2 * r[1:-1] + r[:-2]).norm(dim=-1).max())
assert accel(eased) < accel(abrupt)  # easing the speed change softens it

# edit.plant_feet: a sliding foot is held still, frames outside its window and the other leg don't change
still = loc77[:1].repeat(120, 1, 1, 1)
slid = rt[:1].repeat(120, 1)
slid[:, 0] += torch.clamp(torch.linspace(-0.2, 0.3, 120), 0.0, 0.1)  # the whole body drifts 10 cm while standing
g2, p2, _ = sk77.fk(still, slid)
fi = nm_77.index("LeftFoot")
pf, missed = edit.plant_feet(still, slid, sk77, {"LeftFoot": [[60, 89]]}, 3)
_, p3, _ = sk77.fk(pf, slid)
assert not missed["LeftFoot"] and float((p3[60:90, fi] - p3[60:90, fi].mean(0)).norm(dim=-1).max()) < 5e-3
assert float((p2[60:90, fi] - p2[60:90, fi].mean(0)).norm(dim=-1).max()) > 0.03  # it did slide before
assert float((p3[:55] - p2[:55]).abs().max()) < 1e-4 and float((p3[:, nm_77.index("RightFoot")] - p2[:, nm_77.index("RightFoot")]).abs().max()) < 1e-4
assert edit.plant_windows(np.c_[np.zeros(50), np.full(50, 0.1), np.zeros(50)]) == [[0, 49]]
assert edit.plant_windows(np.c_[np.arange(50) * 1.0, np.full(50, 0.1), np.zeros(50)]) == []  # moving
assert edit.plant_windows(np.c_[np.zeros(50), np.full(50, 0.5), np.zeros(50)]) == []  # in the air

# edit.join_clips: the second clip continues from the first, whichever way it faced and wherever it started
turned_l, turned_r = edit._turned(loc77, rt, 0, 1.2, torch.zeros(3), torch.tensor([3.0, 0.0, -2.0]))
jl, jr, starts, errs = edit.join_clips([(loc77[:90], rt[:90]), (turned_l[80:], turned_r[80:])], sk77, 10)
assert starts == [0, 80] and len(jl) == 150 and errs[0] < 0.02  # the same motion, so the join should recover it
assert float((jr - rt).norm(dim=-1).max()) < 0.02
assert float((jr[1:] - jr[:-1]).norm(dim=-1).max()) < 2 * float((rt[1:] - rt[:-1]).norm(dim=-1).max())  # no jump
assert torch.equal(jl[:80], loc77[:80])
kl, kr, _, _ = edit.join_clips([(loc77[:90], rt[:90]), (turned_l[80:], turned_r[80:])], sk77, 10, keep_facing=True)
assert torch.equal(kl[-1, 0], turned_l[-1, 0])  # keep_facing doesn't turn the second clip

# edit.make_loop: the last frame leads smoothly into the first; in_place removes the travel
ml, mr, travel = edit.make_loop(loc77, rt, 30, 90, 10)
assert len(ml) == 60 and abs(travel[1]) > 0.5
_, mp0, _ = sk77.fk(ml, mr)
closed = metrics._normalized(mp0.numpy().astype(np.float64))
before = metrics._normalized(p0.numpy().astype(np.float64))
assert np.linalg.norm(closed[-1] - closed[0], axis=-1).mean() < 0.5 * np.linalg.norm(before[89] - before[30], axis=-1).mean() + 0.05
ml2, mr2, _ = edit.make_loop(loc77, rt, 30, 90, 10, in_place=True)
assert float((mr2[-1, [0, 2]] - mr2[0, [0, 2]]).norm()) < 0.1 and float(mr[-1, 2] - mr[0, 2]) > 0.5
bl = metrics.best_loop(np.tile(p0[:40].numpy(), (3, 1, 1)), 30)
assert bl["mismatch_m"] < 0.15 and bl["end"] - bl["start"] >= 30

# metrics.rifle_check: a left hand fixed relative to the right has no gap; the rifle drawn is rigid and in the hand
held = {"posed_joints": p0.numpy(), "global_rot_mats": g0.numpy()}
rc = metrics.rifle_check(held, 60)
assert rc["left_hand_gap_max_m"] > 0.05 and rc["left_hand_off_frames"]  # this motion isn't holding anything
rp = metrics.rifle_pose(held, 60)
assert float(np.abs(rp["rot"] @ rp["rot"].transpose(0, 2, 1) - np.eye(3)).max()) < 1e-3
assert np.linalg.norm(rp["grip"][60] - p0[60, nm_77.index("LeftHand")].numpy()) < 1e-5  # exact at the reference frame
rv, rf = render.rifle_mesh(held, 60)
assert rv.shape[1] == 8 * len(render.RIFLE_BOXES) and rf.shape == (12 * len(render.RIFLE_BOXES), 3)
assert np.abs(np.linalg.norm(rv[10, 0] - rv[10, 1]) - np.linalg.norm(rv[80, 0] - rv[80, 1])) < 1e-4
rifle_sheet = np.asarray(render.contact_sheet(held, [60], tile=300, cols=1, prop=(rv, rf))).astype(int)
assert (np.abs(rifle_sheet - (render.PROP * 0.7).astype(int)).sum(-1) < 90).sum() > 50  # dark rifle pixels were drawn

# blender_retarget.auto_map's naming rules, without Blender
import ast  # noqa: E402

src = (Path(__file__).parent / "kimodo_mcp/blender_retarget.py").read_text()
names_only = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(src).body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") in ("CENTRE", "LIMBS")}
ns = {"re": __import__("re"), **names_only}
for n in ast.parse(src).body:
    if isinstance(n, ast.FunctionDef) and n.name in ("clean", "auto_map"):
        exec(compile(ast.Module([n], []), "x", "exec"), ns)
mix = ns["auto_map"]([f"mixamorig:{b}" for b in ("Hips", "Spine", "Spine1", "Spine2", "Neck", "Head", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase")])
assert mix["LeftLeg"].endswith("LeftUpLeg") and mix["LeftShin"].endswith(":LeftLeg") and mix["Spine1"].endswith(":Spine") and mix["Chest"].endswith("Spine2")
ue = ns["auto_map"](["pelvis", "spine_01", "spine_02", "spine_03", "neck_01", "head", "clavicle_l", "upperarm_l", "lowerarm_l", "hand_l", "thigh_l", "calf_l", "foot_l", "ball_l"])
assert ue["Hips"] == "pelvis" and ue["LeftArm"] == "upperarm_l" and ue["LeftForeArm"] == "lowerarm_l" and ue["LeftShin"] == "calf_l" and ue["LeftToeBase"] == "ball_l" and ue["Chest"] == "spine_03"
soma = ns["auto_map"](["Hips", "LeftLeg", "LeftShin", "RightLeg", "RightShin"])
assert soma["LeftLeg"] == "LeftLeg" and soma["LeftShin"] == "LeftShin"

# demo panel: flaws are listed worst kind first, then by frame, with readable labels
from kimodo_mcp.demo_live import flaw_rows  # noqa: E402

rows = flaw_rows([
    {"type": "foot_slide", "frames": [5, 20], "detail": "a"}, {"type": "floor_sink", "frames": [40, 50], "detail": "b"},
    {"type": "floor_sink", "frames": [3, 3], "detail": "c"}, {"type": "new_kind", "frames": [0, 1], "detail": "d"}])
assert [r[0] for r in rows] == ["Floor sink, f3", "Floor sink, f40-50", "Foot slide, f5-20", "New kind, f0-1"] and rows[1][1]["detail"] == "b"

# demo panel: a motion from the view becomes a take, and is then recognised instead of being saved again
import tempfile  # noqa: E402

from kimodo_mcp import demo_live, server  # noqa: E402

with tempfile.TemporaryDirectory() as tmp:
    server.TAKES = demo_live.TAKES = Path(tmp)
    view = {"posed_joints": p0.numpy(), "global_rot_mats": g0.numpy(), "local_rot_mats": loc77.numpy()}
    imported = server.import_motion(view, [("first", 100), ("second", 50)])
    assert demo_live.take_names() == [imported] and demo_live.matching_take(p0.numpy(), [imported]) == imported
    assert demo_live.matching_take(p0.numpy() + 0.01, [imported]) is None and demo_live.matching_take(p0.numpy()[:50], [imported]) is None
    assert demo_live.matching_take(p0.numpy(), ["t999-missing"]) is None
    meta = json.loads((Path(tmp) / imported / "meta.json").read_text())
    assert meta["texts"] == ["first", "second"] and sum(meta["durations"]) * 30 == 150
    saved = np.load(Path(tmp) / imported / "motion.npz")
    assert np.abs(saved["posed_joints"] - p0.numpy()).max() < 1e-3
    second = server.import_motion(view, [])
    assert demo_live.take_names() == sorted([imported, second], reverse=True)
    assert json.loads((Path(tmp) / second / "meta.json").read_text())["texts"] == ["Motion from the demo"]
    try:
        server.replay_recipe(imported)
        raise AssertionError("an imported take has no recipe to replay")
    except server.ToolError:
        pass

print("check passed")
