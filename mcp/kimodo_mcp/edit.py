"""Edits on finished motions: additive rotation layers and splicing a regenerated frame range.

Pure functions (no model, no MCP). Rotations are torch [T,J,3,3], root positions torch [T,3].
"""

import math

import numpy as np
import torch

FPS = 30

GROUPS = {
    "spine": ["Spine1", "Spine2", "Chest"],
    "neck_head": ["Neck1", "Neck2", "Head"],
    "upper_body": ["Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head"],
    "left_arm": ["LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand"],
    "right_arm": ["RightShoulder", "RightArm", "RightForeArm", "RightHand"],
    "hips": ["Hips"],
    "left_leg": ["LeftLeg", "LeftShin", "LeftFoot"],
    "right_leg": ["RightLeg", "RightShin", "RightFoot"],
}
FEET = ["LeftFoot", "RightFoot", "LeftToeBase", "RightToeBase"]


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def curve_weights(num_frames: int, curve, start: int, end: int | None = None, attack_frames: int = 3, decay_frames: int = 20) -> np.ndarray:
    """Weight per frame, 0 outside the layer.

    spring: fast rise over attack_frames to 1, then a damped settle that reaches 0 at end.
    ease:   smooth rise and fall over start..end, peaking in the middle.
    hold:   smooth rise over attack_frames, stays at 1; if end is given, falls over decay_frames to 0 at end.
    A list of [frame, weight] points is interpolated linearly.
    """
    f = np.arange(num_frames, dtype=np.float64)
    if not isinstance(curve, str):
        pts = np.asarray(curve, dtype=np.float64)
        return np.interp(f, pts[:, 0], pts[:, 1], left=0.0, right=0.0)
    if curve == "spring":
        end = start + attack_frames + decay_frames if end is None else end
        decay = max(end - start - attack_frames, 1)
        u = (f - start - attack_frames) / decay
        w = np.where(u < 0, _smoothstep((f - start) / max(attack_frames, 1)), np.exp(-3 * u) * np.cos(1.5 * math.pi * u))
        w[(f < start) | (f > end)] = 0.0
        return w
    if curve == "ease":
        end = start + attack_frames + decay_frames if end is None else end
        s = (f - start) / max(end - start, 1)
        w = 0.5 * (1 - np.cos(2 * math.pi * s))
        w[(f < start) | (f > end)] = 0.0
        return w
    if curve == "hold":
        w = _smoothstep((f - start) / max(attack_frames, 1))
        if end is not None:
            w = w * (1 - _smoothstep((f - (end - decay_frames)) / max(decay_frames, 1)))
            w[f > end] = 0.0
        return w
    raise ValueError("curve must be spring, ease, hold, or a list of [frame, weight] points")


def joint_indices(joints, names: list[str]) -> list[int]:
    wanted = GROUPS.get(joints, [joints]) if isinstance(joints, str) else list(joints)
    missing = [n for n in wanted if n not in names]
    if missing:
        raise ValueError(f"Unknown joints {missing}. Use a group ({', '.join(GROUPS)}) or joint names.")
    return [names.index(n) for n in wanted]


def facing_at(posed_joints: torch.Tensor, names: list[str], frame: int) -> torch.Tensor:
    """Unit horizontal facing (x, 0, z) from the line across the hips; stays valid when lying down."""
    across = posed_joints[frame, names.index("LeftLeg")] - posed_joints[frame, names.index("RightLeg")]
    forward = torch.stack([-across[2], torch.zeros_like(across[0]), across[0]])
    return forward / (forward.norm() + 1e-9)


def apply_layers(local_rots: torch.Tensor, root_positions: torch.Tensor, skeleton, layers: list[dict]) -> torch.Tensor:
    """Apply rotation layers in order and return new local rotations.

    Each layer: joints, start, and optional end, pitch_deg (+ = back), twist_deg (+ = right shoulder back),
    side_deg (+ = toward the right), curve, attack_frames, decay_frames. The rotation is split evenly
    across the layer's joints, about axes taken from the body's facing at start.
    """
    from kimodo.geometry import axis_angle_to_matrix

    names = list(skeleton.bone_index)
    parents = skeleton.joint_parents.tolist()
    local = local_rots.clone()
    T = len(local)
    up = torch.tensor([0.0, 1.0, 0.0], dtype=local.dtype)
    for layer in layers:
        idx = joint_indices(layer["joints"], names)
        global_rots, posed, _ = skeleton.fk(local, root_positions)
        start = int(layer["start"])
        fwd = facing_at(posed, names, min(start, T - 1)).to(local.dtype)
        left = torch.linalg.cross(up, fwd)
        vec = (
            math.radians(layer.get("pitch_deg", 0.0)) * -left
            + math.radians(layer.get("twist_deg", 0.0)) * -up
            + math.radians(layer.get("side_deg", 0.0)) * fwd
        ) / len(idx)
        weights = curve_weights(T, layer.get("curve", "spring"), start, layer.get("end"), layer.get("attack_frames", 3), layer.get("decay_frames", 20))
        delta = axis_angle_to_matrix(torch.as_tensor(weights, dtype=local.dtype)[:, None] * vec)  # [T,3,3]
        for j in idx:
            if parents[j] < 0:
                local[:, j] = delta @ local[:, j]
            else:
                p = global_rots[:, parents[j]]
                local[:, j] = p.transpose(1, 2) @ delta @ p @ local[:, j]
    return local


IK_CHAINS = {
    "LeftHand": ("LeftArm", "LeftForeArm", "LeftHand"),
    "RightHand": ("RightArm", "RightForeArm", "RightHand"),
    "LeftFoot": ("LeftLeg", "LeftShin", "LeftFoot"),
    "RightFoot": ("RightLeg", "RightShin", "RightFoot"),
}


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / (v.norm(dim=-1, keepdim=True) + 1e-9)


def two_bone_ik(local_rots, root_positions, skeleton, joint: str, targets, weights):
    """Move a hand or foot toward target positions by bending its limb; the end joint keeps its world orientation.

    targets: [T,3] world positions. weights: [T] from 0 (unchanged) to 1 (at target). Returns (new local
    rotations, frames whose target was out of reach; those stop at full stretch).
    """
    from kimodo.geometry import axis_angle_to_matrix

    if joint not in IK_CHAINS:
        raise ValueError(f"joint must be one of {', '.join(IK_CHAINS)}")
    names = list(skeleton.bone_index)
    parents = skeleton.joint_parents.tolist()
    ia, ib, ic = (names.index(n) for n in IK_CHAINS[joint])
    local = local_rots.clone()
    gr, pos, _ = skeleton.fk(local, root_positions)
    w = torch.as_tensor(np.asarray(weights), dtype=local.dtype)
    frames = torch.nonzero(w > 0).flatten()
    if len(frames) == 0:
        return local, []
    a, b, c = pos[frames, ia], pos[frames, ib], pos[frames, ic]
    t = c + w[frames, None] * (torch.as_tensor(targets, dtype=local.dtype)[frames] - c)
    ga, gb, gc = gr[frames, ia], gr[frames, ib], gr[frames, ic]
    lab, lbc = (b - a).norm(dim=-1), (c - b).norm(dim=-1)
    reach = (t - a).norm(dim=-1)
    lat = torch.minimum(torch.maximum(reach, (lab - lbc).abs() + 1e-3), lab + lbc - 1e-3)
    unreachable = [int(f) for f, r, l in zip(frames, reach, lat) if r > l + 1e-3]

    cos = lambda x: torch.acos(torch.clamp(x, -1 + 1e-6, 1 - 1e-6))
    ac_ab0 = cos((_unit(c - a) * _unit(b - a)).sum(-1))
    ba_bc0 = cos((_unit(a - b) * _unit(c - b)).sum(-1))
    ac_ab1 = cos((lbc**2 - lab**2 - lat**2) / (-2 * lab * lat))
    ba_bc1 = cos((lat**2 - lab**2 - lbc**2) / (-2 * lab * lbc))
    bend = torch.linalg.cross(c - a, b - a)
    degenerate = bend.norm(dim=-1) < 1e-5  # a straight limb has no bend plane; use the joint's own axis
    bend[degenerate] = (ga[degenerate] @ torch.tensor([1.0, 0.0, 0.0], dtype=local.dtype))
    axis0 = _unit(bend)
    swing = torch.linalg.cross(c - a, t - a)
    has_swing = swing.norm(dim=-1) > 1e-7
    axis1 = torch.where(has_swing[:, None], _unit(swing), axis0)
    ac_at = torch.where(has_swing, cos((_unit(c - a) * _unit(t - a)).sum(-1)), torch.zeros_like(lab))

    to_local = lambda rot, v: (rot.transpose(1, 2) @ v[..., None])[..., 0]
    r_a1 = axis_angle_to_matrix(to_local(ga, axis0) * (ac_ab1 - ac_ab0)[:, None])
    r_b = axis_angle_to_matrix(to_local(gb, axis0) * (ba_bc1 - ba_bc0)[:, None])
    r_a2 = axis_angle_to_matrix(to_local(ga, axis1) * ac_at[:, None])
    local[frames, ia] = local[frames, ia] @ r_a1 @ r_a2
    local[frames, ib] = local[frames, ib] @ r_b
    parent_a = gr[frames, parents[ia]]
    new_b = parent_a @ local[frames, ia] @ local[frames, ib]
    local[frames, ic] = new_b.transpose(1, 2) @ gc
    return local, unreachable


def preset_layers(name: str, start: int, strength: float = 1.0) -> list[dict]:
    """Impact reactions as plain layers. front/back = where the bullet comes from; left/right = shoulder hit."""
    s = strength
    spring = {"start": start, "curve": "spring", "attack_frames": 4, "decay_frames": 30}
    presets = {
        "impact_front": [("spine", {"pitch_deg": 38}), ("neck_head", {"pitch_deg": 28}), ("left_arm", {"pitch_deg": -20}), ("right_arm", {"pitch_deg": -20})],
        # from behind: chest driven forward while the head lags back
        "impact_back": [("spine", {"pitch_deg": -18}), ("neck_head", {"pitch_deg": 45}), ("left_arm", {"pitch_deg": -15}), ("right_arm", {"pitch_deg": -15})],
        "impact_left": [("spine", {"twist_deg": -30, "side_deg": 8}), ("neck_head", {"twist_deg": -12}), ("left_arm", {"pitch_deg": 15})],
        "impact_right": [("spine", {"twist_deg": 30, "side_deg": -8}), ("neck_head", {"twist_deg": 12}), ("right_arm", {"pitch_deg": 15})],
    }
    if name not in presets:
        raise ValueError(f"Unknown preset '{name}'. Available: {', '.join(presets)}")
    return [{"joints": j, **{k: v * s for k, v in rot.items()}, **spring} for j, rot in presets[name]]


def splice_range(src_local, src_root, new_local, new_root, start: int, end: int, blend_frames: int = 6):
    """Source frames outside [start, end], new frames inside, blended over blend_frames at the inner edges.

    Frames outside the range are copied, so they match the source exactly.
    """
    from kimodo.geometry import matrix_to_quaternion, quaternion_to_matrix

    T = len(src_local)
    w = np.zeros(T)
    w[start : end + 1] = 1.0
    b = max(blend_frames, 1)
    if start > 0:  # the first new frame is already partly blended, so the source's last frame stays continuous
        w[start : start + b] = np.minimum(w[start : start + b], _smoothstep((np.arange(b) + 1) / (b + 1)))
    if end < T - 1:
        w[end - b + 1 : end + 1] = np.minimum(w[end - b + 1 : end + 1], _smoothstep((np.arange(b)[::-1] + 1) / (b + 1)))
    wt = torch.as_tensor(w, dtype=src_local.dtype)

    qa, qb = matrix_to_quaternion(src_local), matrix_to_quaternion(new_local)
    qb = torch.where((qa * qb).sum(-1, keepdim=True) < 0, -qb, qb)  # shortest path
    q = (1 - wt)[:, None, None] * qa + wt[:, None, None] * qb
    local = quaternion_to_matrix(q / q.norm(dim=-1, keepdim=True))
    root = (1 - wt)[:, None] * src_root + wt[:, None] * new_root
    outside, inside = wt == 0, wt == 1  # exact copies where nothing is blended
    local[outside], root[outside] = src_local[outside], src_root[outside]
    local[inside], root[inside] = new_local[inside], new_root[inside]
    return local, root


def gaussian_smooth(local_rots, root_positions, sigma_frames: float, joints=None, smooth_root: bool = True):
    """Smooth motion over time with a Gaussian window; returns new (local rotations, root).

    joints: indices to smooth (default all). Rotations are averaged as quaternions, on the shortest path.
    """
    from kimodo.geometry import matrix_to_quaternion, quaternion_to_matrix

    r = max(int(round(sigma_frames * 3)), 1)
    weights = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_frames) ** 2)
    weights = torch.as_tensor(weights / weights.sum(), dtype=local_rots.dtype)
    T = len(local_rots)
    q = matrix_to_quaternion(local_rots)
    out_q, out_root = q.clone(), root_positions.clone()
    for t in range(T):
        idx = torch.clamp(torch.arange(t - r, t + r + 1), 0, T - 1)  # the ends repeat the first and last frames
        window = q[idx]
        aligned = torch.where((window * q[t]).sum(-1, keepdim=True) < 0, -window, window)
        out_q[t] = (weights[:, None, None] * aligned).sum(0)
        if smooth_root:
            out_root[t] = (weights[:, None] * root_positions[idx]).sum(0)
    smoothed = quaternion_to_matrix(out_q / out_q.norm(dim=-1, keepdim=True))
    if joints is None:
        return smoothed, out_root
    result = local_rots.clone()
    result[:, joints] = smoothed[:, joints]
    return result, out_root


def ground_lift(lowest: np.ndarray, start: int = 0, end: int | None = None, smooth_frames: int = 8) -> np.ndarray:
    """Upward root offset per frame that keeps the body out of the floor, without sudden pops.

    lowest: the lowest mesh height per frame. Taking the running maximum before smoothing makes the
    smoothed lift cover every frame that needs it. Frames outside start..end get no lift of their own,
    but the lift eases in and out around them.
    """
    need = np.clip(-np.asarray(lowest, dtype=np.float64), 0.0, None)
    need[:start] = 0.0
    if end is not None:
        need[end + 1 :] = 0.0
    w = max(int(smooth_frames), 1)
    n = len(need)
    held = np.array([need[max(i - w, 0) : i + w + 1].max() for i in range(n)])
    return np.array([held[max(i - w, 0) : i + w + 1].mean() for i in range(n)])


def motion_parts(motion: dict):
    """(skeleton, local rotations, root positions) from a saved motion dict; local rotations derived if absent."""
    from kimodo.skeleton.registry import build_skeleton

    gr = torch.as_tensor(np.asarray(motion["global_rot_mats"]), dtype=torch.float32)
    skeleton = build_skeleton(gr.shape[1])
    root = torch.as_tensor(np.asarray(motion["posed_joints"])[:, skeleton.root_idx], dtype=torch.float32)
    if "local_rot_mats" in motion and np.asarray(motion["local_rot_mats"]).shape[1] == gr.shape[1]:
        local = torch.as_tensor(np.asarray(motion["local_rot_mats"]), dtype=torch.float32)
    else:
        local = skeleton.global_rots_to_local_rots(gr)
    return skeleton, local, root


def rebuild_motion(local: torch.Tensor, root: torch.Tensor, skeleton) -> dict:
    """All saved motion arrays from local rotations + root, as numpy."""
    from kimodo.exports.motion_io import complete_motion_dict, motion_dict_to_numpy

    return motion_dict_to_numpy(complete_motion_dict(local, root, skeleton, FPS))


def foot_drift(before: np.ndarray, after: np.ndarray, names: list[str]) -> float:
    """Largest movement of any foot joint between two [T,J,3] position arrays, in metres."""
    idx = [names.index(n) for n in FEET if n in names]
    return float(np.linalg.norm(after[:, idx] - before[:, idx], axis=-1).max())


def _nlerp(qa: torch.Tensor, qb: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Blend quaternions on the shortest path. w broadcasts against qa."""
    qb = torch.where((qa * qb).sum(-1, keepdim=True) < 0, -qb, qb)
    q = (1 - w) * qa + w * qb
    return q / q.norm(dim=-1, keepdim=True)


def retime(local_rots, root_positions, ranges: list[dict], ease_frames: int = 4):
    """Play frame ranges faster or slower; returns (local, root, new time of each source frame).

    ranges: [{start, end, speed}], speed 2 = twice as fast. Speed changes are eased over about ease_frames, because a
    sudden change is itself a jerk.
    """
    from kimodo.geometry import matrix_to_quaternion, quaternion_to_matrix

    T = len(local_rots)
    log_speed = np.zeros(T)
    for r in ranges:
        log_speed[int(r["start"]) : int(r["end"]) + 1] = math.log(r["speed"])
    if ease_frames > 0:
        sigma = ease_frames / 2
        n = int(math.ceil(sigma * 3))
        kernel = np.exp(-0.5 * (np.arange(-n, n + 1) / sigma) ** 2)
        log_speed = np.convolve(np.pad(log_speed, n, mode="edge"), kernel / kernel.sum(), mode="valid")
    tau = np.r_[0.0, np.cumsum(np.exp(-log_speed[:-1]))]
    new_len = max(int(round(tau[-1])) + 1, 2)
    tau *= (new_len - 1) / tau[-1]  # first and last frames stay exactly where they are
    src = np.interp(np.arange(new_len), tau, np.arange(T))
    i0 = np.minimum(np.floor(src).astype(int), T - 2)
    w = torch.as_tensor(src - i0, dtype=local_rots.dtype)
    q = matrix_to_quaternion(local_rots)
    local = quaternion_to_matrix(_nlerp(q[i0], q[i0 + 1], w[:, None, None]))
    root = (1 - w)[:, None] * root_positions[i0] + w[:, None] * root_positions[i0 + 1]
    return local, root, tau


def plant_windows(foot: np.ndarray, min_frames: int = 6, max_height: float = 0.2, max_speed: float = 0.6, gap: int = 2) -> list[list[int]]:
    """[first, last] frames where a foot ([T,3] ankle positions) is low and slow; the same test audit uses for a slide."""
    speed = np.r_[np.linalg.norm(np.diff(foot[:, [0, 2]], axis=0), axis=1) * FPS, 0.0]
    low = (foot[:, 1] < max_height) & (speed < max_speed)
    edges = np.flatnonzero(np.diff(np.r_[0, low.astype(int), 0]))
    runs = [[int(a), int(b) - 1] for a, b in zip(edges[::2], edges[1::2])]
    merged = []
    for run in runs:
        if merged and run[0] - merged[-1][1] - 1 <= gap:
            merged[-1][1] = run[1]
        else:
            merged.append(run)
    return [r for r in merged if r[1] - r[0] + 1 >= min_frames]


def plant_feet(local_rots, root_positions, skeleton, windows: dict[str, list[list[int]]], blend_frames: int = 4):
    """Hold each foot at its average position through its windows. Returns (new local rotations, unreachable frames).

    windows is updated in place to the frames actually held.
    """
    names = list(skeleton.bone_index)
    local = local_rots.clone()
    T = len(local)
    unreachable = {}
    for joint, runs in windows.items():
        _, pos, _ = skeleton.fk(local, root_positions)
        foot = pos[:, names.index(joint)]
        hip_joint = pos[:, names.index(IK_CHAINS[joint][0])]
        leg = float((pos[0, names.index(IK_CHAINS[joint][1])] - hip_joint[0]).norm() + (foot[0] - pos[0, names.index(IK_CHAINS[joint][1])]).norm())
        targets, weights = foot.clone(), np.zeros(T)
        kept = []
        for a, b in runs:
            # Let go of the foot on frames the leg can't reach, instead of stretching the knee straight.
            for _ in range(3):
                reach = (hip_joint[a : b + 1] - foot[a : b + 1].mean(0)).norm(dim=-1).numpy()
                already = (hip_joint[a : b + 1] - foot[a : b + 1]).norm(dim=-1).numpy()  # a standing leg is nearly straight anyway
                ok = reach < np.maximum(0.98 * leg, already + 0.01)
                edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
                if len(edges) == 0:
                    a = None
                    break
                longest = int(np.argmax(edges[1::2] - edges[::2]))
                a, b = a + int(edges[2 * longest]), a + int(edges[2 * longest + 1]) - 1
            if a is None or b - a < 2:
                continue
            kept.append([a, b])
            w = np.zeros(T)
            w[a : b + 1] = 1.0
            for i in range(1, blend_frames + 1):
                fade = float(_smoothstep(np.array(1 - i / (blend_frames + 1))))
                if a - i >= 0:
                    w[a - i] = fade
                if b + i < T:
                    w[b + i] = fade
            use = w > weights
            targets[use] = foot[a : b + 1].mean(0)
            weights[use] = w[use]
        local, missed = two_bone_ik(local, root_positions, skeleton, joint, targets, weights)
        unreachable[joint] = missed
        windows[joint] = kept
    return local, unreachable


def _turned(local, root, root_idx: int, yaw: float, pivot_from, pivot_to):
    """Rotate a clip about the vertical axis through pivot_from, then move that point to pivot_to."""
    c, s = math.cos(yaw), math.sin(yaw)
    ry = torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=local.dtype)
    out = local.clone()
    out[:, root_idx] = ry @ local[:, root_idx]
    return out, (root - pivot_from) @ ry.T + pivot_to


def join_clips(clips: list[tuple], skeleton, blend_frames: int = 10):
    """Join clips [(local, root), ...] end to start, each turned and moved to continue from the last, cross-faded.

    Returns (local, root, first frame of each clip in the result, pose mismatch in metres at the middle of each fade).
    """
    from kimodo.geometry import matrix_to_quaternion, quaternion_to_matrix

    from .metrics import pose_distance

    names = list(skeleton.bone_index)
    local, root = clips[0]
    starts, seams = [0], []
    b = blend_frames
    w = torch.as_tensor(_smoothstep((np.arange(b) + 1) / (b + 1)), dtype=local.dtype)
    for next_local, next_root in clips[1:]:
        if b >= min(len(local), len(next_local)):
            raise ValueError(f"blend_frames={b} is longer than a clip")
        start = len(local) - b
        at_prev, at_next = start + b // 2, b // 2
        _, pa, _ = skeleton.fk(local[[at_prev]], root[[at_prev]])
        _, pb, _ = skeleton.fk(next_local[[at_next]], next_root[[at_next]])
        fa, fb = facing_at(pa, names, 0), facing_at(pb, names, 0)
        yaw = float(torch.atan2(fa[0], fa[2]) - torch.atan2(fb[0], fb[2]))
        ground = torch.tensor([1.0, 0.0, 1.0], dtype=root.dtype)
        moved_local, moved_root = _turned(next_local, next_root, skeleton.root_idx, yaw, next_root[at_next] * ground, root[at_prev] * ground)
        qa, qb = matrix_to_quaternion(local[start:]), matrix_to_quaternion(moved_local[:b])
        mixed = quaternion_to_matrix(_nlerp(qa, qb, w[:, None, None]))
        mixed_root = (1 - w)[:, None] * root[start:] + w[:, None] * moved_root[:b]
        seams.append(round(pose_distance(pa[0].numpy(), pb[0].numpy()), 3))
        local = torch.cat([local[:start], mixed, moved_local[b:]])
        root = torch.cat([root[:start], mixed_root, moved_root[b:]])
        starts.append(start)
    return local, root, starts, seams


def make_loop(local_rots, root_positions, start: int, end: int, blend_frames: int = 10, in_place: bool = False):
    """Turn frames start..end into a cycle: end is eased onto start's pose, then dropped, so the last frame leads into the first.

    Returns (local, root, root travel per loop as [x, z]). in_place removes that travel evenly, keeping the sway.
    """
    from kimodo.geometry import matrix_to_quaternion, quaternion_to_matrix

    local, root = local_rots[start : end + 1].clone(), root_positions[start : end + 1].clone()
    n = len(local)
    b = min(blend_frames, n - 1)
    w = np.zeros(n)
    w[n - b - 1 :] = _smoothstep(np.arange(b + 1) / b)
    wt = torch.as_tensor(w, dtype=local.dtype)
    corrected = local @ (local[-1].transpose(-1, -2) @ local[0])[None]
    local = quaternion_to_matrix(_nlerp(matrix_to_quaternion(local), matrix_to_quaternion(corrected), wt[:, None, None]))
    root[:, 1] += wt * (root[0, 1] - root[-1, 1])
    travel = root[-1, [0, 2]] - root[0, [0, 2]]
    root[:, [0, 2]] -= root[0, [0, 2]].clone()
    if in_place:
        root[:, [0, 2]] -= torch.linspace(0, 1, n, dtype=root.dtype)[:, None] * travel
    return local[:-1], root[:-1], [round(float(v), 3) for v in travel]
