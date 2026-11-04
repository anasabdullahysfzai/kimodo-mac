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
