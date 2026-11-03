"""Numbers Claude can check a motion against (Y-up, metres, 30 fps)."""

import numpy as np

FPS = 30
KEY_JOINTS = ["Head", "LeftHand", "RightHand", "LeftFoot", "RightFoot", "LeftToeBase", "RightToeBase"]  # Foot = ankle


def _joint_names(nbjoints: int) -> list[str]:
    from kimodo.skeleton.registry import build_skeleton

    return list(build_skeleton(nbjoints).bone_index)


def analyze(motion: dict) -> dict:
    pj = np.asarray(motion["posed_joints"], dtype=np.float64)  # [T,J,3]
    T = len(pj)
    idx = {n: i for i, n in enumerate(_joint_names(pj.shape[1]))}
    head_y = pj[:, idx["Head"], 1]

    # torso lean from vertical (0 = upright, 90 = horizontal), hips -> neck
    spine = pj[:, idx["Neck1"]] - pj[:, idx["Hips"]]
    lean = np.degrees(np.arccos(np.clip(spine[:, 1] / np.linalg.norm(spine, axis=1), -1, 1)))

    def knee_angle(side):  # 180 = straight leg; SOMA names: Leg = hip, Shin = knee, Foot = ankle
        hip, knee, ankle = (pj[:, idx[side + j]] for j in ("Leg", "Shin", "Foot"))
        a, b = hip - knee, ankle - knee
        cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
        return np.degrees(np.arccos(np.clip(cos, -1, 1)))

    knee = {"Left": knee_angle("Left"), "Right": knee_angle("Right")}
    # Facing from the line across the hips (left hip -> right hip), which stays horizontal even when
    # the body lies down; the hips' own forward axis points at the floor when prone and gives noise.
    across = (pj[:, idx["LeftLeg"]] - pj[:, idx["RightLeg"]])[:, [0, 2]]
    facing = np.stack([-across[:, 1], across[:, 0]], axis=1)  # rotate 90 deg: starts as +z for a +z-facing body
    facing /= np.linalg.norm(facing, axis=1, keepdims=True) + 1e-9
    turn = np.degrees(np.unwrap(np.arctan2(facing[:, 0], facing[:, 1])))
    turn -= turn[0]

    peaks = {}
    for name in KEY_JOINTS:
        y = pj[:, idx[name], 1]
        f = int(y.argmax())
        peaks[name] = {
            "max_height_m": round(float(y[f]), 3),
            "frame": f,
            "time_s": round(f / FPS, 2),
            "minus_head_height_m": round(float(y[f] - head_y[f]), 3),
            "torso_lean_deg": round(float(lean[f]), 0),
            "heading_deg": round(float(turn[f]), 0),
        }
        side = "Left" if name.startswith("Left") else "Right" if name.startswith("Right") else None
        if side and ("Foot" in name or "Toe" in name):
            peaks[name]["knee_angle_deg"] = round(float(knee[side][f]), 0)
            # horizontal angle between torso lean and hips->foot: ~180 = leaning away from the kick (counterbalance)
            to_foot = (pj[f, idx[name]] - pj[f, idx["Hips"]])[[0, 2]]
            lean_dir = spine[f, [0, 2]]
            cos = to_foot @ lean_dir / (np.linalg.norm(to_foot) * np.linalg.norm(lean_dir) + 1e-9)
            peaks[name]["lean_vs_foot_deg"] = round(float(np.degrees(np.arccos(np.clip(cos, -1, 1)))), 0)

    hips = pj[:, idx["Hips"]]
    step = np.linalg.norm(np.diff(hips[:, [0, 2]], axis=0), axis=1)
    out = {
        "frames": T,
        "duration_s": round(T / FPS, 2),
        "peaks": peaks,
        "root": {
            "start_xz": [round(float(c), 3) for c in hips[0, [0, 2]]],
            "end_xz": [round(float(c), 3) for c in hips[-1, [0, 2]]],
            "net_travel_m": round(float(np.linalg.norm(hips[-1, [0, 2]] - hips[0, [0, 2]])), 3),
            "path_length_m": round(float(step.sum()), 3),
            "min_height_m": round(float(hips[:, 1].min()), 3),
            "speed_first_1s_m_per_s": round(float(step[: FPS - 1].sum() * FPS / (FPS - 1)), 2),
        },
        "lowest_joint_height_m": round(float(pj[..., 1].min()), 3),
    }
    # mean joint acceleration per frame; higher means jerkier. Spikes often sit on locked keyframes.
    acc = np.linalg.norm(np.diff(pj, 2, axis=0), axis=-1).mean(1) * FPS**2
    out["jitter_m_per_s2"] = round(float(acc.mean()), 3)
    out["jitter_spikes"] = [{"frame": int(i) + 1, "m_per_s2": round(float(acc[i]), 1)} for i in np.argsort(-acc)[:3]]

    out["max_torso_lean_deg"] = round(float(lean.max()), 0)

    # final pose (useful for falls/deaths): where the body ends up and whether it has come to rest
    start_facing = facing[0]
    tail = pj[-FPS // 2 :]
    end = {
        "head_height_m": round(float(head_y[-1]), 3),
        "hips_height_m": round(float(pj[-1, idx["Hips"], 1]), 3),
        "torso_lean_deg": round(float(lean[-1]), 0),
        # net joint movement over the last 0.5 s (summing per-frame speed would mostly measure model noise)
        "last_0_5s_net_movement_m": round(float(np.linalg.norm(tail[-1] - tail[0], axis=-1).mean()), 3),
    }
    if lean[-1] > 30:
        # 0 = torso points the way the character first faced (fell forward), 180 = fell backward
        d = spine[-1, [0, 2]] / (np.linalg.norm(spine[-1, [0, 2]]) + 1e-9)
        end["lean_direction_vs_start_facing_deg"] = round(float(np.degrees(np.arccos(np.clip(d @ start_facing, -1, 1)))), 0)
    disp = pj[-1, idx["Hips"], [0, 2]] - pj[0, idx["Hips"], [0, 2]]
    end["travel_along_start_facing_m"] = round(float(disp @ start_facing), 3)
    # how fast the body went down: from the head last being above 1.2 m to first being below 0.4 m
    low = np.flatnonzero(head_y < 0.4)
    if len(low) and (head_y[: low[0]] > 1.2).any():
        last_up = int(np.flatnonzero(head_y[: low[0]] > 1.2)[-1])
        end["fall_frames"] = [last_up, int(low[0])]
        end["fall_time_s"] = round((low[0] - last_up) / FPS, 2)
    out["end_pose"] = end
    out["heading_change_deg"] = round(float(turn[-1]), 1)
    out["max_heading_turn_deg"] = round(float(np.abs(turn).max()), 1)
    # when the turning happens: frames where it passes 10% and 90% of the largest turn
    big = np.abs(turn).max()
    if big > 45:
        out["turn_window_frames"] = [int(np.argmax(np.abs(turn) >= 0.1 * big)), int(np.argmax(np.abs(turn) >= 0.9 * big))]
    out["heading_every_0_5s_deg"] = {int(f): round(float(turn[f]), 0) for f in range(0, T, FPS // 2)}

    fc = motion.get("foot_contacts")
    if fc is not None:
        fc = np.asarray(fc, dtype=bool)
        out["foot_contact_ratio"] = round(float(fc.any(axis=1).mean()), 3)
        out["airborne_frames"] = int((~fc.any(axis=1)).sum())
    return out


def one_line(m: dict) -> str:
    p = m["peaks"]
    return (
        f"{m['duration_s']}s; R foot peak {p['RightFoot']['max_height_m']} m @f{p['RightFoot']['frame']} "
        f"({p['RightFoot']['minus_head_height_m']:+} vs head), R toe {p['RightToeBase']['minus_head_height_m']:+} vs head; "
        f"L foot peak {p['LeftFoot']['max_height_m']} m; root travel {m['root']['net_travel_m']} m; "
        f"turn {m.get('heading_change_deg', 'n/a')} deg"
        + (f" (frames {m['turn_window_frames'][0]}-{m['turn_window_frames'][1]})" if "turn_window_frames" in m else "")
    )
