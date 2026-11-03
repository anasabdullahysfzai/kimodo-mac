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

    peaks = {}
    for name in KEY_JOINTS:
        y = pj[:, idx[name], 1]
        f = int(y.argmax())
        peaks[name] = {
            "max_height_m": round(float(y[f]), 3),
            "frame": f,
            "time_s": round(f / FPS, 2),
            "minus_head_height_m": round(float(y[f] - head_y[f]), 3),
        }

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
        },
        "lowest_joint_height_m": round(float(pj[..., 1].min()), 3),
        # mean joint acceleration; higher means jerkier motion
        "jitter_m_per_s2": round(float(np.linalg.norm(np.diff(pj, 2, axis=0), axis=-1).mean() * FPS**2), 3),
    }

    gr = motion.get("global_rot_mats")
    if gr is not None:
        fwd = np.asarray(gr)[:, idx["Hips"], :, 2]  # hips local +Z in world
        heading = np.unwrap(np.arctan2(fwd[:, 0], fwd[:, 2]))
        out["heading_change_deg"] = round(float(np.degrees(heading[-1] - heading[0])), 1)
        out["max_heading_turn_deg"] = round(float(np.degrees(np.abs(heading - heading[0]).max())), 1)

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
        f"({p['RightFoot']['minus_head_height_m']:+} vs head); L foot peak {p['LeftFoot']['max_height_m']} m; "
        f"root travel {m['root']['net_travel_m']} m; turn {m.get('heading_change_deg', 'n/a')} deg"
    )
