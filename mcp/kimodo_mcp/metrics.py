"""Numbers Claude can check a motion against (Y-up, metres, 30 fps)."""

import numpy as np

FPS = 30
KEY_JOINTS = ["Head", "LeftHand", "RightHand", "LeftFoot", "RightFoot", "LeftToeBase", "RightToeBase"]  # Foot = ankle


def _joint_names(nbjoints: int) -> list[str]:
    from kimodo.skeleton.registry import build_skeleton

    return list(build_skeleton(nbjoints).bone_index)


def segments(motion: dict, bounds: list[tuple[str, int, int]]) -> list[dict]:
    """Per prompt segment: torso twist and how far each shoulder moved back relative to the hips.

    twist_deg: signed angle of the shoulder line vs the hip line seen from above;
    positive = right shoulder back (upper body turned right), negative = left shoulder back.
    """
    pj = np.asarray(motion["posed_joints"], dtype=np.float64)
    idx = {n: i for i, n in enumerate(_joint_names(pj.shape[1]))}
    xz = lambda name: pj[:, idx[name]][:, [0, 2]]  # noqa: E731
    sh, hp = xz("LeftArm") - xz("RightArm"), xz("LeftLeg") - xz("RightLeg")
    twist = np.degrees(np.arctan2(sh[:, 1], sh[:, 0]) - np.arctan2(hp[:, 1], hp[:, 0]))
    twist = (twist + 180) % 360 - 180
    hips = xz("Hips")
    out = []
    for text, a, b in bounds:
        b = min(b, len(pj) - 1)
        facing = np.array([-hp[a, 1], hp[a, 0]]) / (np.linalg.norm(hp[a]) + 1e-9)
        seg = {"prompt": text, "frames": [a, b], "twist_start_deg": round(float(twist[a]), 0),
               "twist_max_deg": round(float(twist[a:b + 1].max()), 0), "twist_max_frame": a + int(twist[a:b + 1].argmax()),
               "twist_min_deg": round(float(twist[a:b + 1].min()), 0), "twist_min_frame": a + int(twist[a:b + 1].argmin()),
               "twist_end_deg": round(float(twist[b]), 0)}
        for side in ("Right", "Left"):
            rel = (xz(f"{side}Arm") - hips)[a:b + 1] @ facing  # shoulder ahead of hips, along start facing
            back = rel[0] - rel  # positive = moved backward
            seg[f"{side.lower()}_shoulder_max_back_m"] = round(float(back.max()), 3)
            seg[f"{side.lower()}_shoulder_max_back_frame"] = a + int(back.argmax())
        # posture (means over the segment): where the hands are, torso lean, knee bend
        hands = (pj[a:b + 1, idx["LeftHand"]] + pj[a:b + 1, idx["RightHand"]]) / 2
        seg["hands_height_m"] = round(float(hands[:, 1].mean()), 3)
        seg["hands_forward_of_hips_m"] = round(float(((hands[:, [0, 2]] - hips[a:b + 1]) @ facing).mean()), 3)
        seg["hands_apart_m"] = round(float(np.linalg.norm(pj[a:b + 1, idx["LeftHand"]] - pj[a:b + 1, idx["RightHand"]], axis=1).mean()), 3)
        # how much the hands wander around their mean position (small = steady, e.g. holding an aim)
        seg["hands_wander_m"] = round(float(np.linalg.norm(hands - hands.mean(0), axis=1).mean()), 3)
        seg["hips_height_m"] = round(float(pj[a:b + 1, idx["Hips"], 1].mean()), 3)
        spine = pj[a:b + 1, idx["Neck1"]] - pj[a:b + 1, idx["Hips"]]
        seg["torso_lean_deg"] = round(float(np.degrees(np.arccos(spine[:, 1] / np.linalg.norm(spine, axis=1))).mean()), 0)
        knees = []
        for side in ("Left", "Right"):
            hip_, knee_, ankle_ = (pj[a:b + 1, idx[side + j]] for j in ("Leg", "Shin", "Foot"))
            u, v = hip_ - knee_, ankle_ - knee_
            knees.append(np.degrees(np.arccos(np.clip((u * v).sum(1) / np.linalg.norm(u, axis=1) / np.linalg.norm(v, axis=1), -1, 1))))
        seg["knee_angle_deg"] = round(float(np.mean(knees)), 0)  # 180 = straight legs
        seg["feet_apart_m"] = round(float(np.linalg.norm(pj[a:b + 1, idx["LeftFoot"]] - pj[a:b + 1, idx["RightFoot"]], axis=1).mean()), 3)
        seg["head_min_m"] = round(float(pj[a:b + 1, idx["Head"], 1].min()), 3)
        seg["root_travel_m"] = round(float(np.linalg.norm(hips[b] - hips[a])), 3)
        out.append(seg)
    return out


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
    head_speed = np.linalg.norm(np.diff(pj[:, idx["Head"]], axis=0), axis=1) * FPS
    out["head_peak_speed"] = {"m_per_s": round(float(head_speed.max()), 2), "frame": int(head_speed.argmax())}

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


FINGERS = ("Thumb", "Index", "Middle", "Ring", "Pinky")


def _runs(mask: np.ndarray) -> list[list[int]]:
    """[first, last] frame of each run of True."""
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return [[int(a), int(b) - 1] for a, b in zip(edges[::2], edges[1::2])]


def audit(motion: dict, lowest: np.ndarray | None = None) -> dict:
    """List things worth looking at, with frames. These are prompts to check, not verdicts: a freeze is a flaw in a
    hit reaction but correct in a held pose, and a reversal is a flaw in a fall but normal in a dance.

    lowest: lowest mesh height per frame, for the floor check (joints alone miss knees and hands).
    """
    pj = np.asarray(motion["posed_joints"], dtype=np.float64)
    T = len(pj)
    names = _joint_names(pj.shape[1])
    idx = {n: i for i, n in enumerate(names)}
    body = [i for i, n in enumerate(names) if not any(k in n for k in FINGERS)]
    findings = []

    def add(kind, frames, detail):
        findings.append({"type": kind, "frames": frames, "detail": detail})

    if lowest is not None:
        for a, b in _runs(np.asarray(lowest) < -0.02):
            add("floor_sink", [a, b], f"mesh up to {-float(np.min(lowest[a : b + 1])) * 100:.0f} cm below the floor")

    # planted feet that slide: little height change and low height, but moving sideways
    for side in ("Left", "Right"):
        foot = pj[:, idx[f"{side}Foot"]]
        step = np.linalg.norm(np.diff(foot[:, [0, 2]], axis=0), axis=1)
        planted = (foot[:-1, 1] < 0.2) & (step * FPS < 0.6)
        for a, b in _runs(planted):
            dist = float(np.linalg.norm(foot[b + 1, [0, 2]] - foot[a, [0, 2]]))
            if b - a >= 10 and dist > 0.05:
                add("foot_slide", [a, b], f"{side} foot slides {dist * 100:.0f} cm while planted")

    body_acc = np.linalg.norm(np.diff(pj[:, body], 2, axis=0), axis=-1).mean(1) * FPS**2
    for a, b in _runs(body_acc > 25):
        add("jerk", [a + 1, b + 1], f"body acceleration up to {body_acc[a : b + 1].max():.0f} m/s2")

    hands = (pj[:, idx["LeftHand"]] + pj[:, idx["RightHand"]]) / 2
    speed = np.linalg.norm(np.diff(hands, axis=0), axis=1) * FPS
    hand_acc = np.abs(np.diff(speed)) * FPS
    for a, b in _runs(hand_acc > 35):
        add("whip", [a + 1, b + 1], f"hands accelerate up to {hand_acc[a : b + 1].max():.0f} m/s2")

    # freeze: fast, nearly stopped, then fast again
    f = 1
    while f < len(speed) - 1:
        before = speed[max(f - 10, 0) : f]
        after = speed[f + 1 : f + 16]
        if speed[f] == speed[max(f - 3, 0) : f + 4].min() and len(before) and len(after):
            if before.max() >= 1.5 and after.max() >= 1.0 and speed[f] < 0.25 * min(before.max(), after.max()):
                add("freeze", [f, f], f"hands go from {before.max():.1f} to {speed[f]:.1f} m/s, then back to {after.max():.1f}")
                f += 10
        f += 1

    # reversal: the head travels one way, then back, each by at least 25 cm, relative to the hips
    across = pj[:, idx["LeftLeg"]] - pj[:, idx["RightLeg"]]
    fwd = np.array([-across[0, 2], across[0, 0]]) / (np.linalg.norm(across[0, [0, 2]]) + 1e-9)
    z = pj[:, idx["Head"]][:, [0, 2]] @ fwd
    f = 1
    while f < T - 1:
        w = z[max(f - 8, 0) : f + 9]
        before, after = z[max(f - FPS, 0) : f + 1], z[f : f + FPS + 1]
        if z[f] == w.min() and before.max() - z[f] >= 0.25 and after.max() - z[f] >= 0.25:
            add("reversal", [f, f], f"head moves {(before.max() - z[f]) * 100:.0f} cm back, then {(after.max() - z[f]) * 100:.0f} cm forward again")
            f += FPS
        elif z[f] == w.max() and z[f] - before.min() >= 0.25 and z[f] - after.min() >= 0.25:
            add("reversal", [f, f], f"head moves {(z[f] - before.min()) * 100:.0f} cm forward, then {(z[f] - after.min()) * 100:.0f} cm back again")
            f += FPS
        f += 1

    final = np.linalg.norm(pj[-1] - pj, axis=-1).mean(1)
    moving = np.flatnonzero(final >= 0.03)
    settled = int(moving[-1]) + 1 if len(moving) else 0
    if (T - settled) / FPS > 1.0:
        add("idle_tail", [settled, T - 1], f"{(T - settled) / FPS:.1f} s of almost no movement at the end")

    counts = {}
    for item in findings:
        counts[item["type"]] = counts.get(item["type"], 0) + 1
    return {
        "findings": sorted(findings, key=lambda x: x["frames"][0]),
        "counts": counts,
        "floor_sink_max_m": None if lowest is None else round(float(max(-np.min(lowest), 0)), 3),
        "hand_accel_peak_m_per_s2": round(float(hand_acc.max()), 1),
        "body_accel_peak_m_per_s2": round(float(body_acc.max()), 1),
    }


def match_pose(motion: dict, reference: dict, reference_frame: int) -> dict:
    """Find the frames of a motion closest to one pose of another motion.

    Compares body joints relative to the hips, with both poses turned to the same facing, so where the
    character stands and which way it faces don't count. Error is the mean joint distance in metres.
    """
    def norm(pj):
        names = _joint_names(pj.shape[1])
        idx = {n: i for i, n in enumerate(names)}
        keep = [i for i, n in enumerate(names) if not any(k in n for k in FINGERS)]
        across = pj[:, idx["LeftLeg"]] - pj[:, idx["RightLeg"]]
        angle = np.arctan2(across[:, 2], across[:, 0])
        rel = pj - pj[:, idx["Hips"]][:, None]
        c, s = np.cos(angle), np.sin(angle)
        x, z = rel[..., 0], rel[..., 2]
        out = np.stack([x * c[:, None] + z * s[:, None], rel[..., 1], -x * s[:, None] + z * c[:, None]], -1)
        return out[:, keep]

    a = norm(np.asarray(motion["posed_joints"], dtype=np.float64))
    b = norm(np.asarray(reference["posed_joints"], dtype=np.float64)[[reference_frame]])
    if a.shape[1] != b.shape[1]:
        raise ValueError("Takes use different skeletons")
    err = np.linalg.norm(a - b, axis=-1).mean(1)
    best = [int(i) for i in np.argsort(err)[:3]]
    return {"best_frame": best[0], "error_m": round(float(err[best[0]]), 3), "closest": [{"frame": i, "error_m": round(float(err[i]), 3)} for i in best]}


def resolve_metric(data: dict, path: str):
    """A value from nested dicts by dotted path, e.g. end_pose.head_height_m or audit.counts.freeze."""
    node = data
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def score(data: dict, criteria: list[dict]) -> dict:
    """Check criteria ({metric, op, value}) against metrics. A missing audit count means zero findings of that type."""
    ops = {"<": lambda a, b: a < b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b, ">=": lambda a, b: a >= b, "==": lambda a, b: a == b}
    results = []
    for c in criteria:
        if c["op"] not in ops:
            raise ValueError(f"op must be one of {', '.join(ops)}")
        value = resolve_metric(data, c["metric"])
        if value is None and c["metric"].startswith("audit.counts."):
            value = 0
        results.append({**c, "actual": value, "passed": value is not None and bool(ops[c["op"]](value, c["value"]))})
    return {"passed": sum(r["passed"] for r in results), "total": len(results), "results": results}
