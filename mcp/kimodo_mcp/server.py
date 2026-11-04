"""Kimodo MCP server: generate motions, render them headlessly, measure them, refine.

Run by an MCP host (e.g. Claude Code) over stdio. Needs the Kimodo venv and a running
text encoder service (`kimodo_textencoder`, default http://127.0.0.1:9550/).

Env vars:
  KIMODO_MCP_TAKES   where takes are saved (default ~/kimodo-local/motions)
  KIMODO_MODEL       model name (default kimodo-soma-rp)
  KIMODO_DEVICE      torch device for the motion model (default cpu; mps is untested)
"""

import contextlib
import functools
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Annotated

import numpy as np
import torch
from mcp.server import MCPServer
from mcp.server.mcpserver import Image
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, Field

from . import edit, metrics, render
from .demo_live import SAVED, saved_names

FPS = 30
TAKES = Path(os.environ.get("KIMODO_MCP_TAKES", Path.home() / "kimodo-local" / "motions")).expanduser()
MODEL_NAME = os.environ.get("KIMODO_MODEL", "kimodo-soma-rp")
DEVICE = os.environ.get("KIMODO_DEVICE", "cpu")
os.environ.setdefault("TEXT_ENCODER_MODE", "api")  # reuse the running encoder; never load a 2nd Llama

mcp = MCPServer("kimodo")


def tool(fn):
    """Register an MCP tool whose failures reach the caller with their reason.

    The SDK reports unexpected exceptions only as "Error executing tool X"; re-raising them as
    ToolError keeps the type and message, so the caller can act on them.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ToolError:
            raise
        except Exception as e:
            raise ToolError(f"{type(e).__name__}: {e}") from e

    return mcp.tool()(wrapper)
_model = None


class Segment(BaseModel):
    prompt: str = Field(description='What the body does, e.g. "A person in a fighting stance kicks their right leg extremely high."')
    seconds: float = Field(ge=1.0, le=10.0, description="Length of this segment in seconds.")


class Keyframes(BaseModel):
    take_id: str = Field(description="Existing take to copy full-body poses from.")
    frames: list[int] = Field(description="Frame numbers (30 fps) in that take whose full-body pose (incl. root position) is copied.")
    at_frames: list[int] | None = Field(
        default=None,
        description="Where to place those poses in the new take (same length as frames). Default: the same frame numbers.",
    )
    offset_xz: tuple[float, float] = Field(
        default=(0.0, 0.0),
        description="Metres added to the copied root x and z, to place a borrowed pose where the character is (see analyze_motion root positions).",
    )
    mirror: bool = Field(
        default=False,
        description="Swap left and right (mirror across world x = 0, the start facing), e.g. turn a left-side hit into a right-side one. Applied before offset_xz.",
    )
    face_heading_deg: float | None = Field(
        default=None,
        description="Turn each copied pose about the vertical axis so its hips face this heading (0 = +z, the start facing; "
        "90 = +x). Keeps the pose's shape, incl. torso twist relative to the hips; removes a whole-body swivel.",
    )
    turn_deg: float = Field(
        default=0.0,
        description="Rotate each copied pose by this many degrees about the vertical axis through the root "
        "(positive = toward +x, the character's left at the start). Works for any pose, incl. lying down, "
        "e.g. turn a forward fall's lying pose 90 degrees to make a sideways fall. Applied after mirror.",
    )
    upper_body_from: tuple[str, int] | None = Field(
        default=None,
        description="[take_id, frame]: use the spine, head and arms of that pose on top of these hips and legs, "
        "e.g. a standing impact arch on a kneeling lower body.",
    )
    hips_heading_deg: float | None = Field(
        default=None,
        description="Turn only the hips and legs (about the root) so the hips face this heading; the upper body keeps "
        "its world orientation. Converts a whole-body swivel into a torso twist (e.g. a hit that should leave the feet planted). "
        "Applied after face_heading_deg.",
    )
    only: list[str] | None = Field(
        default=None,
        description="Lock only these parts instead of the full body: any of LeftHand, RightHand, LeftFoot, RightFoot, Hips. "
        "The hips' position and heading are always held too, so the root can't wander; the rest of the body is free.",
    )


def face_heading(global_rot_mats, posed_joints, root_idx: int, heading_deg: float, joints=None, attach=None, by_deg=None):
    """Rotate poses about the vertical axis through the root so the hips face heading_deg (torch).

    joints: indices to rotate (default all); the rest keep their world orientation. attach: the joint where
    the unrotated part hangs off the rotated part (e.g. Spine1); the unrotated part is shifted with it so the
    skeleton stays connected.
    """
    import torch

    fwd = global_rot_mats[:, root_idx, :, 2]  # hips local +z in world
    delta = torch.deg2rad(torch.tensor(heading_deg, dtype=fwd.dtype)) - torch.atan2(fwd[:, 0], fwd[:, 2])
    if by_deg is not None:  # rotate by a fixed angle instead of aiming the hips at heading_deg
        delta = torch.full_like(delta, float(torch.deg2rad(torch.tensor(by_deg))))
    c, s, zero, one = torch.cos(delta), torch.sin(delta), torch.zeros_like(delta), torch.ones_like(delta)
    ry = torch.stack([torch.stack([c, zero, s], -1), torch.stack([zero, one, zero], -1), torch.stack([-s, zero, c], -1)], -2)  # [N,3,3]
    root = posed_joints[:, root_idx : root_idx + 1]
    sel = slice(None) if joints is None else joints
    rots, pos = global_rot_mats.clone(), posed_joints.clone()
    rots[:, sel] = ry[:, None] @ global_rot_mats[:, sel]
    pos[:, sel] = (posed_joints[:, sel] - root) @ ry.transpose(1, 2) + root
    if joints is not None and attach is not None:
        rest = [i for i in range(posed_joints.shape[1]) if i not in set(joints)]
        offset = posed_joints[:, attach : attach + 1] - root
        pos[:, rest] += offset @ ry.transpose(1, 2) - offset
    return rots, pos


def mix_upper_body(local_rots, upper_local_rots, names: list[str]):
    """Local rotations [N,J,3,3] with everything above the hips replaced by upper_local_rots (leg joints kept)."""
    keep = set(lower_body(names))
    upper = [i for i in range(len(names)) if i not in keep]
    mixed = local_rots.clone()
    mixed[:, upper] = upper_local_rots[:, upper]
    return mixed


def lower_body(names: list[str]) -> list[int]:
    """Hips and leg joints (SOMA names): the part hips_heading_deg turns."""
    return [i for i, n in enumerate(names) if n == "Hips" or any(k in n for k in ("Leg", "Shin", "Foot", "Toe"))]


def mirror_pose(global_rot_mats, posed_joints, names: list[str]):
    """Mirror poses left<->right across the x = 0 plane: [N,J,3,3] rotations and [N,J,3] positions (torch)."""
    import torch

    perm = [names.index(n.replace("Left", "#").replace("Right", "Left").replace("#", "Right")) for n in names]
    m = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=global_rot_mats.dtype))
    return m @ global_rot_mats[:, perm] @ m, posed_joints[:, perm] * torch.tensor([-1.0, 1.0, 1.0], dtype=posed_joints.dtype)


class Layer(BaseModel):
    joints: str | list[str] = Field(description=f"A group ({', '.join(edit.GROUPS)}) or joint names.")
    start: int = Field(description="First frame (30 fps).")
    end: int | None = Field(default=None, description="Last frame. Default: start + attack_frames + decay_frames; for hold, the end of the clip.")
    pitch_deg: float = Field(default=0.0, description="+ = back, - = forward. Split across the joints.")
    twist_deg: float = Field(default=0.0, description="+ = right shoulder back (turns right), - = left shoulder back.")
    side_deg: float = Field(default=0.0, description="+ = lean toward the right, - = toward the left.")
    curve: str | list[list[float]] = Field(
        default="spring",
        description="spring (fast rise, damped settle), ease (smooth in and out), hold (ease in and stay), or [[frame, weight], ...].",
    )
    attack_frames: int = Field(default=3, ge=0, description="Frames to reach full strength.")
    decay_frames: int = Field(default=20, ge=0, description="Frames to settle (spring) or fade out before end (hold).")


class Criterion(BaseModel):
    metric: str = Field(description="Dotted path into analyze_motion or audit_motion results, e.g. end_pose.head_height_m, peaks.RightFoot.max_height_m, audit.counts.freeze.")
    op: str = Field(description="One of <, <=, >, >=, ==.")
    value: float


class PathPoint(BaseModel):
    frame: int
    x: float = Field(description="World x in metres (character starts at the origin facing +z).")
    z: float = Field(description="World z in metres.")


def _get_model():
    global _model
    if _model is None:
        with contextlib.redirect_stdout(sys.stderr):
            from kimodo import load_model
            from kimodo.demo.embedding_cache import CachedTextEncoder

            try:
                model, name = load_model(MODEL_NAME, device=DEVICE, return_resolved_name=True)
            except Exception as e:
                raise RuntimeError(
                    f"Could not load {MODEL_NAME}: {e}. Is the text encoder running? Start it with: "
                    'TEXT_ENCODER_DEVICE=mps TEXT_ENCODER_TMP_FOLDER="$TMPDIR/text_encoder" kimodo_textencoder'
                ) from e
            model.text_encoder = CachedTextEncoder(model.text_encoder, model_name=name)
            _model = model
    return _model


def _take_dir(take_id: str) -> Path:
    d = TAKES / take_id
    if not (d / "motion.npz").exists():
        raise ValueError(f"No take '{take_id}'. Use list_takes to see available takes.")
    return d


def _load(take_id: str) -> dict:
    return dict(np.load(_take_dir(take_id) / "motion.npz"))


def _live(take_id: str, title: str, body: str = "", frames: list[int] | None = None, play: bool = False) -> None:
    """Tell a running Kimodo demo (patched with kimodo/demo/live.py) what is happening, so it can follow along."""
    _event({"take_id": take_id, "path": str(TAKES / take_id), "title": title, "body": body,
            "frames": frames, "play": play, "time": time.time()})


def _event(event: dict) -> None:
    TAKES.mkdir(parents=True, exist_ok=True)
    tmp = TAKES / "_live.json.tmp"
    tmp.write_text(json.dumps({"time": time.time(), **event}))
    tmp.replace(TAKES / "_live.json")  # atomic, so the demo never reads half a file


def _new_id(prompt: str) -> str:
    TAKES.mkdir(parents=True, exist_ok=True)
    n = 1 + max((int(p.name[1:4]) for p in TAKES.glob("t[0-9][0-9][0-9]-*")), default=0)
    slug = "-".join(re.findall(r"[a-z0-9]+", prompt.lower())[:5])
    return f"t{n:03d}-{slug}"


END_EFFECTORS = ("LeftHand", "RightHand", "LeftFoot", "RightFoot", "Hips")


def _keyframe_constraint(kf: Keyframes) -> dict:
    import torch
    from kimodo.geometry import matrix_to_axis_angle
    from kimodo.skeleton.registry import build_skeleton

    at = kf.at_frames or kf.frames
    if len(at) != len(kf.frames):
        raise ValueError("keyframes_from.at_frames must have the same length as frames")
    src = _load(kf.take_id)
    skel = build_skeleton(src["global_rot_mats"].shape[1])
    rots, pos = torch.from_numpy(src["global_rot_mats"][kf.frames]), torch.from_numpy(src["posed_joints"][kf.frames])
    if kf.mirror:
        rots, pos = mirror_pose(rots, pos, list(skel.bone_index))
    if kf.turn_deg:
        rots, pos = face_heading(rots, pos, skel.root_idx, 0.0, by_deg=kf.turn_deg)
    if kf.face_heading_deg is not None:
        rots, pos = face_heading(rots, pos, skel.root_idx, kf.face_heading_deg)
    if kf.hips_heading_deg is not None:
        names = list(skel.bone_index)
        rots, pos = face_heading(rots, pos, skel.root_idx, kf.hips_heading_deg, lower_body(names), attach=names.index("Spine1"))
    local = skel.global_rots_to_local_rots(rots)
    if kf.upper_body_from:
        upper = _load(kf.upper_body_from[0])["global_rot_mats"][[kf.upper_body_from[1]]]
        local = mix_upper_body(local, skel.global_rots_to_local_rots(torch.from_numpy(upper)), list(skel.bone_index))
    if kf.only:
        bad = [j for j in kf.only if j not in END_EFFECTORS]
        if bad:
            raise ValueError(f"only can't lock {bad}: use any of {', '.join(END_EFFECTORS)}")
    return {
        **({"type": "end-effector", "joint_names": kf.only} if kf.only else {"type": "fullbody"}),
        "frame_indices": at,
        "local_joints_rot": matrix_to_axis_angle(local).tolist(),
        "root_positions": (pos[:, skel.root_idx].numpy() + [kf.offset_xz[0], 0.0, kf.offset_xz[1]]).tolist(),
    }


def _status(text: str, done: bool = False) -> None:
    """Show generation progress in a running demo, without loading anything."""
    _event({"status": text, "done": done})


def _progress(label: str, total_segments: int):
    """A tqdm stand-in for Kimodo's denoising loop that reports each 10% to the demo."""
    segment = [0]

    def bar(steps):
        steps = list(steps)
        segment[0] += 1
        for i, step in enumerate(steps):
            if i % max(len(steps) // 10, 1) == 0:
                part = f" (prompt {segment[0]} of {total_segments})" if total_segments > 1 else ""
                _status(f"{label}{part}: {100 * i // len(steps)}%")
            yield step

    return bar


def _run_model(texts, num_frames, dicts, seed, diffusion_steps, text_weight, constraint_weight, num_samples, transition_frames=5):
    """Run Kimodo; returns (output arrays with a batch dimension, seconds taken)."""
    from kimodo.constraints import load_constraints_lst
    from kimodo.tools import seed_everything

    model = _get_model()
    t0 = time.time()
    with contextlib.redirect_stdout(sys.stderr):
        constraint_lst = load_constraints_lst(dicts, model.skeleton) if dicts else []
        seed_everything(seed)
        out = model(
            texts, num_frames, num_denoising_steps=diffusion_steps, multi_prompt=True,
            constraint_lst=constraint_lst, cfg_type="separated", cfg_weight=[text_weight, constraint_weight],
            num_samples=num_samples, num_transition_frames=transition_frames,
            post_processing=False,  # needs the x86-only motion_correction extension
            return_numpy=True,
            progress_bar=_progress(f"Generating {num_samples} sample(s)", len(texts)),
        )
    seconds = round(time.time() - t0, 1)
    if out["posed_joints"].ndim == 3:  # unbatched single sample
        out = {k: v[None] for k, v in out.items()}
    return out, seconds


def _save_edit(parent_id: str, motion: dict, label: str, edit: dict) -> dict:
    """Save an edited motion as a new take, keeping the parent's prompts so the demo can load it."""
    take_id = _new_id(f"{label} {parent_id[:4]}")
    d = TAKES / take_id
    d.mkdir()
    np.savez(d / "motion.npz", **motion)
    m = metrics.analyze(motion)
    parent_meta = json.loads((_take_dir(parent_id) / "meta.json").read_text())
    meta = {k: v for k, v in parent_meta.items() if k != "mcp"}
    meta["mcp"] = {"parent": parent_id, **edit, "summary": metrics.one_line(m)}
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    return {"take_id": take_id, "parent": parent_id, "summary": metrics.one_line(m), "path": str(d)}


def import_motion(motion: dict, prompts: list[tuple[str, int]]) -> str:
    """Save a motion from the demo's view as a new take, so it can be edited. prompts: (text, frames)."""
    skeleton, local, root = edit.motion_parts(motion)
    new = edit.rebuild_motion(local, root, skeleton)
    take_id = _new_id(prompts[0][0] if prompts else "demo motion")
    d = TAKES / take_id
    d.mkdir()
    np.savez(d / "motion.npz", **new)
    meta = {"num_samples": 1, "mcp": {"imported_from_demo": True, "summary": metrics.one_line(metrics.analyze(new))}}
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    _write_prompts(take_id, prompts or [("Motion from the demo", len(local))])
    return take_id


def _prompt_spans(meta: dict) -> list[tuple[str, int, int]]:
    """(prompt, first frame, frame after last) for each of a take's prompts."""
    texts, secs = meta.get("texts") or [meta.get("text")], meta.get("durations") or [meta.get("duration")]
    edges = np.cumsum([0] + [round(x * FPS) for x in secs])
    return [(t, int(edges[i]), int(edges[i + 1])) for i, t in enumerate(texts)]


def _write_prompts(take_id: str, spans: list[tuple[str, int]]) -> None:
    """Rewrite a take's prompts as (prompt, frames) so they add up to its length, for the demo and analyze_motion."""
    path = TAKES / take_id / "meta.json"
    meta = json.loads(path.read_text())
    kept = [(t, int(n)) for t, n in spans if n > 0]
    meta["texts"], meta["durations"] = [t for t, _ in kept], [n / FPS for _, n in kept]
    meta.pop("text", None), meta.pop("duration", None)
    path.write_text(json.dumps(meta, indent=2))


@tool
def generate_motion(
    segments: Annotated[list[Segment], Field(min_length=1, description="One or more prompt segments played in order.")],
    seed: int = 42,
    diffusion_steps: Annotated[int, Field(ge=2, le=1000, description="100 is the demo default (~60 s on CPU); 30-50 for quick drafts.")] = 100,
    text_weight: Annotated[float, Field(ge=0, le=5, description="How strictly to follow the text (demo default 2.0).")] = 2.0,
    constraint_weight: Annotated[float, Field(ge=0, le=5, description="How strictly to follow constraints (demo default 2.0).")] = 2.0,
    num_samples: Annotated[int, Field(ge=1, le=10, description="Takes generated at once; each gets its own take id.")] = 1,
    transition_frames: Annotated[int, Field(ge=1, le=10)] = 5,
    keyframes_from: Annotated[list[Keyframes] | None, Field(description="Poses to lock, copied from earlier takes; several sources can be combined.")] = None,
    root_path: Annotated[list[PathPoint] | None, Field(description="Root positions the character must pass through.")] = None,
    constraints: Annotated[list[dict] | None, Field(description="Raw Kimodo constraint dicts (constraints.json format).")] = None,
    criteria: Annotated[list[Criterion] | None, Field(description="Rank the new takes against these, best first (see score_takes).")] = None,
) -> dict:
    """Generate motion takes with Kimodo and save each one. Returns take ids and key numbers.

    Each take is saved as a Kimodo demo example folder (motion.npz, meta.json, constraints.json),
    so it can also be opened in the web demo. Inspect results with render_motion and analyze_motion.
    """
    texts = [s.prompt for s in segments]
    num_frames = [round(s.seconds * FPS) for s in segments]
    dicts = list(constraints or [])
    for kf in keyframes_from or []:
        dicts.append(_keyframe_constraint(kf))
    if root_path:
        dicts.append({"type": "root2d", "frame_indices": [p.frame for p in root_path], "smooth_root_2d": [[p.x, p.z] for p in root_path]})

    out, seconds = _run_model(texts, num_frames, dicts, seed, diffusion_steps, text_weight, constraint_weight, num_samples, transition_frames)

    takes = []
    for i in range(out["posed_joints"].shape[0]):
        take_id = _new_id(texts[0])
        d = TAKES / take_id
        d.mkdir()
        motion = {k: np.asarray(v[i]) for k, v in out.items() if isinstance(v, np.ndarray)}
        np.savez(d / "motion.npz", **motion)
        m = metrics.analyze(motion)
        meta = {
            "texts": texts, "durations": [s.seconds for s in segments],
            "num_samples": 1, "seed": seed, "diffusion_steps": diffusion_steps,
            "cfg": {"enabled": True, "text_weight": text_weight, "constraint_weight": constraint_weight},
            "mcp": {"sample": i, "of": num_samples, "transition_frames": transition_frames,
                    "keyframes_from": [kf.model_dump() for kf in keyframes_from or []],
                    "root_path": [p.model_dump() for p in root_path or []], "raw_constraints": constraints or [],
                    "generated_s": seconds, "summary": metrics.one_line(m)},
        }
        (d / "meta.json").write_text(json.dumps(meta, indent=2))
        if dicts:
            (d / "constraints.json").write_text(json.dumps(dicts))
        takes.append({"take_id": take_id, "summary": metrics.one_line(m), "path": str(d)})
    others = f" (+{len(takes) - 1} more: {', '.join(t['take_id'] for t in takes[1:])})" if len(takes) > 1 else ""
    _live(takes[0]["take_id"], f"New take {takes[0]['take_id']}", takes[0]["summary"] + others, play=True)
    result = {"generation_seconds": seconds, "takes": takes}
    if criteria:
        result["ranking"] = sorted((_score(t["take_id"], criteria) for t in takes), key=lambda r: -r["passed"])
    return result


@tool
def render_motion(
    take_id: str,
    frames: Annotated[int | list[int], Field(description="Number of evenly spaced frames, or exact frame numbers (30 fps).")] = 8,
    view: Annotated[str, Field(description="side (faces right), front, three-quarter, or top.")] = "side",
    rifle_frame: Annotated[int | None, Field(description="Draw a rifle fixed in the right hand, aimed through the left hand at this frame (see check_rifle).")] = None,
) -> list:
    """Render frames of a take as one contact-sheet image (body mesh, 0.5 m height guides, frame labels)."""
    motion = _load(take_id)
    picked = render.pick_frames(len(motion["posed_joints"]), frames)
    prop = None if rifle_frame is None else render.rifle_mesh(motion, rifle_frame)
    img = render.contact_sheet(motion, picked, view=view, cols=min(4, len(picked)), prop=prop)
    data = render.png_bytes(img)
    path = _take_dir(take_id) / f"sheet-{view}-{'-'.join(map(str, picked[:12]))}.png"
    path.write_bytes(data)
    _live(take_id, f"Claude is looking at {take_id}", f"Frames {picked}, {view} view", frames=picked)
    return [Image(data=data, format="png"), f"{take_id}: frames {picked}, view {view}. Saved {path}"]


@tool
def analyze_motion(take_id: str) -> dict:
    """Measure a take: peak heights (and frame) of head, hands and feet relative to the head,
    root travel, heading change, end pose, lowest point, foot contact, and jitter. Y-up metres, 30 fps.
    Multi-prompt takes also get per-segment torso twist and shoulder movement ("segments")."""
    motion = _load(take_id)
    m = metrics.analyze(motion)
    meta_path = _take_dir(take_id) / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        texts, secs = meta.get("texts") or [meta.get("text")], meta.get("durations") or [meta.get("duration")]
        if len(texts) > 1 and None not in secs:
            starts = np.cumsum([0] + [round(s * FPS) for s in secs])
            m["segments"] = metrics.segments(motion, [(t, int(starts[i]), int(starts[i + 1]) - 1) for i, t in enumerate(texts)])
    _live(take_id, f"Measuring {take_id}", metrics.one_line(m))
    return m


@tool
def export_motion(
    take_id: str,
    format: Annotated[str, Field(description="bvh (SOMA skeleton, cm), npz (Kimodo), or mp4 (rendered video).")] = "bvh",
    view: Annotated[str, Field(description="Camera for mp4.")] = "three-quarter",
) -> str:
    """Write a take to a file and return its path."""
    d = _take_dir(take_id)
    if format == "npz":
        return str(d / "motion.npz")
    motion = _load(take_id)
    if format == "mp4":
        path = d / f"{take_id}-{view}.mp4"
        render.video(motion, str(path), view=view)
        return str(path)
    if format == "bvh":
        import torch
        from kimodo.exports.bvh import save_motion_bvh
        from kimodo.skeleton.registry import build_skeleton

        skel = build_skeleton(motion["global_rot_mats"].shape[1])
        local = skel.global_rots_to_local_rots(torch.from_numpy(motion["global_rot_mats"]))
        path = d / f"{take_id}.bvh"
        save_motion_bvh(str(path), local, torch.from_numpy(motion["posed_joints"][:, skel.root_idx]), skeleton=skel, fps=FPS)
        return str(path)
    raise ValueError("format must be bvh, npz or mp4")


@tool
def add_layer(
    take_id: str,
    layers: Annotated[list[Layer] | None, Field(description="Rotation layers, applied in order.")] = None,
    preset: Annotated[str | None, Field(description="impact_front, impact_back, impact_left or impact_right; added after any layers.")] = None,
    start: Annotated[int | None, Field(description="Frame where the preset begins (required with preset).")] = None,
    strength: Annotated[float, Field(ge=0, le=3, description="Multiplier for the preset's angles.")] = 1.0,
) -> dict:
    """Add curve-driven rotations to chosen joints of a finished take and save a new take; no model run.

    Works on any motion: impacts, recoil, a slumped posture, a head turn, exaggerating a swing. Frames outside each
    layer are unchanged. Layers on hips or legs move the feet (there is no IK), so check max_foot_drift_m.
    """
    if preset and start is None:
        raise ValueError("preset needs start")
    motion = _load(take_id)
    skeleton, local, root = edit.motion_parts(motion)
    spec = [lay.model_dump() for lay in layers or []]
    if preset:
        spec += edit.preset_layers(preset, start, strength)
    if not spec:
        raise ValueError("Give layers or a preset")
    new = edit.rebuild_motion(edit.apply_layers(local, root, skeleton, spec), root, skeleton)
    drift = round(edit.foot_drift(np.asarray(motion["posed_joints"]), new["posed_joints"], list(skeleton.bone_index)), 3)
    result = _save_edit(take_id, new, "layer on", {"edit": "add_layer", "layers": spec, "preset": preset, "strength": strength})
    result["max_foot_drift_m"] = drift
    _live(result["take_id"], f"New layer take {result['take_id']}", result["summary"], play=True)
    return result


def _redo_indices(joints, names: list[str]) -> set[int]:
    lower = set(lower_body(names))
    if joints == "lower_body":
        return lower
    if joints == "above_hips":
        return set(range(len(names))) - lower
    return set(edit.joint_indices(joints, names))


@tool
def regenerate_range(
    take_id: str,
    start: Annotated[int, Field(ge=0, description="First frame to redo (30 fps).")],
    end: Annotated[int, Field(description="Last frame to redo.")],
    prompt: Annotated[str | None, Field(description="What should happen in the range. Default: the source prompt(s) overlapping it.")] = None,
    seed: int = 42,
    diffusion_steps: Annotated[int, Field(ge=2, le=1000)] = 100,
    text_weight: Annotated[float, Field(ge=0, le=5)] = 2.0,
    constraint_weight: Annotated[float, Field(ge=0, le=5)] = 2.0,
    num_samples: Annotated[int, Field(ge=1, le=10)] = 1,
    blend_frames: Annotated[int, Field(ge=1, le=30, description="Frames blended at each inner edge of the range.")] = 6,
    keyframes_from: Annotated[list[Keyframes] | None, Field(description="Extra poses to lock inside the range.")] = None,
    redo_joints: Annotated[str | list[str] | None, Field(description=f"Redo only these joints: above_hips, lower_body, a group ({', '.join(edit.GROUPS)}) or joint names. Every other joint and the root stay exactly as they were. Default: the whole body.")] = None,
) -> dict:
    """Redo only frames start..end of a take with Kimodo; every other frame stays exactly as it was.

    The surrounding frames are locked to guide the model, then the new section is spliced in and blended at its edges.
    With redo_joints, hands, feet and hips that aren't being redone are also locked inside the range. Each result is saved as a new take.
    """
    motion = _load(take_id)
    T = len(motion["posed_joints"])
    if not 0 <= start <= end < T:
        raise ValueError(f"Range {start}..{end} is outside the take's frames 0..{T - 1}")
    if blend_frames * 2 > end - start + 1:
        raise ValueError(f"Range is too short for blend_frames={blend_frames}: use a range of at least {blend_frames * 2} frames or a smaller blend")
    if prompt is None:
        meta = json.loads((_take_dir(take_id) / "meta.json").read_text())
        texts, secs = meta.get("texts") or [meta.get("text")], meta.get("durations") or [meta.get("duration")]
        edges = np.cumsum([0] + [round(x * FPS) for x in secs])
        prompt = " ".join(t for i, t in enumerate(texts) if edges[i] <= end and edges[i + 1] > start) or " ".join(texts)
    outside = [f for f in range(T) if f < start or f > end]
    dicts = [_keyframe_constraint(Keyframes(take_id=take_id, frames=outside))] if outside else []
    dicts += [_keyframe_constraint(kf) for kf in keyframes_from or []]
    skeleton, src_local, src_root = edit.motion_parts(motion)
    names = list(skeleton.bone_index)
    redo = None if redo_joints is None else _redo_indices(redo_joints, names)
    if redo is not None:
        hold = [j for j in END_EFFECTORS if names.index(j) not in redo]
        if hold:
            dicts.append(_keyframe_constraint(Keyframes(take_id=take_id, frames=list(range(start, end + 1)), only=hold)))
    out, seconds = _run_model([prompt], [T], dicts, seed, diffusion_steps, text_weight, constraint_weight, num_samples)

    takes = []
    for i in range(out["posed_joints"].shape[0]):
        _, new_local, new_root = edit.motion_parts({k: np.asarray(v[i]) for k, v in out.items() if isinstance(v, np.ndarray)})
        local, root = edit.splice_range(src_local, src_root, new_local, new_root, start, end, blend_frames)
        if redo is not None:
            keep = [j for j in range(len(names)) if j not in redo]
            local[:, keep], root = src_local[:, keep], src_root
        takes.append(_save_edit(take_id, edit.rebuild_motion(local, root, skeleton), "redo of", {
            "edit": "regenerate_range", "range": [start, end], "prompt": prompt, "seed": seed, "diffusion_steps": diffusion_steps,
            "text_weight": text_weight, "constraint_weight": constraint_weight, "blend_frames": blend_frames,
            "keyframes_from": [kf.model_dump() for kf in keyframes_from or []], "redo_joints": redo_joints}))
    _live(takes[0]["take_id"], f"Redid f{start}-{end}: {takes[0]['take_id']}", takes[0]["summary"], frames=[max(start - 5, 0), start, (start + end) // 2, end, min(end + 5, T - 1)])
    return {"generation_seconds": seconds, "takes": takes}


@tool
def smooth_motion(
    take_id: str,
    start: Annotated[int, Field(ge=0, description="First frame to smooth (30 fps).")],
    end: Annotated[int, Field(description="Last frame to smooth.")],
    sigma_frames: Annotated[float, Field(gt=0, le=15, description="Smoothing strength in frames: 2 is light, 4 is strong.")] = 3.0,
    joints: Annotated[str | list[str] | None, Field(description="A group or joint names to smooth. Default: the whole body, including root movement.")] = None,
    blend_frames: Annotated[int, Field(ge=1, le=30, description="Frames over which smoothing fades in and out at the range edges.")] = 6,
) -> dict:
    """Soften jerks in a frame range and save a new take; no model run. Frames outside the range are unchanged.

    Also softens intended peaks: check the result with analyze_motion and render_motion.
    """
    motion = _load(take_id)
    T = len(motion["posed_joints"])
    if not 0 <= start <= end < T:
        raise ValueError(f"Range {start}..{end} is outside the take's frames 0..{T - 1}")
    if blend_frames * 2 > end - start + 1:
        raise ValueError(f"Range is too short for blend_frames={blend_frames}")
    skeleton, local, root = edit.motion_parts(motion)
    idx = None if joints is None else edit.joint_indices(joints, list(skeleton.bone_index))
    smoothed, smooth_root = edit.gaussian_smooth(local, root, sigma_frames, idx, smooth_root=joints is None)
    new_local, new_root = edit.splice_range(local, root, smoothed, smooth_root, start, end, blend_frames)
    new = edit.rebuild_motion(new_local, new_root, skeleton)
    result = _save_edit(take_id, new, "smooth of", {"edit": "smooth_motion", "range": [start, end], "sigma_frames": sigma_frames, "joints": joints, "blend_frames": blend_frames})
    result["max_foot_drift_m"] = round(edit.foot_drift(np.asarray(motion["posed_joints"]), new["posed_joints"], list(skeleton.bone_index)), 3)
    _live(result["take_id"], f"Smoothed f{start}-{end}: {result['take_id']}", result["summary"], play=True)
    return result


def _audit(motion: dict) -> dict:
    verts, _ = render.skin_vertices(motion["global_rot_mats"], motion["posed_joints"])
    return metrics.audit(motion, verts[:, :, 1].min(1))


@tool
def audit_motion(take_id: str) -> dict:
    """List possible flaws in a take with their frames: floor_sink, foot_slide, jerk, whip, freeze, reversal, idle_tail.

    These are things to look at, not verdicts: a freeze is a flaw in a hit but correct in a held pose, and a reversal
    is a flaw in a fall but normal in a dance. Floor sink is nearly always a real flaw.
    """
    return _audit(_load(take_id))


@tool
def view_motion(
    take_id: str,
    start: Annotated[int, Field(ge=0, description="First frame to show.")] = 0,
    end: Annotated[int | None, Field(description="Last frame to show. Default: the end of the take.")] = None,
    joints: Annotated[list[str] | None, Field(description="Joints to plot. Default: head, chest, hands and feet.")] = None,
    view: Annotated[str, Field(description="Camera for the onion skin: side, front, three-quarter or top.")] = "side",
    poses: Annotated[int, Field(ge=2, le=12, description="Poses overlaid in the onion skin.")] = 6,
) -> list:
    """Show how a take moves over time: curves of speed, height and front-back position, plus an onion skin with joint paths.

    Reads motion from images: a spike is a jerk, a dip between peaks is a freeze, and a hump in front-back is a reversal.
    Dashed lines mark prompt boundaries. On the paths, dots are 5 frames apart: wide gaps mean fast.
    """
    motion = _load(take_id)
    T = len(motion["posed_joints"])
    end = T - 1 if end is None else min(end, T - 1)
    if start >= end:
        raise ValueError(f"Need start < end within frames 0..{T - 1}")
    marks = []
    meta_path = _take_dir(take_id) / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        secs = meta.get("durations") or []
        marks = [int(x) for x in np.cumsum([round(s * FPS) for s in secs])[:-1]] if len(secs) > 1 else []
    use = joints or ["Head", "Chest", "LeftHand", "RightHand", "LeftFoot", "RightFoot"]
    curves = render.motion_curves(motion, use, start, end, marks)
    picked = [int(round(x)) for x in np.linspace(start, end, poses)]
    onion = render.onion_skin(motion, picked, view, ["Head", "RightHand"] if joints is None else joints[:3])
    _live(take_id, f"Claude is studying {take_id}", f"Frames {start}-{end}", frames=picked)
    return [Image(data=render.png_bytes(curves), format="png"), Image(data=render.png_bytes(onion), format="png"), f"{take_id}: frames {start}-{end}, poses at {picked}"]


@tool
def compare_takes(
    take_ids: Annotated[list[str], Field(min_length=2, max_length=6)],
    frames: Annotated[int | list[int], Field(description="Evenly spaced frames per take, or exact frame numbers.")] = 6,
    view: str = "side",
) -> list:
    """Show several takes one above the other at the same frames, with a table of their key numbers and flagged flaws."""
    rows, table = [], []
    for tid in take_ids:
        motion = _load(tid)
        picked = render.pick_frames(len(motion["posed_joints"]), frames)
        sheet = render.contact_sheet(motion, picked, view=view, tile=220, cols=len(picked))
        rows.append((tid, sheet))
        m, a = metrics.analyze(motion), _audit(motion)
        table.append({"take_id": tid, "seconds": m["duration_s"], "head_peak_m_per_s": m["head_peak_speed"]["m_per_s"], "jitter": m["jitter_m_per_s2"],
                      "root_travel_m": m["root"]["net_travel_m"], "end_head_m": m["end_pose"]["head_height_m"],
                      "fall_direction_deg": m["end_pose"].get("lean_direction_vs_start_facing_deg"), "still_end_m": m["end_pose"]["last_0_5s_net_movement_m"],
                      "floor_sink_m": a["floor_sink_max_m"], "flagged": a["counts"], "body_accel_peak": a["body_accel_peak_m_per_s2"], "hand_accel_peak": a["hand_accel_peak_m_per_s2"]})
    return [Image(data=render.png_bytes(render.stack_labelled(rows)), format="png"), json.dumps(table, indent=1)]


def _score(take_id: str, criteria: list[Criterion]) -> dict:
    motion = _load(take_id)
    data = {**metrics.analyze(motion), "audit": _audit(motion)}
    return {"take_id": take_id, **metrics.score(data, [c.model_dump() for c in criteria])}


@tool
def score_takes(take_ids: Annotated[list[str], Field(min_length=1)], criteria: list[Criterion]) -> list[dict]:
    """Score takes against criteria and rank them, best first. A missing audit count means no findings of that type."""
    return sorted((_score(t, criteria) for t in take_ids), key=lambda r: -r["passed"])


@tool
def match_pose(take_id: str, reference_take_id: str, reference_frame: int) -> dict:
    """Find the frames of a take closest to a pose from another take, ignoring position and facing. Error in metres."""
    return metrics.match_pose(_load(take_id), _load(reference_take_id), reference_frame)


@tool
def set_joint_target(
    take_id: str,
    joint: Annotated[str, Field(description="LeftHand, RightHand, LeftFoot or RightFoot.")],
    start: int,
    end: int,
    offset: Annotated[tuple[float, float, float] | None, Field(description="Move by this many metres (x, y, z; y = up, the character starts facing +z), following the motion.")] = None,
    position: Annotated[tuple[float, float, float] | None, Field(description="Move to this fixed world position in metres.")] = None,
    pin: Annotated[bool, Field(description="Hold the joint where it is at pin_frame, e.g. to stop a planted foot sliding.")] = False,
    pin_frame: Annotated[int | None, Field(description="Frame to hold the joint at. Default: start.")] = None,
    curve: Annotated[str | list[list[float]], Field(description="How strongly the target applies over time: spring, ease, hold, or [[frame, weight], ...].")] = "hold",
    attack_frames: Annotated[int, Field(ge=0, description="Frames to reach full strength.")] = 8,
    decay_frames: Annotated[int, Field(ge=0, description="Frames to let go before end.")] = 8,
    follow: Annotated[str | None, Field(description="Keep this joint where it was relative to another joint at follow_frame, e.g. LeftHand follows RightHand to hold a two-handed rifle grip.")] = None,
    follow_frame: Annotated[int | None, Field(description="Frame whose relative position to keep. Default: start.")] = None,
) -> dict:
    """Move or pin a hand or foot by bending its limb (two-bone IK) and save a new take; no model run.

    Give one of offset, position, pin or follow. The hand or foot keeps its world orientation. A target out of reach stops at
    full stretch; those frames are reported. Only that limb changes, so a rifle grip between two hands needs both moved.
    """
    if sum([offset is not None, position is not None, bool(pin), follow is not None]) != 1:
        raise ValueError("Give exactly one of offset, position, pin or follow")
    motion = _load(take_id)
    skeleton, local, root = edit.motion_parts(motion)
    names = list(skeleton.bone_index)
    T = len(local)
    if not 0 <= start < end < T:
        raise ValueError(f"Range {start}..{end} is outside the take's frames 0..{T - 1}")
    if joint not in edit.IK_CHAINS:
        raise ValueError(f"joint must be one of {', '.join(edit.IK_CHAINS)}")
    current = torch.as_tensor(np.asarray(motion["posed_joints"])[:, names.index(joint)], dtype=root.dtype)
    if offset is not None:
        targets = current + torch.tensor(offset, dtype=root.dtype)
    elif position is not None:
        targets = torch.tensor(position, dtype=root.dtype).expand(T, 3)
    elif follow is not None:
        f = start if follow_frame is None else follow_frame
        pj, gr = (torch.as_tensor(np.asarray(motion[k])[:, names.index(follow)], dtype=root.dtype) for k in ("posed_joints", "global_rot_mats"))
        offset_local = gr[f].T @ (current[f] - pj[f])
        targets = pj + (gr @ offset_local[..., None])[..., 0]
    else:
        targets = current[start if pin_frame is None else pin_frame].expand(T, 3)
    weights = edit.curve_weights(T, curve, start, end, attack_frames, decay_frames)
    new_local, unreachable = edit.two_bone_ik(local, root, skeleton, joint, targets, weights)
    new = edit.rebuild_motion(new_local, root, skeleton)
    result = _save_edit(take_id, new, "ik on", {"edit": "set_joint_target", "joint": joint, "start": start, "end": end, "offset": offset, "position": position,
                                                 "pin": pin, "pin_frame": pin_frame, "curve": curve, "attack_frames": attack_frames, "decay_frames": decay_frames,
                                                 "follow": follow, "follow_frame": follow_frame})
    result["unreachable_frames"] = unreachable[:20]
    result["joint_moved_max_m"] = round(float(np.linalg.norm(new["posed_joints"][:, names.index(joint)] - np.asarray(motion["posed_joints"])[:, names.index(joint)], axis=1).max()), 3)
    _live(result["take_id"], f"IK on {joint}: {result['take_id']}", result["summary"], play=True)
    return result


@tool
def fix_ground(
    take_id: str,
    start: Annotated[int, Field(ge=0, description="First frame to correct.")] = 0,
    end: Annotated[int | None, Field(description="Last frame to correct. Default: the end of the take.")] = None,
    smooth_frames: Annotated[int, Field(ge=1, le=30, description="How gradually the lift eases in and out.")] = 8,
) -> dict:
    """Raise the whole body smoothly wherever its mesh goes through the floor, and save a new take.

    Lifts everything, so a planted foot can rise with a sunken knee; limit start..end to avoid that (see max_foot_lift_m).
    Reports the floor sink before and after.
    """
    motion = _load(take_id)
    skeleton, local, root = edit.motion_parts(motion)
    verts, _ = render.skin_vertices(motion["global_rot_mats"], motion["posed_joints"])
    low = verts[:, :, 1].min(1)
    lift = edit.ground_lift(low, start, end, smooth_frames)
    new_root = root.clone()
    new_root[:, 1] += torch.as_tensor(lift, dtype=root.dtype)
    new = edit.rebuild_motion(local, new_root, skeleton)
    after = render.skin_vertices(new["global_rot_mats"], new["posed_joints"])[0][:, :, 1].min(1)
    result = _save_edit(take_id, new, "ground on", {"edit": "fix_ground", "range": [start, end], "smooth_frames": smooth_frames})
    result |= {"sink_before_m": round(float(max(-low.min(), 0)), 3), "sink_after_m": round(float(max(-after[start:end].min(), 0)), 3),
               "max_lift_m": round(float(lift.max()), 3), "max_foot_lift_m": round(edit.foot_drift(np.asarray(motion["posed_joints"]), new["posed_joints"], list(skeleton.bone_index)), 3)}
    _live(result["take_id"], f"Ground fixed: {result['take_id']}", f"Sink {result['sink_before_m']} -> {result['sink_after_m']} m", play=True)
    return result


@tool
def trim_take(
    take_id: str,
    end: Annotated[int, Field(ge=1, description="Last frame to keep.")],
    start: Annotated[int, Field(ge=0, description="First frame to keep.")] = 0,
) -> dict:
    """Cut a take to frames start..end and save a new take, e.g. to remove an idle tail. Nothing is regenerated."""
    motion = _load(take_id)
    T = len(motion["posed_joints"])
    if not 0 <= start < end < T:
        raise ValueError(f"Range {start}..{end} is outside the take's frames 0..{T - 1}")
    skeleton, local, root = edit.motion_parts(motion)
    new = edit.rebuild_motion(local[start : end + 1], root[start : end + 1], skeleton)
    result = _save_edit(take_id, new, "trim of", {"edit": "trim_take", "range": [start, end]})
    spans = _prompt_spans(json.loads((_take_dir(take_id) / "meta.json").read_text()))
    _write_prompts(result["take_id"], [(t, min(b, end + 1) - max(a, start)) for t, a, b in spans])
    _live(result["take_id"], f"Trimmed: {result['take_id']}", result["summary"], play=True)
    return result


class SpeedRange(BaseModel):
    start: int = Field(ge=0, description="First frame, in the source take.")
    end: int = Field(description="Last frame, in the source take.")
    speed: float = Field(gt=0.05, le=10, description="2 = twice as fast, 0.5 = half speed.")


@tool
def retime_motion(
    take_id: str,
    ranges: Annotated[list[SpeedRange], Field(min_length=1)],
    ease_frames: Annotated[int, Field(ge=0, le=30, description="How gradually speed changes, in frames. 0 is abrupt, which can itself look like a jerk.")] = 4,
) -> dict:
    """Play frame ranges faster or slower and save a new take; no model run. Poses are unchanged; only timing moves.

    A hit gets more impact from a fast strike followed by a slower collapse. The take's length changes:
    frame_map says where source frames ended up.
    """
    motion = _load(take_id)
    T = len(motion["posed_joints"])
    for r in ranges:
        if not 0 <= r.start <= r.end < T:
            raise ValueError(f"Range {r.start}..{r.end} is outside the take's frames 0..{T - 1}")
    skeleton, local, root = edit.motion_parts(motion)
    spec = [r.model_dump() for r in ranges]
    new_local, new_root, tau = edit.retime(local, root, spec, ease_frames)
    result = _save_edit(take_id, edit.rebuild_motion(new_local, new_root, skeleton), "retime of", {"edit": "retime_motion", "ranges": spec, "ease_frames": ease_frames})
    time_at = lambda f: int(round(tau[min(f, T - 1)]))
    spans = _prompt_spans(json.loads((_take_dir(take_id) / "meta.json").read_text()))
    _write_prompts(result["take_id"], [(t, time_at(b) - time_at(a) + (1 if b >= T else 0)) for t, a, b in spans])
    result["frames_before_after"] = [T, len(new_local)]
    result["frame_map"] = {str(f): time_at(f) for r in ranges for f in (r.start, r.end)}
    _live(result["take_id"], f"Retimed: {result['take_id']}", result["summary"], play=True)
    return result


class FootWindow(BaseModel):
    foot: str = Field(description="LeftFoot or RightFoot.")
    start: int
    end: int


def _foot_slide(pj: np.ndarray, windows: dict[str, list[list[int]]], names: list[str]) -> float:
    """Largest distance a foot travels within one of its windows, in metres."""
    worst = 0.0
    for foot, runs in windows.items():
        p = pj[:, names.index(foot)]
        for a, b in runs:
            worst = max(worst, float(np.linalg.norm(p[a : b + 1] - p[a : b + 1].mean(0), axis=1).max()) * 2)
    return round(worst, 3)


@tool
def plant_feet(
    take_id: str,
    windows: Annotated[list[FootWindow] | None, Field(description="Exact frames to hold each foot. Default: find them, wherever a foot is low and slow.")] = None,
    min_frames: Annotated[int, Field(ge=2, description="Shortest contact to treat as planted, when finding them.")] = 6,
    blend_frames: Annotated[int, Field(ge=1, le=15, description="Frames to ease in and out of each plant.")] = 4,
) -> dict:
    """Stop planted feet sliding: hold each foot still during its contacts by bending the leg, and save a new take. No model run.

    Reports the contacts and the slide before and after. A foot is let go on frames the leg can't reach. Check the
    result: a slow step can read as a contact, and on a fall a foot may need to drag.
    """
    motion = _load(take_id)
    skeleton, local, root = edit.motion_parts(motion)
    names = list(skeleton.bone_index)
    pj = np.asarray(motion["posed_joints"])
    if windows:
        found: dict[str, list[list[int]]] = {}
        for w in windows:
            if w.foot not in ("LeftFoot", "RightFoot"):
                raise ValueError("foot must be LeftFoot or RightFoot")
            if not 0 <= w.start < w.end < len(pj):
                raise ValueError(f"Window {w.start}..{w.end} is outside the take's frames 0..{len(pj) - 1}")
            found.setdefault(w.foot, []).append([w.start, w.end])
    else:
        found = {f: edit.plant_windows(pj[:, names.index(f)], min_frames) for f in ("LeftFoot", "RightFoot")}
    if not any(found.values()):
        raise ValueError("No planted feet found: nothing to do. Pass windows to choose frames.")
    new_local, missed = edit.plant_feet(local, root, skeleton, found, blend_frames)  # trims found to the frames it could hold
    new = edit.rebuild_motion(new_local, root, skeleton)
    spec = [{"foot": f, "start": a, "end": b} for f, runs in found.items() for a, b in runs]
    result = _save_edit(take_id, new, "feet on", {"edit": "plant_feet", "windows": spec, "manual": bool(windows), "min_frames": min_frames, "blend_frames": blend_frames})
    result |= {"contacts": found, "slide_before_m": _foot_slide(pj, found, names), "slide_after_m": _foot_slide(new["posed_joints"], found, names),
               "unreachable_frames": {f: m[:20] for f, m in missed.items() if m}}
    _live(result["take_id"], f"Feet planted: {result['take_id']}", f"Slide {result['slide_before_m']} -> {result['slide_after_m']} m", play=True)
    return result


@tool
def join_takes(
    take_ids: Annotated[list[str], Field(min_length=2, max_length=12, description="Takes in play order. Use trim_take first to cut them.")],
    blend_frames: Annotated[int, Field(ge=1, le=60, description="Frames of cross-fade at each join; the result is shorter by this much per join.")] = 10,
) -> dict:
    """Join takes into one, turning and moving each to continue from the last, cross-faded. Saves a new take; no model run.

    seam_pose_error_m is how far apart the poses are in the middle of each fade. Under about 0.1 m blends cleanly;
    above that the fade can look like a slide, so redo the seam with regenerate_range.
    """
    parts = [edit.motion_parts(_load(t)) for t in take_ids]
    if len({p[1].shape[1] for p in parts}) != 1:
        raise ValueError("Takes use different skeletons")
    skeleton = parts[0][0]
    local, root, starts, errors = edit.join_clips([(p[1], p[2]) for p in parts], skeleton, blend_frames)
    new = edit.rebuild_motion(local, root, skeleton)
    result = _save_edit(take_ids[0], new, "join of", {"edit": "join_takes", "takes": take_ids, "blend_frames": blend_frames})
    spans, ends = [], starts[1:] + [len(local)]
    for t, first, end in zip(take_ids, starts, ends):
        own = _prompt_spans(json.loads((_take_dir(t) / "meta.json").read_text()))
        used = 0
        for k, (text, a, b) in enumerate(own):
            n = end - first - used if k == len(own) - 1 else b - a
            n = max(min(n, end - first - used), 0)
            spans.append((text, n))
            used += n
    _write_prompts(result["take_id"], spans)
    result |= {"frames": len(local), "clip_starts": starts, "seam_pose_error_m": errors}
    _live(result["take_id"], f"Joined: {result['take_id']}", result["summary"], play=True)
    return result


@tool
def make_loop(
    take_id: str,
    start: Annotated[int | None, Field(description="First frame of the cycle. Default: find the frames that match best.")] = None,
    end: Annotated[int | None, Field(description="Frame that should lead back into start.")] = None,
    blend_frames: Annotated[int, Field(ge=1, le=60, description="Frames over which the end eases onto the start's pose.")] = 10,
    in_place: Annotated[bool, Field(description="Remove the root's travel, for a treadmill-style loop.")] = False,
    min_frames: Annotated[int, Field(ge=10, description="Shortest cycle, when finding frames.")] = 30,
) -> dict:
    """Make a seamless cycle, e.g. a walk or an idle, and save a new take; no model run. It plays last frame -> first frame.

    Reports root_travel_per_loop_m (x, z) for root motion. Compare seam_step_m with typical_step_m: a seam well above
    typical will show as a pop.
    """
    motion = _load(take_id)
    pj = np.asarray(motion["posed_joints"])
    T = len(pj)
    if (start is None) != (end is None):
        raise ValueError("Give both start and end, or neither")
    found = None
    if start is None:
        if T <= min_frames + 1:
            raise ValueError(f"The take has {T} frames: too short for a cycle of at least {min_frames}")
        found = metrics.best_loop(pj, min_frames)
        start, end = found["start"], found["end"]
    if not 0 <= start < end < T or end - start < 3:
        raise ValueError(f"Range {start}..{end} is outside the take's frames 0..{T - 1} or too short")
    skeleton, local, root = edit.motion_parts(motion)
    new_local, new_root, travel = edit.make_loop(local, root, start, end, blend_frames, in_place)
    new = edit.rebuild_motion(new_local, new_root, skeleton)
    result = _save_edit(take_id, new, "loop of", {"edit": "make_loop", "start": start, "end": end, "blend_frames": blend_frames, "in_place": in_place})
    _write_prompts(result["take_id"], [(" ".join(t for t, _, _ in _prompt_spans(json.loads((_take_dir(take_id) / "meta.json").read_text()))), len(new_local))])
    norm = metrics._normalized(np.asarray(new["posed_joints"], dtype=np.float64))
    steps = np.linalg.norm(np.diff(norm, axis=0), axis=-1).mean(1)
    result |= {"range": [start, end], "frames": len(new_local), "root_travel_per_loop_m": travel,
               "seam_step_m": round(float(np.linalg.norm(norm[-1] - norm[0], axis=-1).mean()), 4), "typical_step_m": round(float(np.median(steps)), 4)}
    if found:
        result["mismatch_before_blend_m"] = found["mismatch_m"]
    _live(result["take_id"], f"Loop: {result['take_id']}", f"Frames {start}-{end}", play=True)
    return result


@tool
def check_rifle(
    take_id: str,
    reference_frame: Annotated[int, Field(ge=0, description="A frame with a good two-handed hold. It defines the rifle: fixed in the right hand, aimed through the left.")] = 0,
    start: Annotated[int, Field(ge=0, description="First frame to check.")] = 0,
    end: Annotated[int | None, Field(description="Last frame to check, e.g. before the character lets go. Default: the end of the take.")] = None,
    max_gap_m: Annotated[float, Field(gt=0, description="How far the left hand can stray from its grip before a frame is flagged.")] = 0.05,
    frames: Annotated[int | list[int], Field(description="Frames to draw.")] = 8,
    view: str = "side",
) -> list:
    """Check a two-handed rifle hold and draw it: left hand gap from its grip, aim change, and muzzle height.

    To fix a drifting left hand, use set_joint_target with joint=LeftHand, follow=RightHand. The rifle is a stand-in
    placed from the wrist joints, so its place in the hands is approximate.
    """
    motion = _load(take_id)
    T = len(motion["posed_joints"])
    if not 0 <= reference_frame < T:
        raise ValueError(f"reference_frame is outside the take's frames 0..{T - 1}")
    report = metrics.rifle_check(motion, reference_frame, start, end, max_gap_m)
    mesh = render.rifle_mesh(motion, reference_frame)
    below = mesh[0][:, :, 1].min(1) < 0
    below[:start] = False
    if end is not None:
        below[end + 1 :] = False
    report["rifle_below_floor_frames"] = metrics._runs(below)
    picked = render.pick_frames(T, frames)
    img = render.contact_sheet(motion, picked, view=view, cols=min(4, len(picked)), prop=mesh)
    return [Image(data=render.png_bytes(img), format="png"), json.dumps(report, indent=1)]


def _blender() -> str:
    for p in (os.environ.get("KIMODO_BLENDER"), shutil.which("blender"), "/Applications/Blender.app/Contents/MacOS/Blender"):
        if p and Path(p).exists():
            return p
    raise ValueError("Blender not found. Install it or set KIMODO_BLENDER to its executable.")


@tool
def retarget_preview(
    take_id: str,
    rig_path: Annotated[str, Field(description="Your character: an .fbx, .glb or .gltf with an armature.")],
    frames: Annotated[int | list[int], Field(description="Frames to render.")] = 8,
    view: Annotated[str, Field(description="side, front, three-quarter or top.")] = "three-quarter",
    export_format: Annotated[str, Field(description="File to write with the animation on your rig: fbx or glb.")] = "fbx",
    joint_map: Annotated[dict[str, str] | None, Field(description="SOMA joint -> your bone name, for bones the automatic match misses, e.g. {\"LeftShin\": \"calf_L\"}.")] = None,
    lift_to_floor: Annotated[bool, Field(description="Raise the whole body, frame by frame with easing, so the rig's mesh stays on or above the floor. Reports the lift and any gap it leaves.")] = False,
) -> list:
    """Retarget a take onto your own character in headless Blender, render it, and write an animated file. The rig file isn't changed.

    Bone names are matched automatically for common styles (Mixamo, Unreal, Unity). Reports which bones were matched,
    how far planted feet slide on the rig, and how far its mesh goes below the floor at the rendered frames. Fingers are left in the rest pose.
    """
    rig = Path(rig_path).expanduser()
    if not rig.is_file() or rig.suffix.lower() not in (".fbx", ".glb", ".gltf"):
        raise ValueError(f"No .fbx, .glb or .gltf file at {rig}")
    if export_format not in ("fbx", "glb"):
        raise ValueError("export_format must be fbx or glb")
    if view not in render.VIEWS:
        raise ValueError(f"view must be one of {sorted(render.VIEWS)}")
    motion = _load(take_id)
    from kimodo.skeleton.registry import build_skeleton

    pj = np.asarray(motion["posed_joints"])
    skel = build_skeleton(pj.shape[1])
    names = list(skel.bone_index)
    out = _take_dir(take_id) / "retarget"
    out.mkdir(exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9_-]+", "-", rig.stem)
    source = out / f"{stem}-source.npz"
    np.savez(source, global_rot_mats=np.asarray(motion["global_rot_mats"]), posed_joints=pj, rest_joints=skel.neutral_joints.cpu().numpy())
    picked = render.pick_frames(len(pj), frames)
    job = {"rig": str(rig), "motion": str(source), "names": names, "joint_map": joint_map, "frames": picked, "out_dir": str(out), "stem": stem,
           "format": export_format, "fps": FPS, "camera_dir": list(render.VIEWS[view][0]), "lift_to_floor": lift_to_floor}
    job_path = out / f"{stem}-job.json"
    job_path.write_text(json.dumps(job))
    run = subprocess.run([_blender(), "-b", "--factory-startup", "--python-exit-code", "3", "--python", str(Path(__file__).with_name("blender_retarget.py")), "--", str(job_path)],
                         capture_output=True, text=True, timeout=600)
    errors = re.findall(r"KIMODO_ERROR: .*", run.stdout)
    if errors:
        raise ValueError(errors[0].removeprefix("KIMODO_ERROR: "))
    report_path = out / f"{stem}-report.json"
    if run.returncode or not report_path.exists():
        raise RuntimeError(f"Blender failed ({run.returncode}): {(run.stdout + run.stderr)[-1500:]}")
    report = json.loads(report_path.read_text())
    joints = np.load(out / f"{stem}-joints.npz")
    slide = {}
    for foot in ("LeftFoot", "RightFoot"):
        runs = edit.plant_windows(pj[:, names.index(foot)])
        slide[foot] = {"contacts": runs, "soma_scaled_m": round(_foot_slide(pj * report["scale_vs_soma"], {foot: runs}, names), 3),
                       "rig_m": round(max((float(np.linalg.norm(joints[foot][a : b + 1] - joints[foot][a : b + 1].mean(0), axis=1).max()) * 2 for a, b in runs), default=0.0), 3)}
    report["foot_slide"] = slide
    paths = report.pop("images")
    content = []
    if paths:
        from PIL import Image as PILImage

        tiles = [PILImage.open(p).convert("RGB") for p in paths]
        size, cols = tiles[0].width, min(4, len(tiles))
        sheet = PILImage.new("RGB", (cols * size, (len(tiles) + cols - 1) // cols * size), "white")
        for i, (tile, f) in enumerate(zip(tiles, picked)):
            render.ImageDraw.Draw(tile).text((8, 6), f"f{f}", fill=render.INK, font=render._font(15))
            sheet.paste(tile, (i % cols * size, i // cols * size))
        data = render.png_bytes(sheet)
        (out / f"{stem}-sheet.png").write_bytes(data)
        content.append(Image(data=data, format="png"))
        for p in paths:
            Path(p).unlink()
    else:
        report["note"] = "The rig has no mesh, so nothing was rendered."
    source.unlink()
    return content + [json.dumps(report, indent=1)]


def _recipe_chain(take_id: str) -> list[tuple[str, dict]]:
    """(take id, meta) from the original generation to this take."""
    chain = []
    current = take_id
    while current:
        meta = json.loads((_take_dir(current) / "meta.json").read_text())
        chain.append((current, meta))
        current = meta.get("mcp", {}).get("parent")
    return chain[::-1]


@tool
def get_recipe(take_id: str) -> list[dict]:
    """The steps that made a take, oldest first: its generation settings, then each edit."""
    steps = []
    for tid, meta in _recipe_chain(take_id):
        info = dict(meta.get("mcp", {}))
        info.pop("summary", None)
        if not info.get("parent"):
            info = {"edit": "generate_motion", "texts": meta.get("texts") or [meta.get("text")], "seconds": meta.get("durations") or [meta.get("duration")],
                    "seed": meta.get("seed"), "diffusion_steps": meta.get("diffusion_steps"), "cfg": meta.get("cfg"), **info}
        steps.append({"take_id": tid, **info})
    return steps


@tool
def replay_recipe(
    take_id: str,
    seed: Annotated[int | None, Field(description="New seed for the first generation. Default: the original, which should reproduce the take.")] = None,
    keep_edits: Annotated[bool, Field(description="Apply the take's later edits after generating.")] = True,
) -> dict:
    """Rerun a take's whole chain, e.g. with a new seed to make a variation, and return the final take.

    Locked poses still come from the takes they were copied from. Edits are aimed at fixed frames, so on a very
    different seed they may land at the wrong moment: check the result.
    """
    chain = _recipe_chain(take_id)
    root_id, root = chain[0]
    info = root.get("mcp", {})
    if info.get("imported_from_demo"):
        raise ValueError(f"{root_id} came from the demo, whose generation settings weren't recorded, so it can't be replayed.")
    texts, secs = root.get("texts") or [root.get("text")], root.get("durations") or [root.get("duration")]
    cfg = root.get("cfg") or {}
    kf = [Keyframes(**k) for k in (info.get("keyframes_from") or [])]
    raw = info.get("raw_constraints")
    if raw is None and not kf and (_take_dir(root_id) / "constraints.json").exists():
        raw = json.loads((_take_dir(root_id) / "constraints.json").read_text())  # takes saved before root_path was recorded
    samples = info.get("of") or 1
    result = generate_motion(
        segments=[Segment(prompt=t, seconds=x) for t, x in zip(texts, secs)], seed=root.get("seed", 42) if seed is None else seed,
        diffusion_steps=root.get("diffusion_steps", 100), text_weight=cfg.get("text_weight", 2.0), constraint_weight=cfg.get("constraint_weight", 2.0),
        num_samples=samples, transition_frames=info.get("transition_frames") or 5, keyframes_from=kf or None,
        root_path=[PathPoint(**p) for p in info.get("root_path") or []] or None, constraints=raw or None,
    )
    current = result["takes"][info.get("sample") or 0]["take_id"]
    steps = [current]
    for _, meta in chain[1:] if keep_edits else []:
        e = meta["mcp"]
        kind = e.get("edit")
        if kind == "add_layer":
            r = add_layer(current, layers=[Layer(**lay) for lay in e["layers"]])
        elif kind == "regenerate_range":
            r = regenerate_range(current, e["range"][0], e["range"][1], prompt=e.get("prompt"), seed=e.get("seed", 42), diffusion_steps=e.get("diffusion_steps", 100),
                                 text_weight=e.get("text_weight", 2.0), constraint_weight=e.get("constraint_weight", 2.0), blend_frames=e.get("blend_frames", 6),
                                 keyframes_from=[Keyframes(**k) for k in e.get("keyframes_from") or []] or None, redo_joints=e.get("redo_joints"))["takes"][0]
        elif kind == "smooth_motion":
            r = smooth_motion(current, e["range"][0], e["range"][1], e["sigma_frames"], e.get("joints"), e.get("blend_frames", 6))
        elif kind == "fix_ground":
            r = fix_ground(current, e["range"][0], e["range"][1], e.get("smooth_frames", 8))
        elif kind == "trim_take":
            r = trim_take(current, e["range"][1], e["range"][0])
        elif kind == "set_joint_target":
            r = set_joint_target(current, e["joint"], e["start"], e["end"], e.get("offset"), e.get("position"), e.get("pin", False), e.get("pin_frame"), e.get("curve", "hold"), e.get("attack_frames", 8), e.get("decay_frames", 8), e.get("follow"), e.get("follow_frame"))
        elif kind == "retime_motion":
            r = retime_motion(current, [SpeedRange(**x) for x in e["ranges"]], e.get("ease_frames", 4))
        elif kind == "plant_feet":
            r = plant_feet(current, [FootWindow(**w) for w in e["windows"]] if e.get("manual") else None, e.get("min_frames", 6), e.get("blend_frames", 4))
        elif kind == "join_takes":
            r = join_takes([current, *e["takes"][1:]], e.get("blend_frames", 10))
        elif kind == "make_loop":
            r = make_loop(current, e["start"], e["end"], e.get("blend_frames", 10), e.get("in_place", False))
        else:
            raise ValueError(f"Unknown edit '{kind}' in the recipe")
        current = r["take_id"]
        steps.append(current)
    return {"take_id": current, "steps": steps, "summary": metrics.one_line(metrics.analyze(_load(current)))}


def safe_name(name: str) -> str:
    """Validate a saved animation's folder name."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name):
        raise ValueError("Name must start with a letter or digit, and use only letters, digits, - and _")
    return name


@tool
def save_animation(
    take_id: str,
    name: Annotated[str, Field(description="Folder name, e.g. kneel-death-back: letters, digits, - and _.")],
    overwrite: Annotated[bool, Field(description="Replace an animation already saved under this name.")] = False,
    video_view: Annotated[str, Field(description="Camera for the saved video.")] = "three-quarter",
) -> dict:
    """Save a take as an approved animation, with a BVH and a video, so it shows in the demo's Saved Animations dropdown.

    The folder is a copy: later edits to takes don't change it. Edit with the take id in source-take.txt.
    """
    source = _take_dir(take_id)
    target = SAVED / safe_name(name)
    if target.exists() and not overwrite:
        raise ValueError(f"'{name}' is already saved. Pass overwrite=true to replace it, or use another name.")
    bvh = Path(export_motion(take_id, "bvh"))
    mp4 = Path(export_motion(take_id, "mp4", video_view))
    tmp = SAVED / f".{name}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    shutil.copy2(source / "motion.npz", tmp / "motion.npz")
    shutil.copy2(source / "meta.json", tmp / "meta.json")
    if (source / "constraints.json").exists():
        shutil.copy2(source / "constraints.json", tmp / "constraints.json")
    shutil.copy2(bvh, tmp / f"{name}.bvh")
    shutil.copy2(mp4, tmp / f"{name}.mp4")
    (tmp / "source-take.txt").write_text(take_id + "\n")
    if target.exists():
        shutil.rmtree(target)
    tmp.replace(target)
    _event({"refresh_saved": True})
    return {"name": name, "path": str(target), "source_take": take_id, "files": sorted(f.name for f in target.iterdir())}


@tool
def list_saved_animations() -> list[dict]:
    """List saved animations and the takes they came from."""
    out = []
    for name in saved_names():
        src = SAVED / name / "source-take.txt"
        out.append({"name": name, "source_take": src.read_text().strip() if src.exists() else None, "path": str(SAVED / name)})
    return out


@tool
def open_in_demo(
    name: Annotated[str, Field(description="A saved animation's name or a take id.")],
    play: bool = True,
) -> str:
    """Load a saved animation or a take in every open demo tab. Needs the demo running with mcp/kimodo.patch applied."""
    saved = SAVED / name
    path = saved if name and "/" not in name and (saved / "motion.npz").exists() else _take_dir(name)
    event = {"take_id": name, "path": str(path), "title": f"Opened {name}", "body": "", "frames": None, "play": play, "reload": True, "time": time.time()}
    _event(event)
    return f"Sent {name} to the demo ({path}). Tabs that are open load it within a second."


@tool
def list_takes() -> list[dict]:
    """List saved takes (newest last) with prompts, settings and a one-line summary."""
    out = []
    for d in sorted(TAKES.glob("t[0-9][0-9][0-9]-*")):
        try:
            meta = json.loads((d / "meta.json").read_text())
        except (OSError, ValueError):
            continue
        out.append({
            "take_id": d.name, "prompts": meta.get("texts"), "seconds": meta.get("durations"),
            "seed": meta.get("seed"), "steps": meta.get("diffusion_steps"), "cfg": meta.get("cfg"),
            "keyframes_from": meta.get("mcp", {}).get("keyframes_from"),
            "parent": meta.get("mcp", {}).get("parent"), "edit": meta.get("mcp", {}).get("edit"), "summary": meta.get("mcp", {}).get("summary"),
        })
    return out


def main():
    mcp.run()


if __name__ == "__main__":
    main()
