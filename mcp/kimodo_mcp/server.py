"""Kimodo MCP server: generate motions, render them headlessly, measure them, refine.

Run by an MCP host (e.g. Claude Code) over stdio. Needs the Kimodo venv and a running
text encoder service (`kimodo_textencoder`, default http://127.0.0.1:9550/).

Env vars:
  KIMODO_MCP_TAKES   where takes are saved (default ~/kimodo-local/motions)
  KIMODO_MODEL       model name (default kimodo-soma-rp)
  KIMODO_DEVICE      torch device for the motion model (default cpu; mps is untested)
"""

import contextlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Annotated

import numpy as np
from mcp.server import MCPServer
from mcp.server.mcpserver import Image
from pydantic import BaseModel, Field

from . import metrics, render

FPS = 30
TAKES = Path(os.environ.get("KIMODO_MCP_TAKES", Path.home() / "kimodo-local" / "motions")).expanduser()
MODEL_NAME = os.environ.get("KIMODO_MODEL", "kimodo-soma-rp")
DEVICE = os.environ.get("KIMODO_DEVICE", "cpu")
os.environ.setdefault("TEXT_ENCODER_MODE", "api")  # reuse the running encoder; never load a 2nd Llama

mcp = MCPServer("kimodo")
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
    event = {"take_id": take_id, "path": str(TAKES / take_id), "title": title, "body": body,
             "frames": frames, "play": play, "time": time.time()}
    tmp = TAKES / "_live.json.tmp"
    tmp.write_text(json.dumps(event))
    tmp.replace(TAKES / "_live.json")  # atomic, so the demo never reads half a file


def _new_id(prompt: str) -> str:
    TAKES.mkdir(parents=True, exist_ok=True)
    n = 1 + max((int(p.name[1:4]) for p in TAKES.glob("t[0-9][0-9][0-9]-*")), default=0)
    slug = "-".join(re.findall(r"[a-z0-9]+", prompt.lower())[:5])
    return f"t{n:03d}-{slug}"


def _keyframe_constraint(kf: Keyframes) -> dict:
    import torch
    from kimodo.geometry import matrix_to_axis_angle
    from kimodo.skeleton.registry import build_skeleton

    at = kf.at_frames or kf.frames
    if len(at) != len(kf.frames):
        raise ValueError("keyframes_from.at_frames must have the same length as frames")
    src = _load(kf.take_id)
    skel = build_skeleton(src["global_rot_mats"].shape[1])
    local = skel.global_rots_to_local_rots(torch.from_numpy(src["global_rot_mats"][kf.frames]))
    return {
        "type": "fullbody",
        "frame_indices": at,
        "local_joints_rot": matrix_to_axis_angle(local).tolist(),
        "root_positions": (src["posed_joints"][kf.frames, skel.root_idx] + [kf.offset_xz[0], 0.0, kf.offset_xz[1]]).tolist(),
    }


@mcp.tool()
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
) -> dict:
    """Generate motion takes with Kimodo and save each one. Returns take ids and key numbers.

    Each take is saved as a Kimodo demo example folder (motion.npz, meta.json, constraints.json),
    so it can also be opened in the web demo. Inspect results with render_motion and analyze_motion.
    """
    from kimodo.constraints import load_constraints_lst
    from kimodo.tools import seed_everything

    model = _get_model()
    texts = [s.prompt for s in segments]
    num_frames = [round(s.seconds * FPS) for s in segments]
    dicts = list(constraints or [])
    for kf in keyframes_from or []:
        dicts.append(_keyframe_constraint(kf))
    if root_path:
        dicts.append({"type": "root2d", "frame_indices": [p.frame for p in root_path], "smooth_root_2d": [[p.x, p.z] for p in root_path]})

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
        )
    seconds = round(time.time() - t0, 1)
    if out["posed_joints"].ndim == 3:  # unbatched single sample
        out = {k: v[None] for k, v in out.items()}

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
                    "generated_s": seconds, "summary": metrics.one_line(m)},
        }
        (d / "meta.json").write_text(json.dumps(meta, indent=2))
        if dicts:
            (d / "constraints.json").write_text(json.dumps(dicts))
        takes.append({"take_id": take_id, "summary": metrics.one_line(m), "path": str(d)})
    others = f" (+{len(takes) - 1} more: {', '.join(t['take_id'] for t in takes[1:])})" if len(takes) > 1 else ""
    _live(takes[0]["take_id"], f"New take {takes[0]['take_id']}", takes[0]["summary"] + others, play=True)
    return {"generation_seconds": seconds, "takes": takes}


@mcp.tool()
def render_motion(
    take_id: str,
    frames: Annotated[int | list[int], Field(description="Number of evenly spaced frames, or exact frame numbers (30 fps).")] = 8,
    view: Annotated[str, Field(description="side (faces right), front, three-quarter, or top.")] = "side",
) -> list:
    """Render frames of a take as one contact-sheet image (body mesh, 0.5 m height guides, frame labels)."""
    motion = _load(take_id)
    picked = render.pick_frames(len(motion["posed_joints"]), frames)
    img = render.contact_sheet(motion, picked, view=view, cols=min(4, len(picked)))
    data = render.png_bytes(img)
    path = _take_dir(take_id) / f"sheet-{view}-{'-'.join(map(str, picked[:12]))}.png"
    path.write_bytes(data)
    _live(take_id, f"Claude is looking at {take_id}", f"Frames {picked}, {view} view", frames=picked)
    return [Image(data=data, format="png"), f"{take_id}: frames {picked}, view {view}. Saved {path}"]


@mcp.tool()
def analyze_motion(take_id: str) -> dict:
    """Measure a take: peak heights (and frame) of head, hands and feet relative to the head,
    root travel, heading change, lowest point, foot contact, and jitter. Y-up metres, 30 fps."""
    m = metrics.analyze(_load(take_id))
    _live(take_id, f"Measuring {take_id}", metrics.one_line(m))
    return m


@mcp.tool()
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


@mcp.tool()
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
            "keyframes_from": meta.get("mcp", {}).get("keyframes_from"), "summary": meta.get("mcp", {}).get("summary"),
        })
    return out


def main():
    mcp.run()


if __name__ == "__main__":
    main()
