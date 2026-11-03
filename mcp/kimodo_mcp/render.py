"""Headless renderer for Kimodo SOMA motions: numpy + PIL, no OpenGL or browser.

Draws the skinned SOMA body as flat-shaded polygons (orthographic camera,
back-face culling, far-to-near painter's order) onto a contact sheet of frames.
"""

import io
from functools import lru_cache

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

FPS = 30
BODY = np.array([154, 166, 232], dtype=np.float32)  # Kimodo viewer's periwinkle
BG = (244, 245, 250)
GRID = (214, 218, 232)
INK = (40, 44, 70)

# Camera direction (origin -> camera) and the world "up" hint. The character starts facing +Z.
VIEWS = {
    "side": ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),  # faces right
    "front": ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    "three-quarter": ((-0.7, 0.35, 0.7), (0.0, 1.0, 0.0)),
    "top": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),  # facing direction points up on screen
}


@lru_cache(maxsize=2)
def _skin(nbjoints: int):
    from kimodo.skeleton.registry import build_skeleton
    from kimodo.viz.soma_skin import SOMASkin

    return SOMASkin(build_skeleton(nbjoints))


def skin_vertices(global_rot_mats: np.ndarray, posed_joints: np.ndarray):
    """[T,J,3,3] global rotations + [T,J,3] positions -> ([T,V,3] vertices, [F,3] faces)."""
    skin = _skin(posed_joints.shape[1])
    verts = skin.skin(
        torch.as_tensor(global_rot_mats, dtype=torch.float32),
        torch.as_tensor(posed_joints, dtype=torch.float32),
        rot_is_global=True,
    )
    return verts.numpy(), skin.faces.numpy()


def _camera(view: str):
    if view not in VIEWS:
        raise ValueError(f"view must be one of {sorted(VIEWS)}")
    d, hint = (np.array(x) for x in VIEWS[view])
    forward = -d / np.linalg.norm(d)
    right = np.cross(forward, hint)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return forward, right, up


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def contact_sheet(motion: dict, frames: list[int], view: str = "side", tile: int = 300, cols: int = 4) -> Image.Image:
    """Render the given frames of one motion sample into a labelled grid.

    All tiles share one scale and ground level so heights compare across frames.
    Light horizontal guides mark every 0.5 m in level views (side, front). Tilted views
    (three-quarter, top) have no true height axis, so they get no guides and are centred on the root.
    """
    pj, gr = motion["posed_joints"], motion["global_rot_mats"]
    verts, faces = skin_vertices(gr[frames], pj[frames])
    forward, right, up = _camera(view)
    light = -forward + np.array([-0.3, 0.8, 0.2])
    light /= np.linalg.norm(light)

    u = verts @ right  # [N,V]
    v = verts @ up
    depth = verts @ forward
    root = pj[frames, 0]
    cu, cv = root @ right, root @ up  # per-tile centring on the root
    level = abs(up[1] - 1) < 1e-6  # camera not tilted: screen-up is world height
    half_w = np.abs(u - cu[:, None]).max() + 0.15
    if not level:
        half_h = np.abs(v - cv[:, None]).max() + 0.15
        span_v = 2 * half_h
    else:
        v_lo, v_hi = min(v.min(), 0.0) - 0.08, v.max() + 0.12
        span_v = v_hi - v_lo
    label_h = 22
    scale = min((tile - 16) / (2 * half_w), (tile - 16 - label_h) / span_v)

    rows = (len(frames) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile, rows * tile), BG)
    draw = ImageDraw.Draw(sheet)
    font, small = _font(15), _font(11)

    for k, f in enumerate(frames):
        ox, oy = (k % cols) * tile, (k // cols) * tile
        cx = ox + tile / 2

        def to_px(uu, vv):
            if not level:
                return cx + (uu - cu[k]) * scale, oy + label_h + (tile - label_h) / 2 - (vv - cv[k]) * scale
            return cx + (uu - cu[k]) * scale, oy + tile - 8 - (vv - v_lo) * scale

        if level:
            for h in np.arange(0.0, v_hi, 0.5):
                y = to_px(0, h)[1]
                draw.line([(ox + 4, y), (ox + tile - 4, y)], fill=GRID if h else INK, width=1)
                if h:
                    draw.text((ox + 6, y - 13), f"{h:.1f} m", fill=(150, 156, 186), font=small)
        else:
            gx, gy = to_px(cu[k], cv[k])
            draw.ellipse([gx - 4, gy - 4, gx + 4, gy + 4], outline=GRID)

        tri = verts[k][faces]  # [F,3,3]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
        visible = n @ forward < 0
        order = np.argsort(-depth[k][faces[visible]].mean(1))  # far to near
        shade = 0.4 + 0.6 * np.clip(n[visible] @ light, 0, 1)
        colors = (BODY[None] * shade[:, None]).clip(0, 255).astype(np.uint8)
        px = np.stack(to_px(u[k], v[k]), axis=-1)[faces[visible]]  # [Fv,3,2]
        for i in order:
            draw.polygon([tuple(p) for p in px[i]], fill=tuple(colors[i]))

        draw.text((ox + 8, oy + 4), f"f{f}  {f / FPS:.2f}s", fill=INK, font=font)
        draw.rectangle([ox, oy, ox + tile - 1, oy + tile - 1], outline=GRID)
    return sheet


def pick_frames(num_frames: int, frames) -> list[int]:
    """An int means that many evenly spaced frames; a list is used as given (clamped)."""
    if isinstance(frames, int):
        return sorted({round(i) for i in np.linspace(0, num_frames - 1, max(1, frames))})
    return [min(max(int(f), 0), num_frames - 1) for f in frames]


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def video(motion: dict, path: str, view: str = "side", size: int = 480, step: int = 1) -> None:
    """Render every `step`-th frame to an mp4 (H.264 via PyAV)."""
    import imageio.v3 as iio

    frames = list(range(0, len(motion["posed_joints"]), step))
    imgs = [np.asarray(contact_sheet(motion, [f], view=view, tile=size, cols=1)) for f in frames]
    iio.imwrite(path, np.stack(imgs), extension=".mp4", fps=FPS / step, codec="h264", plugin="pyav")
