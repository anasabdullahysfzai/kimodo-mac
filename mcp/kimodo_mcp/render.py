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


CURVE_COLORS = [(214, 48, 49), (9, 132, 227), (0, 148, 100), (225, 112, 40), (108, 92, 231), (120, 120, 130), (232, 67, 147), (0, 160, 170)]


def motion_curves(motion: dict, joints: list[str], start: int = 0, end: int | None = None, marks: list[int] | None = None, width: int = 1200) -> Image.Image:
    """Speed, height and front-back position of joints over time, one panel each, so motion can be read from one image.

    marks: frames to draw as dashed lines, e.g. prompt boundaries. Front-back is measured along the starting
    facing, relative to the hips, so a rocking torso shows up even when the character travels.
    """
    from .metrics import _joint_names

    pj = np.asarray(motion["posed_joints"], dtype=np.float64)
    names = _joint_names(pj.shape[1])
    missing = [j for j in joints if j not in names]
    if missing:
        raise ValueError(f"Unknown joints {missing}")
    end = len(pj) - 1 if end is None else min(end, len(pj) - 1)
    frames = np.arange(start, end + 1)
    across = pj[0, names.index("LeftLeg")] - pj[0, names.index("RightLeg")]
    fwd = np.array([-across[2], across[0]]) / (np.linalg.norm(across[[0, 2]]) + 1e-9)
    hips = pj[:, names.index("Hips")]
    panels = []
    for j in joints:
        p = pj[:, names.index(j)]
        speed = np.r_[np.linalg.norm(np.diff(p, axis=0), axis=1) * FPS_RENDER, 0.0]
        panels.append((j, speed[frames], p[frames, 1], ((p - hips)[:, [0, 2]] @ fwd)[frames]))
    titles = ["speed (m/s)", "height (m)", "front-back vs hips (m, + = forward)"]
    panel_h, top, left, right = 210, 26, 62, 20
    img = Image.new("RGB", (width, 3 * panel_h + 30), "white")
    draw = ImageDraw.Draw(img)
    font, small = _font(14), _font(11)
    x_at = lambda f: left + (f - start) / max(end - start, 1) * (width - left - right)
    for k, title in enumerate(titles):
        y0 = k * panel_h + top
        y1 = y0 + panel_h - top - 16
        series = [p[k + 1] for p in panels]
        lo, hi = min(s.min() for s in series), max(s.max() for s in series)
        pad = max((hi - lo) * 0.08, 0.02)
        lo, hi = lo - pad, hi + pad
        y_at = lambda v: y1 - (v - lo) / (hi - lo) * (y1 - y0)
        draw.text((left, y0 - 18), title, fill=INK, font=font)
        step = next(s for s in (0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 50) if (hi - lo) / s <= 6)
        for v in np.arange(np.ceil(lo / step) * step, hi, step):
            draw.line([(left, y_at(v)), (width - right, y_at(v))], fill=(232, 234, 240))
            draw.text((6, y_at(v) - 6), f"{v:g}", fill=(90, 94, 115), font=small)
        tick = next(s for s in (5, 10, 15, 30, 60, 90, 150, 300) if (end - start) / s <= 14)
        for f in range(start - start % tick + (tick if start % tick else 0), end + 1, tick):
            draw.line([(x_at(f), y0), (x_at(f), y1)], fill=(232, 234, 240))
            draw.text((x_at(f) - 8, y1 + 2), f"f{f}", fill=(90, 94, 115), font=small)
        for f in marks or []:
            if start <= f <= end:
                for y in range(int(y0), int(y1), 8):
                    draw.line([(x_at(f), y), (x_at(f), y + 4)], fill=(40, 44, 70))
        for c, (_, *vals) in enumerate(panels):
            pts = [(x_at(f), y_at(v)) for f, v in zip(frames, vals[k])]
            draw.line(pts, fill=CURVE_COLORS[c % len(CURVE_COLORS)], width=2)
    x = left + 330
    for c, (j, *_) in enumerate(panels):
        draw.line([(x, 14), (x + 18, 14)], fill=CURVE_COLORS[c % len(CURVE_COLORS)], width=3)
        draw.text((x + 22, 6), j, fill=INK, font=small)
        x += 34 + int(draw.textlength(j, font=small))
    return img


FPS_RENDER = 30


def onion_skin(motion: dict, frames: list[int], view: str = "side", trail_joints: list[str] | None = None, size: int = 760) -> Image.Image:
    """Overlay several poses in one picture (later poses more solid), with the paths of chosen joints.

    The camera stays fixed, so travel is visible, unlike the contact sheet, which recentres each tile.
    """
    from .metrics import _joint_names

    pj, gr = np.asarray(motion["posed_joints"]), np.asarray(motion["global_rot_mats"])
    verts, faces = skin_vertices(gr[frames], pj[frames])
    forward, right, up = _camera(view)
    light = -forward + np.array([-0.3, 0.8, 0.2])
    light /= np.linalg.norm(light)
    names = _joint_names(pj.shape[1])
    first, last = frames[0], frames[-1]
    span = pj[first : last + 1, [names.index(j) for j in (trail_joints or [])]] if trail_joints else np.empty((0, 0, 3))
    all_pts = np.concatenate([verts.reshape(-1, 3), span.reshape(-1, 3)]) if span.size else verts.reshape(-1, 3)
    u, v = all_pts @ right, all_pts @ up
    level = abs(up[1] - 1) < 1e-6
    v_lo = min(v.min(), 0.0) if level else v.min()
    scale = min((size - 40) / max(u.max() - u.min(), 0.1), (size - 60) / max(v.max() - v_lo, 0.1))
    cx, cy = (u.max() + u.min()) / 2, v_lo
    to_px = lambda a, b: (size / 2 + (a - cx) * scale, size - 30 - (b - cy) * scale)
    img = Image.new("RGBA", (size, size), BG + (255,))
    draw = ImageDraw.Draw(img)
    if level:
        for h in np.arange(0.0, v.max() + 0.5, 0.5):
            y = to_px(0, h)[1]
            if y > 0:
                draw.line([(10, y), (size - 10, y)], fill=GRID if h else INK)
                if h:
                    draw.text((12, y - 13), f"{h:.1f} m", fill=(150, 156, 186), font=_font(11))
    for k in range(len(frames)):
        layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        tri = verts[k][faces]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
        vis = n @ forward < 0
        d = (verts[k] @ forward)[faces[vis]].mean(1)
        shade = (0.4 + 0.6 * np.clip(n[vis] @ light, 0, 1))[:, None]
        cols = (BODY[None] * shade).clip(0, 255).astype(np.uint8)
        px = np.stack(to_px(verts[k] @ right, verts[k] @ up), axis=-1)[faces[vis]]
        for i in np.argsort(-d):
            ld.polygon([tuple(q) for q in px[i]], fill=tuple(cols[i]) + (255,))
        alpha = 70 + int(185 * (k + 1) / len(frames)) if k < len(frames) - 1 else 255
        layer.putalpha(layer.getchannel("A").point(lambda a: a * alpha // 255))
        img = Image.alpha_composite(img, layer)
    draw = ImageDraw.Draw(img)
    for c, j in enumerate(trail_joints or []):
        path = pj[first : last + 1, names.index(j)]
        pts = [to_px(q @ right, q @ up) for q in path]
        col = CURVE_COLORS[c % len(CURVE_COLORS)]
        if len(pts) > 1:
            draw.line(pts, fill=col, width=2)
        for q in pts[::5]:  # a dot every 5 frames: wide gaps mean fast, bunched dots mean slow
            draw.ellipse([q[0] - 2, q[1] - 2, q[0] + 2, q[1] + 2], fill=col)
        draw.text((12, 8 + 15 * c), j, fill=col, font=_font(12))
    draw.text((size - 190, 8), f"f{first}-f{last}, {len(frames)} poses", fill=INK, font=_font(13))
    return img.convert("RGB")


def stack_labelled(rows: list[tuple[str, Image.Image]]) -> Image.Image:
    """Stack images vertically with a label above each."""
    width = max(i.width for _, i in rows)
    height = sum(i.height + 24 for _, i in rows)
    out = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(out)
    y = 0
    for label, img in rows:
        draw.text((8, y + 4), label, fill=INK, font=_font(15))
        out.paste(img, (0, y + 24))
        y += img.height + 24
    return out
