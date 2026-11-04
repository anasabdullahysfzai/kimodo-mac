"""Make the Kimodo web demo follow the MCP server live.

The MCP server writes <takes>/_live.json on every generate / render / analyze call. This watcher
(started by the demo, see kimodo.patch) loads that take into every open demo tab, plays new takes,
and steps through the frames Claude is inspecting. Set KIMODO_LIVE=0 to turn it off.

It also adds the demo's Saved Animations, Takes and Edit Motion panels. Their buttons call the same functions as the
MCP tools, and results come back through the same file, so they load like any other edit.
"""

import json
import os
import threading
import time
from pathlib import Path

import numpy as np

LIVE = Path(os.environ.get("KIMODO_MCP_TAKES", Path.home() / "kimodo-local" / "motions")).expanduser() / "_live.json"
STEP_S = 0.6  # how long each inspected frame stays on screen
SAVED = Path(os.environ.get("KIMODO_SAVED", Path.home() / "kimodo-local" / "approved-animations")).expanduser()
TAKES = LIVE.parent
_panels = {}  # client id -> function that refreshes the panels' dropdowns
_current = {}  # client id -> folder of the motion last loaded in that tab
_busy = set()  # client ids with an edit running
_resets = {}  # client id -> function that sets the edit range to the whole motion


_status_handles = {}  # client id -> the notification showing generation progress


def _show_status(client, cid: int, event: dict) -> None:
    handle = _status_handles.get(cid)
    if handle is None:
        _status_handles[cid] = client.add_notification(title="Working", body=event["status"], loading=True, with_close_button=False, auto_close_seconds=None)
    else:
        handle.body = event["status"]


def saved_names() -> list[str]:
    """Saved animations: folders holding a motion.npz, sorted by name."""
    if not SAVED.is_dir():
        return []
    return sorted(d.name for d in SAVED.iterdir() if (d / "motion.npz").exists())


def take_names() -> list[str]:
    """Takes, newest first."""
    return sorted((d.name for d in TAKES.glob("t[0-9][0-9][0-9]-*") if (d / "motion.npz").exists()), reverse=True)


def matching_take(positions: np.ndarray, candidates: list[str]) -> str | None:
    """The first candidate take whose joint positions are the ones given, else None."""
    for take in candidates:
        path = TAKES / take / "motion.npz"
        if path.exists():
            saved = np.load(path)["posed_joints"]
            if saved.shape == positions.shape and np.abs(saved - positions).max() < 1e-3:
                return take
    return None


def _view_take(session, cid: int) -> str:
    """The take holding the motion in this tab's view, saving the motion as a new take if there is none."""
    from . import server

    if not session.motions:
        raise ValueError("There's no motion in the view. Generate one or open a take first.")
    motion = list(session.motions.values())[0]
    pos = motion.joints_pos.detach().cpu().numpy()
    if pos.ndim != 3 or pos.shape[1] != 77:
        raise ValueError("Editing only supports the SOMA skeleton.")
    candidates = []
    current = Path(_current[cid]) if cid in _current else None
    if current is not None:
        source = current / "source-take.txt"
        candidates = [source.read_text().strip()] if source.exists() else [current.name]
    take = matching_take(pos, candidates)
    if take:
        return take
    spans = sorted(session.client.timeline._prompts.values(), key=lambda x: x.start_frame)
    prompts = [(x.text, x.end_frame - x.start_frame + (1 if i == len(spans) - 1 else 0)) for i, x in enumerate(spans)]
    arrays = {"posed_joints": pos, "global_rot_mats": motion.joints_rot.detach().cpu().numpy(), "local_rot_mats": motion.joints_local_rot.detach().cpu().numpy()}
    take = server.import_motion(arrays, prompts)
    _current[cid] = str(TAKES / take)
    return take


def add_saved_panel(demo, client) -> None:
    """Add the Saved Animations, Takes and Edit Motion panels to a client's GUI."""
    cid = client.client_id
    none = "<none saved>"
    with client.gui.add_folder("Saved Animations", expand_by_default=True):
        names = saved_names()
        dropdown = client.gui.add_dropdown("Animation", options=names or [none], initial_value=(names or [none])[0])
        button = client.gui.add_button("Open Animation", hint="Load the selected saved animation.", disabled=not names)
        save_name = client.gui.add_text("Save As", initial_value="", hint="Letters, digits, - and _.")
        save_replace = client.gui.add_checkbox("Replace Existing", initial_value=False)
        save_button = client.gui.add_button("Save Animation", hint="Save the motion in the view, with a BVH and a video.")
    no_takes = "<no takes>"
    with client.gui.add_folder("Takes", expand_by_default=True):
        takes = take_names()
        take_dropdown = client.gui.add_dropdown("Take", options=takes or [no_takes], initial_value=(takes or [no_takes])[0], hint="Every generation and edit, newest first. Open an earlier one to undo.")
        take_button = client.gui.add_button("Open Take", disabled=not takes)
    with client.gui.add_folder("Edit Motion", expand_by_default=False):
        start = client.gui.add_number("Start Frame", initial_value=0, min=0, step=1)
        end = client.gui.add_number("End Frame", initial_value=0, min=0, step=1)
        playhead = client.gui.add_button_group("Set From Playhead", ["Start", "End"])
        trim = client.gui.add_button("Trim To Frames", hint="Keep only these frames.")
        sigma = client.gui.add_slider("Smooth Strength", min=1.0, max=6.0, step=0.5, initial_value=2.0)
        smooth = client.gui.add_button("Smooth Frames", hint="Soften jerks in these frames. Also softens intended peaks.")
        speed = client.gui.add_slider("Speed", min=0.25, max=3.0, step=0.05, initial_value=1.5)
        retime = client.gui.add_button("Retime Frames", hint="Play these frames at this speed. Poses don't change; the length does.")
        feet = client.gui.add_button("Plant Feet", hint="Stop planted feet sliding, across the whole motion.")
        ground = client.gui.add_button("Fix Ground", hint="Raise the body wherever it goes through the floor, across the whole motion.")
        prompt = client.gui.add_text("Redo Prompt", initial_value="", hint="What should happen in these frames. Empty: the motion's own prompts.")
        upper = client.gui.add_checkbox("Upper Body Only", initial_value=False, hint="Keep the hips and legs exactly as they are.")
        seed = client.gui.add_number("Redo Seed", initial_value=42, step=1)
        redo = client.gui.add_button("Redo Frames", hint="Run the model on these frames only; every other frame stays exactly as it is.")

    def last_frame(session) -> int:
        """The motion's own last frame. The timeline can be longer, since it shows short prompts as 2 s."""
        return list(session.motions.values())[0].length - 1 if session.motions else 0

    def frames(session) -> tuple[int, int]:
        last = last_frame(session)
        return min(max(int(start.value), 0), last), min(max(int(end.value), 0), last)

    def action(control, label: str, run) -> None:
        """Run an edit on the motion in the view, off the GUI thread. run(take id, session) does the work."""

        @control.on_click
        def _(event) -> None:
            if cid in _busy:
                client.add_notification(title="Please wait", body="Another edit is still running.", auto_close_seconds=4.0, color="yellow")
                return
            _busy.add(cid)

            def work() -> None:
                note = client.add_notification(title=label, body="Working...", loading=True, with_close_button=False, auto_close_seconds=None)
                try:
                    session = demo.client_sessions[cid]
                    run(_view_take(session, cid), session)
                except Exception as e:  # show the reason; never take the demo down
                    client.add_notification(title=f"{label} failed", body=str(e).removeprefix("ValueError: "), auto_close_seconds=12.0, color="red")
                    if cid in _status_handles:
                        _status_handles.pop(cid).remove()
                finally:
                    note.remove()
                    _busy.discard(cid)

            threading.Thread(target=work, daemon=True).start()

    def blend(a: int, b: int) -> int:
        return max(min(6, (b - a + 1) // 2), 1)

    def run_trim(take, session):
        from . import server

        a, b = frames(session)
        server.trim_take(take, b, a)

    def run_smooth(take, session):
        from . import server

        a, b = frames(session)
        server.smooth_motion(take, a, b, float(sigma.value), None, blend(a, b))

    def run_retime(take, session):
        from . import server

        a, b = frames(session)
        server.retime_motion(take, [server.SpeedRange(start=a, end=b, speed=float(speed.value))])

    def run_feet(take, session):
        from . import server

        server.plant_feet(take)

    def run_ground(take, session):
        from . import server

        server.fix_ground(take)

    def run_redo(take, session):
        from . import server

        a, b = frames(session)
        with demo._generation_lock:  # one model run at a time, shared with Generate
            server._model = demo.load_model(session.model_name).model
            server.regenerate_range(take, a, b, prompt=prompt.value.strip() or None, seed=int(seed.value), blend_frames=blend(a, b), redo_joints="above_hips" if upper.value else None)

    def run_save(take, session):
        from . import server

        name = save_name.value.strip()
        if name in saved_names() and not save_replace.value:
            raise ValueError(f"'{name}' is already saved. Tick Replace Existing, or use another name.")
        server.save_animation(take, name, overwrite=bool(save_replace.value))
        client.add_notification(title="Animation saved", body=name, auto_close_seconds=5.0, color="green")

    for control, label, run in ((trim, "Trim", run_trim), (smooth, "Smooth", run_smooth), (retime, "Retime", run_retime), (feet, "Plant Feet", run_feet),
                                (ground, "Fix Ground", run_ground), (redo, "Redo Frames", run_redo), (save_button, "Save Animation", run_save)):
        action(control, label, run)

    @playhead.on_click
    def _(event) -> None:
        frame = int(demo.client_sessions[cid].frame_idx)
        (start if event.target.value == "Start" else end).value = frame

    def open_motion(event_client, path: str) -> None:
        demo.live_loaders[cid](event_client, path)
        _current[cid] = path
        reset_range()

    def reset_range() -> None:
        start.value, end.value = 0, last_frame(demo.client_sessions[cid])

    @button.on_click
    def _(event) -> None:
        if dropdown.value != none:
            open_motion(event.client, str(SAVED / dropdown.value))

    @take_button.on_click
    def _(event) -> None:
        if take_dropdown.value != no_takes:
            open_motion(event.client, str(TAKES / take_dropdown.value))

    def refresh(select_take: str | None = None) -> None:
        current = saved_names()
        keep = dropdown.value if dropdown.value in current else None
        dropdown.options = current or [none]
        dropdown.value = keep or (current or [none])[0]
        button.disabled = not current
        all_takes = take_names()
        keep = select_take if select_take in all_takes else take_dropdown.value if take_dropdown.value in all_takes else None
        take_dropdown.options = all_takes or [no_takes]
        take_dropdown.value = keep or (all_takes or [no_takes])[0]
        take_button.disabled = not all_takes

    _panels[cid] = refresh
    _resets[cid] = reset_range


def _apply(demo, cid: int, event: dict) -> None:
    session = demo.client_sessions[cid]
    client = session.client
    if event.get("refresh_saved"):
        if cid in _panels:
            _panels[cid]()
        return
    if "status" in event:
        _show_status(client, cid, event)
        return
    if cid in _status_handles:
        _status_handles.pop(cid).remove()
    if event.get("reload") or _current.get(cid) != event["path"]:
        demo.live_loaders[cid](client, event["path"])  # the demo's own Load Example (motion, prompts, settings)
        _current[cid] = event["path"]
        if cid in _resets:
            _resets[cid]()
    if cid in _panels:
        _panels[cid](Path(event["path"]).name)
    client.add_notification(title=event["title"], body=event.get("body", ""), auto_close_seconds=8.0, color="blue")
    if event.get("play"):
        demo.set_frame(cid, 0)
        session.playing = True
    elif event.get("frames"):
        session.playing = False
        for f in event["frames"]:
            demo.set_frame(cid, min(int(f), session.max_frame_idx))
            time.sleep(STEP_S)


def _watch(demo) -> None:
    seen = None
    while True:
        time.sleep(0.5)
        try:
            stamp = LIVE.stat().st_mtime_ns
            if stamp == seen:
                continue
            first = seen is None
            seen = stamp
            if first and time.time() - stamp / 1e9 > 10:
                continue  # don't replay an old event when the demo starts
            event = json.loads(LIVE.read_text())
        except (OSError, ValueError):
            continue
        for cid in list(demo.live_loaders):
            if not demo.client_active(cid):
                demo.live_loaders.pop(cid, None)
                _current.pop(cid, None)
                _panels.pop(cid, None)
                _resets.pop(cid, None)
                continue
            try:
                _apply(demo, cid, event)
            except Exception as e:  # never take the demo down
                print(f"[kimodo-mcp live] {type(e).__name__}: {e}")


def start(demo) -> None:
    if os.environ.get("KIMODO_LIVE", "1") == "0":
        return
    threading.Thread(target=_watch, args=(demo,), daemon=True, name="kimodo-mcp-live").start()
    print(f"[kimodo-mcp live] following {LIVE}")
