"""Make the Kimodo web demo follow the MCP server live.

The MCP server writes <takes>/_live.json on every generate / render / analyze call. This watcher
(started by the demo, see demo.patch) loads that take into every open demo tab, plays new takes,
and steps through the frames Claude is inspecting. Set KIMODO_LIVE=0 to turn it off.
"""

import json
import os
import threading
import time
from pathlib import Path

LIVE = Path(os.environ.get("KIMODO_MCP_TAKES", Path.home() / "kimodo-local" / "motions")).expanduser() / "_live.json"
STEP_S = 0.6  # how long each inspected frame stays on screen
SAVED = Path(os.environ.get("KIMODO_SAVED", Path.home() / "kimodo-local" / "approved-animations")).expanduser()
_panels = {}  # client id -> function that refreshes the Saved Animations dropdown


_status_handles = {}  # client id -> the notification showing generation progress


def _show_status(client, cid: int, event: dict) -> None:
    handle = _status_handles.get(cid)
    if handle is None:
        _status_handles[cid] = client.add_notification(title="Claude is working", body=event["status"], loading=True, with_close_button=False, auto_close_seconds=None)
    else:
        handle.body = event["status"]


def saved_names() -> list[str]:
    """Saved animations: folders holding a motion.npz, sorted by name."""
    if not SAVED.is_dir():
        return []
    return sorted(d.name for d in SAVED.iterdir() if (d / "motion.npz").exists())


def add_saved_panel(demo, client) -> None:
    """Add the Saved Animations dropdown to a client's panel. It refreshes when an animation is saved."""
    cid = client.client_id
    none = "<none saved>"
    with client.gui.add_folder("Saved Animations", expand_by_default=True):
        names = saved_names()
        dropdown = client.gui.add_dropdown("Animation", options=names or [none], initial_value=(names or [none])[0])
        button = client.gui.add_button("Open Animation", hint="Load the selected saved animation.", disabled=not names)

    @button.on_click
    def _(event) -> None:
        if dropdown.value != none:
            demo.live_loaders[cid](event.client, str(SAVED / dropdown.value))

    def refresh() -> None:
        current = saved_names()
        keep = dropdown.value if dropdown.value in current else None
        dropdown.options = current or [none]
        dropdown.value = keep or (current or [none])[0]
        button.disabled = not current

    _panels[cid] = refresh


def _apply(demo, cid: int, event: dict, loaded: dict) -> None:
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
    if event.get("reload") or loaded.get(cid) != event["path"]:
        demo.live_loaders[cid](client, event["path"])  # the demo's own Load Example (motion, prompts, settings)
        loaded[cid] = event["path"]
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
    seen, loaded = None, {}
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
                loaded.pop(cid, None)
                _panels.pop(cid, None)
                continue
            try:
                _apply(demo, cid, event, loaded)
            except Exception as e:  # never take the demo down
                print(f"[kimodo-mcp live] {type(e).__name__}: {e}")


def start(demo) -> None:
    if os.environ.get("KIMODO_LIVE", "1") == "0":
        return
    threading.Thread(target=_watch, args=(demo,), daemon=True, name="kimodo-mcp-live").start()
    print(f"[kimodo-mcp live] following {LIVE}")
