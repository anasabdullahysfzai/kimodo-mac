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


def _apply(demo, cid: int, event: dict, loaded: dict) -> None:
    session = demo.client_sessions[cid]
    client = session.client
    if loaded.get(cid) != event["path"]:
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
