"""Retarget a Kimodo motion onto a rigged character. Runs inside Blender, not the Kimodo venv:

    Blender -b --factory-startup --python blender_retarget.py -- job.json

SOMA's rest pose has identity global rotations, so a motion's global rotation at a frame is that joint's change from
rest. The rig's bone gets the same change in world space, after a per-bone correction that lines up its rest direction
with SOMA's, so T-pose and A-pose rigs both work. Bones with no SOMA match just follow their parents.
"""

import json
import re
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

KIMODO_TO_BLENDER = Matrix(((1, 0, 0), (0, 0, -1), (0, 1, 0)))  # Y-up facing +Z -> Z-up facing -Y
CHAINS = [
    ["Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head"],
    ["LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand"],
    ["RightShoulder", "RightArm", "RightForeArm", "RightHand"],
    ["LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase"],
    ["RightLeg", "RightShin", "RightFoot", "RightToeBase"],
]
REQUIRED = ["Hips", "Head", "LeftArm", "RightArm", "LeftForeArm", "RightForeArm", "LeftHand", "RightHand", "LeftLeg", "RightLeg", "LeftShin", "RightShin", "LeftFoot", "RightFoot"]
CENTRE = {"Hips": ["hips", "pelvis", "hip"], "Spine1": ["spine01", "spine"], "Spine2": ["spine02", "spine1"], "Chest": ["spine03", "spine2", "chest"], "Neck1": ["neck01", "neck1", "neck"], "Neck2": ["neck02", "neck2"], "Head": ["head"]}
LIMBS = {"Shoulder": ["shoulder", "clavicle"], "Arm": ["upperarm", "arm"], "ForeArm": ["forearm", "lowerarm"], "Hand": ["hand"], "Leg": ["upleg", "thigh", "upperleg"], "Shin": ["shin", "calf", "lowerleg"], "Foot": ["foot"], "ToeBase": ["toebase", "ball", "toe"]}


def clean(name: str) -> str:
    name = name.split(":")[-1].lower().replace("mixamorig", "")
    return re.sub(r"[^a-z0-9]", "", name)


def auto_map(bone_names: list[str]) -> dict:
    """SOMA joint -> bone, from common naming styles (Mixamo, Unreal, Unity, Rigify-like limbs)."""
    found = {}
    for b in bone_names:
        found.setdefault(clean(b), b)
    out = {soma: next((found[a] for a in aliases if a in found), None) for soma, aliases in CENTRE.items()}
    for limb, aliases in LIMBS.items():
        for side, s, sh in (("Left", "left", "l"), ("Right", "right", "r")):
            forms = [f for a in aliases for f in (s + a, a + s, sh + a, a + sh)]
            if limb == "Leg" and not any(f in found for f in forms):  # "LeftLeg" is the thigh only when there is a separate shin
                forms = [s + "leg", "leg" + s, sh + "leg", "leg" + sh]
            if limb == "Shin" and not any(f in found for f in forms):  # Mixamo's LeftLeg is the shin, beside LeftUpLeg
                forms = [s + "leg", "leg" + s, sh + "leg", "leg" + sh]
            out[side + limb] = next((found[f] for f in forms if f in found), None)
    return {k: v for k, v in out.items() if v}


def shortest(a: Vector, b: Vector) -> Matrix:
    return a.normalized().rotation_difference(b.normalized()).to_matrix()


def main():
    job = json.loads(Path(sys.argv[sys.argv.index("--") + 1]).read_text())
    data = np.load(job["motion"])
    names = list(job["names"])
    idx = {n: i for i, n in enumerate(names)}
    gr, pj, rest = data["global_rot_mats"], data["posed_joints"], data["rest_joints"]

    bpy.ops.wm.read_factory_settings(use_empty=True)
    rig = job["rig"]
    if rig.lower().endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=rig, automatic_bone_orientation=False, ignore_leaf_bones=False)
    else:
        bpy.ops.import_scene.gltf(filepath=rig)
    armatures = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    if not armatures:
        raise SystemExit("KIMODO_ERROR: no armature in the rig file")
    arm = max(armatures, key=lambda o: len(o.data.bones))
    arm.animation_data_clear()
    bones = arm.data.bones

    mapping = auto_map([b.name for b in bones])
    mapping.update(job.get("joint_map") or {})
    bad = [v for v in mapping.values() if v not in bones]
    if bad:
        raise SystemExit(f"KIMODO_ERROR: joint_map names bones the rig doesn't have: {bad}")
    missing = [j for j in REQUIRED if j not in mapping]
    if missing:
        raise SystemExit(f"KIMODO_ERROR: no bone found for {missing}. Pass joint_map. Bones in this rig: {[b.name for b in bones][:120]}")

    a_loc, a_rot, _ = arm.matrix_world.decompose()
    a_rot3 = a_rot.to_matrix()
    a_inv = arm.matrix_world.inverted()
    world_head = lambda b: arm.matrix_world @ b.head_local
    bone_of = {j: bones[n] for j, n in mapping.items()}

    # Turn SOMA to the rig's facing, and scale by hip height.
    across = world_head(bone_of["LeftArm"]) - world_head(bone_of["RightArm"])
    yaw = Matrix.Rotation(float(np.arctan2(across.y, across.x)), 3, "Z")
    W = yaw @ KIMODO_TO_BLENDER
    W_inv = W.inverted()
    toe = "LeftToeBase" if "LeftToeBase" in bone_of else "LeftFoot"
    rig_height = (world_head(bone_of["Hips"]) - world_head(bone_of[toe])).z
    scale = rig_height / float(rest[idx["Hips"], 1] - rest[idx[toe], 1])
    floor = 0.0  # the file's ground plane, which engines and other clips for the rig also use; a rig's rest mesh can sit off it

    # Per-bone correction: rig rest direction -> SOMA rest direction, both in world space.
    correction = {}
    for chain in CHAINS:
        used = [j for j in chain if j in bone_of]
        previous = Matrix.Identity(3)
        for k, j in enumerate(used):
            if k + 1 < len(used) and j != "Hips":
                child = used[k + 1]
                src = W @ Vector(rest[idx[child]] - rest[idx[j]])
                dst = world_head(bone_of[child]) - world_head(bone_of[j])
                previous = shortest(dst, src)
            correction[j] = previous  # hands, head and toes keep their parent's correction
    joint_of = {b.name: j for j, b in bone_of.items()}

    def walk(b):
        yield b
        for c in b.children:
            yield from walk(c)

    order = [b for root in bones if root.parent is None for b in walk(root)]
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"
        pb.matrix_basis = Matrix.Identity(4)
    T = len(gr)
    scene = bpy.context.scene
    scene.render.fps, scene.frame_start, scene.frame_end = job.get("fps", 30), 0, T - 1
    tracked = {j: np.zeros((T, 3)) for j in bone_of}
    hips_name = bone_of["Hips"].name
    def bake(lift) -> None:
        """Key every frame, with the hips raised by lift[f]."""
        for f in range(T):
            pose = {}
            for b in order:
                offset = b.parent.matrix_local.inverted() @ b.matrix_local if b.parent else b.matrix_local
                chain = pose[b.parent.name] @ offset if b.parent else offset
                j = joint_of.get(b.name)
                if j is None:
                    pose[b.name] = chain
                    continue
                S = W @ Matrix(gr[f, idx[j]].tolist()) @ W_inv
                world_rot = S @ correction[j] @ a_rot3 @ b.matrix_local.to_3x3()
                desired = (a_rot3.inverted() @ world_rot).to_4x4()
                if b.name == hips_name:
                    p = Vector(pj[f, idx["Hips"]]) * scale
                    world = W @ p + Vector((world_head(b).x, world_head(b).y, floor + lift[f]))
                    desired.translation = a_inv @ world
                else:
                    desired.translation = chain.translation
                pb = arm.pose.bones[b.name]
                pb.matrix_basis = chain.inverted() @ desired
                pb.keyframe_insert("rotation_quaternion", frame=f)
                if b.name == hips_name:
                    pb.keyframe_insert("location", frame=f)
                pose[b.name] = desired
                tracked[j][f] = arm.matrix_world @ desired.translation

    def mesh_lowest() -> float:
        """The lowest world height of any mesh at the current frame."""
        depsgraph = bpy.context.evaluated_depsgraph_get()
        low = np.inf
        for o in (o for o in bpy.data.objects if o.type == "MESH"):
            ev = o.evaluated_get(depsgraph)
            m = ev.to_mesh()
            co = np.empty(len(m.vertices) * 3)
            m.vertices.foreach_get("co", co)
            mw = np.array(o.matrix_world)
            low = min(low, float((co.reshape(-1, 3) @ mw[2, :3] + mw[2, 3]).min()))
            ev.to_mesh_clear()
        return low

    def every_frame_lowest() -> np.ndarray:
        lows = []
        for f in range(T):
            scene.frame_set(f)
            lows.append(mesh_lowest())
        return np.array(lows)

    bake(np.zeros(T))
    lift_report = {}
    if job.get("lift_to_floor") and any(o.type == "MESH" for o in bpy.data.objects):
        before = every_frame_lowest()
        need = np.clip(floor - before, 0.0, None)
        w = 8  # frames either side: taking the widest need first means the eased lift still covers every frame
        held = np.array([need[max(i - w, 0) : i + w + 1].max() for i in range(T)])
        lift = np.array([held[max(i - w, 0) : i + w + 1].mean() for i in range(T)])
        bake(lift)
        after = every_frame_lowest()
        lift_report = {"lift": {"max_m": round(float(lift.max()), 3), "first_frame_m": round(float(lift[0]), 3), "last_frame_m": round(float(lift[-1]), 3),
                                "lowest_before_m": round(float(before.min()), 3), "lowest_after_m": round(float(after.min()), 3), "largest_gap_above_floor_m": round(float((after - floor).max()), 3)}}
    out = Path(job["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    stem = job["stem"]
    if job["format"] == "glb":
        exported = out / f"{stem}.glb"
        bpy.ops.export_scene.gltf(filepath=str(exported), export_format="GLB", export_animations=True)
    else:
        exported = out / f"{stem}.fbx"
        bpy.ops.export_scene.fbx(filepath=str(exported), bake_anim=True, bake_anim_use_all_actions=False, bake_anim_use_nla_strips=False, bake_anim_simplify_factor=0.0, add_leaf_bones=False, object_types={"ARMATURE", "MESH"})

    # Render on a floor, with the camera following the hips.
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    report = {"mapped": mapping, "unmapped_bones": [b.name for b in bones if b.name not in joint_of], "scale_vs_soma": round(scale, 3), "exported": str(exported), "has_mesh": bool(meshes), "floor_z": round(floor, 3), **lift_report}
    frames, images, sinks = job["frames"], [], {}
    if meshes:
        thickness = 0.01 * rig_height
        bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, floor - thickness / 2))  # a slab, so level cameras see it as a line
        bpy.context.object.scale = (40 * rig_height, 40 * rig_height, thickness)
        floor_material = bpy.data.materials.new("floor")
        floor_material.diffuse_color = (0.82, 0.84, 0.9, 1.0)
        bpy.context.object.data.materials.append(floor_material)
        scene.world = bpy.data.worlds.new("bg")
        scene.world.color = (0.9, 0.91, 0.95)
        scene.display.shading.background_type = "WORLD"
        cam_data = bpy.data.cameras.new("cam")
        cam_data.type = "ORTHO"
        height = (world_head(bone_of["Head"]) - world_head(bone_of[toe])).z
        cam_data.ortho_scale = height * 1.9
        cam = bpy.data.objects.new("cam", cam_data)
        scene.collection.objects.link(cam)
        scene.camera = cam
        scene.render.engine = "BLENDER_WORKBENCH"
        scene.display.shading.light = "STUDIO"
        scene.display.shading.color_type = "MATERIAL"
        scene.render.resolution_x = scene.render.resolution_y = job.get("size", 360)
        scene.render.image_settings.file_format = "PNG"
        view = W @ Vector(job["camera_dir"])
        for f in frames:
            scene.frame_set(f)
            target = Vector(tracked["Hips"][f])
            cam.location = target + view.normalized() * 10
            cam.rotation_euler = (-view).to_track_quat("-Z", "Y").to_euler()
            bpy.context.view_layer.update()
            path = out / f"{stem}-f{f}.png"
            scene.render.filepath = str(path)
            bpy.ops.render.render(write_still=True)
            images.append(str(path))
            low = mesh_lowest()
            sinks[f] = round(max(floor - low, 0.0), 3)
    report |= {"images": images, "frames": frames, "mesh_below_floor_m": sinks}
    np.savez(out / f"{stem}-joints.npz", **tracked)
    (out / f"{stem}-report.json").write_text(json.dumps(report, indent=1))


try:
    main()
except SystemExit as e:
    print(e, flush=True)
    sys.exit(3)
