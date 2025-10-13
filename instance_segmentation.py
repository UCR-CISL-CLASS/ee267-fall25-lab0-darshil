import carla
import random
import queue
from pathlib import Path
import json

# ---------------- config ----------------
CAM_LOC = carla.Location(x=0.0, y=0.0, z=2.2)
CAM_ROT = carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
IMG_W, IMG_H, FOV = 1280, 720, 90
SPAWN_RADIUS_M = 80.0
TARGET_NPCS = 40
FIXED_DT = 0.05
OUT_PNG = "instance_segmentation.png"
POSE_JSON = "instance_camera_pose.json"

def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(20.0)
    world = client.get_world()

    # --- enable synchronous mode ---
    original = world.get_settings()
    s = world.get_settings()
    s.synchronous_mode = True
    s.fixed_delta_seconds = FIXED_DT
    world.apply_settings(s)

    spawned = []
    try:
        bp_lib = world.get_blueprint_library()

        # --- instance segmentation camera ---
        cam_bp = bp_lib.find("sensor.camera.instance_segmentation")
        cam_bp.set_attribute("image_size_x", str(IMG_W))
        cam_bp.set_attribute("image_size_y", str(IMG_H))
        cam_bp.set_attribute("fov", str(FOV))

        cam_tf = carla.Transform(CAM_LOC, CAM_ROT)
        camera = world.spawn_actor(cam_bp, cam_tf)
        spawned.append(camera)

        # save camera pose for the report
        Path(".").mkdir(parents=True, exist_ok=True)
        with open(POSE_JSON, "w") as f:
            json.dump(
                {
                    "location": {"x": CAM_LOC.x, "y": CAM_LOC.y, "z": CAM_LOC.z},
                    "rotation": {"pitch": CAM_ROT.pitch, "yaw": CAM_ROT.yaw, "roll": CAM_ROT.roll},
                },
                f,
                indent=2,
            )
        print(f"[Info] Saved camera pose -> {POSE_JSON}")

        # --- spawn vehicles within 80 m of the camera ---
        vehicle_bps = bp_lib.filter("vehicle.*")
        spawns = world.get_map().get_spawn_points()
        random.shuffle(spawns)

        count = 0
        for sp in spawns:
            if count >= TARGET_NPCS:
                break
            if sp.location.distance(CAM_LOC) <= SPAWN_RADIUS_M:
                bp = random.choice(vehicle_bps)
                if bp.has_attribute("color"):
                    bp.set_attribute("color", random.choice(bp.get_attribute("color").recommended_values))
                npc = world.try_spawn_actor(bp, sp)
                if npc:
                    npc.set_autopilot(True)
                    spawned.append(npc)
                    count += 1
        print(f"[Info] Spawned {count} vehicles near camera (<= {SPAWN_RADIUS_M} m)")

        # --- instance segmentation capture ---
        q = queue.Queue()
        camera.listen(q.put)

        # warm up then grab one frame
        world.tick(); _ = q.get(True, 5)
        world.tick(); seg_image = q.get(True, 5)

        seg_image.save_to_disk(OUT_PNG, carla.ColorConverter.Raw)
        print(f"[Info] Saved instance segmentation image -> {OUT_PNG}")
        print("[Info] Done. (No screenshot section included.)")

    finally:
        # --- cleanup and restore ---
        try:
            world.apply_settings(original)
        except Exception:
            pass
        try:
            for a in spawned:
                if a and a.is_alive and a.type_id.startswith("sensor."):
                    a.stop()
        except Exception:
            pass
        try:
            client.apply_batch([carla.command.DestroyActor(a) for a in spawned if a and a.is_alive])
        except Exception:
            pass

if __name__ == "__main__":
    main()
