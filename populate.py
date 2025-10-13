import carla
import random
import json
from pathlib import Path

CAM_LOC = carla.Location(x=0.0, y=0.0, z=2.2)           
CAM_ROT = carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0) 
IMG_W, IMG_H, FOV = 1280, 720, 90
SPAWN_RADIUS_M = 80.0
TARGET_VEHICLES = 40
FIXED_DT = 0.05
POSE_JSON = "instance_camera_pose.json"


def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(20.0)
    world = client.get_world()

    # Switch to synchronous mode for stability
    original = world.get_settings()
    s = world.get_settings()
    s.synchronous_mode = True
    s.fixed_delta_seconds = FIXED_DT
    world.apply_settings(s)

    spawned = []
    try:
        bp_lib = world.get_blueprint_library()

        # ---- Step 1: Set up the instance segmentation camera ----
        cam_bp = bp_lib.find("sensor.camera.instance_segmentation")
        cam_bp.set_attribute("image_size_x", str(IMG_W))
        cam_bp.set_attribute("image_size_y", str(IMG_H))
        cam_bp.set_attribute("fov", str(FOV))

        cam_tf = carla.Transform(CAM_LOC, CAM_ROT)
        camera = world.spawn_actor(cam_bp, cam_tf)
        spawned.append(camera)

        # Save the camera transform (for your report)
        Path(".").mkdir(parents=True, exist_ok=True)
        with open(POSE_JSON, "w") as f:
            json.dump(
                {
                    "location": {"x": CAM_LOC.x, "y": CAM_LOC.y, "z": CAM_LOC.z},
                    "rotation": {"pitch": CAM_ROT.pitch, "yaw": CAM_ROT.yaw, "roll": CAM_ROT.roll},
                    "image_size": {"width": IMG_W, "height": IMG_H, "fov": FOV},
                },
                f, indent=2
            )
        print(f"[Step 1] Instance camera spawned. Pose saved -> {POSE_JSON}")

        # ---- Step 2: Populate the scene around the camera (<= 80 m) ----
        vehicle_bps = bp_lib.filter("vehicle.*")
        spawns = world.get_map().get_spawn_points()
        random.shuffle(spawns)

        count = 0
        for sp in spawns:
            if count >= TARGET_VEHICLES:
                break
            if sp.location.distance(CAM_LOC) <= SPAWN_RADIUS_M:
                bp = random.choice(vehicle_bps)
                if bp.has_attribute("color"):
                    bp.set_attribute('color', random.choice(bp.get_attribute('color').recommended_values))
                npc = world.try_spawn_actor(bp, sp)
                if npc:
                    npc.set_autopilot(True)
                    spawned.append(npc)
                    count += 1
        print(f"[Step 2] Spawned {count} vehicles within {SPAWN_RADIUS_M} m of the camera.")

        spectator = world.get_spectator()
        spectator.set_transform(
            carla.Transform(
                carla.Location(x=CAM_LOC.x, y=CAM_LOC.y, z=120.0),
                carla.Rotation(pitch=-90.0)
            )
        )
        print("[Info] Spectator positioned overhead. Take your screenshot from your viewer.")
        print("[Info] Press Ctrl+C here when done.")

        # Keep the world running so you can frame & capture the shot
        while True:
            world.tick()

    finally:
        # Cleanup & restore settings
        try:
            world.apply_settings(original)
        except Exception:
            pass
        try:
            client.apply_batch([carla.command.DestroyActor(a) for a in spawned if a and a.is_alive])
        except Exception:
            pass
        print("[Info] Cleaned up and restored settings.")

if __name__ == "__main__":
    main()
