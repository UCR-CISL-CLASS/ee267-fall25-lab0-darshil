import carla
import random
import queue
import numpy as np
import cv2
import json
from pathlib import Path

# ---------- Config (edit if needed) ----------
CAM_LOC = carla.Location(x=0.0, y=0.0, z=2.2)          # Step 1: camera location
CAM_ROT = carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0) # Step 1: camera rotation
IMG_W, IMG_H, FOV = 1280, 720, 90
SPAWN_RADIUS_M = 80.0                                   # Step 2: radius
TARGET_NPCS = 40                                        # Step 2: how many vehicles to try
FIXED_DT = 0.05
POSE_JSON = "instance_camera_pose.json"
OPTIONAL_SAVE_PNG = "instance_segmentation.png"         # Saved only if you press 's'


def put_instructions(bgr, step):
    h, w = bgr.shape[:2]
    lines = [
        f"STEP {step}/5",
        "n = next step   s = save PNG (Raw IDs)   q = quit",
    ]
    if step == 1:
        lines += [
            "Step 1: Camera spawned (instance segmentation).",
            "Transform saved to instance_camera_pose.json.",
            "Take your screenshot now if needed, then press 'n'.",
        ]
    elif step == 2:
        lines += [
            f"Step 2: Spawning vehicles within {SPAWN_RADIUS_M:.0f} m of camera.",
            "Wait for cars to appear; take your screenshot; press 'n' to continue.",
        ]
    elif step == 3:
        lines += [
            "Step 3: Live instance-seg feed running (IDs as colors).",
            "You can take screenshots anytime; press 'n' to set spectator for overview.",
        ]
    elif step == 4:
        lines += [
            "Step 4: Spectator positioned overhead (good for overview screenshots).",
            "Fly around in your viewer if open; take screenshot; press 'n' to finish.",
        ]
    elif step == 5:
        lines += [
            "Step 5: Finished. Press 'q' to quit (cleanup & restore).",
        ]
    y = 28
    cv2.putText(bgr, TITLE, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 230, 30), 2, cv2.LINE_AA)
    y += 30
    for ln in lines:
        cv2.putText(bgr, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 24
    return bgr

def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(20.0)
    world = client.get_world()

    # Enable synchronous mode
    original = world.get_settings()
    s = world.get_settings()
    s.synchronous_mode = True
    s.fixed_delta_seconds = FIXED_DT
    world.apply_settings(s)

    spawned = []
    q = queue.Queue()
    step = 1
    did_step2_spawns = False

    try:
        bp_lib = world.get_blueprint_library()

        # STEP 1: Create blueprint, choose pose, spawn camera, save transform
        cam_bp = bp_lib.find("sensor.camera.instance_segmentation")
        cam_bp.set_attribute("image_size_x", str(IMG_W))
        cam_bp.set_attribute("image_size_y", str(IMG_H))
        cam_bp.set_attribute("fov", str(FOV))
        cam_tf = carla.Transform(CAM_LOC, CAM_ROT)
        camera = world.spawn_actor(cam_bp, cam_tf)
        spawned.append(camera)
        camera.listen(q.put)

        # Save camera transform for your report
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
        print(f"[Step 1] Camera spawned. Pose saved -> {POSE_JSON}")

        # Prepare vehicles blueprints & spawn points ahead of Step 2
        vehicle_bps = bp_lib.filter("vehicle.*")
        spawns = world.get_map().get_spawn_points()
        random.shuffle(spawns)

        # OpenCV window
        cv2.namedWindow("InstanceSeg (IDs as colors)", cv2.WINDOW_AUTOSIZE)

        # Warm-up sensor
        world.tick(); _ = q.get(True, 5)

        # Loop until user presses q. Use 'n' to advance steps.
        while True:
            world.tick()
            image = q.get(True, 5)

            # Instance segmentation frames are BGRA bytes (RGB encodes instance IDs)
            frame = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
            bgr = frame[:, :, :3]  # drop alpha for display

            # On Step 2 trigger: spawn vehicles within 80 m (only once)
            if step >= 2 and not did_step2_spawns:
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
                did_step2_spawns = True
                print(f"[Step 2] Spawned {count} vehicles within {SPAWN_RADIUS_M} m of camera.")

            # On Step 4 trigger: place spectator for an overview
            if step >= 4:
                spectator = world.get_spectator()
                spectator.set_transform(
                    carla.Transform(
                        carla.Location(x=CAM_LOC.x, y=CAM_LOC.y, z=120.0),
                        carla.Rotation(pitch=-90.0)
                    )
                )

            # Overlay instructions
            bgr = put_instructions(bgr, min(step, 5))

            # Show live
            cv2.imshow("InstanceSeg (IDs as colors)", bgr)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                # Step 5: finish/cleanup
                print("[Step 5] Quit requested.")
                break
            elif key == ord('n'):
                # Advance to next step
                if step < 5:
                    step += 1
                    print(f"[Info] Moved to Step {step}.")
            elif key == ord('s'):
                # Optional: save one PNG of the raw IDs
                image.save_to_disk(OPTIONAL_SAVE_PNG, carla.ColorConverter.Raw)
                print(f"[Info] Saved {OPTIONAL_SAVE_PNG}")

    finally:
        cv2.destroyAllWindows()
        # cleanup & restore
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
        print("[Info] Cleaned up and restored settings.")

if __name__ == "__main__":
    main()
