import carla
import random
import queue
import numpy as np
import cv2
from pathlib import Path

# ---------- Config ----------
CAM_LOC = carla.Location(x=0.0, y=0.0, z=2.2)
CAM_ROT = carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
IMG_W, IMG_H, FOV = 1280, 720, 90
SPAWN_RADIUS_M = 80.0
TARGET_NPCS = 40
FIXED_DT = 0.05
INSTANCE_PNG = "instance_segmentation.png"  # saved automatically once, or press 'i' to re-save

def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(20.0)
    world = client.get_world()

    # --- Synchronous mode ---
    original = world.get_settings()
    s = world.get_settings()
    s.synchronous_mode = True
    s.fixed_delta_seconds = FIXED_DT
    world.apply_settings(s)

    spawned = []
    rgb_q = queue.Queue()
    inst_q = queue.Queue()
    instance_saved = False

    try:
        bp_lib = world.get_blueprint_library()

        # ---------- Sensors (same pose): RGB + Instance ----------
        # RGB
        rgb_bp = bp_lib.find("sensor.camera.rgb")
        rgb_bp.set_attribute("image_size_x", str(IMG_W))
        rgb_bp.set_attribute("image_size_y", str(IMG_H))
        rgb_bp.set_attribute("fov", str(FOV))
        cam_tf = carla.Transform(CAM_LOC, CAM_ROT)
        rgb_cam = world.spawn_actor(rgb_bp, cam_tf)
        spawned.append(rgb_cam)
        rgb_cam.listen(rgb_q.put)

        # Instance segmentation
        inst_bp = bp_lib.find("sensor.camera.instance_segmentation")
        inst_bp.set_attribute("image_size_x", str(IMG_W))
        inst_bp.set_attribute("image_size_y", str(IMG_H))
        inst_bp.set_attribute("fov", str(FOV))
        inst_cam = world.spawn_actor(inst_bp, cam_tf)
        spawned.append(inst_cam)
        inst_cam.listen(inst_q.put)

        # ---------- Populate scene: spawn NPCs within 80 m ----------
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
        print(f"[Info] Spawned {count} vehicles within {SPAWN_RADIUS_M} m of the camera.")

        # Optional: set spectator overhead to help you frame screenshots in a viewer
        spectator = world.get_spectator()
        spectator.set_transform(
            carla.Transform(
                carla.Location(x=CAM_LOC.x, y=CAM_LOC.y, z=120.0),
                carla.Rotation(pitch=-90.0)
            )
        )

        # Warm up both sensors
        world.tick(); _ = rgb_q.get(True, 5)
        world.tick(); _ = inst_q.get(True, 5)

        # Live RGB window (normal world view)
        cv2.namedWindow("RGB View (Press i: save instance PNG, q: quit)", cv2.WINDOW_AUTOSIZE)
        print("RGB live view running. Take your screenshots manually from this window.")
        print("Press 'i' to save an instance-seg image, 'q' to quit.")

        # Automatically save one instance frame once things are populated
        def save_instance_frame():
            nonlocal instance_saved
            # advance one tick to ensure a fresh instance frame
            world.tick()
            inst_img = inst_q.get(True, 5)
            inst_img.save_to_disk(INSTANCE_PNG, carla.ColorConverter.Raw)
            instance_saved = True
            print(f"[Info] Saved instance segmentation PNG -> {INSTANCE_PNG}")

        # Save once automatically after a couple extra ticks
        world.tick(); _ = inst_q.get(True, 5)
        save_instance_frame()

        while True:
            world.tick()
            rgb_img = rgb_q.get(True, 5)

            # Convert BGRA to BGR for display
            frame = np.frombuffer(rgb_img.raw_data, dtype=np.uint8).reshape((rgb_img.height, rgb_img.width, 4))
            bgr = frame[:, :, :3]

            # UI hint
            cv2.putText(bgr, "RGB view | 'i': save instance PNG | 'q': quit",
                        (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 240, 40), 2, cv2.LINE_AA)

            if instance_saved:
                cv2.putText(bgr, f"Saved: {INSTANCE_PNG}", (10, 52),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2, cv2.LINE_AA)

            cv2.imshow("RGB View (Press i: save instance PNG, q: quit)", bgr)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('i'):
                save_instance_frame()

    finally:
        cv2.destroyAllWindows()
        # Restore settings and cleanup
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
