import carla
import random
import queue
import numpy as np
import cv2

# -------- Config (edit if you want) --------
CAM_LOC = carla.Location(x=0.0, y=0.0, z=2.2)
CAM_ROT = carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
IMG_W, IMG_H, FOV = 1280, 720, 90
SPAWN_RADIUS_M = 80.0
TARGET_NPCS = 40
FIXED_DT = 0.05
OPTIONAL_SAVE_PATH = "instance_segmentation.png"  # saved when you press 's'

def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(20.0)
    world = client.get_world()

    # --- sync mode ---
    original = world.get_settings()
    s = world.get_settings()
    s.synchronous_mode = True
    s.fixed_delta_seconds = FIXED_DT
    world.apply_settings(s)

    spawned = []
    try:
        bp_lib = world.get_blueprint_library()

        # --- instance segmentation camera (world-fixed) ---
        cam_bp = bp_lib.find("sensor.camera.instance_segmentation")
        cam_bp.set_attribute("image_size_x", str(IMG_W))
        cam_bp.set_attribute("image_size_y", str(IMG_H))
        cam_bp.set_attribute("fov", str(FOV))

        cam_tf = carla.Transform(CAM_LOC, CAM_ROT)
        camera = world.spawn_actor(cam_bp, cam_tf)
        spawned.append(camera)

        # --- spawn vehicles within 80 m of camera ---
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

        # --- put spectator above camera for easier manual screenshots (optional) ---
        spectator = world.get_spectator()
        spectator.set_transform(
            carla.Transform(carla.Location(x=CAM_LOC.x, y=CAM_LOC.y, z=120.0),
                            carla.Rotation(pitch=-90.0))
        )

        # --- live feed (OpenCV window) ---
        q = queue.Queue()
        camera.listen(q.put)

        # warm up
        world.tick(); _ = q.get(True, 5)

        cv2.namedWindow("InstanceSeg (IDs as colors)", cv2.WINDOW_AUTOSIZE)
        print("Live instance segmentation running.")
        print("Take your screenshots manually. Press 's' to save one PNG, 'q' to quit.")

        while True:
            world.tick()
            image = q.get(True, 5)

            # Instance segmentation sensor gives BGRA bytes with instance IDs encoded as colors
            frame = np.frombuffer(image.raw_data, dtype=np.uint8)
            frame = frame.reshape((image.height, image.width, 4))  # BGRA
            # Drop alpha for display
            bgr = frame[:, :, :3]

            cv2.imshow("InstanceSeg (IDs as colors)", bgr)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s'):
                # Save the original sensor image so IDs stay exact (no palette conversion)
                image.save_to_disk(OPTIONAL_SAVE_PATH, carla.ColorConverter.Raw)
                print(f"[Info] Saved {OPTIONAL_SAVE_PATH}")

    finally:
        cv2.destroyAllWindows()
        # cleanup and restore
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
