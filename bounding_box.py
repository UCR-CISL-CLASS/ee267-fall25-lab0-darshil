import carla
import math
import random
import queue
import numpy as np
import cv2

def build_projection_matrix(w, h, fov, is_behind_camera=False):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = (-focal if is_behind_camera else focal)
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_image_point(loc, K, w2c):
    # loc: carla.Location or Vector3D
    p = np.array([loc.x, loc.y, loc.z, 1.0])
    # world -> camera
    pc = np.dot(w2c, p)
    # UE4 coords (x,y,z) -> conventional camera (y, -z, x)
    pc = np.array([pc[1], -pc[2], pc[0]])
    pix = np.dot(K, pc)
    pix[0] /= pix[2]; pix[1] /= pix[2]
    return pix[0:2]

def point_in_canvas(p, H, W):
    return 0 <= p[0] < W and 0 <= p[1] < H

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(20.0)
    world  = client.get_world()
    bp_lib = world.get_blueprint_library()

    # Spawn ego vehicle
    spawn_points = world.get_map().get_spawn_points()
    vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
    vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
    if vehicle is None:
        raise RuntimeError("Failed to spawn ego vehicle; try rerunning.")

    # Spawn RGB camera on ego
    camera_bp = bp_lib.find('sensor.camera.rgb')
    # (use whatever default resolution is set in the blueprint)
    camera_init_trans = carla.Transform(carla.Location(z=2.0))
    camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)

    # Enable autopilot for ego (optional, like tutorial)
    vehicle.set_autopilot(True)

    # Switch to synchronous mode
    settings = world.get_settings()
    original = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    # Image queue
    image_queue = queue.Queue()
    camera.listen(image_queue.put)

    # Get camera attributes / intrinsics
    image_w = camera_bp.get_attribute("image_size_x").as_int()
    image_h = camera_bp.get_attribute("image_size_y").as_int()
    fov     = camera_bp.get_attribute("fov").as_float()

    K   = build_projection_matrix(image_w, image_h, fov)
    K_b = build_projection_matrix(image_w, image_h, fov, is_behind_camera=True)

    # Level bounding boxes (traffic lights + traffic signs)
    level_bbs = list(world.get_level_bbs(carla.CityObjectLabel.TrafficLight))
    level_bbs.extend(world.get_level_bbs(carla.CityObjectLabel.TrafficSigns))

    # Box edges order
    edges = [[0,1], [1,3], [3,2], [2,0],
             [0,4], [4,5], [5,1], [5,7],
             [7,6], [6,4], [6,2], [7,3]]

    # Prime first frame
    world.tick()
    _ = image_queue.get(True, 5)

    cv2.namedWindow('BoundingBoxes', cv2.WINDOW_AUTOSIZE)
    print("Running… press 'q' to quit")

    try:
        while True:
            # Get next frame
            world.tick()
            image = image_queue.get(True, 5)

            # Raw BGRA -> numpy
            img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

            # world->camera matrix for current camera pose
            w2c = np.array(camera.get_transform().get_inverse_matrix())

            # -------------------- Level objects: traffic lights + signs (3D wireframe) --------------------
            for bb in level_bbs:
                # distance & FOV filters like the tutorial
                if bb.location.distance(vehicle.get_transform().location) < 50:
                    fwd = vehicle.get_transform().get_forward_vector()
                    ray = bb.location - vehicle.get_transform().location
                    if fwd.dot(ray) > 0:
                        verts = [v for v in bb.get_world_vertices(carla.Transform())]
                        for e in edges:
                            p1 = get_image_point(verts[e[0]], K, w2c)
                            p2 = get_image_point(verts[e[1]], K, w2c)

                            # behind-camera handling (use K_b when necessary)
                            cam_fwd = camera.get_transform().get_forward_vector()
                            r0 = verts[e[0]] - camera.get_transform().location
                            r1 = verts[e[1]] - camera.get_transform().location
                            if cam_fwd.dot(r0) <= 0: p1 = get_image_point(verts[e[0]], K_b, w2c)
                            if cam_fwd.dot(r1) <= 0: p2 = get_image_point(verts[e[1]], K_b, w2c)

                            if point_in_canvas(p1, image_h, image_w) or point_in_canvas(p2, image_h, image_w):
                                cv2.line(img, (int(p1[0]), int(p1[1])),
                                         (int(p2[0]), int(p2[1])), (0,0,255,255), 1)

            # -------------------- Vehicle bounding boxes (3D wireframe + 2D rect) --------------------
            for npc in world.get_actors().filter('vehicle.*'):
                if npc.id == vehicle.id:
                    continue

                dist = npc.get_transform().location.distance(vehicle.get_transform().location)
                if dist >= 50:
                    continue

                fwd = vehicle.get_transform().get_forward_vector()
                ray = npc.get_transform().location - vehicle.get_transform().location
                if fwd.dot(ray) <= 0:
                    continue

                bb = npc.bounding_box
                verts = [v for v in bb.get_world_vertices(npc.get_transform())]

                # 3D wireframe (blue)
                for e in edges:
                    p1 = get_image_point(verts[e[0]], K, w2c)
                    p2 = get_image_point(verts[e[1]], K, w2c)

                    cam_fwd = camera.get_transform().get_forward_vector()
                    r0 = verts[e[0]] - camera.get_transform().location
                    r1 = verts[e[1]] - camera.get_transform().location
                    if cam_fwd.dot(r0) <= 0: p1 = get_image_point(verts[e[0]], K_b, w2c)
                    if cam_fwd.dot(r1) <= 0: p2 = get_image_point(verts[e[1]], K_b, w2c)

                    if point_in_canvas(p1, image_h, image_w) or point_in_canvas(p2, image_h, image_w):
                        cv2.line(img, (int(p1[0]), int(p1[1])),
                                 (int(p2[0]), int(p2[1])), (255,0,0,255), 1)

                # 2D rectangle (green) from projected vertices
                xs, ys = [], []
                for v in verts:
                    p = get_image_point(v, K, w2c)
                    xs.append(p[0]); ys.append(p[1])
                x_min = int(max(0, min(xs))); x_max = int(min(image_w-1, max(xs)))
                y_min = int(max(0, min(ys))); y_max = int(min(image_h-1, max(ys)))
                if x_min < x_max and y_min < y_max:
                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0,255,0,255), 1)

            # Show frame
            cv2.imshow('BoundingBoxes', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cv2.destroyAllWindows()
        # restore async mode and destroy actors
        world.apply_settings(original)
        if camera.is_alive: camera.stop()
        for a in [camera, vehicle]:
            try:
                if a.is_alive: a.destroy()
            except Exception:
                pass

if __name__ == "__main__":
    main()
