import carla
import random
import queue
import numpy as np
import cv2

# ---------- math helpers from tutorial ----------
def build_projection_matrix(w, h, fov, is_behind_camera=False):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = (-focal if is_behind_camera else focal)
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_image_point(loc, K, w2c):
    p = np.array([loc.x, loc.y, loc.z, 1.0])
    pc = np.dot(w2c, p)  # world -> camera
    pc = np.array([pc[1], -pc[2], pc[0]])  # UE4 -> conventional camera coords
    pix = np.dot(K, pc)
    pix[0] /= pix[2]; pix[1] /= pix[2]
    return pix[0:2]

def point_in_canvas(p, H, W):
    return 0 <= p[0] < W and 0 <= p[1] < H

# box edges (tutorial order)
EDGES = [[0,1],[1,3],[3,2],[2,0],[0,4],[4,5],[5,1],[5,7],[7,6],[6,4],[6,2],[7,3]]

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(20.0)
    world  = client.get_world()
    bp_lib = world.get_blueprint_library()

    # ---------- spawn ego vehicle ----------
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    ego_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
    ego = world.try_spawn_actor(ego_bp, spawn_points[0])
    if ego is None:
        raise RuntimeError("Failed to spawn ego vehicle; try rerunning.")
    ego.set_autopilot(True)

    # ---------- attach RGB camera ----------
    cam_bp = bp_lib.find('sensor.camera.rgb')
    cam_tf = carla.Transform(carla.Location(z=2.0))
    camera = world.spawn_actor(cam_bp, cam_tf, attach_to=ego)

    # ---------- sync mode ----------
    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    # ---------- simple NPC cars (this is the only addition) ----------
    npc_list = []
    vehicle_bps = bp_lib.filter('vehicle.*')
    target_n = 50  # change this number if you want more/less cars
    idx = 1
    for sp in spawn_points[1:]:
        if len(npc_list) >= target_n:
            break
        bp = random.choice(vehicle_bps)
        if bp.has_attribute('color'):
            bp.set_attribute('color', random.choice(bp.get_attribute('color').recommended_values))
        npc = world.try_spawn_actor(bp, sp)
        if npc:
            npc.set_autopilot(True)
            npc_list.append(npc)
        idx += 1
    print(f"[Info] Spawned NPC vehicles: {len(npc_list)}")

    # ---------- camera stream ----------
    q = queue.Queue()
    camera.listen(q.put)

    # camera intrinsics
    image_w = cam_bp.get_attribute("image_size_x").as_int()
    image_h = cam_bp.get_attribute("image_size_y").as_int()
    fov     = cam_bp.get_attribute("fov").as_float()
    K   = build_projection_matrix(image_w, image_h, fov)
    K_b = build_projection_matrix(image_w, image_h, fov, is_behind_camera=True)

    # level objects (traffic lights + signs)
    level_bbs = list(world.get_level_bbs(carla.CityObjectLabel.TrafficLight))
    level_bbs += list(world.get_level_bbs(carla.CityObjectLabel.TrafficSigns))

    # prime one frame
    world.tick(); _ = q.get(True, 5)
    cv2.namedWindow('BoundingBoxes', cv2.WINDOW_AUTOSIZE)
    print("Running… press 'q' to quit")

    try:
        while True:
            world.tick()
            image = q.get(True, 5)
            img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

            # world->camera matrix for current camera pose
            w2c = np.array(camera.get_transform().get_inverse_matrix())

            # --------- draw level object boxes (red) ---------
            for bb in level_bbs:
                if bb.location.distance(ego.get_transform().location) < 50:
                    fwd = ego.get_transform().get_forward_vector()
                    ray = bb.location - ego.get_transform().location
                    if fwd.dot(ray) > 0:
                        verts = [v for v in bb.get_world_vertices(carla.Transform())]
                        for e in EDGES:
                            p1 = get_image_point(verts[e[0]], K, w2c)
                            p2 = get_image_point(verts[e[1]], K, w2c)
                            cam_fwd = camera.get_transform().get_forward_vector()
                            r0 = verts[e[0]] - camera.get_transform().location
                            r1 = verts[e[1]] - camera.get_transform().location
                            if cam_fwd.dot(r0) <= 0: p1 = get_image_point(verts[e[0]], K_b, w2c)
                            if cam_fwd.dot(r1) <= 0: p2 = get_image_point(verts[e[1]], K_b, w2c)
                            if point_in_canvas(p1, image_h, image_w) or point_in_canvas(p2, image_h, image_w):
                                cv2.line(img, (int(p1[0]), int(p1[1])),
                                         (int(p2[0]), int(p2[1])), (0,0,255,255), 1)

            # --------- draw vehicle boxes (blue 3D, green 2D) ---------
            for npc in world.get_actors().filter('vehicle.*'):
                if npc.id == ego.id:
                    continue
                dist = npc.get_transform().location.distance(ego.get_transform().location)
                if dist >= 50:
                    continue
                fwd = ego.get_transform().get_forward_vector()
                ray = npc.get_transform().location - ego.get_transform().location
                if fwd.dot(ray) <= 0:
                    continue

                bb = npc.bounding_box
                verts = [v for v in bb.get_world_vertices(npc.get_transform())]

                # 3D wireframe (blue)
                for e in EDGES:
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

                # 2D rectangle (green)
                xs, ys = [], []
                for v in verts:
                    p = get_image_point(v, K, w2c)
                    xs.append(p[0]); ys.append(p[1])
                x_min = int(max(0, min(xs))); x_max = int(min(image_w-1, max(xs)))
                y_min = int(max(0, min(ys))); y_max = int(min(image_h-1, max(ys)))
                if x_min < x_max and y_min < y_max:
                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0,255,0,255), 1)

            cv2.imshow('BoundingBoxes', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cv2.destroyAllWindows()
        # stop/destroy actors and restore settings
        world.apply_settings(original)
        try:
            if camera.is_alive: camera.stop()
        except Exception:
            pass
        for a in npc_list + [camera, ego]:
            try:
                if a.is_alive: a.destroy()
            except Exception:
                pass

if __name__ == "__main__":
    main()
