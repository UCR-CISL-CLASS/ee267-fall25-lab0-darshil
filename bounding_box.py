import carla
import random
import queue
import numpy as np
import cv2
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, ElementTree

# ------------------ config ------------------
OUTDIR = "output"      # where images/XML will be saved
MAX_DIST = 50.0        # meters: draw/filter vehicles within this distance
FIXED_DT = 0.05        # sync step
TARGET_NPC = 50        # how many extra cars to try to spawn

# ------------- tutorial math helpers -------------
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
    # world -> camera (4x4)
    pc = np.dot(w2c, p)
    # UE4 coords (x,y,z) -> conventional camera coords (y, -z, x)
    pc = np.array([pc[1], -pc[2], pc[0]])
    pix = np.dot(K, pc)
    pix[0] /= pix[2]; pix[1] /= pix[2]
    return pix[0:2]

def point_in_canvas(p, H, W):
    return 0 <= p[0] < W and 0 <= p[1] < H

# box edges (tutorial order)
EDGES = [[0,1],[1,3],[3,2],[2,0],[0,4],[4,5],[5,1],[5,7],[7,6],[6,4],[6,2],[7,3]]

# ------------- Pascal VOC writer -------------
def write_voc_xml(xml_path, img_path, w, h, boxes):
    """
    boxes: [{'name':'vehicle','xmin':..,'ymin':..,'xmax':..,'ymax':..}, ...]
    """
    ann = Element('annotation')
    SubElement(ann, 'folder').text = Path(img_path).parent.name
    SubElement(ann, 'filename').text = Path(img_path).name
    SubElement(ann, 'path').text = str(Path(img_path).resolve())

    src = SubElement(ann, 'source')
    SubElement(src, 'database').text = 'CARLA'

    sz = SubElement(ann, 'size')
    SubElement(sz, 'width').text = str(w)
    SubElement(sz, 'height').text = str(h)
    SubElement(sz, 'depth').text = '3'

    SubElement(ann, 'segmented').text = '0'

    for b in boxes:
        obj = SubElement(ann, 'object')
        SubElement(obj, 'name').text = b.get('name', 'vehicle')
        SubElement(obj, 'pose').text = 'Unspecified'
        SubElement(obj, 'truncated').text = '0'
        SubElement(obj, 'difficult').text = '0'
        bb = SubElement(obj, 'bndbox')
        SubElement(bb, 'xmin').text = str(int(b['xmin']))
        SubElement(bb, 'ymin').text = str(int(b['ymin']))
        SubElement(bb, 'xmax').text = str(int(b['xmax']))
        SubElement(bb, 'ymax').text = str(int(b['ymax']))
    ElementTree(ann).write(xml_path, encoding='utf-8', xml_declaration=True)

# ------------- main -------------
def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(20.0)
    world  = client.get_world()
    bp_lib = world.get_blueprint_library()

    # Spawn ego vehicle
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    ego_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
    ego = world.try_spawn_actor(ego_bp, spawn_points[0])
    if ego is None:
        raise RuntimeError("Failed to spawn ego vehicle; re-run or change blueprint.")
    ego.set_autopilot(True)

    # Attach RGB camera to ego
    cam_bp = bp_lib.find('sensor.camera.rgb')
    cam_tf = carla.Transform(carla.Location(z=2.0))
    camera = world.spawn_actor(cam_bp, cam_tf, attach_to=ego)

    # Switch world to synchronous mode
    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = FIXED_DT
    world.apply_settings(settings)

    # Simple NPC car spawner
    npc_list = []
    vehicle_bps = bp_lib.filter('vehicle.*')
    for sp in spawn_points[1:]:
        if len(npc_list) >= TARGET_NPC:
            break
        bp = random.choice(vehicle_bps)
        if bp.has_attribute('color'):
            bp.set_attribute('color', random.choice(bp.get_attribute('color').recommended_values))
        npc = world.try_spawn_actor(bp, sp)
        if npc:
            npc.set_autopilot(True)
            npc_list.append(npc)
    print(f"[Info] Spawned NPC vehicles: {len(npc_list)}")

    # Camera stream queue
    q = queue.Queue()
    camera.listen(q.put)

    # Camera intrinsics
    image_w = cam_bp.get_attribute("image_size_x").as_int()
    image_h = cam_bp.get_attribute("image_size_y").as_int()
    fov     = cam_bp.get_attribute("fov").as_float()
    K   = build_projection_matrix(image_w, image_h, fov)
    K_b = build_projection_matrix(image_w, image_h, fov, is_behind_camera=True)

    # Prime first frame
    world.tick(); _ = q.get(True, 5)
    cv2.namedWindow('BoundingBoxes', cv2.WINDOW_AUTOSIZE)
    Path(OUTDIR).mkdir(parents=True, exist_ok=True)
    print("Running… press 'q' to quit")

    try:
        while True:
            world.tick()
            image = q.get(True, 5)
            img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

            # world->camera matrix for current camera pose
            w2c = np.array(camera.get_transform().get_inverse_matrix())

            # Collect 2D boxes for VOC export this frame
            boxes_for_voc = []

            # --------- draw vehicle boxes (blue 3D wireframe + green 2D rect) ---------
            for npc in world.get_actors().filter('vehicle.*'):
                if npc.id == ego.id:
                    continue

                # distance + FOV filtering (as in tutorial)
                dist = npc.get_transform().location.distance(ego.get_transform().location)
                if dist >= MAX_DIST:
                    continue
                fwd = ego.get_transform().get_forward_vector()
                ray = npc.get_transform().location - ego.get_transform().location
                if fwd.dot(ray) <= 0:
                    continue

                # 3D vertices of the vehicle bounding box
                bb = npc.bounding_box
                verts = [v for v in bb.get_world_vertices(npc.get_transform())]

                # draw 3D wireframe (blue)
                for e in EDGES:
                    p1 = get_image_point(verts[e[0]], K, w2c)
                    p2 = get_image_point(verts[e[1]], K, w2c)

                    # handle vertices behind the camera (use K_b)
                    cam_fwd = camera.get_transform().get_forward_vector()
                    r0 = verts[e[0]] - camera.get_transform().location
                    r1 = verts[e[1]] - camera.get_transform().location
                    if cam_fwd.dot(r0) <= 0: p1 = get_image_point(verts[e[0]], K_b, w2c)
                    if cam_fwd.dot(r1) <= 0: p2 = get_image_point(verts[e[1]], K_b, w2c)

                    if point_in_canvas(p1, image_h, image_w) or point_in_canvas(p2, image_h, image_w):
                        cv2.line(img, (int(p1[0]), int(p1[1])),
                                 (int(p2[0]), int(p2[1])), (255,0,0,255), 1)

                # 2D rectangle from projected vertices (for export + optional draw)
                xs, ys = [], []
                for v in verts:
                    p = get_image_point(v, K, w2c)
                    xs.append(p[0]); ys.append(p[1])
                x_min = int(max(0, min(xs))); x_max = int(min(image_w-1, max(xs)))
                y_min = int(max(0, min(ys))); y_max = int(min(image_h-1, max(ys)))
                if x_min < x_max and y_min < y_max:
                    # draw green 2D rectangle
                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0,255,0,255), 1)
                    # store for VOC
                    boxes_for_voc.append({'name':'vehicle','xmin':x_min,'ymin':y_min,'xmax':x_max,'ymax':y_max})

            # Show
            cv2.imshow('BoundingBoxes', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Export PNG + VOC XML for this frame
            frame_name = f"{image.frame:06d}"
            img_path = f"{OUTDIR}/{frame_name}.png"
            xml_path = f"{OUTDIR}/{frame_name}.xml"
            image.save_to_disk(img_path)
            if boxes_for_voc:
                write_voc_xml(xml_path, img_path, image.width, image.height, boxes_for_voc)

    finally:
        cv2.destroyAllWindows()
        # restore settings and cleanup actors
        try:
            world.apply_settings(original)
        except Exception:
            pass
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
