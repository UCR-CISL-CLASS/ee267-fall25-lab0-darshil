import argparse
import atexit
import math
import os
import queue
import random
import sys
import time
import json
from pathlib import Path

import cv2
import numpy as np
import carla


# ----------------------------- Helpers -----------------------------

def trace(*a): print("[TRACE]", *a)
def info(*a):  print("[Info]", *a)
def warn(*a):  print("[Warn]", *a, file=sys.stderr)

EDGES = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]

def build_projection_matrix(w, h, fov, is_behind_camera=False):
    """Camera intrinsics K."""
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = (-focal if is_behind_camera else focal)
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_image_point(loc, K, w2c):
    """Project 3D world point (carla.Location/Vector3D) to image pixel coordinates."""
    pt = np.array([loc.x, loc.y, loc.z, 1.0])
    # world -> camera
    pt_cam = np.dot(w2c, pt)
    # UE4 (x,y,z) -> conventional camera coords (y, -z, x)
    pt_cam = np.array([pt_cam[1], -pt_cam[2], pt_cam[0]])
    # project
    p = np.dot(K, pt_cam)
    p[0] /= p[2]
    p[1] /= p[2]
    return p[0:2]

def point_in_canvas(p, H, W):
    return 0 <= p[0] < W and 0 <= p[1] < H

def setup_world_sync(world, fixed_dt=0.05):
    """Enable synchronous mode (safe for varying CARLA builds)."""
    original = world.get_settings()
    new = world.get_settings()
    new.synchronous_mode = True
    new.fixed_delta_seconds = fixed_dt
    world.apply_settings(new)
    s = world.get_settings()
    trace("World sync:", s.synchronous_mode, "fixed_dt:", s.fixed_delta_seconds)
    return original

def world_tick(world, steps=1):
    for _ in range(steps):
        world.tick()

def ensure_outdir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

# ----------------------------- Exporters -----------------------------

def write_voc_xml(xml_path, img_path, w, h, boxes):
    """
    boxes: list of dicts [{'name':'vehicle','xmin':..,'ymin':..,'xmax':..,'ymax':..}, ...]
    """
    from xml.etree.ElementTree import Element, SubElement, ElementTree

    ann = Element('annotation')
    SubElement(ann, 'folder').text = str(Path(img_path).parent.name)
    SubElement(ann, 'filename').text = Path(img_path).name
    SubElement(ann, 'path').text = str(Path(img_path).resolve())

    source = SubElement(ann, 'source')
    SubElement(source, 'database').text = 'CARLA'

    size = SubElement(ann, 'size')
    SubElement(size, 'width').text = str(w)
    SubElement(size, 'height').text = str(h)
    SubElement(size, 'depth').text = '3'

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

class CocoWriter:
    def __init__(self, out_json_path):
        self.out = out_json_path
        self.images = []
        self.annotations = []
        self.categories = [{'id': 1, 'name': 'vehicle', 'supercategory': 'vehicle'}]
        self._next_img_id = 1
        self._next_ann_id = 1

    def add_image(self, file_name, w, h):
        img_id = self._next_img_id; self._next_img_id += 1
        self.images.append({
            'id': img_id, 'file_name': file_name, 'width': w, 'height': h,
            'license': 1, 'date_captured': ''
        })
        return img_id

    def add_box(self, img_id, xmin, ymin, xmax, ymax, cat_id=1):
        x = float(xmin); y = float(ymin)
        w = float(max(0, xmax - xmin)); h = float(max(0, ymax - ymin))
        ann = {
            'id': self._next_ann_id, 'image_id': img_id,
            'category_id': cat_id, 'bbox': [x, y, w, h],
            'area': w * h, 'iscrowd': 0, 'segmentation': []
        }
        self._next_ann_id += 1
        self.annotations.append(ann)

    def save(self):
        data = {
            'info': {}, 'licenses': [{'id':1, 'name':'', 'url':''}],
            'categories': self.categories,
            'images': self.images, 'annotations': self.annotations
        }
        with open(self.out, 'w') as f:
            json.dump(data, f, indent=2)


# ----------------------------- Main -----------------------------

def main(args):
    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    world = client.get_world()

    # Town
    if args.town and world.get_map().name.split('/')[-1] != args.town:
        info(f"Loading {args.town}…")
        world = client.load_world(args.town)
        world_tick(world, 10)

    original_settings = setup_world_sync(world, args.fixed_dt)

    spawned = []
    def cleanup():
        try:
            info("Cleaning up actors and restoring settings…")
            client.apply_batch([carla.command.DestroyActor(a) for a in spawned if a and a.is_alive])
            # restore sync settings
            world.apply_settings(original_settings)
        except Exception as e:
            warn("cleanup:", e)
    atexit.register(cleanup)

    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    # Ego vehicle
    ego_bp = bp_lib.find(args.ego_blueprint)
    ego = world.try_spawn_actor(ego_bp, random.choice(spawn_points))
    assert ego, "Failed to spawn ego vehicle—retry or change blueprint."
    spawned.append(ego)
    if args.ego_autopilot:
        ego.set_autopilot(True)

    # Camera on ego
    cam_bp = bp_lib.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', str(args.width))
    cam_bp.set_attribute('image_size_y', str(args.height))
    cam_bp.set_attribute('fov', str(args.fov))
    cam_tf = carla.Transform(carla.Location(x=0.0, y=0.0, z=args.cam_height))
    camera = world.spawn_actor(cam_bp, cam_tf, attach_to=ego)
    spawned.append(camera)

    # Optional: extra NPC vehicles
    if args.extra_vehicles > 0:
        veh_bps = bp_lib.filter('vehicle.*')
        count = 0
        for sp in spawn_points:
            if count >= args.extra_vehicles: break
            bp = random.choice(veh_bps)
            if bp.has_attribute('color'):
                bp.set_attribute('color', random.choice(bp.get_attribute('color').recommended_values))
            npc = world.try_spawn_actor(bp, sp)
            if npc:
                npc.set_autopilot(True)
                spawned.append(npc)
                count += 1
        info(f"Spawned extra vehicles: {count}/{args.extra_vehicles}")

    # Queue for camera images
    img_q = queue.Queue()
    camera.listen(img_q.put)

    # Level bounding boxes (traffic lights + traffic signs)
    level_bbs = []
    if args.draw_level:
        bbt = world.get_level_bbs(carla.CityObjectLabel.TrafficLight)
        bbs = world.get_level_bbs(carla.CityObjectLabel.TrafficSigns)
        level_bbs = list(bbt) + list(bbs)
        info(f"Level BBoxes loaded: traffic lights={len(bbt)}, signs={len(bbs)}")

    # Prepare projection matrices
    K   = build_projection_matrix(args.width, args.height, args.fov, is_behind_camera=False)
    K_b = build_projection_matrix(args.width, args.height, args.fov, is_behind_camera=True)

    # Exporters
    coco = None
    if args.export_coco:
        ensure_outdir(args.outdir)
        coco = CocoWriter(str(Path(args.outdir) / "annotations_coco.json"))

    ensure_outdir(args.outdir)

    # Warm-up tick to receive first frame
    world_tick(world, 2)
    _ = img_q.get(True, 5)

    frame_idx = 0
    info("Running… press 'q' in the OpenCV window to quit.")
    cv2.namedWindow('BoundingBoxes', cv2.WINDOW_AUTOSIZE)

    while True:
        world_tick(world, 1)
        image = img_q.get(True, 5)

        # Raw BGRA -> copy (we draw on it)
        img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
        H, W = image.height, image.width

        # Current camera extrinsics (world->camera)
        w2c = np.array(camera.get_transform().get_inverse_matrix())

        # ------------- Draw level objects (3D wireframe) -------------
        if args.draw_level:
            for bb in level_bbs:
                # distance filter
                if bb.location.distance(ego.get_transform().location) > args.max_dist:
                    continue
                # only in front of ego
                fwd = ego.get_transform().get_forward_vector()
                ray = bb.location - ego.get_transform().location
                if fwd.dot(ray) <= 0:
                    continue
                verts = [v for v in bb.get_world_vertices(carla.Transform())]
                for e in EDGES:
                    p1 = get_image_point(verts[e[0]], K, w2c)
                    p2 = get_image_point(verts[e[1]], K, w2c)
                    if (point_in_canvas(p1, H, W) or point_in_canvas(p2, H, W)):
                        # handle behind-camera vertices
                        cam_fwd = camera.get_transform().get_forward_vector()
                        r0 = verts[e[0]] - camera.get_transform().location
                        r1 = verts[e[1]] - camera.get_transform().location
                        if cam_fwd.dot(r0) <= 0: p1 = get_image_point(verts[e[0]], K_b, w2c)
                        if cam_fwd.dot(r1) <= 0: p2 = get_image_point(verts[e[1]], K_b, w2c)
                        cv2.line(img, (int(p1[0]), int(p1[1])),
                                      (int(p2[0]), int(p2[1])), (0,0,255,255), 1)

        # ------------- Draw vehicle boxes (3D wireframe) + optional 2D -------------
        boxes_2d_for_export = []
        for npc in world.get_actors().filter('vehicle.*'):
            if npc.id == ego.id:  # skip ego
                continue

            dist = npc.get_transform().location.distance(ego.get_transform().location)
            if dist > args.max_dist:
                continue

            fwd = ego.get_transform().get_forward_vector()
            ray = npc.get_transform().location - ego.get_transform().location
            if fwd.dot(ray) <= 0:
                continue

            bb = npc.bounding_box
            verts = [v for v in bb.get_world_vertices(npc.get_transform())]

            # 3D wireframe
            if args.draw_3d:
                for e in EDGES:
                    p1 = get_image_point(verts[e[0]], K, w2c)
                    p2 = get_image_point(verts[e[1]], K, w2c)

                    # behind-camera handling
                    cam_fwd = camera.get_transform().get_forward_vector()
                    r0 = verts[e[0]] - camera.get_transform().location
                    r1 = verts[e[1]] - camera.get_transform().location
                    if cam_fwd.dot(r0) <= 0: p1 = get_image_point(verts[e[0]], K_b, w2c)
                    if cam_fwd.dot(r1) <= 0: p2 = get_image_point(verts[e[1]], K_b, w2c)

                    if (point_in_canvas(p1, H, W) or point_in_canvas(p2, H, W)):
                        cv2.line(img, (int(p1[0]), int(p1[1])),
                                      (int(p2[0]), int(p2[1])), (255,0,0,255), 1)

            # 2D rectangle from projected vertices
            if args.draw_2d or args.export_voc or args.export_coco:
                xs, ys = [], []
                for v in verts:
                    p = get_image_point(v, K, w2c)
                    xs.append(p[0]); ys.append(p[1])
                x_min, x_max = int(max(0, min(xs))), int(min(W-1, max(xs)))
                y_min, y_max = int(max(0, min(ys))), int(min(H-1, max(ys)))

                # draw if requested
                if args.draw_2d and x_min < x_max and y_min < y_max:
                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0,255,0,255), 1)

                # collect for export (inside image)
                if 0 <= x_min < x_max < W and 0 <= y_min < y_max < H:
                    boxes_2d_for_export.append({
                        'name': 'vehicle',
                        'xmin': x_min, 'ymin': y_min,
                        'xmax': x_max, 'ymax': y_max
                    })

        # Show image
        cv2.imshow('BoundingBoxes', img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break

        # Optional save/export per frame
        if args.save_images or args.export_voc or args.export_coco:
            ensure_outdir(args.outdir)
            frame_name = f"{image.frame:06d}.png"
            img_path = str(Path(args.outdir) / frame_name)
            # save the sensor image directly (lossless)
            image.save_to_disk(img_path)

            if args.export_voc and boxes_2d_for_export:
                xml_path = str(Path(args.outdir) / f"{image.frame:06d}.xml")
                write_voc_xml(xml_path, img_path, W, H, boxes_2d_for_export)

            if args.export_coco and boxes_2d_for_export:
                if coco is None:
                    warn("COCO writer missing unexpectedly")
                else:
                    img_id = coco.add_image(Path(img_path).name, W, H)
                    for b in boxes_2d_for_export:
                        coco.add_box(img_id, b['xmin'], b['ymin'], b['xmax'], b['ymax'], cat_id=1)

        frame_idx += 1
        if args.max_frames and frame_idx >= args.max_frames:
            break

    # finalize COCO
    if args.export_coco and coco is not None:
        coco.save()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='localhost')
    ap.add_argument('--port', type=int, default=2000)
    ap.add_argument('--town', default='', help='e.g., Town03')
    ap.add_argument('--fixed-dt', type=float, default=0.05)

    ap.add_argument('--ego-blueprint', default='vehicle.lincoln.mkz_2020')
    ap.add_argument('--ego-autopilot', action='store_true')
    ap.add_argument('--extra-vehicles', type=int, default=30, help='spawn extra NPC vehicles')

    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--fov', type=float, default=90.0)
    ap.add_argument('--cam-height', type=float, default=2.0)

    ap.add_argument('--draw-level', action='store_true', help='draw traffic light/sign 3D boxes')
    ap.add_argument('--draw-3d', action='store_true', help='draw 3D wireframe boxes for vehicles')
    ap.add_argument('--draw-2d', action='store_true', help='draw 2D rectangles for vehicles')
    ap.add_argument('--max-dist', type=float, default=50.0, help='meters visibility filter')

    ap.add_argument('--save-images', action='store_true', help='save RGB frames to outdir')
    ap.add_argument('--export-voc', action='store_true')
    ap.add_argument('--export-coco', action='store_true')
    ap.add_argument('--outdir', default='output', help='where to save images/annotations')
    ap.add_argument('--max-frames', type=int, default=0, help='stop after N frames (0 = infinite)')

    args = ap.parse_args()
    main(args)
