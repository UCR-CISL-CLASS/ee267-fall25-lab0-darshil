#!/usr/bin/env python3
# Part 3 – Traffic Manager: spawn background traffic and save a screenshot
# Works with CARLA 0.9.15

import argparse
import atexit
import random
import time
import carla


def setup_world_sync(world, fixed_dt=0.05):
    """Enable synchronous mode and return a snapshot of original settings."""
    settings = world.get_settings()
    original = carla.WorldSettings(
        synchronous_mode=settings.synchronous_mode,
        fixed_delta_seconds=settings.fixed_delta_seconds,
        no_rendering_mode=settings.no_rendering_mode,
        max_substeps=settings.max_substeps,
        max_substep_delta_time=settings.max_substep_delta_time,
    )
    new_settings = world.get_settings()
    new_settings.synchronous_mode = True
    new_settings.fixed_delta_seconds = fixed_dt
    world.apply_settings(new_settings)
    return original


def tick(world):
    """Advance one synchronous frame."""
    world.tick()


def main(args):
    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)

    world = client.get_world()
    if args.town and world.get_map().name.split('/')[-1] != args.town:
        world = client.load_world(args.town)

    # --- Traffic Manager (TM) ---
    tm = client.get_trafficmanager(args.tm_port)
    tm.set_synchronous_mode(True)
    tm.set_global_distance_to_leading_vehicle(2.5)
    tm.set_respawn_dormant_vehicles(True)
    tm.set_random_device_seed(args.seed)
    random.seed(args.seed)

    # --- Synchronous world for stable spawning and screenshots ---
    original_settings = setup_world_sync(world, fixed_dt=0.05)

    # --- Cleanup handler ---
    spawned = []

    def cleanup():
        try:
            print("\n[Cleanup] Destroying actors and restoring settings...")
            # stop walker controllers first
            for a in spawned:
                if a and a.is_alive and 'controller.ai.walker' in a.type_id:
                    try:
                        a.stop()
                    except Exception:
                        pass
            client.apply_batch([carla.command.DestroyActor(a.id) for a in spawned if a and a.is_alive])
            tm.set_synchronous_mode(False)
            world.apply_settings(original_settings)
        except Exception as e:
            print("[Cleanup] Warning:", e)

    atexit.register(cleanup)

    # --- Spawn VEHICLES ---
    bp_lib = world.get_blueprint_library()
    vehicle_bps = bp_lib.filter('vehicle.*')
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    nveh = min(args.vehicles, len(spawn_points))
    print(f"[Info] Spawning up to {nveh} vehicles...")
    batch = []
    for i in range(nveh):
        bp = random.choice(vehicle_bps)
        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        batch.append(carla.command.SpawnActor(bp, spawn_points[i]))

    results = client.apply_batch_sync(batch, True)
    vehicle_ids = [r.actor_id for r in results if r.error is None]
    vehicles = [world.get_actor(i) for i in vehicle_ids]
    spawned += vehicles

    # Hand vehicles to TM
    for v in vehicles:
        v.set_autopilot(True, args.tm_port)

    print(f"[Info] Spawned {len(vehicles)} vehicles.")

    # --- Spawn PEDESTRIANS (walkers + controllers) ---
    walker_bps = bp_lib.filter('walker.pedestrian.*')
    controller_bp = bp_lib.find('controller.ai.walker')

    walker_spawns = []
    for _ in range(args.walkers):
        loc = world.get_random_location_from_navigation()
        if loc:
            walker_spawns.append(carla.Transform(loc, carla.Rotation(yaw=random.uniform(-180, 180))))

    w_batch = [carla.command.SpawnActor(random.choice(walker_bps), tf) for tf in walker_spawns]
    w_res = client.apply_batch_sync(w_batch, True)
    walker_ids = [r.actor_id for r in w_res if r.error is None]

    c_batch = [carla.command.SpawnActor(controller_bp, carla.Transform(), wid) for wid in walker_ids]
    c_res = client.apply_batch_sync(c_batch, True)
    controller_ids = [r.actor_id for r in c_res if r.error is None]

    walkers = [world.get_actor(w) for w in walker_ids]
    controllers = [world.get_actor(c) for c in controller_ids]
    spawned += walkers + controllers

    for c in controllers:
        c.start()
        c.go_to_location(world.get_random_location_from_navigation())
        c.set_max_speed(random.uniform(1.0, 2.5))  # m/s
    print(f"[Info] Spawned {len(walkers)} walkers.")

    # --- Spectator + RGB camera for screenshot ---
    spectator = world.get_spectator()
    spectator.set_transform(carla.Transform(carla.Location(x=0, y=0, z=120), carla.Rotation(pitch=-90)))

    cam_bp = bp_lib.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', str(args.width))
    cam_bp.set_attribute('image_size_y', str(args.height))
    cam_bp.set_attribute('fov', '90')
    camera = world.spawn_actor(cam_bp, carla.Transform(carla.Location(z=120)))
    spawned.append(camera)

    saved = {'done': False}

    def on_image(img):
        if not saved['done']:
            img.save_to_disk(args.output)
            saved['done'] = True
            print(f"[Info] Screenshot saved -> {args.output}")

    camera.listen(on_image)

    # Let traffic spread for a few seconds and capture a frame
    steps = int(10 / world.get_settings().fixed_delta_seconds)
    for _ in range(steps):
        tick(world)

    print("[Info] Running… Press Ctrl+C to stop.")
    try:
        while True:
            tick(world)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='localhost')
    ap.add_argument('--port', type=int, default=2000)
    ap.add_argument('--tm-port', type=int, default=8000)
    ap.add_argument('--town', default='')                 # e.g., Town03
    ap.add_argument('--vehicles', type=int, default=60)
    ap.add_argument('--walkers', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--output', default='tm_overview.png')
    args = ap.parse_args()
    main(args)
