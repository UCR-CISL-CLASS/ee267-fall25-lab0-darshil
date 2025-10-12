import argparse, atexit, random, time
import carla

def main(args):
    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)

    world = client.get_world()
    if args.town:
        if world.get_map().name.split('/')[-1] != args.town:
            world = client.load_world(args.town)

    tm = client.get_trafficmanager(args.tm_port)
    tm.set_global_distance_to_leading_vehicle(2.5)
    tm.set_synchronous_mode(True)  # keep TM in lockstep with the world
    tm.set_respawn_dormant_vehicles(True)

    settings = world.get_settings()
    original_settings = carla.WorldSettings(
        synchronous_mode=settings.synchronous_mode,
        fixed_delta_seconds=settings.fixed_delta_seconds,
        no_rendering_mode=settings.no_rendering_mode,
        max_substeps=settings.max_substeps,
        max_substep_delta_time=settings.max_substep_delta_time,
    )

    # Clean up on exit
    spawned_actors = []

    def cleanup():
        try:
            print("Cleaning up…")
            client.apply_batch([carla.command.DestroyActor(x.id) for x in spawned_actors if x.is_alive])
            tm.set_synchronous_mode(False)
            world.apply_settings(original_settings)
        except Exception as e:
            print("Cleanup warning:", e)

    atexit.register(cleanup)

    # Switch world to synchronous for stability
    new_settings = world.get_settings()
    new_settings.synchronous_mode = True
    new_settings.fixed_delta_seconds = 0.05
    world.apply_settings(new_settings)

    # Helper: advance one frame
    def tick():
        world.tick()

    blueprint_library = world.get_blueprint_library()

    # ------------------ VEHICLES ------------------
    vehicle_bps = blueprint_library.filter('vehicle.*')
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    num_vehicles = min(args.vehicles, len(spawn_points))

    batch = []
    for i in range(num_vehicles):
        bp = random.choice(vehicle_bps)
        # randomize color if available
        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        batch.append(carla.command.SpawnActor(bp, spawn_points[i]))

    results = client.apply_batch_sync(batch, True)
    vehicles = [r.actor_id for r in results if r.error is None]
    spawned_actors += [world.get_actor(vid) for vid in vehicles]

    # hand vehicles to the Traffic Manager
    for v in spawned_actors:
        v.set_autopilot(True, args.tm_port)

    print(f"Spawned {len(vehicles)} vehicles")

    # ------------------ WALKERS ------------------
    walker_bp = blueprint_library.filter('walker.pedestrian.*')
    controller_bp = blueprint_library.find('controller.ai.walker')

    walker_transforms = []
    for _ in range(args.walkers):
        loc = world.get_random_location_from_navigation()
        if loc:
            yaw = random.uniform(-180.0, 180.0)
            walker_transforms.append(carla.Transform(loc, carla.Rotation(yaw=yaw)))

    walker_batch = [carla.command.SpawnActor(random.choice(walker_bp), tf)
                    for tf in walker_transforms]
    walker_results = client.apply_batch_sync(walker_batch, True)
    walker_ids = [r.actor_id for r in walker_results if r.error is None]

    controller_batch = [carla.command.SpawnActor(controller_bp, carla.Transform(), wid)
                        for wid in walker_ids]
    controller_results = client.apply_batch_sync(controller_batch, True)
    controller_ids = [r.actor_id for r in controller_results if r.error is None]

    walkers = [world.get_actor(w) for w in walker_ids]
    controllers = [world.get_actor(c) for c in controller_ids]
    spawned_actors += walkers + controllers

    for c in controllers:
        c.start()
        c.go_to_location(world.get_random_location_from_navigation())
        c.set_max_speed(random.uniform(1.0, 2.5))  # m/s

    print(f"Spawned {len(walkers)} walkers")

    # ------------------ SPECTATOR + SCREENSHOT ------------------
    # Put spectator high above the town center and save an overview shot
    spectator = world.get_spectator()
    # pick a decent vantage point
    sp = carla.Transform(carla.Location(x=0, y=0, z=120), carla.Rotation(pitch=-90))
    spectator.set_transform(sp)

    # Spawn a RGB camera to capture one image
    cam_bp = blueprint_library.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', str(args.width))
    cam_bp.set_attribute('image_size_y', str(args.height))
    cam_bp.set_attribute('fov', '90')

    camera = world.spawn_actor(cam_bp, carla.Transform(carla.Location(z=120)), attach_to=None)
    spawned_actors.append(camera)

    image_saved = {'done': False}
    def _on_img(img):
        if not image_saved['done']:
            img.save_to_disk('traffic_overview.png')
            image_saved['done'] = True
    camera.listen(_on_img)

    # run a few seconds so AI spreads out and camera grabs a frame
    for _ in range(int(10 / new_settings.fixed_delta_seconds)):
        tick()

    print("Saved screenshot -> traffic_overview.png")
    print("Press Ctrl+C to stop; cleanup will run automatically.")
    # keep running until user stops
    try:
        while True:
            tick()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='localhost')
    ap.add_argument('--port', type=int, default=2000)
    ap.add_argument('--tm-port', type=int, default=8000)
    ap.add_argument('--vehicles', type=int, default=60)
    ap.add_argument('--walkers', type=int, default=40)
    ap.add_argument('--town', default='')        # e.g., Town03
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=720)
    args = ap.parse_args()
    main(args)
