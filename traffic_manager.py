import argparse
import atexit
import random
import carla


def setup_world_sync(world, fixed_dt=0.05):
    """Enable synchronous mode and return original settings."""
    original = world.get_settings()
    new_settings = world.get_settings()
    new_settings.synchronous_mode = True
    new_settings.fixed_delta_seconds = fixed_dt
    world.apply_settings(new_settings)
    return original


def tick(world):
    world.tick()


def main(args):
    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    world = client.get_world()

    if args.town and world.get_map().name.split('/')[-1] != args.town:
        world = client.load_world(args.town)

    # Setup Traffic Manager
    tm = client.get_trafficmanager(args.tm_port)
    tm.set_synchronous_mode(True)
    tm.set_global_distance_to_leading_vehicle(2.5)
    tm.set_respawn_dormant_vehicles(True)
    tm.set_random_device_seed(args.seed)
    random.seed(args.seed)

    original_settings = setup_world_sync(world, fixed_dt=0.05)
    spawned = []

    def cleanup():
        try:
            print("\n[Cleanup] Destroying spawned actors and restoring settings...")
            client.apply_batch([carla.command.DestroyActor(a.id)
                                for a in spawned if a and a.is_alive])
            tm.set_synchronous_mode(False)
            world.apply_settings(original_settings)
        except Exception as e:
            print("[Cleanup] Warning:", e)
        print("[Cleanup] Done.")

    atexit.register(cleanup)

    bp_lib = world.get_blueprint_library()

    # --- Spawn Vehicles ---
    vehicle_bps = bp_lib.filter('vehicle.*')
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    nveh = min(args.vehicles, len(spawn_points))

    print(f"[Info] Spawning {nveh} vehicles...")
    v_batch = []
    for i in range(nveh):
        bp = random.choice(vehicle_bps)
        if bp.has_attribute('color'):
            bp.set_attribute('color', random.choice(bp.get_attribute('color').recommended_values))
        v_batch.append(carla.command.SpawnActor(bp, spawn_points[i]))

    v_results = client.apply_batch_sync(v_batch, True)
    vehicles = [world.get_actor(r.actor_id) for r in v_results if r.error is None]
    spawned += vehicles

    for v in vehicles:
        v.set_autopilot(True, args.tm_port)

    print(f"[Info] Spawned {len(vehicles)} vehicles.")

    # --- Spawn Walkers ---
    walker_bps = bp_lib.filter('walker.pedestrian.*')
    controller_bp = bp_lib.find('controller.ai.walker')
    walker_spawns = []
    for _ in range(args.walkers):
        loc = world.get_random_location_from_navigation()
        if loc:
            walker_spawns.append(carla.Transform(loc, carla.Rotation(yaw=random.uniform(-180, 180))))

    w_batch = [carla.command.SpawnActor(random.choice(walker_bps), tf)
               for tf in walker_spawns]
    w_results = client.apply_batch_sync(w_batch, True)
    walker_ids = [r.actor_id for r in w_results if r.error is None]

    c_batch = [carla.command.SpawnActor(controller_bp, carla.Transform(), wid)
               for wid in walker_ids]
    c_results = client.apply_batch_sync(c_batch, True)
    controller_ids = [r.actor_id for r in c_results if r.error is None]

    walkers = [world.get_actor(w) for w in walker_ids]
    controllers = [world.get_actor(c) for c in controller_ids]
    spawned += walkers + controllers

    for c in controllers:
        c.start()
        c.go_to_location(world.get_random_location_from_navigation())
        c.set_max_speed(random.uniform(1.0, 2.5))

    print(f"[Info] Spawned {len(walkers)} walkers.")
    print("[Info] Traffic Manager is running. Use the spectator to view traffic.")
    print("[Info] Press Ctrl+C to stop.")

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
    args = ap.parse_args()
    main(args)
