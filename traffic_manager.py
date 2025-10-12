import argparse
import atexit
import random
import sys
import time
from typing import List

import carla


def trace(*a):
    print("[TRACE]", *a)


def info(*a):
    print("[Info]", *a)


def warn(*a):
    print("[Warn]", *a, file=sys.stderr)


def err(*a):
    print("[Error]", *a, file=sys.stderr)


def setup_world_sync(world: carla.World, fixed_dt: float = 0.05) -> carla.WorldSettings:
    """
    Save original world settings, then flip to synchronous mode.
    Avoid constructing WorldSettings(...) directly (signature varies between builds).
    """
    original = world.get_settings()
    new_settings = world.get_settings()
    new_settings.synchronous_mode = True
    new_settings.fixed_delta_seconds = fixed_dt
    world.apply_settings(new_settings)
    s = world.get_settings()
    trace("World settings -> sync:", s.synchronous_mode, "fixed_dt:", s.fixed_delta_seconds)
    return original


def world_tick(world: carla.World, steps: int = 1):
    """Advance the world deterministically."""
    for _ in range(steps):
        world.tick()


def clear_npcs(client: carla.Client, world: carla.World):
    """Destroy all vehicle.* and walker.* actors (leave hero/sensors alone)."""
    actors = world.get_actors()
    targets = [a for a in actors if a.is_alive and (
        a.type_id.startswith('vehicle.') or a.type_id.startswith('walker.')
    )]
    if not targets:
        info("No existing NPCs to clear.")
        return
    info(f"Clearing {len(targets)} NPC actors…")
    client.apply_batch([carla.command.DestroyActor(a) for a in targets])


# ----------------------------- Spawners -----------------------------

def spawn_vehicles_robust(client: carla.Client,
                          world: carla.World,
                          tm: carla.TrafficManager,
                          desired: int,
                          tm_port: int,
                          max_attempts_per_vehicle: int = 5) -> List[carla.Actor]:
    """Try hard to spawn N vehicles; print failures; hand them to TM."""
    bp_lib = world.get_blueprint_library()
    veh_bps = bp_lib.filter('vehicle.*')
    points = world.get_map().get_spawn_points()
    random.shuffle(points)
    if not points:
        warn("Map has zero spawn points; load a larger town (e.g., Town03).")
        return []

    target = min(desired, len(points))
    spawned_ids = []
    attempts = 0
    i = 0
    info(f"Spawning up to {target} vehicles…")

    # Try multiple points for each vehicle to overcome overlaps/occupied spots
    while i < target and attempts < target * max_attempts_per_vehicle:
        attempts += 1
        bp = random.choice(veh_bps)
        if bp.has_attribute('color'):
            bp.set_attribute('color', random.choice(bp.get_attribute('color').recommended_values))
        sp = points[attempts % len(points)]

        res = client.apply_batch_sync([carla.command.SpawnActor(bp, sp)], True)[0]
        if res.error:
            warn(f"[vehicle] fail @attempt {attempts}: {res.error}")
            continue
        spawned_ids.append(res.actor_id)
        i += 1

    vehicles = [world.get_actor(i) for i in spawned_ids if i]
    for v in vehicles:
        v.set_autopilot(True, tm_port)
    info(f"Spawned {len(vehicles)}/{target} vehicles")
    return vehicles


def spawn_walkers_robust(client: carla.Client,
                         world: carla.World,
                         desired: int,
                         max_attempts_per_walker: int = 10) -> (List[carla.Actor], List[carla.Actor]):
    """
    Spawn pedestrians with AI controllers; returns (walkers, controllers).
    Requires a navmesh; if get_random_location_from_navigation() returns None repeatedly,
    switch to a town with navigation data (e.g., Town03/Town05).
    """
    bp_lib = world.get_blueprint_library()
    walker_bps = bp_lib.filter('walker.pedestrian.*')
    ctrl_bp = bp_lib.find('controller.ai.walker')

    walker_ids, ctrl_ids = [], []
    attempts = 0

    info(f"Spawning up to {desired} walkers…")
    while len(walker_ids) < desired and attempts < desired * max_attempts_per_walker:
        attempts += 1
        loc = world.get_random_location_from_navigation()
        if not loc:
            if attempts % 10 == 0:
                warn("[walker] nav location None (map may lack navmesh); consider Town03/Town05.")
            continue
        tf = carla.Transform(loc, carla.Rotation(yaw=random.uniform(-180, 180)))
        res = client.apply_batch_sync([carla.command.SpawnActor(random.choice(walker_bps), tf)], True)[0]
        if res.error:
            warn(f"[walker] fail @attempt {attempts}: {res.error}")
            continue
        walker_ids.append(res.actor_id)

    if walker_ids:
        ctrl_batch = [carla.command.SpawnActor(ctrl_bp, carla.Transform(), wid) for wid in walker_ids]
        ctrl_res = client.apply_batch_sync(ctrl_batch, True)
        for r in ctrl_res:
            if r.error:
                warn(f"[walker-ctrl] fail: {r.error}")
            else:
                ctrl_ids.append(r.actor_id)

    walkers = [world.get_actor(w) for w in walker_ids if w]
    controllers = [world.get_actor(c) for c in ctrl_ids if c]

    for c in controllers:
        try:
            c.start()
            dest = world.get_random_location_from_navigation()
            if dest:
                c.go_to_location(dest)
            c.set_max_speed(random.uniform(1.0, 2.5))  # m/s
        except Exception as e:
            warn("walker controller start/go_to_location error:", e)

    info(f"Spawned {len(walkers)}/{desired} walkers")
    return walkers, controllers


# ----------------------------- Main -----------------------------

def main(args):
    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)

    world = client.get_world()
    current_map = world.get_map().name
    trace("Connected. Map:", current_map)

    # Load requested town if different
    if args.town:
        want = args.town
        if current_map.split('/')[-1] != want:
            info(f"Loading town: {want}")
            world = client.load_world(want)
            world_tick(world, 10)  # settle a bit
            trace("Loaded map:", world.get_map().name)

    # Traffic Manager
    tm = client.get_trafficmanager(args.tm_port)
    tm.set_synchronous_mode(True)
    tm.set_global_distance_to_leading_vehicle(2.5)
    tm.set_respawn_dormant_vehicles(True)
    tm.set_random_device_seed(args.seed)
    random.seed(args.seed)
    trace("TM port:", args.tm_port, "seed:", args.seed)

    # Switch world to synchronous
    original_settings = setup_world_sync(world, fixed_dt=0.05)

    # Prepare cleanup
    spawned = []

    def cleanup():
        try:
            info("Cleaning up…")
            # stop walker controllers first
            for a in spawned:
                try:
                    if a and a.is_alive and 'controller.ai.walker' in a.type_id:
                        a.stop()
                except Exception:
                    pass
            client.apply_batch([carla.command.DestroyActor(a.id)
                                for a in spawned if a and a.is_alive])
        except Exception as e:
            warn("Destroy warning:", e)
        try:
            tm.set_synchronous_mode(False)
        except Exception:
            pass
        try:
            world.apply_settings(original_settings)
        except Exception as e:
            warn("Restore settings warning:", e)
        info("Done.")
    atexit.register(cleanup)

    # Optionally clear existing NPCs (helps when spawn points are occupied)
    if args.clear:
        clear_npcs(client, world)
        world_tick(world, 5)

    # Diagnostics
    spawn_points = world.get_map().get_spawn_points()
    trace("#spawn points:", len(spawn_points))
    trace("nav test:", world.get_random_location_from_navigation())

    # Spawn vehicles
    vehicles = spawn_vehicles_robust(client, world, tm, args.vehicles, args.tm_port)
    spawned += vehicles

    # Spawn walkers
    walkers, controllers = spawn_walkers_robust(client, world, args.walkers)
    spawned += walkers + controllers

    # Optional: position spectator at a top-down vantage point
    if args.spectator_topdown:
        spectator = world.get_spectator()
        spectator.set_transform(
            carla.Transform(
                carla.Location(x=0.0, y=0.0, z=120.0),
                carla.Rotation(pitch=-90.0)
            )
        )

    info("Traffic Manager running. Open a visual client (e.g., manual_control.py) to view.")
    info("Press Ctrl+C to stop.")
    try:
        while True:
            world_tick(world, 1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--tm-port", type=int, default=8000)
    ap.add_argument("--town", default="", help="e.g., Town03, Town05")
    ap.add_argument("--vehicles", type=int, default=60)
    ap.add_argument("--walkers", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clear", action="store_true", help="Clear existing NPCs before spawning")
    ap.add_argument("--spectator-topdown", action="store_true", help="Jump spectator to a top-down view")
    args = ap.parse_args()
    main(args)
