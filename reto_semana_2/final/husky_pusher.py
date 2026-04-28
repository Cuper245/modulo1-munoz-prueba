# Test de control de Husky para empujar cajas fuera del corredor usando solo LiDAR
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from robots import HuskyA200
from watcher import Watcher

dt = 0.05
T = 45

K_omega = 2.0
v_forward = 0.6

CORRIDOR_X = [0, 6]
CORRIDOR_Y = [-1, 1]

LIDAR_RANGE = 1.5
LIDAR_FOV = np.pi / 6
N_RAYS = 32
BOX_SIZE = 0.35
N_BOXES = 3


def in_corridor(pos):
    return (
        CORRIDOR_X[0] <= pos[0] <= CORRIDOR_X[1]
        and CORRIDOR_Y[0] <= pos[1] <= CORRIDOR_Y[1]
    )


def create_random_boxes(n_boxes=N_BOXES, seed=None):
    rng = np.random.default_rng(seed)
    boxes = []

    x_positions = np.linspace(1.5, 5.0, n_boxes)

    for i in range(n_boxes):
        x = x_positions[i] + rng.uniform(-0.25, 0.25)
        y = rng.uniform(-0.65, 0.65)

        boxes.append({
            "pos": np.array([x, y], dtype=float),
            "pushed": False
        })

    return boxes


def ray_segment_intersect(ox, oy, dx, dy, ax, ay, bx, by):
    denom = dx * (by - ay) - dy * (bx - ax)

    if abs(denom) < 1e-10:
        return np.inf

    t = ((ax - ox) * (by - ay) - (ay - oy) * (bx - ax)) / denom
    s = ((ax - ox) * dy - (ay - oy) * dx) / denom

    if t >= 0 and 0 <= s <= 1:
        return t

    return np.inf


def box_to_segments(box_pos, size=BOX_SIZE):
    x, y = box_pos
    r = size / 2

    x0, x1 = x - r, x + r
    y0, y1 = y - r, y + r

    return [
        ((x0, y0), (x1, y0)),
        ((x1, y0), (x1, y1)),
        ((x1, y1), (x0, y1)),
        ((x0, y1), (x0, y0)),
    ]


def lidar_scan(husky, boxes):
    readings = np.full(N_RAYS, LIDAR_RANGE)
    hit_boxes = [None for _ in range(N_RAYS)]
    ray_lines = []

    relative_angles = np.linspace(
        -LIDAR_FOV,
        LIDAR_FOV,
        N_RAYS
    )

    angles = husky.theta + relative_angles

    all_segments = []

    for box_idx, box in enumerate(boxes):
        if not in_corridor(box["pos"]):
            continue

        for segment in box_to_segments(box["pos"]):
            all_segments.append((box_idx, segment))

    for i, angle in enumerate(angles):
        ray_dx = np.cos(angle)
        ray_dy = np.sin(angle)

        min_distance = LIDAR_RANGE
        detected_box = None

        for box_idx, ((ax, ay), (bx, by)) in all_segments:
            distance = ray_segment_intersect(
                husky.x,
                husky.y,
                ray_dx,
                ray_dy,
                ax,
                ay,
                bx,
                by
            )

            if distance < min_distance:
                min_distance = distance
                detected_box = box_idx

        readings[i] = min_distance
        hit_boxes[i] = detected_box

        x_end = husky.x + ray_dx * min_distance
        y_end = husky.y + ray_dy * min_distance

        ray_lines.append([
            [husky.x, husky.y],
            [x_end, y_end],
            detected_box
        ])

    return readings, hit_boxes, ray_lines, relative_angles


def lidar_navigation_control(husky, lidar_readings, hit_boxes, t):
    detected = lidar_readings < LIDAR_RANGE

    # Si está cerca de los bordes del corredor, corrige hacia el centro
    margin = 0.25

    if husky.y > CORRIDOR_Y[1] - margin:
        return 0.25, -1.2, False, None

    if husky.y < CORRIDOR_Y[0] + margin:
        return 0.25, 1.2, False, None

    # Si detecta caja, gira hacia el rayo más cercano
    if np.any(detected):
        closest_ray = int(np.argmin(lidar_readings))

        relative_angles = np.linspace(
            -LIDAR_FOV,
            LIDAR_FOV,
            N_RAYS
        )

        target_angle = relative_angles[closest_ray]

        v = 0.25 if abs(target_angle) > 0.25 else 0.7
        omega = 2.5 * target_angle

        return v, omega, True, closest_ray

    # Si no detecta nada, avanza casi recto con exploración leve
    v = 0.45

    # Corrección suave hacia el centro del corredor
    center_y = 0.0
    error_y = center_y - husky.y

    omega = 0.8 * error_y + 0.15 * np.sin(1.5 * t)

    return v, omega, False, None


def simulate_husky_mission(show_lidar=True, return_dataset=False, seed=None):

    boxes = create_random_boxes(seed=seed)

    watcher = Watcher(
        boxes,
        (CORRIDOR_X[0], CORRIDOR_X[1], CORRIDOR_Y[0], CORRIDOR_Y[1])
    )

    husky = HuskyA200()
    husky.set_terrain("grass")

    trajectory = []
    box_traj = [[] for _ in boxes]
    lidar_rays_traj = []

    v_cmd_log = []
    v_real_log = []
    omega_log = []
    lidar_log = []

    lidar_dataset = []

    success = False
    pushing_box_idx = None

    for t in np.arange(0, T, dt):

        if watcher.check():
            success = True
            break

        boxes_in_corridor = [b for b in boxes if in_corridor(b["pos"])]

        if len(boxes_in_corridor) == 0:
            success = True
            break

        lidar_readings, hit_boxes, ray_lines, relative_angles = lidar_scan(
            husky,
            boxes
        )

        detections_lidar = int(np.sum(lidar_readings < LIDAR_RANGE))
        box_detected = 1 if detections_lidar > 0 else 0

        detected = False
        closest_ray = None

        if pushing_box_idx is not None and in_corridor(boxes[pushing_box_idx]["pos"]):
            box = boxes[pushing_box_idx]

            v_cmd = 0.45
            omega_cmd = 0.0
            detected = True

        else:
            pushing_box_idx = None

            v_cmd, omega_cmd, detected, closest_ray = lidar_navigation_control(
                lidar_readings,
                relative_angles,
                t
            )

        wR1, wR2, wL1, wL2 = husky.inverse_kinematics(v_cmd, omega_cmd)
        v_real, omega_real = husky.forward_kinematics(wR1, wR2, wL1, wL2)

        husky.update_pose(v_real, omega_real, dt)

        if pushing_box_idx is None and detected and closest_ray is not None:
            detected_box_idx = hit_boxes[closest_ray]
            detected_distance = lidar_readings[closest_ray]

            if detected_box_idx is not None and detected_distance < 0.38:
                pushing_box_idx = detected_box_idx
                boxes[pushing_box_idx]["pushed"] = True

        if pushing_box_idx is not None:
            box = boxes[pushing_box_idx]

            if box["pos"][1] >= 0:
                push_direction = np.array([0, 1])
            else:
                push_direction = np.array([0, -1])

            push_gain = min(1.0, abs(v_real) * 1.8)
            box["pos"] += push_direction * push_gain * dt

            if not in_corridor(box["pos"]):
                pushing_box_idx = None

        trajectory.append([husky.x, husky.y])
        lidar_rays_traj.append(ray_lines)

        v_cmd_log.append(v_cmd)
        v_real_log.append(v_real)
        omega_log.append(omega_real)
        lidar_log.append(detections_lidar)

        row = {
            "t": t,
            "husky_x": husky.x,
            "husky_y": husky.y,
            "husky_theta": husky.theta,
            "v_cmd": v_cmd,
            "v_real": v_real,
            "omega_real": omega_real,
            "d_min": lidar_readings.min(),
            "d_mean": lidar_readings.mean(),
            "d_std": lidar_readings.std(),
            "detections_lidar": detections_lidar,
            "box_detected": box_detected,
            "pushing_box": -1 if pushing_box_idx is None else pushing_box_idx
        }

        for k, d in enumerate(lidar_readings):
            row[f"lidar_{k:02d}"] = d

        lidar_dataset.append(row)

        for i, b in enumerate(boxes):
            box_traj[i].append(b["pos"].copy())

    trajectory = np.array(trajectory)

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)

    if len(trajectory) > 0:
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'k-', label="Husky")

    if show_lidar and len(lidar_rays_traj) > 0:
        last_rays = lidar_rays_traj[-1]

        for start, end, detected_box in last_rays:
            color = "orange" if detected_box is not None else "gray"
            alpha = 0.9 if detected_box is not None else 0.25
            linewidth = 1.4 if detected_box is not None else 0.8

            plt.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color=color,
                alpha=alpha,
                linewidth=linewidth
            )

    colors = ['red', 'blue', 'green', 'purple', 'brown']

    for i, traj in enumerate(box_traj):
        traj = np.array(traj)

        if len(traj) > 0:
            plt.plot(
                traj[:, 0],
                traj[:, 1],
                '--',
                color=colors[i % len(colors)],
                label=f"Caja {i}"
            )

            last_x, last_y = traj[-1]
            plt.gca().add_patch(
                Rectangle(
                    (last_x - BOX_SIZE / 2, last_y - BOX_SIZE / 2),
                    BOX_SIZE,
                    BOX_SIZE,
                    fill=True,
                    alpha=0.35,
                    color=colors[i % len(colors)]
                )
            )

    plt.plot(
        [0, 6, 6, 0, 0],
        [-1, -1, 1, 1, -1],
        'r-',
        linewidth=2,
        label="Corredor"
    )

    plt.title("Husky limpiando corredor usando LiDAR")
    plt.legend()
    plt.grid()
    plt.axis("equal")

    plt.subplot(2, 1, 2)
    plt.plot(v_cmd_log, label="v_cmd")
    plt.plot(v_real_log, label="v_real")
    plt.plot(omega_log, label="omega_real")
    plt.plot(lidar_log, label="detecciones_lidar")

    plt.legend()
    plt.title("Control basado en LiDAR")
    plt.grid()

    plt.tight_layout()
    plt.show()

    if return_dataset:
        return success, lidar_dataset

    return success


if __name__ == "__main__":
    success, dataset = simulate_husky_mission(
        show_lidar=True,
        return_dataset=True,
        seed=None
    )

    print("Misión exitosa:", success)
    print("Samples LiDAR:", len(dataset))