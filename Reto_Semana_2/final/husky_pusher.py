import numpy as np
import matplotlib.pyplot as plt
from robots import HuskyA200
from watcher import Watcher

# CONFIGURACION
dt = 0.05
T = 30

K_omega = 2.0
v_forward = 0.6

CORRIDOR_X = [0, 6]
CORRIDOR_Y = [-1, 1]

LIDAR_RANGE = 1.5
LIDAR_FOV = np.pi / 6


# FUNCIONES

def in_corridor(pos):
    return (CORRIDOR_X[0] <= pos[0] <= CORRIDOR_X[1] and
            CORRIDOR_Y[0] <= pos[1] <= CORRIDOR_Y[1])


def go_to_target(husky, target):
    dx = target[0] - husky.x
    dy = target[1] - husky.y

    theta_target = np.arctan2(dy, dx)

    error_theta = np.arctan2(np.sin(theta_target - husky.theta),
                             np.cos(theta_target - husky.theta))

    v = 0.2 if abs(error_theta) > 0.3 else v_forward
    omega = K_omega * error_theta

    return v, omega, error_theta


def lidar_detect(husky, boxes):
    detected = []

    for box in boxes:
        dx = box["pos"][0] - husky.x
        dy = box["pos"][1] - husky.y

        dist = np.linalg.norm([dx, dy])

        angle = np.arctan2(dy, dx) - husky.theta
        angle = np.arctan2(np.sin(angle), np.cos(angle))

        if dist < LIDAR_RANGE and abs(angle) < LIDAR_FOV:
            detected.append(box)

    return detected


# SIMULACION PRINCIPAL

def simulate_husky_mission():

    boxes = [
        {"pos": np.array([2.0, 0.0]), "pushed": False},
        {"pos": np.array([3.5, 0.5]), "pushed": False},
        {"pos": np.array([4.5, -0.5]), "pushed": False},
    ]

    watcher = Watcher(
        boxes,
        (CORRIDOR_X[0], CORRIDOR_X[1], CORRIDOR_Y[0], CORRIDOR_Y[1])
    )

    husky = HuskyA200()
    husky.set_terrain("grass")

    trajectory = []
    box_traj = [[] for _ in boxes]

    v_cmd_log, v_real_log = [], []
    omega_log = []
    lidar_log = []

    success = False

    for t in np.arange(0, T, dt):

        # condición REAL de terminación
        if watcher.check():
            success = True
            break

        # seleccionar cajas dentro del corredor
        remaining_boxes = [b for b in boxes if in_corridor(b["pos"])]

        if len(remaining_boxes) == 0:
            success = True
            break

        # seleccionar la más cercana
        distances = [
            np.linalg.norm(b["pos"] - np.array([husky.x, husky.y]))
            for b in remaining_boxes
        ]

        target_box = remaining_boxes[np.argmin(distances)]
        box_pos = target_box["pos"]

        # control
        v_cmd, omega_cmd, angle_error = go_to_target(husky, box_pos)

        # cinemática real
        wR1, wR2, wL1, wL2 = husky.inverse_kinematics(v_cmd, omega_cmd)
        v_real, omega_real = husky.forward_kinematics(wR1, wR2, wL1, wL2)

        husky.update_pose(v_real, omega_real, dt)

        # contacto
        dist = np.linalg.norm(box_pos - np.array([husky.x, husky.y]))
        aligned = abs(angle_error) < 0.25

        if dist < 0.35 and aligned:
            target_box["pushed"] = True

        # empuje
        if target_box["pushed"]:

            if box_pos[1] >= 0:
                push_direction = np.array([0, 1])
            else:
                push_direction = np.array([0, -1])

            push_gain = min(1.0, abs(v_real) * 1.5)
            target_box["pos"] += push_direction * push_gain * dt

        # logs
        trajectory.append([husky.x, husky.y])

        v_cmd_log.append(v_cmd)
        v_real_log.append(v_real)
        omega_log.append(omega_real)

        visible_boxes = lidar_detect(husky, boxes)
        lidar_log.append(len(visible_boxes))

        for i, b in enumerate(boxes):
            box_traj[i].append(b["pos"].copy())

    trajectory = np.array(trajectory)

    # GRAFICAS
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'k-', label="Husky")

    colors = ['red', 'blue', 'green']
    for i, traj in enumerate(box_traj):
        traj = np.array(traj)
        if len(traj) > 0:
            plt.plot(traj[:, 0], traj[:, 1], '--',
                     color=colors[i], label=f"Caja {i}")

    plt.plot([0, 6, 6, 0, 0], [-1, -1, 1, 1, -1],
             'r-', linewidth=2)

    plt.title("Husky empujando cajas")
    plt.legend()
    plt.grid()
    plt.axis("equal")

    plt.subplot(2, 1, 2)
    plt.plot(v_cmd_log, label="v_cmd")
    plt.plot(v_real_log, label="v_real")
    plt.plot(omega_log, label="omega_real")
    plt.plot(lidar_log, label="detecciones_lidar")

    plt.legend()
    plt.title("Velocidades y detección")
    plt.grid()

    plt.tight_layout()
    plt.show()

    return success