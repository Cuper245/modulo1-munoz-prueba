import numpy as np
import matplotlib.pyplot as plt
from robots import HuskyA200
from watcher import Watcher

# ==============================
# CONFIGURACION
# ==============================
dt = 0.05
T = 30

K_omega = 2.0
v_forward = 0.6

CORRIDOR_X = [0, 6]
CORRIDOR_Y = [-1, 1]

# ==============================
# FUNCIONES
# ==============================

def in_corridor(pos):
    return (CORRIDOR_X[0] <= pos[0] <= CORRIDOR_X[1] and
            CORRIDOR_Y[0] <= pos[1] <= CORRIDOR_Y[1])


def go_to_target(husky, target):
    dx = target[0] - husky.x
    dy = target[1] - husky.y

    theta_target = np.arctan2(dy, dx)

    error_theta = theta_target - husky.theta
    error_theta = np.arctan2(np.sin(error_theta), np.cos(error_theta))

    # control tipo unicycle
    v = 0.2 if abs(error_theta) > 0.3 else v_forward
    omega = K_omega * error_theta

    return v, omega, error_theta


# ==============================
# SIMULACION PRINCIPAL
# ==============================

def simulate_husky_mission():

    # 🔥 IMPORTANTE: resetear cajas cada vez
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
    husky.set_terrain("grass")  # slip realista

    trajectory = []
    box_traj = [[] for _ in boxes]

    # logs
    v_cmd_log, v_real_log = [], []
    omega_log = []

    current_box = 0

    for t in np.arange(0, T, dt):

        if current_box >= len(boxes):
            print("🎯 Todas las cajas procesadas")
            break

        box = boxes[current_box]
        box_pos = box["pos"]

        # ==============================
        # CONTROL
        # ==============================
        v_cmd, omega_cmd, angle_error = go_to_target(husky, box_pos)

        # ==============================
        # CINEMATICA REAL
        # ==============================
        wR1, wR2, wL1, wL2 = husky.inverse_kinematics(v_cmd, omega_cmd)
        v_real, omega_real = husky.forward_kinematics(wR1, wR2, wL1, wL2)

        husky.update_pose(v_real, omega_real, dt)

        # ==============================
        # CONTACTO
        # ==============================
        dist = np.linalg.norm(box_pos - np.array([husky.x, husky.y]))

        if dist < 0.35 and abs(angle_error) < 0.25:
            box["pushed"] = True

        pushing = box["pushed"]

        # ==============================
        # EMPUJE MEJORADO
        # ==============================
        if pushing:

            # 🔥 empujar en dirección normal al corredor
            if box_pos[1] >= 0:
                push_direction = np.array([0, 1])
            else:
                push_direction = np.array([0, -1])

            box["pos"] += push_direction * abs(v_real) * dt

        # ==============================
        # SALIO DEL CORREDOR
        # ==============================
        if not in_corridor(box["pos"]):
            print(f"✅ Caja {current_box} fuera del corredor")
            current_box += 1

        # ==============================
        # WATCHER
        # ==============================
        if watcher.check():
            print("🎯 FASE HUSKY COMPLETADA")
            success = True
            break

        # ==============================
        # LOGS
        # ==============================
        trajectory.append([husky.x, husky.y])

        v_cmd_log.append(v_cmd)
        v_real_log.append(v_real)
        omega_log.append(omega_real)

        for i, b in enumerate(boxes):
            box_traj[i].append(b["pos"].copy())

        print(f"t={t:.2f} | v_cmd={v_cmd:.2f}, v_real={v_real:.2f}, omega={omega_real:.2f}")

    else:
        success = False

    # ==============================
    # GRAFICAS
    # ==============================
    trajectory = np.array(trajectory)

    plt.figure(figsize=(10,8))

    # Trayectoria
    plt.subplot(2,1,1)
    plt.plot(trajectory[:,0], trajectory[:,1], 'k-', label="Husky")

    colors = ['red', 'blue', 'green']
    for i, traj in enumerate(box_traj):
        traj = np.array(traj)
        if len(traj) > 0:
            plt.plot(traj[:,0], traj[:,1], '--', color=colors[i], label=f"Caja {i}")

    plt.plot([0,6,6,0,0], [-1,-1,1,1,-1], 'r-', linewidth=2)

    plt.title("Husky empujando cajas")
    plt.legend()
    plt.grid()
    plt.axis("equal")

    # Velocidades
    plt.subplot(2,1,2)
    plt.plot(v_cmd_log, label="v_cmd")
    plt.plot(v_real_log, label="v_real")
    plt.plot(omega_log, label="omega_real")

    plt.legend()
    plt.title("Velocidades")
    plt.grid()

    plt.tight_layout()
    plt.show()

    return success