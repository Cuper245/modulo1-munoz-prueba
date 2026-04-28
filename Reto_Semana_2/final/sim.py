import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from robots import HuskyA200
from anymal import simulate_anymal_transport
from puzzlebot import PuzzleBot, PuzzleCoordinator

# =========================================================
# CONFIG GLOBAL
# =========================================================
dt = 0.05

CORRIDOR_X = [0, 8]   # 🔥 MÁS GRANDE (conecta con trabajo)
CORRIDOR_Y = [-1, 1]

START_ZONE = [-2, 0]
WORK_ZONE = [10, 0]

# =========================================================
# HUSKY CON LOGS
# =========================================================
def simulate_husky():

    husky = HuskyA200()
    husky.set_terrain("grass")

    boxes = [
        {"pos": np.array([2.0, 0.0]), "pushed": False},
        {"pos": np.array([4.0, 0.6]), "pushed": False},
        {"pos": np.array([6.0, -0.6]), "pushed": False},
    ]

    traj = []
    v_log = []
    omega_log = []
    box_traj = [[] for _ in boxes]

    current_box = 0

    for t in np.arange(0, 30, dt):

        if current_box >= len(boxes):
            break

        box = boxes[current_box]

        dx = box["pos"][0] - husky.x
        dy = box["pos"][1] - husky.y

        theta_target = np.arctan2(dy, dx)
        error = np.arctan2(np.sin(theta_target - husky.theta),
                           np.cos(theta_target - husky.theta))

        v = 0.7 if abs(error) < 0.3 else 0.2
        omega = 2.5 * error

        wR1, wR2, wL1, wL2 = husky.inverse_kinematics(v, omega)
        v_real, omega_real = husky.forward_kinematics(wR1, wR2, wL1, wL2)

        husky.update_pose(v_real, omega_real, dt)

        dist = np.linalg.norm(box["pos"] - np.array([husky.x, husky.y]))

        if dist < 0.35:
            box["pushed"] = True

        if box["pushed"]:
            direction = np.array([0, 1]) if box["pos"][1] >= 0 else np.array([0, -1])
            box["pos"] += direction * abs(v_real) * dt

        if not (CORRIDOR_Y[0] <= box["pos"][1] <= CORRIDOR_Y[1]):
            current_box += 1

        traj.append([husky.x, husky.y])
        v_log.append(v_real)
        omega_log.append(omega_real)

        for i, b in enumerate(boxes):
            box_traj[i].append(b["pos"].copy())

    return np.array(traj), np.array(v_log), np.array(omega_log), [np.array(b) for b in box_traj]


# =========================================================
# PUZZLEBOTS
# =========================================================
def simulate_puzzlebots():

    bots = [
        PuzzleBot("PB1", [8.5, -1.2]),
        PuzzleBot("PB2", [8.5,  0.0]),
        PuzzleBot("PB3", [8.5,  1.2]),
    ]

    boxes = [
        {"name":"A", "pos":np.array([9.5, -1.0, 0.0]), "stacked":False},
        {"name":"B", "pos":np.array([9.0,  0.8, 0.0]), "stacked":False},
        {"name":"C", "pos":np.array([10.2, 1.2, 0.0]), "stacked":False},
    ]

    stack = np.array([11.0, 0.0, 0.0])

    coord = PuzzleCoordinator(bots, boxes, stack)

    traj_bots = {b.name: [] for b in bots}
    traj_boxes = {b["name"]: [] for b in boxes}

    for _ in range(100):

        coord.step()

        for b in bots:
            traj_bots[b.name].append([b.x, b.y])

        for box in boxes:
            traj_boxes[box["name"]].append(box["pos"][:2].copy())

    return traj_bots, traj_boxes, stack


# =========================================================
# ANIMACIÓN COMPLETA PRO
# =========================================================
def animate_all():

    husky_traj, v_log, omega_log, husky_boxes = simulate_husky()
    anymal_success, anymal_traj = simulate_anymal_transport()
    pb_traj, pb_boxes, stack = simulate_puzzlebots()

    # 🔥 acelerar ANYmal solo visualmente
    anymal_traj = anymal_traj[::3]

    fig, ax = plt.subplots(figsize=(12,6))

    ax.set_xlim(-2, 12)
    ax.set_ylim(-3, 3)
    ax.grid()

    # ZONAS
    ax.add_patch(plt.Rectangle((-2,-2),2,4,fill=False,linestyle='--'))
    ax.add_patch(plt.Rectangle((0,-1),8,2,fill=False,linestyle='--'))
    ax.add_patch(plt.Rectangle((8,-2),4,4,fill=False,linestyle='--'))

    ax.text(-1.5,1.5,"INICIO")
    ax.text(3,1.3,"CORREDOR")
    ax.text(9,1.5,"TRABAJO")

    # ROBOTS
    husky_dot, = ax.plot([], [], 'ko', markersize=8, label="Husky")
    anymal_dot, = ax.plot([], [], 'bo', markersize=6, label="ANYmal")

    pb_dots = {name: ax.plot([], [], 'o')[0] for name in pb_traj}

    # CAJAS VISUALES (🔥 mejor diseño)
    husky_box_dots = [ax.plot([], [], 'o', alpha=0.5)[0] for _ in husky_boxes]
    pb_box_dots = {name: ax.plot([], [], 's')[0] for name in pb_boxes}

    stack_dot, = ax.plot(stack[0], stack[1], 'k*', markersize=12)

    ax.legend()

    total_frames = len(husky_traj) + len(anymal_traj) + len(next(iter(pb_traj.values())))

    def update(frame):

        # ---------------- HUSKY ----------------
        if frame < len(husky_traj):

            x, y = husky_traj[frame]
            husky_dot.set_data([x], [y])

            for i, b in enumerate(husky_boxes):
                if frame < len(b):
                    husky_box_dots[i].set_data([b[frame][0]], [b[frame][1]])

        # ---------------- ANYMAL ----------------
        elif frame < len(husky_traj) + len(anymal_traj):

            idx = frame - len(husky_traj)
            x, y = anymal_traj[idx]

            anymal_dot.set_data([x], [y])

            # 🔥 detener en zona de trabajo
            if x >= 8:
                anymal_dot.set_color('green')

        # ---------------- PUZZLEBOTS ----------------
        else:

            idx = frame - len(husky_traj) - len(anymal_traj)

            for name in pb_traj:
                data = np.array(pb_traj[name])
                if idx < len(data):
                    pb_dots[name].set_data([data[idx][0]], [data[idx][1]])

            for name in pb_boxes:
                data = np.array(pb_boxes[name])
                if idx < len(data):
                    pb_box_dots[name].set_data([data[idx][0]], [data[idx][1]])

        return list(pb_dots.values()) + list(pb_box_dots.values()) + husky_box_dots + [husky_dot, anymal_dot]

    ani = FuncAnimation(fig, update, frames=total_frames, interval=50, blit=True)

    plt.show()


# =========================================================
# GRÁFICAS PRO
# =========================================================
def plot_all():

    husky_traj, v_log, omega_log, _ = simulate_husky()
    _, anymal_traj = simulate_anymal_transport()

    # HUSKY
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(husky_traj[:,0], husky_traj[:,1])
    plt.title("Trayectoria Husky")
    plt.axis("equal")
    plt.grid()

    plt.subplot(1,2,2)
    plt.plot(v_log, label="v")
    plt.plot(omega_log, label="omega")
    plt.legend()
    plt.title("Velocidades Husky")
    plt.grid()

    plt.show()

    # ANYMAL
    plt.figure(figsize=(6,5))
    plt.plot(anymal_traj[:,0], anymal_traj[:,1])
    plt.title("Trayectoria ANYmal")
    plt.axis("equal")
    plt.grid()
    plt.show()


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    print("🚀 SIMULACIÓN MULTIROBOT FINAL PRO")

    plot_all()      # gráficas completas
    animate_all()   # animación por fases