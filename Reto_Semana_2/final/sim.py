import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from husky_pusher import simulate_husky_mission

from robots import HuskyA200
from anymal import simulate_anymal_transport
from puzzlebot import PuzzleBot, PuzzleCoordinator
from husky_pusher import simulate_husky_mission

# =========================================================
# CONFIG GLOBAL
# =========================================================
dt = 0.05

CORRIDOR_X = [0, 8]
CORRIDOR_Y = [-1, 1]

START_ZONE = [-2, 0]
WORK_ZONE = [10, 0]

N_BOXES = 3

LIDAR_RANGE = 1.5
LIDAR_FOV = np.pi / 6
N_RAYS = 32
BOX_SIZE = 0.35


# =========================================================
# CAJAS RANDOM
# =========================================================
def create_random_husky_boxes(n_boxes=N_BOXES, seed=None):
    rng = np.random.default_rng(seed)

    boxes = []
    x_positions = np.linspace(1.5, 6.5, n_boxes)

    for i in range(n_boxes):
        x = x_positions[i] + rng.uniform(-0.3, 0.3)
        y = rng.uniform(-0.65, 0.65)

        boxes.append({
            "pos": np.array([x, y], dtype=float),
            "pushed": False
        })

    return boxes


# =========================================================
# LIDAR
# =========================================================
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

    angles = husky.theta + np.linspace(
        -LIDAR_FOV,
        LIDAR_FOV,
        N_RAYS
    )

    all_segments = []

    for box_idx, box in enumerate(boxes):
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

    return readings, hit_boxes, ray_lines


# =========================================================
# HUSKY CON LOGS + LIDAR (USANDO HUSKY_PUSHER)
# =========================================================
def simulate_husky(seed=None):

    success, logs = simulate_husky_mission(
        show_lidar=False,
        return_dataset=True,
        seed=seed
    )

    traj = logs["trajectory"]
    v_log = logs["v_real_log"]
    omega_log = logs["omega_log"]
    lidar_log = logs["lidar_log"]
    box_traj = logs["box_traj"]
    lidar_rays_traj = logs["lidar_rays_traj"]

    return (
        traj,
        v_log,
        omega_log,
        lidar_log,
        box_traj,
        lidar_rays_traj
    )

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
        {"name": "A", "pos": np.array([9.5, -1.0, 0.0]), "stacked": False},
        {"name": "B", "pos": np.array([9.0,  0.8, 0.0]), "stacked": False},
        {"name": "C", "pos": np.array([10.2, 1.2, 0.0]), "stacked": False},
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
def animate_all(seed=None):

    husky_traj, v_log, omega_log, lidar_log, husky_boxes, lidar_rays = simulate_husky(seed=seed)
    anymal_success, anymal_traj = simulate_anymal_transport()
    pb_traj, pb_boxes, stack = simulate_puzzlebots()

    anymal_traj = anymal_traj[::3]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.set_xlim(-2, 12)
    ax.set_ylim(-3, 3)
    ax.grid()

    ax.add_patch(plt.Rectangle((-2, -2), 2, 4, fill=False, linestyle='--'))
    ax.add_patch(plt.Rectangle((0, -1), 8, 2, fill=False, linestyle='--'))
    ax.add_patch(plt.Rectangle((8, -2), 4, 4, fill=False, linestyle='--'))

    ax.text(-1.5, 1.5, "INICIO")
    ax.text(3, 1.3, "CORREDOR")
    ax.text(9, 1.5, "TRABAJO")

    husky_dot, = ax.plot([], [], 'ko', markersize=8, label="Husky")
    anymal_dot, = ax.plot([], [], 'bo', markersize=6, label="ANYmal")

    pb_dots = {name: ax.plot([], [], 'o')[0] for name in pb_traj}

    husky_box_dots = [
        ax.plot([], [], 'o', alpha=0.5)[0]
        for _ in husky_boxes
    ]

    pb_box_dots = {
        name: ax.plot([], [], 's')[0]
        for name in pb_boxes
    }

    lidar_lines = [
        ax.plot([], [], '-', color="gray", alpha=0.25, linewidth=0.8)[0]
        for _ in range(N_RAYS)
    ]

    stack_dot, = ax.plot(stack[0], stack[1], 'k*', markersize=12)

    ax.legend()

    total_frames = (
        len(husky_traj)
        + len(anymal_traj)
        + len(next(iter(pb_traj.values())))
    )

    def update(frame):

        if frame < len(husky_traj):

            x, y = husky_traj[frame]
            husky_dot.set_data([x], [y])

            for i, b in enumerate(husky_boxes):
                if frame < len(b):
                    husky_box_dots[i].set_data([b[frame][0]], [b[frame][1]])

            if frame < len(lidar_rays):
                rays = lidar_rays[frame]

                for i, ray in enumerate(rays):
                    start, end, detected_box = ray

                    lidar_lines[i].set_data(
                        [start[0], end[0]],
                        [start[1], end[1]]
                    )

                    if detected_box is None:
                        lidar_lines[i].set_color("gray")
                        lidar_lines[i].set_alpha(0.25)
                        lidar_lines[i].set_linewidth(0.8)
                    else:
                        lidar_lines[i].set_color("orange")
                        lidar_lines[i].set_alpha(0.9)
                        lidar_lines[i].set_linewidth(1.4)

        elif frame < len(husky_traj) + len(anymal_traj):

            for line in lidar_lines:
                line.set_data([], [])

            idx = frame - len(husky_traj)
            x, y = anymal_traj[idx]

            anymal_dot.set_data([x], [y])

            if x >= 8:
                anymal_dot.set_color('green')

        else:

            for line in lidar_lines:
                line.set_data([], [])

            idx = frame - len(husky_traj) - len(anymal_traj)

            for name in pb_traj:
                data = np.array(pb_traj[name])
                if idx < len(data):
                    pb_dots[name].set_data([data[idx][0]], [data[idx][1]])

            for name in pb_boxes:
                data = np.array(pb_boxes[name])
                if idx < len(data):
                    pb_box_dots[name].set_data([data[idx][0]], [data[idx][1]])

        return (
            list(pb_dots.values())
            + list(pb_box_dots.values())
            + husky_box_dots
            + lidar_lines
            + [husky_dot, anymal_dot, stack_dot]
        )

    ani = FuncAnimation(
        fig,
        update,
        frames=total_frames,
        interval=50,
        blit=True
    )

    plt.show()


# =========================================================
# GRÁFICAS PRO
# =========================================================
def plot_all(seed=None):

    husky_traj, v_log, omega_log, lidar_log, _, _ = simulate_husky(seed=seed)
    _, anymal_traj = simulate_anymal_transport()

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    if len(husky_traj) > 0:
        plt.plot(husky_traj[:, 0], husky_traj[:, 1])
    plt.title("Trayectoria Husky")
    plt.axis("equal")
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(v_log, label="v")
    plt.plot(omega_log, label="omega")
    plt.plot(lidar_log, label="detecciones_lidar")
    plt.legend()
    plt.title("Velocidades y detección Husky")
    plt.grid()

    plt.show()

    plt.figure(figsize=(6, 5))
    plt.plot(anymal_traj[:, 0], anymal_traj[:, 1])
    plt.title("Trayectoria ANYmal")
    plt.axis("equal")
    plt.grid()
    plt.show()


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    print("🚀 SIMULACIÓN MULTIROBOT FINAL PRO")

    plot_all()
    animate_all()
