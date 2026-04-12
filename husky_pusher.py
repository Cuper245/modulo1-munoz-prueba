import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from robots import HuskyA200

# ==============================
# CONFIGURACION
# ==============================
dt = 0.05
T = 30

K_omega = 2.0
v_forward = 0.6

CORRIDOR_X = [0, 6]
CORRIDOR_Y = [-1, 1]

boxes = [
    {"pos": np.array([2.0, 0.0]), "pushed": False},
    {"pos": np.array([3.5, 0.5]), "pushed": False},
    {"pos": np.array([4.5, -0.5]), "pushed": False},
]

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

    # 🔥 Mejora: gira primero, luego avanza
    if abs(error_theta) > 0.3:
        v = 0.2
    else:
        v = v_forward

    omega = K_omega * error_theta

    return v, omega, error_theta

# ==============================
# SIMULACION
# ==============================

husky = HuskyA200()
husky.set_terrain("asphalt")

trajectory = []
box_traj = [[] for _ in boxes]
push_states = []

current_box = 0

for t in np.arange(0, T, dt):

    if current_box >= len(boxes):
        break

    box = boxes[current_box]
    box_pos = box["pos"]

    v, omega, error_theta = go_to_target(husky, box_pos)

    husky.update_pose(v, omega, dt)

    dist = np.linalg.norm(box_pos - np.array([husky.x, husky.y]))

    if dist < 0.3:
        box["pushed"] = True

    pushing = box["pushed"]

    if pushing:
        direction = np.array([np.cos(husky.theta), np.sin(husky.theta)])
        box["pos"] += direction * v * dt

    if not in_corridor(box["pos"]):
        print(f"✅ Caja {current_box} fuera del corredor")
        current_box += 1

    trajectory.append([husky.x, husky.y])
    push_states.append(pushing)

    for i, b in enumerate(boxes):
        box_traj[i].append(b["pos"].copy())

    print(f"t={t:.2f} | v={v:.2f}, omega={omega:.2f}")

# ==============================
# GRAFICA ESTATICA
# ==============================

trajectory = np.array(trajectory)

plt.figure(figsize=(8,6))

plt.plot(trajectory[:,0], trajectory[:,1],
         linewidth=3, label="Husky", color='black')

colors = ['red', 'blue', 'green']

for i, traj in enumerate(box_traj):
    traj = np.array(traj)

    plt.plot(traj[:,0], traj[:,1],
             linestyle='--', alpha=0.4,
             color=colors[i], label=f"Caja {i}")

    final_pos = traj[-1]
    size = 0.2

    rect = plt.Rectangle(
        (final_pos[0]-size/2, final_pos[1]-size/2),
        size, size,
        color=colors[i], alpha=0.8
    )
    plt.gca().add_patch(rect)

plt.plot([0,6,6,0,0], [-1,-1,1,1,-1],
         'r-', linewidth=2, label="Corredor")

plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.title("Husky empujando cajas")

plt.axis("equal")
plt.show()

# ==============================
# ANIMACION PRO
# ==============================

trajectory = np.array(trajectory)
box_traj = [np.array(traj) for traj in box_traj]

fig, ax = plt.subplots(figsize=(8,6))

husky_point, = ax.plot([], [], 'ko', markersize=8, label="Husky")
husky_path, = ax.plot([], [], 'k-', linewidth=2)

box_points = []
box_paths = []

for i in range(len(box_traj)):
    p, = ax.plot([], [], 's', color=colors[i], markersize=8)
    l, = ax.plot([], [], '--', color=colors[i], alpha=0.4)
    box_points.append(p)
    box_paths.append(l)

ax.plot([0,6,6,0,0], [-1,-1,1,1,-1], 'r-', linewidth=2)

ax.set_xlim(-1, 8)
ax.set_ylim(-3, 3)
ax.set_title("Animación Husky empujando cajas")
ax.grid()
ax.legend()

def update(frame):

    x = trajectory[frame,0]
    y = trajectory[frame,1]

    # 🔥 cambia color si está empujando
    if push_states[frame]:
        husky_point.set_color('orange')
    else:
        husky_point.set_color('black')

    husky_point.set_data([x], [y])
    husky_path.set_data(trajectory[:frame,0], trajectory[:frame,1])

    for i in range(len(box_traj)):
        if frame < len(box_traj[i]):
            bx = box_traj[i][frame,0]
            by = box_traj[i][frame,1]

            box_points[i].set_data([bx], [by])
            box_paths[i].set_data(
                box_traj[i][:frame,0],
                box_traj[i][:frame,1]
            )

    return [husky_point, husky_path] + box_points + box_paths

ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(trajectory),
    interval=50,
    blit=True
)

plt.show()