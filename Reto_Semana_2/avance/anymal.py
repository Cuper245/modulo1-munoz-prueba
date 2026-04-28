import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from base import ANYmal

# ==============================
# CONFIG
# ==============================
dt = 0.01
T = 25.0

DEST = np.array([11.0, 0.0])
TOL = 0.15

STEP_HEIGHT = 0.06
STEP_LENGTH = 0.18
V_FORWARD = 0.6

LATERAL_OFFSET = {
    'LF':  0.10,
    'RF': -0.10,
    'LH':  0.10,
    'RH': -0.10,
}

CYCLE = 0.6

# ==============================
# LOGS
# ==============================
log_t = []
log_pos = []
log_error = []
log_v = []
log_height = []
log_detJ = { 'LF': [], 'RF': [], 'LH': [], 'RH': [] }


def foot_trajectory(t, phase, base_height, leg_name):

    tau = (t / CYCLE + phase) % 1.0

    x = STEP_LENGTH * (tau - 0.5)
    y = LATERAL_OFFSET[leg_name]

    if tau < 0.5:
        z = -base_height + STEP_HEIGHT * np.sin(np.pi * tau * 2)
    else:
        z = -base_height + 0.01

    return np.array([x, y, z])


def clamp_joint_angles(q):
    return np.clip(q, [-1.5, -1.0, -2.5], [1.5, 1.0, -0.3])


def anymal_controller(anymal, t, base_height):

    phases = {
        'LF': 0.0,
        'RH': 0.0,
        'RF': 0.5,
        'LH': 0.5,
    }

    q12 = np.zeros(12)

    for i, name in enumerate(['LF', 'RF', 'LH', 'RH']):
        leg = anymal.legs[name]

        p = foot_trajectory(t, phases[name], base_height, name)

        try:
            q = leg.inverse_kinematics(p)
        except:
            q = np.array([0.0, 0.4, -1.2])

        # evitar singularidad
        if leg.is_singular(q):
            q[1] += 0.3
            q[2] -= 0.5

        # evitar rodilla estirada
        if q[2] > -0.5:
            q[2] = -0.8

        q = clamp_joint_angles(q)

        q12[3*i:3*(i+1)] = q

    return q12


def simulate_anymal_transport():

    anymal = ANYmal()

    x, y = 0.0, 0.0
    trajectory = []

    for t in np.arange(0, T, dt):

        base_height = 0.48 + 0.01*np.sin(2*np.pi*t)

        # control articular
        q12 = anymal_controller(anymal, t, base_height)
        anymal.set_all_joint_angles(q12)

        # navegación
        dx = DEST[0] - x
        dy = DEST[1] - y

        dist = np.linalg.norm([dx, dy])
        theta = np.arctan2(dy, dx)

        # velocidad adaptativa
        if dist > 1.0:
            v = V_FORWARD
        elif dist > 0.3:
            v = 0.4 * dist + 0.05
        else:
            v = 0.15

        v = min(v, V_FORWARD)

        # actualizar posición
        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt

        anymal.base_pos = np.array([x, y, base_height])
        trajectory.append([x, y])

        # ==============================
        # LOGS
        # ==============================
        log_t.append(t)
        log_pos.append([x, y])
        log_error.append(dist)
        log_v.append(v)
        log_height.append(base_height)

        for name in ['LF','RF','LH','RH']:
            leg = anymal.legs[name]
            J = leg.jacobian()
            detJ = np.linalg.det(J)
            log_detJ[name].append(detJ)

        # condiciones de parada
        if dist < 0.2:
            print("✅ ANYmal llegó (snap final)")
            return True, np.array(trajectory)

        if dist < TOL:
            print("✅ ANYmal llegó")
            return True, np.array(trajectory)

        print(f"t={t:.2f} | pos=({x:.2f},{y:.2f}) | error={dist:.2f}")

    print("❌ ANYmal no llegó")
    return False, np.array(trajectory)

# ==============================
# GRAFICAS
# ==============================
def plot_metrics():

    t = np.array(log_t)

    plt.figure(figsize=(12,8))

    plt.subplot(2,2,1)
    plt.plot(t, log_error)
    plt.title("Error al objetivo")

    plt.subplot(2,2,2)
    plt.plot(t, log_v)
    plt.title("Velocidad")

    plt.subplot(2,2,3)
    plt.plot(t, log_height)
    plt.title("Altura base")

    plt.subplot(2,2,4)
    for name in log_detJ:
        plt.plot(t, log_detJ[name], label=name)

    plt.axhline(1e-3, linestyle='--')
    plt.title("det(J)")
    plt.legend()

    plt.tight_layout()
    plt.show()

# ==============================
# ANIMACION
# ==============================
def animate_anymal(traj):

    fig, ax = plt.subplots(figsize=(6,6))

    ax.set_xlim(-1, 12)
    ax.set_ylim(-2, 2)

    # trayectoria
    line, = ax.plot([], [], lw=2)

    # robot (punto)
    body, = ax.plot([], [], 'ro', markersize=8)

    # destino
    ax.plot(DEST[0], DEST[1], 'gx', markersize=12, label="Destino")

    ax.legend()

    def update(frame):
        x, y = traj[frame]

        line.set_data(traj[:frame,0], traj[:frame,1])
        body.set_data([x], [y])  # FIX CLAVE

        return line, body

    ani = FuncAnimation(
        fig,
        update,
        frames=len(traj),
        interval=30,
        blit=False  # FIX CLAVE
    )

    plt.title("Animación ANYmal")
    plt.grid()
    plt.show()

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":

    success, traj = simulate_anymal_transport()

    # trayectoria
    plt.figure()
    plt.plot(traj[:,0], traj[:,1], 'k-', label="Trayectoria")
    plt.scatter(DEST[0], DEST[1], c='red', label="Destino")
    plt.legend()
    plt.title("Trayectoria ANYmal")
    plt.axis("equal")
    plt.grid()
    plt.show()

    # métricas
    plot_metrics()

    # animación
    animate_anymal(traj)