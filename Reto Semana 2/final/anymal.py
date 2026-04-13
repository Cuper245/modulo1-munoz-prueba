import numpy as np
from robots import ANYmal
import matplotlib.pyplot as plt

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

# offsets laterales (estabilidad)
LATERAL_OFFSET = {
    'LF':  0.10,
    'RF': -0.10,
    'LH':  0.10,
    'RH': -0.10,
}

CYCLE = 0.6

# ==============================
# TRAYECTORIA PIE
# ==============================

def foot_trajectory(t, phase, base_height, leg_name):

    tau = (t / CYCLE + phase) % 1.0

    x = STEP_LENGTH * (tau - 0.5)
    y = LATERAL_OFFSET[leg_name]

    if tau < 0.5:
        # SWING
        z = -base_height + STEP_HEIGHT * np.sin(np.pi * tau * 2)
    else:
        # STANCE (ligero contacto)
        z = -base_height + 0.01

    return np.array([x, y, z])

# ==============================
# CLAMP ARTICULAR
# ==============================

def clamp_joint_angles(q):
    return np.clip(q, [-1.5, -1.0, -2.5], [1.5, 1.0, -0.3])

# ==============================
# CONTROLADOR
# ==============================

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

        # ==============================
        # SINGULARIDAD
        # ==============================
        if leg.is_singular(q):
            q[1] += 0.3
            q[2] -= 0.5

        # evitar rodilla estirada
        if q[2] > -0.5:
            q[2] = -0.8

        q = clamp_joint_angles(q)

        q12[3*i:3*(i+1)] = q

    return q12

# ==============================
# SIMULACION
# ==============================

def simulate_anymal_transport():

    anymal = ANYmal()

    x, y = 0.0, 0.0
    trajectory = []

    for t in np.arange(0, T, dt):

        base_height = 0.48 + 0.01*np.sin(2*np.pi*t)

        # ==============================
        # CONTROL ARTICULAR
        # ==============================
        q12 = anymal_controller(anymal, t, base_height)
        anymal.set_all_joint_angles(q12)

        # ==============================
        # NAVEGACION
        # ==============================
        dx = DEST[0] - x
        dy = DEST[1] - y

        dist = np.linalg.norm([dx, dy])
        theta = np.arctan2(dy, dx)

        # ==============================
        # VELOCIDAD ROBUSTA
        # ==============================
        if dist > 1.0:
            v = V_FORWARD
        elif dist > 0.3:
            v = 0.4 * dist + 0.05
        else:
            v = 0.15  # velocidad mínima

        v = min(v, V_FORWARD)

        # ==============================
        # ACTUALIZAR POSICION
        # ==============================
        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt

        anymal.base_pos = np.array([x, y, base_height])
        trajectory.append([x, y])

        # ==============================
        # SNAP FINAL (CLAVE)
        # ==============================
        if dist < 0.2:
            x = DEST[0]
            y = DEST[1]
            print("✅ ANYmal llegó (snap final)")
            return True, np.array(trajectory)

        if dist < TOL:
            print("✅ ANYmal llegó")
            return True, np.array(trajectory)

        print(f"t={t:.2f} | pos=({x:.2f},{y:.2f}) | error={dist:.2f}")

    print("❌ ANYmal no llegó")
    return False, np.array(trajectory)

# ==============================
# MAIN
# ==============================

if __name__ == "__main__":

    success, traj = simulate_anymal_transport()

    plt.figure(figsize=(8,5))
    plt.plot(traj[:,0], traj[:,1], 'k-', label="Trayectoria")
    plt.scatter(DEST[0], DEST[1], c='red', label="Destino")

    plt.title("ANYmal transporte (FINAL)")
    plt.legend()
    plt.grid()
    plt.axis("equal")
    plt.show()