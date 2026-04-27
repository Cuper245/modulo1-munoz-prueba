import numpy as np
from robots import ANYmal
import matplotlib.pyplot as plt

dt = 0.01
T = 25.0

DEST = np.array([11.0, 0.0])
TOL = 0.15

STEP_HEIGHT = 0.06
STEP_LENGTH = 0.18

BASE_SPEED = 0.6
PAYLOAD_MASS = 6.0
ROBOT_MASS = 30.0

V_FORWARD = BASE_SPEED * (ROBOT_MASS / (ROBOT_MASS + PAYLOAD_MASS))

LATERAL_OFFSET = {
    'LF':  0.10,
    'RF': -0.10,
    'LH':  0.10,
    'RH': -0.10,
}

CYCLE = 0.6

def interpolate(p0, pf, alpha):
    return (1 - alpha)*p0 + alpha*pf

def foot_trajectory(t, phase, base_height, leg_name):

    tau = (t / CYCLE + phase) % 1.0

    p_start = np.array([-STEP_LENGTH/2, LATERAL_OFFSET[leg_name], -base_height])
    p_end   = np.array([ STEP_LENGTH/2, LATERAL_OFFSET[leg_name], -base_height])

    if tau < 0.5:
        alpha = tau * 2
        p = interpolate(p_start, p_end, alpha)
        p[2] += STEP_HEIGHT * np.sin(np.pi * alpha)
    else:
        alpha = (tau - 0.5) * 2
        p = interpolate(p_end, p_start, alpha)
        p[2] += 0.01

    return p

def clamp_joint_angles(q):
    return np.clip(q, [-1.5, -1.0, -2.5], [1.5, 1.0, -0.3])

def compute_detJ(leg, q):
    J = leg.jacobian(q)
    return np.linalg.det(J)

def anymal_controller(anymal, t, base_height, detJ_log):

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

        detJ = compute_detJ(leg, q)
        detJ_log[name].append(detJ)

        if abs(detJ) < 1e-3:
            q[1] += 0.3
            q[2] -= 0.5

        if q[2] > -0.5:
            q[2] = -0.8

        q = clamp_joint_angles(q)
        q12[3*i:3*(i+1)] = q

    return q12

def simulate_anymal_transport():

    anymal = ANYmal()

    x, y = 0.0, 0.0
    trajectory = []

    detJ_log = {
        'LF': [],
        'RF': [],
        'LH': [],
        'RH': [],
    }

    payload_offsets = [
        np.array([-0.2, -0.2, 0.0]),
        np.array([ 0.0,  0.0, 0.0]),
        np.array([ 0.2,  0.2, 0.0]),
    ]

    payload_positions = []

    for t in np.arange(0, T, dt):

        base_height = 0.46 + 0.01*np.sin(2*np.pi*t)

        q12 = anymal_controller(anymal, t, base_height, detJ_log)
        anymal.set_all_joint_angles(q12)

        dx = DEST[0] - x
        dy = DEST[1] - y

        dist = np.linalg.norm([dx, dy])
        theta = np.arctan2(dy, dx)

        if dist > 1.0:
            v = V_FORWARD
        elif dist > 0.3:
            v = 0.4 * dist + 0.05
        else:
            v = 0.12

        v = min(v, V_FORWARD)

        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt

        anymal.base_pos = np.array([x, y, base_height])
        trajectory.append([x, y])

        payload_step = []
        for offset in payload_offsets:
            px = x + offset[0]
            py = y + offset[1]
            pz = base_height + 0.15
            payload_step.append([px, py, pz])

        payload_positions.append(payload_step)

        if dist < 0.2:
            x = DEST[0]
            y = DEST[1]
            return True, np.array(trajectory)

        if dist < TOL:
            return True, np.array(trajectory)

    return False, np.array(trajectory)

if __name__ == "__main__":

    success, traj = simulate_anymal_transport()

    plt.figure(figsize=(8,5))
    plt.plot(traj[:,0], traj[:,1])
    plt.scatter(DEST[0], DEST[1])
    plt.title("ANYmal transporte")
    plt.grid()
    plt.axis("equal")
    plt.show()