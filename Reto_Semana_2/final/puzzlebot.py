import numpy as np

# =========================================================================
# PUZZLEBOT (FISICA REAL DEL PROFE)
# =========================================================================

class PuzzleBot:
    def __init__(self, name, start_pos, r=0.05, L=0.19):

        self.name = name

        # Estado
        self.x = start_pos[0]
        self.y = start_pos[1]
        self.theta = 0.0

        # Parámetros físicos
        self.r = r
        self.L = L

        # 🔥 MÁS RÁPIDO
        self.v_max = 1.2
        self.omega_max = 4.0

        self.carrying = None

    def get_pose(self):
        return np.array([self.x, self.y])

    # ===============================
    # CINEMATICA
    # ===============================

    def inverse_kinematics(self, v, omega):
        v = np.clip(v, -self.v_max, self.v_max)
        omega = np.clip(omega, -self.omega_max, self.omega_max)

        wR = (2*v + omega*self.L) / (2*self.r)
        wL = (2*v - omega*self.L) / (2*self.r)

        return wR, wL

    def forward_kinematics(self, wR, wL):
        v = self.r/2 * (wR + wL)
        omega = self.r/self.L * (wR - wL)
        return v, omega

    def update(self, v_cmd, omega_cmd, dt=0.2):  # 🔥 dt mayor
        wR, wL = self.inverse_kinematics(v_cmd, omega_cmd)
        v, omega = self.forward_kinematics(wR, wL)

        self.x += v * np.cos(self.theta) * dt
        self.y += v * np.sin(self.theta) * dt
        self.theta += omega * dt
        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))


# =========================================================================
# CONTROLADOR (TUNEADO)
# =========================================================================

def go_to_target(bot, target):

    dx = target[0] - bot.x
    dy = target[1] - bot.y

    theta_target = np.arctan2(dy, dx)

    error_theta = theta_target - bot.theta
    error_theta = np.arctan2(np.sin(error_theta), np.cos(error_theta))

    dist = np.linalg.norm([dx, dy])

    # 🔥 MÁS AGRESIVO
    k_v = 2.5
    k_w = 6.0

    v = k_v * dist
    omega = k_w * error_theta

    v = np.clip(v, -bot.v_max, bot.v_max)
    omega = np.clip(omega, -bot.omega_max, bot.omega_max)

    # 🔥 SOLO gira si está MUY mal
    if abs(error_theta) > 1.0:
        v = 0.0

    return v, omega, dist


# =========================================================================
# COORDINADOR
# =========================================================================

class PuzzleCoordinator:

    def __init__(self, bots, boxes, stack_pos):

        self.bots = bots
        self.boxes = boxes
        self.stack_pos = stack_pos

        self.state = "ASSIGN"

    # ---------------------------------------------------------------------
    def assign_tasks(self):

        free_bots = [b for b in self.bots if b.carrying is None]
        free_boxes = [b for b in self.boxes if not b["stacked"]]

        for bot, box in zip(free_bots, free_boxes):
            bot.carrying = box
            print(f"🤖 {bot.name} asignado a caja {box['name']}")

    # ---------------------------------------------------------------------
    def step(self):

        if self.state == "ASSIGN":
            self.assign_tasks()
            self.state = "MOVE"
            return False

        for bot in self.bots:

            if bot.carrying is None:
                continue

            box = bot.carrying

            # =============================
            # IR A CAJA
            # =============================
            if not box.get("picked", False):

                target = box["pos"][:2]

                v, omega, dist = go_to_target(bot, target)
                bot.update(v, omega)

                print(f"{bot.name} → caja {box['name']} | dist={dist:.2f}")

                # 🔥 MÁS PERMISIVO
                if dist < 0.25:
                    box["picked"] = True
                    print(f"📦 {bot.name} recogió {box['name']}")

            # =============================
            # IR A STACK
            # =============================
            else:

                target = self.stack_pos[:2]

                v, omega, dist = go_to_target(bot, target)
                bot.update(v, omega)

                box["pos"][:2] = np.array([bot.x, bot.y])

                print(f"{bot.name} llevando {box['name']}")

                if dist < 0.25:
                    box["stacked"] = True
                    bot.carrying = None
                    print(f"🏗️ Caja {box['name']} apilada")

        # =============================
        # TERMINO
        # =============================
        if all(b["stacked"] for b in self.boxes):
            print("🎯 TODAS LAS CAJAS APILADAS")
            return True

        return False