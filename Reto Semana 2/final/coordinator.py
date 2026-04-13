import numpy as np

# IMPORTS CORREGIDOS
from husky_pusher import simulate_husky_mission
from anymal import simulate_anymal_transport
from puzzlebot import PuzzleCoordinator, PuzzleBot


class Coordinator:
    """
    Coordinador de misión multi-robot.

    Arquitectura:
        - Máquina de estados (FSM)
        - Manejo de errores
        - Secuencia:
            HUSKY -> ANYMAL -> PUZZLEBOTS
    """

    def __init__(self):

        self.state = "HUSKY"
        self.max_retries = 2

        # ==============================
        # PUZZLEBOTS
        # ==============================
        self.bots = [
            PuzzleBot("PB1", [0.0, -0.5]),
            PuzzleBot("PB2", [0.0,  0.0]),
            PuzzleBot("PB3", [0.0,  0.5]),
        ]

        self.boxes = [
            {"name":"A", "pos":np.array([1.0, -0.5, 0.0]), "stacked":False},
            {"name":"B", "pos":np.array([1.2,  0.0, 0.0]), "stacked":False},
            {"name":"C", "pos":np.array([1.4,  0.5, 0.0]), "stacked":False},
        ]

        self.stack_pos = np.array([3.0, 0.0, 0.0])

        self.puzzle_coordinator = PuzzleCoordinator(
            self.bots,
            self.boxes,
            self.stack_pos
        )

    # =========================================================================
    # FASE 1: HUSKY
    # =========================================================================
    def run_husky(self):

        for attempt in range(self.max_retries):

            print(f"\n🔧 Intento Husky {attempt+1}")

            success = simulate_husky_mission()

            if success:
                print("✅ Corredor despejado")
                return True

            print("⚠️ Reintentando Husky...")

        return False

    # =========================================================================
    # FASE 2: ANYMAL
    # =========================================================================
    def run_anymal(self):

        for attempt in range(self.max_retries):

            print(f"\n🐕 Intento ANYmal {attempt+1}")

            success, traj = simulate_anymal_transport()

            if success:
                print("✅ ANYmal llegó a zona de trabajo")
                return True

            print("⚠️ Reintentando ANYmal...")

        return False

    # =========================================================================
    # FASE 3: PUZZLEBOTS
    # =========================================================================
    def run_puzzlebots(self):

        done = False
        step_count = 0

        while not done:

            print(f"\n--- STEP {step_count} ---")

            done = self.puzzle_coordinator.step()
            step_count += 1

            # protección
            if step_count > 15:
                print("⚠️ Timeout PuzzleBots")
                return False

        return True

    # =========================================================================
    # RUN COMPLETO (FSM)
    # =========================================================================
    def run(self):

        print("🚀 INICIANDO MISIÓN MULTI-ROBOT\n")

        # ==============================
        # FSM LOOP
        # ==============================
        while True:

            # -------- HUSKY --------
            if self.state == "HUSKY":

                print("🔧 FASE 1: Husky limpiando corredor...")

                if not self.run_husky():
                    print("❌ FALLÓ HUSKY")
                    return

                self.state = "ANYMAL"

            # -------- ANYMAL --------
            elif self.state == "ANYMAL":

                print("\n🐕 FASE 2: ANYmal transportando robots...")

                if not self.run_anymal():
                    print("❌ FALLÓ ANYMAL")
                    return

                self.state = "PUZZLEBOTS"

            # -------- PUZZLEBOTS --------
            elif self.state == "PUZZLEBOTS":

                print("\n🤖 FASE 3: PuzzleBots apilando cajas...")

                if not self.run_puzzlebots():
                    print("❌ FALLÓ PUZZLEBOTS")
                    return

                self.state = "DONE"

            # -------- FIN --------
            elif self.state == "DONE":

                print("\n🏁 MISIÓN COMPLETADA EXITOSAMENTE")
                return


# =========================================================================
# MAIN
# =========================================================================
if __name__ == "__main__":

    coordinator = Coordinator()
    coordinator.run()