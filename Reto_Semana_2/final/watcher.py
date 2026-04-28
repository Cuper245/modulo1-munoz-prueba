class Watcher:
    def __init__(self, boxes, corridor):
        self.boxes = boxes
        self.corridor = corridor  # (xmin, xmax, ymin, ymax)

        self.events = []
        self.already_reported = set()

        self.completed = False  # 🔥 evita múltiples "terminados"

    # =========================================================
    # CHECK SI ESTA FUERA
    # =========================================================
    def is_outside(self, pos):
        xmin, xmax, ymin, ymax = self.corridor

        # 🔥 margen pequeño para evitar ruido numérico
        eps = 1e-6

        return not (
            xmin - eps <= pos[0] <= xmax + eps and
            ymin - eps <= pos[1] <= ymax + eps
        )

    # =========================================================
    # LOOP PRINCIPAL
    # =========================================================
    def check(self):

        if self.completed:
            return True  # 🔥 ya terminó, no recalcular

        count = 0

        for i, b in enumerate(self.boxes):

            if self.is_outside(b["pos"]):
                count += 1

                # 🔥 evento solo una vez
                if i not in self.already_reported:
                    event = {
                        "type": "box_out",
                        "box_id": i,
                        "position": b["pos"].copy()
                    }

                    self.events.append(event)
                    self.already_reported.add(i)

                    print(f"[EVENT] Caja {i} salió en {event['position']}")

        print(f"[WATCHER] {count}/{len(self.boxes)} cajas fuera")

        # =====================================================
        # CONDICION DE TERMINO
        # =====================================================
        if count == len(self.boxes):
            print("🚀 FASE 1 COMPLETADA")

            self.completed = True  # 🔥 congelar estado
            return True

        return False

    # =========================================================
    # EVENTOS
    # =========================================================
    def get_events(self):
        return self.events.copy()

    def clear_events(self):
        self.events = []