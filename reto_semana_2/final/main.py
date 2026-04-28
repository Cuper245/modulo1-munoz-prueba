"""
MAIN - Sistema Multi-Robot
==========================

Ejecuta:
    1. Husky (limpieza de corredor)
    2. ANYmal (transporte)
    3. PuzzleBots (apilado cooperativo)

Uso:
    python main.py
"""

from coordinator import Coordinator


def main():

    print("\n" + "="*60)
    print("🤖 SISTEMA MULTI-ROBOT - EJECUCIÓN COMPLETA")
    print("="*60)

    coordinator = Coordinator()

    try:
        coordinator.run()

    except KeyboardInterrupt:
        print("\n⛔ Interrumpido por el usuario")

    except Exception as e:
        print("\n💥 ERROR EN EJECUCIÓN:")
        print(e)

    finally:
        print("\n" + "="*60)
        print("FIN DEL PROGRAMA")
        print("="*60)


# =========================================================================
# ENTRY POINT
# =========================================================================
if __name__ == "__main__":
    main()