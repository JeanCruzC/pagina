#!/usr/bin/env python3
"""
Versión simplificada sin threads complejos para evitar problemas de Ctrl+C
"""
import os
import signal
import sys
from website import create_app

def signal_handler(sig, frame):
    """Manejo limpio de Ctrl+C"""
    print('\n[APP] Deteniendo aplicación...')
    sys.exit(0)

def main():
    # Configurar manejo de señales
    signal.signal(signal.SIGINT, signal_handler)
    
    print("=== Schedules Generator (Versión Simple) ===")
    print("PuLP y Greedy optimizadores listos")
    print("Aplicación iniciada en: http://127.0.0.1:5000")
    print("Usa Ctrl+C para detener")
    
    app = create_app()
    
    try:
        app.run(
            debug=False,
            use_reloader=False,
            threaded=True,
            host="127.0.0.1",
            port=5000,
        )
    except KeyboardInterrupt:
        print('\n[APP] Aplicación detenida por el usuario')
    except Exception as e:
        print(f'\n[APP] Error: {e}')

if __name__ == "__main__":
    main()