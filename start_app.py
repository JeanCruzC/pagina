#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para iniciar la aplicación con configuración optimizada
"""
import os
from website import create_app

def main():
    print("=== Iniciando Schedules Generator ===")
    print("PuLP y Greedy optimizadores listos")
    
    app = create_app()
    
    print("Aplicación iniciada en: http://127.0.0.1:5000")
    print("Usa Ctrl+C para detener")
    
    app.run(
        debug=False,
        use_reloader=False,
        threaded=True,
        host="127.0.0.1",
        port=5000,
    )

if __name__ == "__main__":
    main()