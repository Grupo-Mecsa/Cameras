"""Punto de entrada de DeCam: ``python app.py`` abre la interfaz gráfica.

Todo el código vive en el paquete ``decam``; este archivo solo existe para que
el lanzamiento (y PyInstaller) tengan un script en la raíz.
"""

from decam.ui.aplicacion import main

if __name__ == "__main__":
    main()
