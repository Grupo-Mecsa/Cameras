"""Genera el ejecutable de DeCam con PyInstaller.

Uso:
    pip install pyinstaller
    python build_exe.py            # carpeta dist/DeCam/ con DeCam.exe (recomendado)
    python build_exe.py --onefile  # un único DeCam.exe (arranca mucho más lento)

El resultado queda en ``dist/``. Se construye en modo ``--onedir`` por defecto:
con PyTorch dentro, un ``--onefile`` supera con facilidad 1 GB y tarda casi un
minuto en arrancar, porque debe descomprimirse en una carpeta temporal.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).parent
NOMBRE = "DeCam"

# Paquetes cuyos datos (archivos .yaml, .onnx, cascadas XML) no detecta
# PyInstaller por sí solo y hay que arrastrar completos.
PAQUETES_COMPLETOS = ("ultralytics", "cv2", "openvino")

# Paquetes cuyos metadatos consulta ultralytics en tiempo de ejecución.
METADATOS = (
    "ultralytics",
    "torch",
    "torchvision",
    "tqdm",
    "numpy",
    "pillow",
    "openvino",
)


def construir_argumentos(onefile: bool, con_consola: bool) -> list[str]:
    """Arma la línea de comandos de PyInstaller."""
    args = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--name",
        NOMBRE,
        "--onefile" if onefile else "--onedir",
        "--console" if con_consola else "--windowed",
    ]

    for paquete in PAQUETES_COMPLETOS:
        args += ["--collect-all", paquete]
    for paquete in METADATOS:
        args += ["--copy-metadata", paquete]

    # Los pesos YOLO y el modelo YuNet se incluyen solo si ya están descargados.
    modelos = RAIZ / "models"
    if modelos.is_dir() and any(modelos.iterdir()):
        args += ["--add-data", f"{modelos}{os_sep()}models"]
    for pesos in RAIZ.glob("yolov8*.pt"):
        args += ["--add-data", f"{pesos}{os_sep()}."]
    # Modelos ya exportados a OpenVINO: evitan reexportar en el equipo destino.
    for carpeta in RAIZ.glob("yolov8*_openvino_model"):
        args += ["--add-data", f"{carpeta}{os_sep()}{carpeta.name}"]

    args.append(str(RAIZ / "app.py"))
    return args


def os_sep() -> str:
    """Separador que usa ``--add-data`` (``;`` en Windows, ``:`` en el resto)."""
    return ";" if sys.platform.startswith("win") else ":"


def main() -> int:
    """Ejecuta PyInstaller y devuelve su código de salida."""
    parser = argparse.ArgumentParser(description="Compila DeCam a ejecutable.")
    parser.add_argument(
        "--onefile", action="store_true", help="Un solo .exe en vez de una carpeta."
    )
    parser.add_argument(
        "--console", action="store_true", help="Mantener la consola visible (para depurar)."
    )
    opciones = parser.parse_args()

    if shutil.which("pyinstaller") is None:
        try:
            import PyInstaller  # noqa: F401
        except ImportError:
            print("Falta PyInstaller. Instálalo con:  pip install pyinstaller")
            return 1

    comando = construir_argumentos(opciones.onefile, opciones.console)
    print("Ejecutando:\n  " + " ".join(comando) + "\n")
    resultado = subprocess.run(comando, cwd=RAIZ)
    if resultado.returncode == 0:
        destino = RAIZ / "dist" / (f"{NOMBRE}.exe" if opciones.onefile else NOMBRE)
        print(f"\nListo. Ejecutable en: {destino}")
    return resultado.returncode


if __name__ == "__main__":
    raise SystemExit(main())
