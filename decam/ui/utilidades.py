"""Utilidades pequeñas de la interfaz sin dependencia de widgets."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def abrir_en_sistema(ruta: str | Path) -> None:
    """Abre un archivo o carpeta con la aplicación predeterminada del sistema."""
    ruta = str(ruta)
    if sys.platform.startswith("win"):
        os.startfile(ruta)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", ruta])
    else:
        subprocess.Popen(["xdg-open", ruta])


def parsear_tiempo(texto: str) -> float:
    """Convierte ``HH:MM:SS``, ``MM:SS`` o un número de segundos a segundos.

    Raises:
        ValueError: si el texto no tiene un formato reconocible.
    """
    partes = texto.strip().split(":")
    if len(partes) > 3:
        raise ValueError(texto)
    total = 0.0
    for parte in partes:
        total = total * 60 + float(parte)
    return total
