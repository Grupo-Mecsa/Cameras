"""Rutas de datos del usuario y registro en fichero.

Las dos cosas van juntas porque comparten el mismo problema: averiguar una
carpeta en la que se pueda escribir. En desarrollo es el propio directorio del
proyecto, pero dentro del ejecutable de PyInstaller no sirve:

* con ``--onefile``, ``__file__`` apunta a la carpeta temporal de extracción
  (``sys._MEIPASS``), que se borra al cerrar la aplicación;
* con ``--onedir``, apunta al directorio de instalación, que puede ser de solo
  lectura (por ejemplo bajo ``C:/Program Files``).

En ambos casos se usa ``%LOCALAPPDATA%/DeCam``, salvo que exista un
``config.json`` junto al ejecutable: eso se interpreta como instalación
portable y se respeta.

El registro en fichero existe porque la aplicación se distribuye como
ejecutable sin consola: sin él, un fallo solo deja un mensaje de una línea en
la interfaz y el traceback se pierde.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import traceback
import types
from pathlib import Path
from typing import Optional

NOMBRE_APP = "DeCam"
NOMBRE_LOG = "decam.log"
NOMBRE_CONFIG = "config.json"

#: Tamaño máximo del log antes de rotar, y número de copias que se conservan.
BYTES_MAX_LOG = 1_000_000
COPIAS_LOG = 3

log = logging.getLogger("decam")

#: Ruta del fichero de log una vez configurado, o ``None`` si no se pudo abrir.
_ruta_log: Optional[Path] = None


def congelado() -> bool:
    """Indica si el código corre dentro del ejecutable de PyInstaller."""
    return getattr(sys, "frozen", False)


def _carpeta_del_programa() -> Path:
    """Carpeta del ejecutable (o la raíz del proyecto, en desarrollo)."""
    if congelado():
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


def carpeta_datos() -> Path:
    """Devuelve la carpeta donde guardar configuración y registro.

    En desarrollo es el directorio del proyecto. En el ejecutable es
    ``%LOCALAPPDATA%/DeCam`` (creándola si hace falta), y si no se puede, la
    carpeta del propio ejecutable como último recurso.
    """
    programa = _carpeta_del_programa()
    if not congelado():
        return programa

    # Instalación portable: el usuario ya tiene su configuración junto al .exe.
    if (programa / NOMBRE_CONFIG).is_file():
        return programa

    base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
    if not base:
        return programa
    carpeta = Path(base) / NOMBRE_APP
    try:
        carpeta.mkdir(parents=True, exist_ok=True)
    except OSError:
        return programa
    return carpeta


def ruta_config() -> Path:
    """Ruta de ``config.json`` en una carpeta escribible."""
    return carpeta_datos() / NOMBRE_CONFIG


def ruta_log() -> Optional[Path]:
    """Ruta del fichero de log, o ``None`` si el registro no está configurado."""
    return _ruta_log


def configurar() -> Optional[Path]:
    """Abre el fichero de log y engancha los manejadores de excepciones.

    Es idempotente: si ya se llamó antes, devuelve la ruta existente. Si el
    fichero no se puede abrir (carpeta de solo lectura, disco lleno), el
    registro queda desactivado y la aplicación sigue funcionando.

    Returns:
        La ruta del log, o ``None`` si no se pudo abrir.
    """
    global _ruta_log
    if _ruta_log is not None:
        return _ruta_log

    log.setLevel(logging.INFO)
    log.propagate = False
    destino = carpeta_datos() / NOMBRE_LOG
    try:
        manejador = logging.handlers.RotatingFileHandler(
            destino,
            maxBytes=BYTES_MAX_LOG,
            backupCount=COPIAS_LOG,
            encoding="utf-8",
        )
    except OSError:
        _instalar_manejadores_excepciones()
        return None

    manejador.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S")
    )
    log.addHandler(manejador)
    _ruta_log = destino

    log.info("--- Inicio de DeCam (congelado=%s) ---", congelado())
    log.info("Python %s en %s", sys.version.split()[0], sys.platform)
    _instalar_manejadores_excepciones()
    return _ruta_log


def _instalar_manejadores_excepciones() -> None:
    """Redirige al log las excepciones que nadie captura."""
    anterior = sys.excepthook

    def en_hilo_principal(tipo, valor, rastro) -> None:
        log.critical(
            "Excepción no capturada:\n%s",
            "".join(traceback.format_exception(tipo, valor, rastro)),
        )
        anterior(tipo, valor, rastro)

    sys.excepthook = en_hilo_principal

    # threading.excepthook existe desde 3.8; cubre los hilos de análisis.
    import threading

    anterior_hilo = threading.excepthook

    def en_otro_hilo(args) -> None:
        log.critical(
            "Excepción no capturada en el hilo %s:\n%s",
            getattr(args.thread, "name", "?"),
            "".join(
                traceback.format_exception(
                    args.exc_type, args.exc_value, args.exc_traceback
                )
            ),
        )
        anterior_hilo(args)

    threading.excepthook = en_otro_hilo


def excepcion_de_tk(
    _tipo: type[BaseException],
    valor: BaseException,
    rastro: Optional[types.TracebackType],
) -> None:
    """Manejador para ``Tk.report_callback_exception``.

    Tkinter se come los fallos de los callbacks imprimiéndolos por ``stderr``,
    que en el ejecutable sin consola no existe. Esto los deja en el log.
    """
    log.error(
        "Fallo en un callback de Tkinter:\n%s",
        "".join(traceback.format_exception(type(valor), valor, rastro)),
    )
