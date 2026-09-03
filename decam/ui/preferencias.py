"""Configuración persistida entre ejecuciones (``config.json``)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from decam import registro
from decam.movimiento import UMBRAL_MOVIMIENTO

RUTA_CONFIG = registro.ruta_config()


@dataclass
class Preferencias:
    """Lo que la interfaz recuerda de una sesión a otra."""

    carpeta_videos: str = ""
    carpeta_salida: str = ""
    zona_puerta: Optional[list[int]] = None
    fps_analisis: float = 1.0
    tolerancia_segundos: float = 3.0
    modelo: str = "yolov8n"
    acelerador: str = "auto"
    decodificacion_hardware: bool = False
    filtro_movimiento: bool = True
    umbral_movimiento: float = UMBRAL_MOVIMIENTO
    criterio_zona: str = "pies"
    min_solape: float = 0.25
    registrar_general: bool = True
    detectar_rostros: bool = False
    backend_rostros: str = "yunet"
    guardar_recortes_rostros: bool = True
    identificar_rostros: bool = False
    carpeta_personas: str = ""
    usar_tracking: bool = True
    incremental: bool = True

    @classmethod
    def cargar(cls, ruta: Path = RUTA_CONFIG) -> "Preferencias":
        """Lee las preferencias del disco; devuelve valores por defecto si falla."""
        try:
            datos: dict[str, Any] = json.loads(ruta.read_text(encoding="utf-8"))
        except FileNotFoundError:
            registro.log.info("Sin configuración previa en %s", ruta)
            return cls()
        except (OSError, ValueError) as exc:
            registro.log.warning("No se pudo leer %s: %s", ruta, exc)
            return cls()
        validos = {k: v for k, v in datos.items() if k in cls.__dataclass_fields__}
        try:
            return cls(**validos)
        except TypeError as exc:
            registro.log.warning("Configuración inválida en %s: %s", ruta, exc)
            return cls()

    def guardar(self, ruta: Path = RUTA_CONFIG) -> None:
        """Escribe las preferencias en disco (los errores se dejan en el log)."""
        try:
            ruta.write_text(
                json.dumps(self.__dict__, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            registro.log.error("No se pudo guardar %s: %s", ruta, exc)
