"""Manifiesto de videos ya analizados, para el análisis incremental.

Es un ``analizados.json`` en la carpeta de resultados con, por cada video, su
tamaño, su fecha de modificación, una huella de la configuración con la que se
analizó y el resultado serializado. Al volver a lanzar el análisis sobre la
misma carpeta, los videos cuya entrada coincide se reutilizan sin procesarlos.

Dos decisiones de diseño:

* La huella excluye los parámetros que no cambian el resultado (el acelerador,
  la decodificación por hardware): cambiar de GPU a CPU no invalida nada, pero
  mover la zona de la puerta o el criterio sí, porque los eventos serían otros.
* Se escribe después de cada video, no al final: si el análisis se cancela o se
  cae a mitad, lo ya hecho se conserva.

Este módulo trabaja con diccionarios y no importa nada de ``decam.analizador`` para que
``decam.analizador`` pueda importarlo sin ciclos.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional

NOMBRE_MANIFIESTO = "analizados.json"

#: Se incrementa cuando cambia el formato del fichero; un manifiesto de otra
#: versión se ignora entero en vez de intentar migrarlo.
VERSION = 1

#: Parámetros de la configuración que no alteran los eventos detectados.
CLAVES_SIN_EFECTO = frozenset({"acelerador", "decodificacion_hardware"})


def huella_config(config: dict[str, Any]) -> str:
    """Resume la configuración en una cadena corta y estable.

    Args:
        config: la configuración como diccionario (``dataclasses.asdict``).
    """
    relevante = {k: v for k, v in config.items() if k not in CLAVES_SIN_EFECTO}
    texto = json.dumps(relevante, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(texto.encode("utf-8")).hexdigest()[:16]


def clave_video(video: str | Path) -> str:
    """Clave con la que se guarda un video: su ruta absoluta normalizada."""
    return os.path.normcase(str(Path(video).resolve()))


class Manifiesto:
    """Lee y escribe el manifiesto de una carpeta de resultados."""

    def __init__(self, ruta: str | Path) -> None:
        """Abre el manifiesto en ``ruta``; si no existe o está dañado, empieza vacío."""
        self.ruta = Path(ruta)
        self._videos: dict[str, dict[str, Any]] = {}
        self.cargar()

    def __len__(self) -> int:
        return len(self._videos)

    def cargar(self) -> None:
        """Lee el fichero. Cualquier problema deja el manifiesto vacío."""
        self._videos = {}
        try:
            datos = json.loads(self.ruta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(datos, dict) or datos.get("version") != VERSION:
            return
        videos = datos.get("videos")
        if isinstance(videos, dict):
            self._videos = {k: v for k, v in videos.items() if isinstance(v, dict)}

    def guardar(self) -> None:
        """Escribe el fichero de forma atómica (temporal + reemplazo)."""
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        contenido = json.dumps(
            {"version": VERSION, "videos": self._videos},
            indent=2,
            ensure_ascii=False,
        )
        temporal = self.ruta.with_suffix(self.ruta.suffix + ".tmp")
        temporal.write_text(contenido, encoding="utf-8")
        os.replace(temporal, self.ruta)

    def buscar(self, video: str | Path, huella: str) -> Optional[dict[str, Any]]:
        """Devuelve el resultado guardado de un video si sigue siendo válido.

        Es válido si el fichero no ha cambiado (mismo tamaño y fecha) y se
        analizó con la misma configuración (misma huella).
        """
        entrada = self._videos.get(clave_video(video))
        if entrada is None:
            return None
        try:
            estado = Path(video).stat()
        except OSError:
            return None
        if (
            entrada.get("tamano") != estado.st_size
            or entrada.get("mtime_ns") != estado.st_mtime_ns
            or entrada.get("huella") != huella
        ):
            return None
        resultado = entrada.get("resultado")
        return resultado if isinstance(resultado, dict) else None

    def registrar(
        self, video: str | Path, huella: str, resultado: dict[str, Any]
    ) -> None:
        """Guarda el resultado de un video y escribe el fichero en el acto."""
        estado = Path(video).stat()
        self._videos[clave_video(video)] = {
            "archivo": Path(video).name,
            "tamano": estado.st_size,
            "mtime_ns": estado.st_mtime_ns,
            "huella": huella,
            "resultado": resultado,
        }
        self.guardar()
