"""Lo que el análisis escribe a disco: miniaturas, recortes de rostros y CSV.

Separado del analizador para que este no sepa nada de rutas ni de JPG, y para
poder analizar sin escribir nada (``salida=None``), que es lo que hacen los
tests.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import cv2

from decam.deteccion import CajaEntera
from decam.eventos import COLUMNAS_CSV, Evento, ResultadoVideo, formatear_tiempo
from decam.seguimiento import EventoCerrado
from decam.zona import Zona

NOMBRE_CSV = "eventos.csv"


class EscritorSalida:
    """Escribe los artefactos de un análisis en una carpeta de resultados."""

    def __init__(self, carpeta: str | Path, zona: Zona, guardar_recortes: bool = True) -> None:
        """Prepara el escritor.

        Args:
            carpeta: carpeta de resultados; se crea si no existe.
            zona: zona de la puerta, para dibujarla en las miniaturas.
            guardar_recortes: si se guardan los recortes de los rostros.
        """
        self.carpeta = Path(carpeta)
        self.carpeta.mkdir(parents=True, exist_ok=True)
        self.zona = zona
        self.guardar_recortes = guardar_recortes

    @property
    def carpeta_miniaturas(self) -> Path:
        return self.carpeta / "miniaturas"

    @property
    def carpeta_rostros(self) -> Path:
        return self.carpeta / "rostros"

    @property
    def ruta_csv(self) -> Path:
        return self.carpeta / NOMBRE_CSV

    def guardar_evento(self, cerrado: EventoCerrado) -> None:
        """Guarda las imágenes de un evento recién cerrado y anota su miniatura."""
        evento = cerrado.evento
        if cerrado.frame_miniatura is not None:
            evento.miniatura = self.guardar_miniatura(
                evento, cerrado.frame_miniatura, cerrado.caja_miniatura,
            )
        if (
            self.guardar_recortes
            and cerrado.frame_rostros is not None
            and cerrado.cajas_rostros
        ):
            self.guardar_recortes_rostros(
                evento, cerrado.frame_rostros, cerrado.cajas_rostros,
            )

    def guardar_miniatura(self, evento: Evento, frame, caja: CajaEntera | None) -> str:
        """Guarda un JPG del primer frame del evento con la zona y la caja dibujadas.

        Returns:
            La ruta del archivo generado, o cadena vacía si no se pudo guardar.
        """
        carpeta = self.carpeta_miniaturas / evento.tipo
        carpeta.mkdir(parents=True, exist_ok=True)
        imagen = frame.copy()
        self.zona.dibujar(imagen)
        if caja is not None:
            cx1, cy1, cx2, cy2 = caja
            cv2.rectangle(imagen, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)
            cv2.putText(
                imagen, "persona", (cx1, max(15, cy1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
            )
        destino = carpeta / f"{self._base(evento)}.jpg"
        return str(destino) if cv2.imwrite(str(destino), imagen) else ""

    def guardar_recortes_rostros(
        self, evento: Evento, frame, cajas: list[CajaEntera],
    ) -> list[str]:
        """Guarda un recorte JPG por rostro más el frame completo señalizado.

        Se usa el frame del evento con más rostros, que no tiene por qué ser el
        mismo de la miniatura.

        Returns:
            Las rutas de los recortes generados.
        """
        carpeta = self.carpeta_rostros / evento.tipo
        carpeta.mkdir(parents=True, exist_ok=True)
        alto, ancho = frame.shape[:2]
        base = self._base(evento)

        rutas: list[str] = []
        señalizado = frame.copy()
        for i, (x1, y1, x2, y2) in enumerate(cajas, start=1):
            cv2.rectangle(señalizado, (x1, y1), (x2, y2), (0, 200, 255), 2)
            # Margen del 30% alrededor del rostro: sirve para identificarlo mejor.
            margen_x = int((x2 - x1) * 0.3)
            margen_y = int((y2 - y1) * 0.3)
            recorte = frame[
                max(0, y1 - margen_y) : min(alto, y2 + margen_y),
                max(0, x1 - margen_x) : min(ancho, x2 + margen_x),
            ]
            if recorte.size == 0:
                continue
            destino = carpeta / f"{base}_rostro{i}.jpg"
            if cv2.imwrite(str(destino), recorte):
                rutas.append(str(destino))

        cv2.imwrite(str(carpeta / f"{base}_frame.jpg"), señalizado)
        return rutas

    def escribir_csv(self, resultados: Sequence[ResultadoVideo]) -> Path:
        """Escribe todos los eventos en ``eventos.csv`` y devuelve su ruta."""
        return escribir_csv(resultados, self.ruta_csv)

    @staticmethod
    def _base(evento: Evento) -> str:
        """Nombre base de los archivos de un evento: ``video_HH-MM-SS``."""
        marca = formatear_tiempo(evento.inicio).replace(":", "-")
        return f"{Path(evento.archivo).stem}_{marca}"


def escribir_csv(resultados: Sequence[ResultadoVideo], destino: str | Path) -> Path:
    """Escribe todos los eventos de ``resultados`` en un CSV (UTF-8 con BOM)."""
    ruta = Path(destino)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", newline="", encoding="utf-8-sig") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(COLUMNAS_CSV))
        escritor.writeheader()
        for resultado in resultados:
            for evento in resultado.eventos:
                escritor.writerow(evento.a_fila_csv())
    return ruta
