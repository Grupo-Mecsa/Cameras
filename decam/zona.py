"""La zona de la puerta y los criterios para decidir si una persona está en ella.

:class:`Zona` es lo único que el resto del análisis conoce: un lugar del frame
que sabe si contiene un punto y qué fracción de una caja cae dentro. Hay dos
implementaciones, :class:`ZonaRectangular` y :class:`ZonaPoligonal`; ni los
criterios ni el analizador distinguen entre ellas.

La zona se persiste (config, preferencias, manifiesto) como un dato plano, la
*especificación*: cuatro enteros para un rectángulo o una secuencia de pares
para un polígono. :func:`normalizar_espec` la canoniza y :func:`zona_desde` la
convierte en objeto.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence, Union

import cv2
import numpy as np

#: Caja ``(x1, y1, x2, y2)``.
Caja = tuple[float, float, float, float]
Punto = tuple[float, float]

#: Especificación persistible de una zona: rectángulo o lista de vértices.
EspecZona = Union[tuple[int, int, int, int], tuple[tuple[int, int], ...]]

#: Criterios para decidir si una persona "está" en la zona.
CRITERIOS_ZONA = ("pies", "centro", "solape")

COLOR_ZONA_BGR = (255, 0, 0)


class Zona(Protocol):
    """Una región del frame contra la que se comparan las personas."""

    def contiene_punto(self, punto: Punto) -> bool:
        """``True`` si el punto cae dentro de la zona (bordes incluidos)."""

    def fraccion_de(self, caja: Caja) -> float:
        """Fracción (0 a 1) del área de ``caja`` que cae dentro de la zona."""

    def validar(self) -> None:
        """Lanza ``ValueError`` si la zona no es utilizable."""

    def dibujar(self, imagen) -> None:
        """Pinta la zona sobre una imagen BGR (para miniaturas)."""


# ------------------------------------------------------------- especificación


def es_poligono(espec: Sequence) -> bool:
    """``True`` si la especificación es una lista de vértices y no un rectángulo."""
    return len(espec) > 0 and isinstance(espec[0], (tuple, list))


def normalizar_zona(zona: Sequence[float]) -> tuple[int, int, int, int]:
    """Ordena y redondea un rectángulo a ``(x1, y1, x2, y2)`` con enteros."""
    x1, y1, x2, y2 = zona
    return (
        int(round(min(x1, x2))),
        int(round(min(y1, y2))),
        int(round(max(x1, x2))),
        int(round(max(y1, y2))),
    )


def normalizar_espec(datos: Sequence) -> EspecZona:
    """Canoniza lo que venga de JSON o de la interfaz.

    Returns:
        Una tupla de cuatro enteros (rectángulo) o una tupla de pares de
        enteros (polígono).

    Raises:
        ValueError: si no tiene forma de zona.
    """
    if len(datos) == 0:
        raise ValueError("La zona está vacía.")
    if es_poligono(datos):
        puntos = tuple((int(round(p[0])), int(round(p[1]))) for p in datos)
        if len(puntos) < 3:
            raise ValueError("Un polígono necesita al menos tres puntos.")
        return puntos
    if len(datos) != 4:
        raise ValueError("Un rectángulo son cuatro coordenadas.")
    return normalizar_zona(datos)


def espec_a_lista(espec: EspecZona) -> list:
    """Forma serializable en JSON de una especificación."""
    if es_poligono(espec):
        return [list(p) for p in espec]  # type: ignore[union-attr]
    return list(espec)


def zona_desde(datos: Sequence) -> "Zona":
    """Construye la zona que corresponda a una especificación."""
    espec = normalizar_espec(datos)
    if es_poligono(espec):
        return ZonaPoligonal(espec)  # type: ignore[arg-type]
    return ZonaRectangular(*espec)  # type: ignore[misc]


# ------------------------------------------------------------------ rectángulo


@dataclass(frozen=True)
class ZonaRectangular:
    """Un rectángulo alineado con los ejes, en píxeles del frame."""

    x1: int
    y1: int
    x2: int
    y2: int

    @classmethod
    def desde(cls, zona: Sequence[float]) -> "ZonaRectangular":
        """Crea la zona a partir de cualquier secuencia de cuatro coordenadas."""
        return cls(*normalizar_zona(zona))

    @property
    def como_tupla(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def ancho(self) -> int:
        return self.x2 - self.x1

    @property
    def alto(self) -> int:
        return self.y2 - self.y1

    def validar(self) -> None:
        if self.ancho <= 0 or self.alto <= 0:
            raise ValueError("La zona de la puerta no es válida (ancho o alto cero).")

    def contiene_punto(self, punto: Punto) -> bool:
        x, y = punto
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def area_solapada(self, caja: Caja) -> float:
        """Área de intersección entre la caja y la zona (0 si no se tocan)."""
        ax1, ay1, ax2, ay2 = caja
        ancho = min(ax2, self.x2) - max(ax1, self.x1)
        alto = min(ay2, self.y2) - max(ay1, self.y1)
        if ancho <= 0 or alto <= 0:
            return 0.0
        return float(ancho * alto)

    def fraccion_de(self, caja: Caja) -> float:
        ancho = caja[2] - caja[0]
        alto = caja[3] - caja[1]
        if ancho <= 0 or alto <= 0:
            return 0.0
        return self.area_solapada(caja) / float(ancho * alto)

    def dibujar(self, imagen) -> None:
        cv2.rectangle(imagen, (self.x1, self.y1), (self.x2, self.y2), COLOR_ZONA_BGR, 2)
        _etiqueta(imagen, (self.x1, self.y1))

    def __str__(self) -> str:
        return f"({self.x1}, {self.y1}) - ({self.x2}, {self.y2})"


# -------------------------------------------------------------------- polígono


@dataclass(frozen=True)
class ZonaPoligonal:
    """Un polígono cualquiera (simple, sin cruces), en píxeles del frame.

    Es lo que hace falta cuando la cámara ve la puerta en perspectiva: el vano
    o el tramo de suelo delante de él no es un rectángulo en la imagen.
    """

    puntos: tuple[tuple[int, int], ...]

    @classmethod
    def desde(cls, puntos: Sequence[Sequence[float]]) -> "ZonaPoligonal":
        return cls(tuple((int(round(x)), int(round(y))) for x, y in puntos))

    @property
    def como_tupla(self) -> tuple[tuple[int, int], ...]:
        return self.puntos

    @property
    def caja_envolvente(self) -> tuple[int, int, int, int]:
        xs = [p[0] for p in self.puntos]
        ys = [p[1] for p in self.puntos]
        return (min(xs), min(ys), max(xs), max(ys))

    def _contorno(self) -> np.ndarray:
        return np.array(self.puntos, dtype=np.int32).reshape(-1, 1, 2)

    def validar(self) -> None:
        if len(self.puntos) < 3:
            raise ValueError("La zona de la puerta necesita al menos tres puntos.")
        if cv2.contourArea(self._contorno()) <= 0:
            raise ValueError("La zona de la puerta no es válida (puntos alineados).")

    def contiene_punto(self, punto: Punto) -> bool:
        x, y = punto
        return cv2.pointPolygonTest(self._contorno(), (float(x), float(y)), False) >= 0

    def fraccion_de(self, caja: Caja) -> float:
        """Fracción de la caja dentro del polígono, por rasterizado.

        Se pinta el polígono en una máscara del tamaño de la intersección entre
        la caja y la caja envolvente del polígono y se cuentan los píxeles: es
        exacto a ±1 px, que sobra para un umbral de solape.
        """
        x1, y1, x2, y2 = caja
        ancho = x2 - x1
        alto = y2 - y1
        if ancho <= 0 or alto <= 0:
            return 0.0
        bx1, by1, bx2, by2 = self.caja_envolvente
        ix1, iy1 = max(x1, bx1), max(y1, by1)
        ix2, iy2 = min(x2, bx2), min(y2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0

        ox, oy = int(math.floor(ix1)), int(math.floor(iy1))
        mascara = np.zeros(
            (int(math.ceil(iy2)) - oy, int(math.ceil(ix2)) - ox), dtype=np.uint8
        )
        desplazado = self._contorno() - np.array([ox, oy], dtype=np.int32)
        cv2.fillPoly(mascara, [desplazado], 1)
        return cv2.countNonZero(mascara) / float(ancho * alto)

    def dibujar(self, imagen) -> None:
        cv2.polylines(imagen, [self._contorno()], True, COLOR_ZONA_BGR, 2)
        _etiqueta(imagen, min(self.puntos, key=lambda p: (p[1], p[0])))

    def __str__(self) -> str:
        return f"polígono de {len(self.puntos)} puntos"


def _etiqueta(imagen, punto: tuple[int, int]) -> None:
    cv2.putText(
        imagen, "puerta", (punto[0], max(15, punto[1] - 6)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_ZONA_BGR, 1, cv2.LINE_AA,
    )


# ------------------------------------------------------------------- criterios


def punto_pies(caja: Caja) -> Punto:
    """Punto de apoyo de una persona: el centro del borde inferior de su caja."""
    x1, _, x2, y2 = caja
    return ((x1 + x2) / 2.0, y2)


def punto_centro(caja: Caja) -> Punto:
    """Centro geométrico de la caja."""
    x1, y1, x2, y2 = caja
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def persona_en_zona(
    caja: Caja,
    zona: Zona,
    criterio: str = "pies",
    min_solape: float = 0.25,
) -> bool:
    """Decide si una persona detectada cuenta como "dentro de la zona".

    El criterio importa mucho en pasillos: por perspectiva, la caja de alguien
    que camina por el centro puede rozar la esquina de una puerta lejana sin que
    la persona esté ni cerca de ella. Por eso el solape simple da falsos
    positivos y no es el criterio por defecto.

    Args:
        caja: caja ``(x1, y1, x2, y2)`` de la persona.
        zona: zona de la puerta.
        criterio:
            ``pies``: el punto de apoyo (centro del borde inferior) cae en la
                zona. Es el más fiable cuando la zona se dibuja sobre el suelo o
                el vano por el que se pisa.
            ``centro``: el centro de la caja cae en la zona.
            ``solape``: al menos ``min_solape`` del área de la persona está
                dentro de la zona.
        min_solape: fracción mínima para el criterio ``solape``.

    Returns:
        ``True`` si la persona cuenta como presente en la zona.
    """
    if criterio == "pies":
        return zona.contiene_punto(punto_pies(caja))
    if criterio == "centro":
        return zona.contiene_punto(punto_centro(caja))
    if criterio == "solape":
        return zona.fraccion_de(caja) >= min_solape
    raise ValueError(f"Criterio de zona desconocido: {criterio}")
