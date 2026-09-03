"""La zona de la puerta y los criterios para decidir si una persona está en ella.

:class:`Zona` es lo único que el resto del análisis conoce: un lugar del frame
que sabe si contiene un punto y qué fracción de una caja cae dentro. Hoy la
única implementación es :class:`ZonaRectangular`; una zona poligonal encajaría
implementando el mismo protocolo sin tocar los criterios ni el analizador.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import cv2

#: Caja ``(x1, y1, x2, y2)``.
Caja = tuple[float, float, float, float]
Punto = tuple[float, float]

#: Criterios para decidir si una persona "está" en la zona.
CRITERIOS_ZONA = ("pies", "centro", "solape")


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


def normalizar_zona(zona: Sequence[float]) -> tuple[int, int, int, int]:
    """Ordena y redondea una zona a ``(x1, y1, x2, y2)`` con enteros."""
    x1, y1, x2, y2 = zona
    return (
        int(round(min(x1, x2))),
        int(round(min(y1, y2))),
        int(round(max(x1, x2))),
        int(round(max(y1, y2))),
    )


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
        color = (255, 0, 0)
        cv2.rectangle(imagen, (self.x1, self.y1), (self.x2, self.y2), color, 2)
        cv2.putText(
            imagen, "puerta", (self.x1, max(15, self.y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )

    def __str__(self) -> str:
        return f"({self.x1}, {self.y1}) - ({self.x2}, {self.y2})"


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
