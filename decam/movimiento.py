"""Filtro de movimiento: decide si un frame merece pasar por el modelo."""

from __future__ import annotations

import cv2

#: Fracción de píxeles que deben cambiar entre dos frames analizados para que
#: valga la pena invocar al modelo. Medido sobre grabaciones de pasillo 1080p:
#: los frames con persona no bajan de 0.009 y los vacíos no pasan de 0.0011 (p90),
#: así que 0.002 deja un margen holgado por ambos lados.
UMBRAL_MOVIMIENTO = 0.002

#: Ancho al que se reduce el frame para comparar. Reducir es lo que hace barata
#: la comparación y, de paso, diluye el reloj sobreimpreso de las cámaras.
ANCHO_MOVIMIENTO = 320


class DetectorMovimiento:
    """Mide cuánto cambia un frame respecto al anterior analizado.

    Sirve para no invocar al modelo en frames donde no ocurre nada, que en una
    cámara de vigilancia son la inmensa mayoría. El frame se pasa a gris, se
    reduce y se difumina antes de compararlo: así un cambio irrelevante —el
    reloj sobreimpreso, el ruido del sensor— no cuenta como movimiento.
    """

    def __init__(
        self,
        ancho: int = ANCHO_MOVIMIENTO,
        umbral_pixel: int = 25,
    ) -> None:
        """Prepara el detector.

        Args:
            ancho: ancho al que se reduce el frame antes de comparar.
            umbral_pixel: diferencia de gris a partir de la cual un píxel cuenta
                como cambiado.
        """
        self.ancho = ancho
        self.umbral_pixel = umbral_pixel
        self._previo = None

    def _preparar(self, frame):
        """Pasa el frame a gris reducido y difuminado."""
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        alto = max(1, int(gris.shape[0] * self.ancho / gris.shape[1]))
        return cv2.GaussianBlur(cv2.resize(gris, (self.ancho, alto)), (5, 5), 0)

    def fraccion_cambiada(self, frame) -> float:
        """Devuelve la fracción de píxeles que cambian respecto al frame previo.

        El primer frame no tiene con qué compararse, así que devuelve ``1.0``
        para que siempre se analice.
        """
        actual = self._preparar(frame)
        if self._previo is None:
            self._previo = actual
            return 1.0

        diferencia = cv2.absdiff(self._previo, actual)
        _, binaria = cv2.threshold(
            diferencia, self.umbral_pixel, 255, cv2.THRESH_BINARY
        )
        self._previo = actual
        return cv2.countNonZero(binaria) / binaria.size

    def reiniciar(self) -> None:
        """Olvida el frame de referencia; se usa al cambiar de video."""
        self._previo = None
