"""Parámetros con los que se ejecuta el análisis."""

from __future__ import annotations

from dataclasses import dataclass

from decam.aceleradores import ACELERADORES
from decam.movimiento import UMBRAL_MOVIMIENTO
from decam.rostros import UMBRAL_SFACE
from decam.zona import CRITERIOS_ZONA, EspecZona, Zona, zona_desde


@dataclass
class ConfiguracionAnalisis:
    """Parámetros con los que se ejecuta el análisis.

    ``zona_puerta`` se guarda como dato plano —cuatro enteros para un
    rectángulo o una tupla de pares para un polígono— para que la configuración
    se serialice tal cual en el manifiesto; :attr:`zona` la convierte en el
    objeto que usa el análisis.
    """

    zona_puerta: EspecZona
    fps_analisis: float = 1.0
    tolerancia_segundos: float = 3.0
    modelo: str = "yolov8n"
    confianza: float = 0.35
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
    umbral_identificacion: float = UMBRAL_SFACE
    #: Seguir a cada persona entre frames (ByteTrack) para contar personas
    #: distintas y saber si entran o salen. Cuesta poco; solo se desactiva para
    #: comparar resultados con versiones anteriores.
    usar_tracking: bool = True

    @property
    def zona(self) -> Zona:
        """La zona de la puerta como objeto (rectángulo o polígono)."""
        return zona_desde(self.zona_puerta)

    def validar(self) -> None:
        """Verifica que los parámetros sean utilizables.

        Raises:
            ValueError: si algún parámetro es inválido.
        """
        self.zona.validar()
        if self.fps_analisis <= 0:
            raise ValueError("Los frames por segundo deben ser mayores que cero.")
        if self.tolerancia_segundos < 0:
            raise ValueError("La tolerancia no puede ser negativa.")
        if self.criterio_zona not in CRITERIOS_ZONA:
            raise ValueError(
                f"Criterio de zona inválido: {self.criterio_zona}. "
                f"Opciones: {', '.join(CRITERIOS_ZONA)}"
            )
        if not 0 < self.min_solape <= 1:
            raise ValueError("El solape mínimo debe estar entre 0 y 1.")
        if self.acelerador not in ACELERADORES:
            raise ValueError(f"Acelerador inválido: {self.acelerador}")
        if not 0 <= self.umbral_movimiento < 1:
            raise ValueError("El umbral de movimiento debe estar entre 0 y 1.")
