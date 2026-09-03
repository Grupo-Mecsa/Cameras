"""Detección de personas en un frame.

:class:`DetectorPersonas` es la abstracción que consume el analizador; lo único
que le pide a un detector es una lista de cajas con su confianza. Eso permite
sustituir YOLO por otro modelo —o por un doble en los tests— sin tocar el
resto del análisis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

from decam.aceleradores import dispositivo_de_prediccion, exportar_a_openvino
from decam.callbacks import CallbackLog

# En el dataset COCO la clase 0 corresponde a "person".
CLASE_PERSONA: int = 0

CajaEntera = tuple[int, int, int, int]


@dataclass(frozen=True)
class DeteccionCruda:
    """Una persona tal como sale del detector, antes del seguimiento."""

    caja: CajaEntera
    confianza: float


@dataclass(frozen=True)
class Deteccion:
    """Una persona detectada en un frame, con su identificador de seguimiento."""

    caja: CajaEntera
    #: ``None`` si no hay seguimiento o la pista aún no se ha confirmado.
    id: Optional[int] = None

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.caja
        return max(0, x2 - x1) * max(0, y2 - y1)


class DetectorPersonas(Protocol):
    """Lo que el analizador necesita de un detector de personas."""

    @property
    def descripcion(self) -> str:
        """Texto para el log: qué modelo y dónde corre."""

    def preparar(self) -> None:
        """Carga lo que haga falta. Se llama una vez antes del primer frame."""

    def detectar(self, frame) -> list[DeteccionCruda]:
        """Devuelve todas las personas del frame (BGR)."""


class DetectorYOLO:
    """Detector de personas con un modelo YOLO de ultralytics.

    El modelo se carga en :meth:`preparar` (o en la primera detección) y se
    reutiliza para todos los videos. Con ``openvino-gpu`` se carga la versión
    OpenVINO del modelo, que se exporta automáticamente la primera vez.
    """

    def __init__(
        self,
        modelo: str,
        confianza: float,
        acelerador: str,
        on_log: Optional[CallbackLog] = None,
    ) -> None:
        """Prepara el detector sin cargar aún el modelo.

        Args:
            modelo: nombre del modelo (``yolov8n``) o ruta a un ``.pt``.
            confianza: confianza mínima de las detecciones.
            acelerador: acelerador ya resuelto (``cuda``, ``openvino-gpu``,
                ``cpu``); ver :func:`decam.aceleradores.resolver_acelerador`.
            on_log: destino de los mensajes de carga.
        """
        self.modelo = modelo
        self.confianza = confianza
        self.acelerador = acelerador
        self.dispositivo = dispositivo_de_prediccion(acelerador)
        self._log = on_log or (lambda _m: None)
        self._modelo = None

    @property
    def descripcion(self) -> str:
        return f"YOLO {self.modelo} en {self.acelerador} (device={self.dispositivo})"

    def preparar(self) -> None:
        if self._modelo is not None:
            return

        from ultralytics import YOLO  # import perezoso: tarda en cargar

        if self.acelerador == "openvino-gpu":
            carpeta = exportar_a_openvino(self.modelo, self._log)
            self._log(f"Cargando {carpeta.name} en GPU Intel (OpenVINO)...")
            self._modelo = YOLO(str(carpeta), task="detect")
        else:
            nombre = self.modelo
            if not nombre.endswith(".pt"):
                nombre = f"{nombre}.pt"
            self._log(f"Cargando modelo {nombre} en {self.acelerador.upper()}...")
            self._modelo = YOLO(nombre)

    def detectar(self, frame) -> list[DeteccionCruda]:
        self.preparar()
        predicciones = self._modelo.predict(
            frame,
            classes=[CLASE_PERSONA],
            conf=self.confianza,
            device=self.dispositivo,
            verbose=False,
        )
        detecciones: list[DeteccionCruda] = []
        for prediccion in predicciones:
            cajas = getattr(prediccion, "boxes", None)
            if cajas is None:
                continue
            cajas = cajas.cpu().numpy()
            for valores, confianza in zip(cajas.xyxy, cajas.conf):
                x1, y1, x2, y2 = (int(v) for v in valores[:4])
                detecciones.append(DeteccionCruda((x1, y1, x2, y2), float(confianza)))
        return detecciones
