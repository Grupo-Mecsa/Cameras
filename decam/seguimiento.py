"""Seguimiento de personas entre frames y agrupación de detecciones en eventos.

Tres piezas, de la más baja a la más alta:

* :class:`Rastreador` (y su implementación :class:`RastreadorPersonas` con
  ByteTrack): da a cada persona un identificador estable entre frames.
* :class:`Trayectorias`: recuerda por dónde ha pasado cada identificador
  respecto a la zona, para decidir si entra o sale.
* :class:`SeguidorEventos`: convierte la secuencia de frames con/sin persona en
  intervalos de tiempo (eventos).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Protocol, Sequence

import numpy as np

from decam.deteccion import CajaEntera, DeteccionCruda
from decam.eventos import Evento

# ------------------------------------------------------------------ rastreador


class Rastreador(Protocol):
    """Asigna identificadores estables a las detecciones de frames consecutivos."""

    def reiniciar(self) -> None:
        """Olvida todas las pistas; se llama al empezar cada video."""

    def asignar_ids(self, detecciones: Sequence[DeteccionCruda]) -> list[Optional[int]]:
        """Devuelve el identificador de cada detección, alineado con la entrada.

        ``None`` para las detecciones cuya pista aún no se ha confirmado. Se
        debe llamar también con la lista vacía, para que el tiempo avance.
        """


class _LoteDetecciones:
    """Lo mínimo que ``BYTETracker.update`` necesita de un lote de detecciones.

    Imita la parte del ``Boxes`` de ultralytics que el tracker consulta
    (``conf``, ``xywh``, ``cls`` e indexado booleano), para no depender del
    formato de resultados de ningún modelo concreto.
    """

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls

    @classmethod
    def desde(cls, detecciones: Sequence[DeteccionCruda]) -> "_LoteDetecciones":
        xyxy = np.array([d.caja for d in detecciones], dtype=np.float32).reshape(-1, 4)
        conf = np.array([d.confianza for d in detecciones], dtype=np.float32)
        return cls(xyxy, conf, np.zeros(len(detecciones), dtype=np.float32))

    @property
    def xywh(self) -> np.ndarray:
        x1, y1, x2, y2 = (self.xyxy[:, i] for i in range(4))
        return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, indice) -> "_LoteDetecciones":
        return _LoteDetecciones(self.xyxy[indice], self.conf[indice], self.cls[indice])


class RastreadorPersonas:
    """Asigna un identificador estable a cada persona entre frames (ByteTrack).

    Se usa el tracker de ultralytics directamente, en vez de ``modelo.track()``,
    por dos motivos: ``track()`` descarta las detecciones cuya pista aún no se
    ha confirmado (una persona vista en un solo frame desaparecería), y no deja
    reiniciar el seguimiento al cambiar de video de forma explícita. Aquí todas
    las detecciones se conservan; las que todavía no tienen pista llevan
    ``None``.
    """

    #: Frames que ByteTrack mantiene viva una pista perdida, como mínimo.
    BUFFER_MINIMO = 30

    def __init__(self, fps_analisis: float, tolerancia_segundos: float) -> None:
        """Crea el tracker.

        Args:
            fps_analisis: frames analizados por segundo de video.
            tolerancia_segundos: tolerancia de agrupación de eventos. Una pista
                perdida se mantiene al menos ese tiempo para que la misma
                persona no reciba dos identificadores si la detección falla
                unos frames.
        """
        from types import SimpleNamespace

        from ultralytics.trackers import BYTETracker  # import perezoso

        # ByteTrack cuenta en frames analizados, no en segundos.
        self.buffer = max(
            self.BUFFER_MINIMO, math.ceil(tolerancia_segundos * fps_analisis) + 1
        )
        argumentos = SimpleNamespace(
            track_high_thresh=0.25,
            track_low_thresh=0.1,
            new_track_thresh=0.25,
            track_buffer=self.buffer,
            match_thresh=0.8,
            fuse_score=True,
        )
        self._tracker = BYTETracker(argumentos)

    def reiniciar(self) -> None:
        self._tracker.reset()

    def asignar_ids(self, detecciones: Sequence[DeteccionCruda]) -> list[Optional[int]]:
        ids: list[Optional[int]] = [None] * len(detecciones)
        pistas = self._tracker.update(_LoteDetecciones.desde(detecciones))
        # Cada fila: x1, y1, x2, y2, id, confianza, clase, índice en la entrada.
        for fila in pistas:
            ids[int(fila[-1])] = int(fila[4])
        return ids


# ---------------------------------------------------------------- trayectorias

#: Etiquetas de dirección, por (estaba dentro al principio, estaba dentro al final).
DIRECCIONES = {
    (False, True): "entra",
    (True, False): "sale",
    (False, False): "cruza",
    (True, True): "permanece",
}
_PLURALES = {"entra": "entran", "sale": "salen", "cruza": "cruzan", "permanece": "permanecen"}


@dataclass
class _Trayectoria:
    """Lo que se sabe del recorrido de una pista respecto a la zona."""

    primero_dentro: bool
    ultimo_dentro: bool
    estuvo_dentro: bool


class Trayectorias:
    """Recorrido de cada pista respecto a la zona de la puerta, en un video.

    Con esto se decide la dirección de cada persona: si la primera vez que se
    la vio estaba fuera de la zona y la última dentro, "entra" (llegó a la
    puerta y desapareció por ella); al revés, "sale"; fuera-fuera, "cruza" la
    zona de paso; dentro-dentro, "permanece".
    """

    def __init__(self) -> None:
        self._pistas: dict[int, _Trayectoria] = {}

    def observar(self, id_pista: int, dentro: bool) -> None:
        """Anota dónde está la pista en este frame."""
        actual = self._pistas.get(id_pista)
        if actual is None:
            self._pistas[id_pista] = _Trayectoria(dentro, dentro, dentro)
        else:
            actual.ultimo_dentro = dentro
            actual.estuvo_dentro = actual.estuvo_dentro or dentro

    def direccion(self, id_pista: int) -> Optional[str]:
        """Dirección de una pista, o ``None`` si nunca tocó la zona."""
        pista = self._pistas.get(id_pista)
        if pista is None or not pista.estuvo_dentro:
            return None
        return DIRECCIONES[(pista.primero_dentro, pista.ultimo_dentro)]

    def resumen(self, ids: Iterable[int]) -> str:
        """Resume las direcciones de varias pistas: ``"2 entran, 1 sale"``."""
        conteo: dict[str, int] = {}
        for id_pista in ids:
            etiqueta = self.direccion(id_pista)
            if etiqueta is not None:
                conteo[etiqueta] = conteo.get(etiqueta, 0) + 1
        partes = []
        for etiqueta in DIRECCIONES.values():  # orden fijo y legible
            n = conteo.get(etiqueta, 0)
            if n == 1:
                partes.append(f"1 {etiqueta}")
            elif n > 1:
                partes.append(f"{n} {_PLURALES[etiqueta]}")
        return ", ".join(partes)


# --------------------------------------------------------------------- eventos


@dataclass
class EventoCerrado:
    """Un evento terminado junto a las imágenes que hay que guardar de él."""

    evento: Evento
    frame_miniatura: Any = None
    caja_miniatura: Optional[CajaEntera] = None
    frame_rostros: Any = None
    cajas_rostros: list[CajaEntera] = field(default_factory=list)
    #: Identificadores de seguimiento de las personas vistas en el evento.
    ids: set[int] = field(default_factory=set)


class SeguidorEventos:
    """Agrupa detecciones sueltas de un mismo tipo en intervalos de tiempo.

    Se le pasa cada frame analizado; devuelve un :class:`EventoCerrado` en el
    momento en que un intervalo termina, es decir cuando pasan más de
    ``tolerancia`` segundos sin ninguna detección.
    """

    def __init__(self, archivo: str, tipo: str, tolerancia: float) -> None:
        """Crea un seguidor para un tipo de evento de un video concreto."""
        self.archivo = archivo
        self.tipo = tipo
        self.tolerancia = tolerancia
        self._evento: Optional[Evento] = None
        self._ultimo_positivo = 0.0
        self._frame_miniatura = None
        self._caja_miniatura: Optional[CajaEntera] = None
        self._frame_rostros = None
        self._cajas_rostros: list[CajaEntera] = []
        self._nombres: set[str] = set()
        self._ids: set[int] = set()
        self._max_simultaneas = 0

    @property
    def activo(self) -> bool:
        """``True`` si hay un evento abierto pendiente de cerrar."""
        return self._evento is not None

    def actualizar(
        self,
        segundo: float,
        caja: Optional[CajaEntera],
        frame,
        rostros: list[CajaEntera],
        nombres: set[str],
        ids: Optional[set[int]] = None,
        simultaneas: int = 1,
    ) -> Optional[EventoCerrado]:
        """Incorpora el resultado de un frame.

        Args:
            segundo: instante del frame dentro del video.
            caja: caja de la persona principal detectada, o ``None`` si no hay
                ninguna.
            frame: imagen del frame (solo se usa si hay detección).
            rostros: cajas de los rostros encontrados en esa persona.
            nombres: nombres del catálogo reconocidos en ese frame.
            ids: identificadores de seguimiento de todas las personas del frame
                que cuentan para este seguidor.
            simultaneas: cuántas personas había en el frame para este seguidor;
                sirve de respaldo para contar cuando no hay seguimiento.

        Returns:
            El evento que se acaba de cerrar, o ``None`` si no se cerró ninguno.
        """
        if caja is not None:
            if self._evento is None:
                self._evento = Evento(
                    archivo=self.archivo, inicio=segundo, fin=segundo, tipo=self.tipo
                )
                self._frame_miniatura = frame.copy()
                self._caja_miniatura = caja
            else:
                self._evento.fin = segundo
            self._ultimo_positivo = segundo
            self._nombres |= nombres
            if ids:
                self._ids |= ids
            self._max_simultaneas = max(self._max_simultaneas, simultaneas)

            # Se conserva el frame con más rostros: es el mejor para los recortes.
            if len(rostros) > self._evento.rostros:
                self._evento.rostros = len(rostros)
                self._frame_rostros = frame.copy()
                self._cajas_rostros = rostros
            return None

        if self._evento is not None and segundo - self._ultimo_positivo > self.tolerancia:
            return self.cerrar()
        return None

    def cerrar(self) -> Optional[EventoCerrado]:
        """Cierra el evento en curso, si lo hay, y reinicia el seguidor."""
        if self._evento is None:
            return None

        self._evento.personas = ", ".join(sorted(self._nombres))
        # Con seguimiento, las pistas distintas; sin él, o si el seguimiento
        # perdió a alguien, al menos las que llegaron a verse a la vez.
        self._evento.n_personas = max(len(self._ids), self._max_simultaneas)
        cerrado = EventoCerrado(
            evento=self._evento,
            frame_miniatura=self._frame_miniatura,
            caja_miniatura=self._caja_miniatura,
            frame_rostros=self._frame_rostros,
            cajas_rostros=self._cajas_rostros,
            ids=self._ids,
        )
        self._evento = None
        self._frame_miniatura = None
        self._caja_miniatura = None
        self._frame_rostros = None
        self._cajas_rostros = []
        self._nombres = set()
        self._ids = set()
        self._max_simultaneas = 0
        return cerrado
