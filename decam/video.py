"""Lectura de videos: búsqueda de archivos, metadatos, saltos y lectura en hilo."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

import cv2

# Extensiones de video soportadas.
EXTENSIONES_VIDEO: tuple[str, ...] = (".mp4", ".avi", ".mkv")


def encontrar_videos(carpeta: str | Path, recursivo: bool = True) -> list[Path]:
    """Devuelve los videos soportados dentro de ``carpeta``, ordenados por nombre.

    Args:
        carpeta: carpeta donde buscar.
        recursivo: si es ``True`` (por defecto) también busca en las subcarpetas,
            como en las exportaciones de NVR que crean una carpeta por descarga.
    """
    ruta = Path(carpeta)
    if not ruta.is_dir():
        return []
    candidatos = ruta.rglob("*") if recursivo else ruta.iterdir()
    videos = [
        archivo
        for archivo in candidatos
        if archivo.is_file() and archivo.suffix.lower() in EXTENSIONES_VIDEO
    ]
    return sorted(videos, key=lambda p: str(p).lower())


def abrir_captura(video: str | Path, hardware: bool = False) -> cv2.VideoCapture:
    """Abre un video, opcionalmente con decodificación acelerada por hardware.

    La aceleración (Quick Sync, D3D11) descomprime ~2.2x más rápido, pero el
    decodificador del driver aplica una conversión de color ligeramente distinta
    a la del decodificador por software: la imagen sale con un sesgo uniforme de
    unos 8 niveles de gris. Las cajas apenas se mueven (±2 px), pero las
    detecciones que rozaban el umbral de confianza pueden caerse. Por eso viene
    desactivada por defecto.
    """
    if not hardware:
        return cv2.VideoCapture(str(video))
    return cv2.VideoCapture(
        str(video),
        cv2.CAP_FFMPEG,
        [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY],
    )


def primer_frame(video: str | Path):
    """Lee el primer frame de un video.

    Returns:
        El frame en formato BGR (``numpy.ndarray``) o ``None`` si no se pudo leer.
    """
    captura = cv2.VideoCapture(str(video))
    try:
        if not captura.isOpened():
            return None
        ok, frame = captura.read()
        return frame if ok else None
    finally:
        captura.release()


@dataclass
class InfoVideo:
    """Metadatos básicos de un video."""

    fps: float
    total_frames: int
    ancho: int
    alto: int

    @property
    def duracion(self) -> float:
        """Duración aproximada en segundos."""
        return self.total_frames / self.fps if self.fps > 0 else 0.0


def info_video(captura: cv2.VideoCapture) -> InfoVideo:
    """Extrae los metadatos de una captura ya abierta.

    Si el contenedor no informa los FPS se asume 25, valor habitual en NVR.
    """
    fps = captura.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0:
        fps = 25.0
    return InfoVideo(
        fps=fps,
        total_frames=int(captura.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        ancho=int(captura.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        alto=int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
    )


#: Frames que se decodifican antes del objetivo para que el códec se
#: resincronice tras un salto. En H.265/HEVC, leer justo después de un
#: ``CAP_PROP_POS_FRAMES`` devuelve imágenes rotas: solo se decodifica la zona
#: en movimiento sobre un fondo gris.
#:
#: El margen necesario depende del GOP del archivo. Medido contra decodificación
#: secuencial (la referencia fiable) sobre grabaciones Hikvision 1080p H.265:
#: con margen 5 el 27 % de los píxeles sigue mal y con margen 15 aún el 9 %;
#: a partir de 30 la imagen es correcta y no mejora subiendo más. Se deja en 30.
MARGEN_RESINCRONIZACION = 30


def leer_frame(
    captura: cv2.VideoCapture,
    indice: int,
    margen: int = MARGEN_RESINCRONIZACION,
):
    """Salta al frame ``indice`` de una captura abierta y lo devuelve.

    Para evitar frames corruptos se salta ``margen`` frames antes del objetivo y
    se decodifica hacia delante hasta llegar a él.

    Returns:
        El frame en BGR, o ``None`` si no se pudo leer.
    """
    indice = max(0, indice)
    inicio = max(0, indice - margen)
    captura.set(cv2.CAP_PROP_POS_FRAMES, inicio)
    frame = None
    for _ in range(indice - inicio + 1):
        ok, leido = captura.read()
        if not ok:
            return frame
        frame = leido
    return frame


#: Frames muestreados que caben en la cola entre el lector y el analizador.
#: Cada uno ocupa ~6 MB en 1080p, así que 8 son unos 50 MB de margen.
TAM_COLA_LECTURA = 8


@dataclass
class FrameMuestreado:
    """Un frame que toca analizar, con su posición en el video."""

    indice: int
    segundo: float
    imagen: Any


class LectorFrames:
    """Decodifica un video en un hilo aparte y entrega los frames a analizar.

    Decodificar e inferir son trabajos distintos —CPU el primero, acelerador el
    segundo—, así que hacerlos en serie desaprovecha ambos. Este lector los
    solapa: mientras el hilo principal infiere sobre un frame, el lector ya está
    decodificando los siguientes. El tiempo total pasa de ser la suma de ambos a
    ser aproximadamente el mayor de los dos.

    La cola es pequeña a propósito: si el analizador se retrasa, el lector se
    frena en lugar de acumular cientos de megas de frames en memoria.

    Se usa como gestor de contexto, que garantiza parar el hilo antes de que el
    llamante libere la captura::

        with LectorFrames(captura, paso, fps, cancelar) as lector:
            for muestra in lector:
                ...
    """

    def __init__(
        self,
        captura: cv2.VideoCapture,
        paso: int,
        fps: float,
        cancelar: threading.Event,
        tam_cola: int = TAM_COLA_LECTURA,
    ) -> None:
        """Prepara el lector.

        Args:
            captura: captura ya abierta; el lector no la cierra.
            paso: se entrega uno de cada ``paso`` frames.
            fps: fotogramas por segundo del video, para calcular el instante.
            cancelar: evento compartido para abortar el análisis.
            tam_cola: frames decodificados por adelantado como máximo.
        """
        self._captura = captura
        self._paso = max(1, paso)
        self._fps = fps if fps > 0 else 25.0
        self._cancelar = cancelar
        self._cola: "queue.Queue[Optional[FrameMuestreado]]" = queue.Queue(
            maxsize=tam_cola
        )
        self._parar = threading.Event()
        self._hilo: Optional[threading.Thread] = None
        self.error: Optional[BaseException] = None

    # ------------------------------------------------------- gestor de contexto

    def __enter__(self) -> "LectorFrames":
        """Arranca el hilo de lectura."""
        self._hilo = threading.Thread(
            target=self._producir, name="lector-frames", daemon=True
        )
        self._hilo.start()
        return self

    def __exit__(self, *_excepcion) -> None:
        """Detiene el hilo de lectura y espera a que termine."""
        self._parar.set()
        # Vaciar la cola desbloquea al lector si estaba esperando hueco.
        while True:
            try:
                self._cola.get_nowait()
            except queue.Empty:
                break
        if self._hilo is not None:
            self._hilo.join(timeout=5.0)
            self._hilo = None

    # ----------------------------------------------------------------- productor

    def _producir(self) -> None:
        """Recorre el video y encola los frames que toca analizar."""
        indice = 0
        try:
            while not self._debe_parar():
                # grab() avanza sin convertir el frame; solo se decodifica del
                # todo (retrieve) el que se va a analizar.
                if not self._captura.grab():
                    break
                if indice % self._paso == 0:
                    ok, imagen = self._captura.retrieve()
                    if not ok:
                        break
                    muestra = FrameMuestreado(
                        indice=indice, segundo=indice / self._fps, imagen=imagen
                    )
                    if not self._encolar(muestra):
                        break
                indice += 1
        except BaseException as exc:  # noqa: BLE001 - se reenvía al consumidor
            self.error = exc
        finally:
            self._encolar(None)  # centinela de fin

    def _debe_parar(self) -> bool:
        """Indica si hay que abandonar la lectura."""
        return self._parar.is_set() or self._cancelar.is_set()

    def _encolar(self, muestra: Optional[FrameMuestreado]) -> bool:
        """Encola un elemento sin bloquearse para siempre si hay que parar.

        Returns:
            ``True`` si se encoló, ``False`` si se abandonó la lectura.
        """
        while not self._debe_parar():
            try:
                self._cola.put(muestra, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    # ---------------------------------------------------------------- consumidor

    def __iter__(self) -> Iterator[FrameMuestreado]:
        """Entrega los frames muestreados en orden, hasta agotar el video."""
        while True:
            try:
                muestra = self._cola.get(timeout=0.5)
            except queue.Empty:
                # Sin datos y sin lector vivo: no va a llegar nada más.
                if self._hilo is None or not self._hilo.is_alive():
                    return
                continue
            if muestra is None:
                return
            yield muestra
