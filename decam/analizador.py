"""Orquestación del análisis: recorre los videos y produce los eventos.

:class:`AnalizadorPuerta` no construye nada por su cuenta: recibe el detector,
el rastreador, el filtro de movimiento y el analizador de rostros ya hechos.
:func:`crear_analizador` es quien sabe montarlos a partir de la configuración;
es la única función del módulo que conoce YOLO y ByteTrack.
"""

from __future__ import annotations

import threading
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Optional

from decam import manifiesto as _manifiesto
from decam.aceleradores import resolver_acelerador
from decam.callbacks import CallbackEvento, CallbackLog, CallbackProgreso
from decam.configuracion import ConfiguracionAnalisis
from decam.deteccion import Deteccion, DetectorPersonas, DetectorYOLO
from decam.eventos import TIPO_GENERAL, TIPO_ZONA, ResultadoVideo
from decam.manifiesto import Manifiesto
from decam.movimiento import DetectorMovimiento
from decam.rostros import AnalizadorRostros, DetectorRostros, IdentificadorRostros
from decam.salida import EscritorSalida
from decam.seguimiento import (
    EventoCerrado,
    Rastreador,
    RastreadorPersonas,
    SeguidorEventos,
    Trayectorias,
)
from decam.video import LectorFrames, abrir_captura, info_video
from decam.zona import Zona, persona_en_zona


class AnalisisCancelado(Exception):
    """Se lanza internamente cuando el usuario cancela el análisis."""


class AnalizadorPuerta:
    """Analiza videos buscando personas dentro de una zona (la puerta)."""

    def __init__(
        self,
        config: ConfiguracionAnalisis,
        detector: DetectorPersonas,
        *,
        rastreador: Optional[Rastreador] = None,
        movimiento: Optional[DetectorMovimiento] = None,
        rostros: Optional[AnalizadorRostros] = None,
        cancelar: Optional[threading.Event] = None,
        on_progreso: Optional[CallbackProgreso] = None,
        on_log: Optional[CallbackLog] = None,
        on_evento: Optional[CallbackEvento] = None,
    ) -> None:
        """Crea el analizador con sus colaboradores ya construidos.

        Args:
            config: parámetros del análisis (zona, fps, tolerancia, criterio).
            detector: detector de personas.
            rastreador: seguimiento entre frames; ``None`` desactiva el conteo
                de personas distintas y la dirección.
            movimiento: filtro de movimiento; ``None`` analiza todos los frames.
            rostros: detección/identificación de rostros; ``None`` la omite.
            cancelar: evento que, al activarse, detiene el análisis.
            on_progreso: recibe ``(nombre_video, porcentaje_video, porcentaje_total)``.
            on_log: recibe mensajes de texto para mostrar en el log.
            on_evento: recibe cada evento en cuanto se cierra.
        """
        config.validar()
        self.config = config
        self.detector = detector
        self.rastreador = rastreador
        self.movimiento = movimiento
        self.rostros = rostros
        self.cancelar = cancelar or threading.Event()
        self._on_progreso = on_progreso
        self._on_log = on_log
        self._on_evento = on_evento

    # ------------------------------------------------------------------ utilidades

    def _log(self, mensaje: str) -> None:
        if self._on_log is not None:
            self._on_log(mensaje)

    def _progreso(self, nombre: str, pct_video: float, pct_total: float) -> None:
        if self._on_progreso is not None:
            self._on_progreso(nombre, pct_video, pct_total)

    def _comprobar_cancelacion(self) -> None:
        """Lanza :class:`AnalisisCancelado` si el usuario pidió cancelar."""
        if self.cancelar.is_set():
            raise AnalisisCancelado()

    # -------------------------------------------------------------------- análisis

    def analizar_video(
        self,
        video: str | Path,
        salida: Optional[EscritorSalida] = None,
        peso_progreso: tuple[float, float] = (0.0, 1.0),
    ) -> ResultadoVideo:
        """Analiza un video y devuelve los eventos detectados.

        Se llevan dos registros en paralelo: las personas que cumplen el criterio
        de la zona (``TIPO_ZONA``) y, si está activado, todas las personas vistas
        en cualquier punto del frame (``TIPO_GENERAL``).

        Args:
            video: ruta del archivo de video.
            salida: dónde guardar miniaturas y recortes; ``None`` no escribe nada.
            peso_progreso: ``(inicio, tamaño)`` del tramo de progreso global que
                ocupa este video, ambos entre 0 y 1.

        Returns:
            Un :class:`ResultadoVideo`; si el video no se pudo abrir, ``error``
            contiene el motivo y ``eventos`` queda vacío.
        """
        ruta = Path(video)
        resultado = ResultadoVideo(archivo=ruta.name)
        captura = abrir_captura(ruta, self.config.decodificacion_hardware)

        if not captura.isOpened():
            captura.release()
            resultado.error = f"No se pudo abrir el video: {ruta.name}"
            return resultado

        try:
            info = info_video(captura)
            if self.movimiento is not None:
                self.movimiento.reiniciar()
            if self.rastreador is not None:
                self.rastreador.reiniciar()
            trayectorias = Trayectorias()
            paso = max(1, int(round(info.fps / self.config.fps_analisis)))
            self.detector.preparar()
            zona = self.config.zona
            base_progreso, ancho_progreso = peso_progreso

            seguidores = [
                SeguidorEventos(ruta.name, TIPO_ZONA, self.config.tolerancia_segundos)
            ]
            if self.config.registrar_general:
                seguidores.append(
                    SeguidorEventos(
                        ruta.name, TIPO_GENERAL, self.config.tolerancia_segundos
                    )
                )

            # La decodificación va en un hilo aparte y se solapa con la
            # inferencia, que es lo que consume el tiempo aquí.
            with LectorFrames(captura, paso, info.fps, self.cancelar) as lector:
                for muestra in lector:
                    self._comprobar_cancelacion()
                    self._procesar_frame(
                        muestra.imagen, muestra.segundo, zona,
                        seguidores, trayectorias, resultado, salida,
                    )

                    pct_video = (
                        min(100.0, muestra.indice / info.total_frames * 100.0)
                        if info.total_frames > 0
                        else 0.0
                    )
                    pct_total = (
                        base_progreso + ancho_progreso * pct_video / 100.0
                    ) * 100.0
                    self._progreso(ruta.name, pct_video, pct_total)

            if lector.error is not None:
                self._log(f"  Aviso al leer {ruta.name}: {lector.error}")

            # Si se canceló mientras el lector se vaciaba, el bucle termina sin
            # lanzar: hay que comprobarlo aquí para no dar el video por terminado
            # (y apuntarlo en el manifiesto) a medias.
            self._comprobar_cancelacion()

            # Los eventos que siguen abiertos al acabar el video se cierran igual.
            for seguidor in seguidores:
                self._registrar(seguidor.cerrar(), trayectorias, resultado, salida)

            self._progreso(ruta.name, 100.0, (base_progreso + ancho_progreso) * 100.0)
            return resultado
        finally:
            captura.release()

    def _procesar_frame(
        self,
        frame,
        segundo: float,
        zona: Zona,
        seguidores: list[SeguidorEventos],
        trayectorias: Trayectorias,
        resultado: ResultadoVideo,
        salida: Optional[EscritorSalida],
    ) -> None:
        """Detecta personas en un frame y alimenta cada seguidor de eventos.

        Si el filtro de movimiento está activo y el frame no ha cambiado, se
        salta la inferencia: es lo que evita ejecutar el modelo sobre horas de
        pasillo vacío. Nunca se salta mientras haya un evento abierto, para no
        perder a alguien que se queda quieto delante de la puerta.
        """
        if self._sin_movimiento(frame, seguidores):
            resultado.frames_omitidos += 1
            for seguidor in seguidores:
                cerrado = seguidor.actualizar(segundo, None, None, [], set())
                self._registrar(cerrado, trayectorias, resultado, salida)
            return

        resultado.frames_analizados += 1
        personas = self._detectar(frame)
        en_zona: list[Deteccion] = []
        for deteccion in personas:
            dentro = persona_en_zona(
                deteccion.caja, zona, self.config.criterio_zona, self.config.min_solape
            )
            if dentro:
                en_zona.append(deteccion)
            if deteccion.id is not None:
                trayectorias.observar(deteccion.id, dentro)

        for seguidor in seguidores:
            candidatas = en_zona if seguidor.tipo == TIPO_ZONA else personas
            if not candidatas:
                cerrado = seguidor.actualizar(segundo, None, None, [], set())
            else:
                # La persona más grande es la más cercana a la cámara y la que
                # mejor se ve en la miniatura.
                principal = max(candidatas, key=lambda d: d.area)
                rostros, nombres = (
                    self.rostros.analizar(frame, principal.caja)
                    if self.rostros is not None
                    else ([], set())
                )
                ids = {d.id for d in candidatas if d.id is not None}
                cerrado = seguidor.actualizar(
                    segundo, principal.caja, frame, rostros, nombres,
                    ids=ids, simultaneas=len(candidatas),
                )
            self._registrar(cerrado, trayectorias, resultado, salida)

    def _detectar(self, frame) -> list[Deteccion]:
        """Detecta las personas del frame y les asigna pista si hay rastreador."""
        crudas = self.detector.detectar(frame)
        ids = (
            self.rastreador.asignar_ids(crudas)
            if self.rastreador is not None
            else [None] * len(crudas)
        )
        return [Deteccion(cruda.caja, id_pista) for cruda, id_pista in zip(crudas, ids)]

    def _sin_movimiento(self, frame, seguidores: list[SeguidorEventos]) -> bool:
        """Indica si el frame puede saltarse por no haber cambiado nada.

        Mientras algún seguidor tenga un evento abierto se devuelve ``False``
        aunque la imagen esté quieta: una persona parada apenas genera cambios,
        y cortarle el evento la partiría en dos o la perdería.
        """
        if self.movimiento is None:
            return False
        cambio = self.movimiento.fraccion_cambiada(frame)
        if any(seguidor.activo for seguidor in seguidores):
            return False
        return cambio < self.config.umbral_movimiento

    def _registrar(
        self,
        cerrado: Optional[EventoCerrado],
        trayectorias: Trayectorias,
        resultado: ResultadoVideo,
        salida: Optional[EscritorSalida],
    ) -> None:
        """Completa un evento recién cerrado, lo guarda y lo notifica."""
        if cerrado is None:
            return
        evento = cerrado.evento
        if cerrado.ids:
            evento.direccion = trayectorias.resumen(cerrado.ids)
        if salida is not None:
            salida.guardar_evento(cerrado)

        resultado.eventos.append(evento)
        self._log(f"  Evento: {evento}")
        if self._on_evento is not None:
            self._on_evento(evento)

    def analizar_videos(
        self,
        videos: Iterable[str | Path],
        salida: Optional[EscritorSalida] = None,
        manifiesto: Optional[Manifiesto] = None,
    ) -> list[ResultadoVideo]:
        """Analiza varios videos y escribe ``eventos.csv`` y las miniaturas.

        Los videos que no se puedan abrir se saltan: su ``ResultadoVideo`` lleva
        el mensaje en ``error`` y el análisis continúa con el siguiente.

        Args:
            videos: rutas de los videos a analizar.
            salida: dónde escribir miniaturas, recortes y CSV; ``None`` no
                escribe nada.
            manifiesto: si se da, los videos que ya figuren en él con esta misma
                configuración y sin cambios se reutilizan sin analizar, y cada
                video analizado se apunta en el acto (una cancelación no pierde
                lo hecho hasta entonces).

        Returns:
            La lista de resultados, uno por video (analizado o reutilizado).
        """
        lista = [Path(v) for v in videos]
        huella = _manifiesto.huella_config(asdict(self.config))

        self._log(f"Detector: {self.detector.descripcion}")
        self._log(f"Videos a analizar: {len(lista)}")

        resultados: list[ResultadoVideo] = []
        reutilizados = 0
        try:
            for i, video in enumerate(lista):
                self._comprobar_cancelacion()
                peso = (i / len(lista), 1 / len(lista))

                previo = manifiesto.buscar(video, huella) if manifiesto else None
                if previo is not None:
                    resultado = ResultadoVideo.desde_dict(previo)
                    reutilizados += 1
                    self._log(
                        f"[{i + 1}/{len(lista)}] {video.name} — ya analizado, "
                        f"se reutiliza ({len(resultado.eventos)} eventos)"
                    )
                    for evento in resultado.eventos:
                        if self._on_evento is not None:
                            self._on_evento(evento)
                    self._progreso(video.name, 100.0, (peso[0] + peso[1]) * 100.0)
                    resultados.append(resultado)
                    continue

                self._log(f"[{i + 1}/{len(lista)}] {video.name}")
                resultado = self.analizar_video(video, salida, peso)
                if resultado.error:
                    self._log(f"  ERROR: {resultado.error}")
                elif resultado.frames_omitidos:
                    self._log(
                        f"  {resultado.frames_analizados} frames analizados, "
                        f"{resultado.frames_omitidos} omitidos sin movimiento "
                        f"({resultado.ahorro_movimiento:.0%} de ahorro)"
                    )
                resultados.append(resultado)
                # Los videos con error no se apuntan: la próxima vez se reintentan.
                if manifiesto is not None and not resultado.error:
                    manifiesto.registrar(video, huella, resultado.a_dict())
        except AnalisisCancelado:
            self._log("Análisis cancelado por el usuario.")

        if reutilizados:
            self._log(
                f"Reutilizados {reutilizados} de {len(lista)} videos ya analizados."
            )
        if salida is not None:
            salida.escribir_csv(resultados)
        return resultados


def crear_analizador(
    config: ConfiguracionAnalisis,
    cancelar: Optional[threading.Event] = None,
    on_progreso: Optional[CallbackProgreso] = None,
    on_log: Optional[CallbackLog] = None,
    on_evento: Optional[CallbackEvento] = None,
) -> AnalizadorPuerta:
    """Monta un :class:`AnalizadorPuerta` con las implementaciones reales.

    Aquí vive toda la lógica de "según la configuración, construye esto o
    aquello": resolver el acelerador, YOLO, ByteTrack, el filtro de movimiento
    y los rostros. Un fallo en una función opcional (rostros, seguimiento) no
    impide el análisis de personas: se avisa por el log y se sigue sin ella.
    """
    config.validar()
    log = on_log or (lambda _m: None)

    acelerador = resolver_acelerador(config.acelerador)
    if config.acelerador not in ("auto", acelerador):
        log(
            f"El acelerador '{config.acelerador}' no está disponible; "
            f"se usará '{acelerador}'."
        )
    detector = DetectorYOLO(config.modelo, config.confianza, acelerador, on_log)

    movimiento = DetectorMovimiento() if config.filtro_movimiento else None

    rastreador: Optional[Rastreador] = None
    if config.usar_tracking:
        try:
            rastreador = RastreadorPersonas(config.fps_analisis, config.tolerancia_segundos)
        except ImportError as exc:
            log(f"Seguimiento de personas desactivado: {exc}")

    rostros: Optional[AnalizadorRostros] = None
    if config.detectar_rostros:
        try:
            detector_rostros = DetectorRostros(config.backend_rostros)
        except (FileNotFoundError, ValueError) as exc:
            log(f"Detección de rostros desactivada: {exc}")
        else:
            identificador = None
            if config.identificar_rostros:
                try:
                    log("Cargando catálogo de personas conocidas...")
                    identificador = IdentificadorRostros(
                        carpeta_personas=config.carpeta_personas,
                        detector=detector_rostros,
                        umbral=config.umbral_identificacion,
                        on_log=on_log,
                    )
                    log(f"Catálogo listo: {len(identificador.catalogo)} persona(s).")
                except (FileNotFoundError, ValueError) as exc:
                    log(f"Identificación desactivada: {exc}")
            rostros = AnalizadorRostros(detector_rostros, identificador)

    return AnalizadorPuerta(
        config,
        detector,
        rastreador=rastreador,
        movimiento=movimiento,
        rostros=rostros,
        cancelar=cancelar,
        on_progreso=on_progreso,
        on_log=on_log,
        on_evento=on_evento,
    )
