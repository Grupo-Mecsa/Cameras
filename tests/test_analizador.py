"""``analizar_video`` completo con un detector guionizado y sin YOLO.

Esto es lo que permite la inyección de dependencias: probar el recorrido
entero —lectura del video, muestreo, criterio de zona, seguimiento, eventos,
dirección, miniaturas— en menos de un segundo y sin ningún modelo.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
import pytest

from decam.analizador import AnalisisCancelado, AnalizadorPuerta
from decam.configuracion import ConfiguracionAnalisis
from decam.deteccion import DeteccionCruda
from decam.eventos import TIPO_GENERAL, TIPO_ZONA
from decam.salida import EscritorSalida
from decam.seguimiento import RastreadorPersonas

ANCHO, ALTO, FPS, FRAMES = 320, 240, 10.0, 60
ZONA = (200, 100, 300, 200)


@pytest.fixture(scope="module")
def video(tmp_path_factory) -> Path:
    """Seis segundos de gris a 10 fps; el contenido no importa, lo ve el doble."""
    ruta = tmp_path_factory.mktemp("videos") / "pasillo.mp4"
    escritor = cv2.VideoWriter(str(ruta), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (ANCHO, ALTO))
    for i in range(FRAMES):
        escritor.write(np.full((ALTO, ANCHO, 3), 40 + i, dtype=np.uint8))
    escritor.release()
    assert ruta.stat().st_size > 0
    return ruta


class DetectorGuionizado:
    """Devuelve, por número de llamada, lo que "ve" en cada frame muestreado.

    A 2 fps sobre un video de 10 fps se muestrea un frame cada 0,5 s. Una
    persona de 60 px de ancho camina desde x=100 hacia la zona en pasos de 25
    px: sus pies (x+30) entran en la zona (x ≥ 200) en la llamada 6.
    """

    descripcion = "detector guionizado"

    def __init__(self) -> None:
        self.llamadas = 0
        self.preparado = False

    def preparar(self) -> None:
        self.preparado = True

    def detectar(self, frame) -> list[DeteccionCruda]:
        k = self.llamadas
        self.llamadas += 1
        if 3 <= k <= 8:
            x = 100 + 25 * (k - 3)
            return [DeteccionCruda((x, 100, x + 60, 200), 0.9)]
        return []


class RastreadorFijo:
    """Otra implementación del protocolo: todo el mundo es la pista 7."""

    def __init__(self) -> None:
        self.reinicios = 0

    def reiniciar(self) -> None:
        self.reinicios += 1

    def asignar_ids(self, detecciones: Sequence[DeteccionCruda]) -> list[Optional[int]]:
        return [7] * len(detecciones)


def config(**kw) -> ConfiguracionAnalisis:
    base = dict(
        zona_puerta=ZONA, fps_analisis=2.0, tolerancia_segundos=1.0,
        acelerador="cpu", filtro_movimiento=False,
    )
    base.update(kw)
    return ConfiguracionAnalisis(**base)


class TestAnalizarVideo:
    def test_recorrido_completo_con_bytetrack(self, video):
        detector = DetectorGuionizado()
        analizador = AnalizadorPuerta(
            config(), detector, rastreador=RastreadorPersonas(2.0, 1.0),
        )
        resultado = analizador.analizar_video(video)

        assert resultado.error is None
        assert detector.preparado
        assert resultado.frames_analizados == detector.llamadas >= 12
        assert resultado.frames_omitidos == 0

        por_tipo = {e.tipo: e for e in resultado.eventos}
        assert set(por_tipo) == {TIPO_ZONA, TIPO_GENERAL}

        zona = por_tipo[TIPO_ZONA]
        assert (zona.inicio, zona.fin) == (3.0, 4.0)  # llamadas 6..8
        assert zona.n_personas == 1
        assert zona.direccion == "1 entra"

        general = por_tipo[TIPO_GENERAL]
        assert (general.inicio, general.fin) == (1.5, 4.0)  # llamadas 3..8
        assert general.direccion == "1 entra"

    def test_funciona_con_cualquier_rastreador(self, video):
        rastreador = RastreadorFijo()
        analizador = AnalizadorPuerta(config(), DetectorGuionizado(), rastreador=rastreador)
        resultado = analizador.analizar_video(video)
        assert rastreador.reinicios == 1
        zona = next(e for e in resultado.eventos if e.tipo == TIPO_ZONA)
        assert zona.n_personas == 1
        assert zona.direccion == "1 entra"

    def test_sin_rastreador_no_hay_direccion_pero_si_conteo(self, video):
        analizador = AnalizadorPuerta(config(), DetectorGuionizado())
        resultado = analizador.analizar_video(video)
        zona = next(e for e in resultado.eventos if e.tipo == TIPO_ZONA)
        assert zona.direccion == ""
        assert zona.n_personas == 1  # máximo simultáneo

    def test_sin_registrar_general_solo_zona(self, video):
        analizador = AnalizadorPuerta(config(registrar_general=False), DetectorGuionizado())
        resultado = analizador.analizar_video(video)
        assert [e.tipo for e in resultado.eventos] == [TIPO_ZONA]

    def test_el_criterio_de_zona_se_aplica(self, video):
        # Con "pies" la persona entra en la llamada 6 (x=175, pies en 205).
        # Con "solape" al 90 % hace falta que casi toda la caja esté dentro, y
        # eso no pasa hasta la llamada 7 (x=200): el evento empieza 0,5 s después.
        analizador = AnalizadorPuerta(
            config(criterio_zona="solape", min_solape=0.9), DetectorGuionizado()
        )
        resultado = analizador.analizar_video(video)
        zona = next(e for e in resultado.eventos if e.tipo == TIPO_ZONA)
        assert (zona.inicio, zona.fin) == (3.5, 4.0)

    def test_escribe_miniaturas_si_hay_salida(self, video, tmp_path):
        cfg = config()
        salida = EscritorSalida(tmp_path / "out", cfg.zona, guardar_recortes=False)
        resultado = AnalizadorPuerta(cfg, DetectorGuionizado()).analizar_video(video, salida)
        zona = next(e for e in resultado.eventos if e.tipo == TIPO_ZONA)
        assert Path(zona.miniatura).is_file()
        assert Path(zona.miniatura).parent == salida.carpeta_miniaturas / TIPO_ZONA
        assert zona.miniatura.endswith("pasillo_00-00-03.jpg")

    def test_progreso_y_eventos_llegan_por_callbacks(self, video):
        progresos, eventos, logs = [], [], []
        analizador = AnalizadorPuerta(
            config(), DetectorGuionizado(),
            on_progreso=lambda n, pv, pt: progresos.append((n, pv, pt)),
            on_evento=eventos.append, on_log=logs.append,
        )
        analizador.analizar_video(video, peso_progreso=(0.5, 0.5))
        assert progresos[-1] == ("pasillo.mp4", 100.0, 100.0)
        assert all(50.0 <= pt <= 100.0 for _, _, pt in progresos)
        assert len(eventos) == 2
        assert any("Evento:" in l for l in logs)

    def test_cancelacion_a_mitad_de_video(self, video):
        cancelar = threading.Event()
        detector = DetectorGuionizado()
        analizador = AnalizadorPuerta(
            config(), detector, cancelar=cancelar,
            # Se cancela tras el primer frame procesado: el siguiente ya no pasa.
            on_progreso=lambda *_: cancelar.set(),
        )
        with pytest.raises(AnalisisCancelado):
            analizador.analizar_video(video)
        assert detector.llamadas == 1

    def test_video_ilegible_devuelve_error(self, tmp_path):
        roto = tmp_path / "roto.mp4"
        roto.write_bytes(b"no soy un video")
        resultado = AnalizadorPuerta(config(), DetectorGuionizado()).analizar_video(roto)
        assert resultado.error and "roto.mp4" in resultado.error
        assert resultado.eventos == []

    def test_analizar_videos_encadena_y_escribe_csv(self, video, tmp_path):
        cfg = config()
        salida = EscritorSalida(tmp_path / "out", cfg.zona, guardar_recortes=False)
        resultados = AnalizadorPuerta(cfg, DetectorGuionizado()).analizar_videos([video], salida)
        assert len(resultados) == 1
        assert salida.ruta_csv.is_file()
        assert "1 entra" not in salida.ruta_csv.read_text(encoding="utf-8-sig"), (
            "sin rastreador no hay dirección"
        )
