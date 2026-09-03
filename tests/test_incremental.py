"""Análisis incremental: manifiesto, serialización y reutilización de resultados."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict

import pytest

from decam import manifiesto
from decam.analizador import AnalisisCancelado, AnalizadorPuerta
from decam.configuracion import ConfiguracionAnalisis
from decam.deteccion import DeteccionCruda
from decam.eventos import TIPO_ZONA, Evento, ResultadoVideo
from decam.manifiesto import Manifiesto, huella_config
from decam.salida import EscritorSalida


def config(**kw) -> ConfiguracionAnalisis:
    base = dict(zona_puerta=(0, 0, 10, 10), acelerador="cpu", filtro_movimiento=False)
    base.update(kw)
    return ConfiguracionAnalisis(**base)


class DetectorNulo:
    """Un detector que nunca ve a nadie; basta para probar la orquestación."""

    descripcion = "detector nulo"

    def preparar(self) -> None:
        pass

    def detectar(self, frame) -> list[DeteccionCruda]:
        return []


class TestSerializacion:
    def test_evento_ida_y_vuelta(self):
        ev = Evento("a.mp4", 1.0, 2.5, miniatura="x.jpg", rostros=1,
                    personas="ana", n_personas=2, direccion="1 entra, 1 sale")
        assert Evento.desde_dict(json.loads(json.dumps(ev.a_dict()))) == ev

    def test_evento_ignora_claves_desconocidas(self):
        ev = Evento.desde_dict({"archivo": "a", "inicio": 0, "fin": 1, "futuro": 1})
        assert ev.archivo == "a" and not hasattr(ev, "futuro")

    def test_resultado_ida_y_vuelta_marca_reutilizado(self):
        r = ResultadoVideo("a.mp4", frames_analizados=10, frames_omitidos=90)
        r.eventos = [Evento("a.mp4", 0, 1), Evento("a.mp4", 5, 6)]
        copia = ResultadoVideo.desde_dict(json.loads(json.dumps(r.a_dict())))
        assert copia.eventos == r.eventos
        assert copia.frames_omitidos == 90 and copia.error is None
        assert copia.reutilizado and not r.reutilizado
        assert "reutilizado" not in r.a_dict()


class TestHuella:
    def test_estable(self):
        assert huella_config(asdict(config())) == huella_config(asdict(config()))

    def test_cambia_con_lo_que_altera_el_resultado(self):
        base = huella_config(asdict(config()))
        assert huella_config(asdict(config(zona_puerta=(0, 0, 10, 11)))) != base
        assert huella_config(asdict(config(criterio_zona="centro"))) != base
        assert huella_config(asdict(config(fps_analisis=2.0))) != base
        assert huella_config(asdict(config(usar_tracking=False))) != base

    def test_ignora_lo_que_no_altera_el_resultado(self):
        base = huella_config(asdict(config()))
        assert huella_config(asdict(config(acelerador="auto"))) == base
        assert huella_config(asdict(config(decodificacion_hardware=True))) == base


class TestManifiesto:
    @pytest.fixture
    def video(self, tmp_path):
        v = tmp_path / "cam.mp4"
        v.write_bytes(b"0" * 100)
        return v

    def test_vacio_si_no_existe(self, tmp_path):
        m = Manifiesto(tmp_path / "analizados.json")
        assert len(m) == 0
        assert m.buscar(tmp_path / "x.mp4", "h") is None

    def test_registrar_y_buscar(self, tmp_path, video):
        ruta = tmp_path / "analizados.json"
        Manifiesto(ruta).registrar(video, "h1", {"archivo": "cam.mp4", "eventos": []})
        assert ruta.is_file()
        assert not ruta.with_suffix(".json.tmp").exists(), "escritura atómica"
        # Se relee desde disco en otra instancia.
        m = Manifiesto(ruta)
        assert len(m) == 1
        assert m.buscar(video, "h1") == {"archivo": "cam.mp4", "eventos": []}

    def test_no_vale_si_cambia_la_configuracion(self, tmp_path, video):
        m = Manifiesto(tmp_path / "a.json")
        m.registrar(video, "h1", {})
        assert m.buscar(video, "h2") is None

    def test_no_vale_si_cambia_el_fichero(self, tmp_path, video):
        m = Manifiesto(tmp_path / "a.json")
        m.registrar(video, "h1", {})
        video.write_bytes(b"0" * 101)  # otro tamaño
        assert m.buscar(video, "h1") is None

    def test_no_vale_si_el_video_ya_no_existe(self, tmp_path, video):
        m = Manifiesto(tmp_path / "a.json")
        m.registrar(video, "h1", {})
        video.unlink()
        assert m.buscar(video, "h1") is None

    def test_json_corrupto_empieza_vacio(self, tmp_path):
        ruta = tmp_path / "a.json"
        ruta.write_text("{roto", encoding="utf-8")
        assert len(Manifiesto(ruta)) == 0

    def test_otra_version_se_ignora(self, tmp_path, video):
        ruta = tmp_path / "a.json"
        ruta.write_text(
            json.dumps({"version": manifiesto.VERSION + 1, "videos": {"x": {}}}),
            encoding="utf-8",
        )
        assert len(Manifiesto(ruta)) == 0

    def test_la_clave_no_distingue_mayusculas_en_windows(self, tmp_path, video):
        import os

        if os.path.normcase("A") == "A":
            pytest.skip("sistema de archivos sensible a mayúsculas")
        m = Manifiesto(tmp_path / "a.json")
        m.registrar(video, "h1", {"ok": 1})
        otra = video.parent / video.name.upper()
        assert m.buscar(otra, "h1") == {"ok": 1}


class TestAnalizarVideosIncremental:
    """``analizar_videos`` con ``analizar_video`` sustituido por un doble."""

    @pytest.fixture
    def entorno(self, tmp_path, monkeypatch):
        videos = []
        for nombre in ("a.mp4", "b.mp4"):
            v = tmp_path / "videos" / nombre
            v.parent.mkdir(exist_ok=True)
            v.write_bytes(b"0" * 10)
            videos.append(v)
        salida = EscritorSalida(tmp_path / "salida", config().zona, guardar_recortes=False)
        llamadas: list[str] = []
        notificados: list[Evento] = []

        def falso_analizar_video(self, video, salida=None, peso_progreso=(0, 1)):
            llamadas.append(video.name)
            r = ResultadoVideo(video.name, frames_analizados=3)
            r.eventos = [Evento(video.name, 1.0, 2.0, n_personas=1)]
            return r

        monkeypatch.setattr(AnalizadorPuerta, "analizar_video", falso_analizar_video)
        analizador = AnalizadorPuerta(config(), DetectorNulo(), on_evento=notificados.append)
        return analizador, videos, salida, llamadas, notificados

    @staticmethod
    def manif(salida: EscritorSalida) -> Manifiesto:
        return Manifiesto(salida.carpeta / manifiesto.NOMBRE_MANIFIESTO)

    def test_segunda_pasada_reutiliza_todo(self, entorno):
        analizador, videos, salida, llamadas, notificados = entorno

        r1 = analizador.analizar_videos(videos, salida, self.manif(salida))
        assert llamadas == ["a.mp4", "b.mp4"]
        assert not any(r.reutilizado for r in r1)
        assert len(self.manif(salida)) == 2

        notificados.clear()
        r2 = analizador.analizar_videos(videos, salida, self.manif(salida))
        assert llamadas == ["a.mp4", "b.mp4"], "no se analizó nada de nuevo"
        assert all(r.reutilizado for r in r2)
        assert [e.archivo for e in notificados] == ["a.mp4", "b.mp4"], (
            "los eventos reutilizados también llegan a la interfaz"
        )
        # El CSV se reescribe con todo, reutilizado o no.
        with salida.ruta_csv.open(encoding="utf-8-sig", newline="") as f:
            filas = list(csv.DictReader(f))
        assert [f["archivo"] for f in filas] == ["a.mp4", "b.mp4"]
        assert filas[0]["personas_distintas"] == "1"

    def test_solo_se_reanaliza_lo_que_cambio(self, entorno):
        analizador, videos, salida, llamadas, _ = entorno
        analizador.analizar_videos(videos, salida, self.manif(salida))
        videos[1].write_bytes(b"0" * 11)
        r = analizador.analizar_videos(videos, salida, self.manif(salida))
        assert llamadas == ["a.mp4", "b.mp4", "b.mp4"]
        assert [x.reutilizado for x in r] == [True, False]

    def test_cambiar_la_configuracion_invalida_todo(self, entorno):
        analizador, videos, salida, llamadas, _ = entorno
        analizador.analizar_videos(videos, salida, self.manif(salida))
        analizador.config.criterio_zona = "centro"
        analizador.analizar_videos(videos, salida, self.manif(salida))
        assert llamadas == ["a.mp4", "b.mp4", "a.mp4", "b.mp4"]

    def test_sin_manifiesto_se_analiza_todo(self, entorno):
        analizador, videos, salida, llamadas, _ = entorno
        analizador.analizar_videos(videos, salida, self.manif(salida))
        analizador.analizar_videos(videos, salida, None)
        assert len(llamadas) == 4

    def test_sin_salida_no_escribe_nada(self, entorno, tmp_path):
        analizador, videos, _, llamadas, _ = entorno
        r = analizador.analizar_videos(videos, None, None)
        assert len(r) == 2 and llamadas == ["a.mp4", "b.mp4"]
        assert not (tmp_path / "salida" / "eventos.csv").exists()

    def test_los_errores_no_se_apuntan(self, entorno, monkeypatch):
        analizador, videos, salida, llamadas, _ = entorno

        def con_error(self, video, salida=None, peso_progreso=(0, 1)):
            llamadas.append(video.name)
            return ResultadoVideo(video.name, error="no abre")

        monkeypatch.setattr(AnalizadorPuerta, "analizar_video", con_error)
        analizador.analizar_videos(videos, salida, self.manif(salida))
        assert len(self.manif(salida)) == 0
        analizador.analizar_videos(videos, salida, self.manif(salida))
        assert len(llamadas) == 4, "se reintentan"

    def test_cancelar_conserva_lo_ya_analizado(self, entorno, monkeypatch):
        analizador, videos, salida, llamadas, _ = entorno
        original = AnalizadorPuerta.analizar_video

        def cancela_en_el_segundo(self, video, salida=None, peso_progreso=(0, 1)):
            if video.name == "b.mp4":
                raise AnalisisCancelado()
            return original(self, video, salida, peso_progreso)

        monkeypatch.setattr(AnalizadorPuerta, "analizar_video", cancela_en_el_segundo)
        r = analizador.analizar_videos(videos, salida, self.manif(salida))
        assert [x.archivo for x in r] == ["a.mp4"]
        assert len(self.manif(salida)) == 1

        # Al relanzar, a.mp4 se reutiliza y solo falta b.mp4.
        monkeypatch.setattr(AnalizadorPuerta, "analizar_video", original)
        r = analizador.analizar_videos(videos, salida, self.manif(salida))
        assert [x.reutilizado for x in r] == [True, False]
        assert llamadas == ["a.mp4", "b.mp4"]

    def test_los_eventos_reutilizados_son_del_tipo_correcto(self, entorno):
        analizador, videos, salida, _, _ = entorno
        analizador.analizar_videos(videos, salida, self.manif(salida))
        r = analizador.analizar_videos(videos, salida, self.manif(salida))
        assert all(e.tipo == TIPO_ZONA for x in r for e in x.eventos)
