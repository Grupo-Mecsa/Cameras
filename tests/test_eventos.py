"""Agrupación de detecciones en eventos, CSV y utilidades de tiempo/archivos."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from decam.configuracion import ConfiguracionAnalisis
from decam.eventos import (
    COLUMNAS_CSV,
    TIPO_GENERAL,
    TIPO_ZONA,
    Evento,
    ResultadoVideo,
    formatear_tiempo,
)
from decam.salida import escribir_csv
from decam.seguimiento import SeguidorEventos
from decam.video import encontrar_videos

FRAME = np.zeros((8, 8, 3), dtype=np.uint8)
CAJA = (1, 1, 5, 7)


def seguidor(tolerancia: float = 3.0) -> SeguidorEventos:
    return SeguidorEventos("cam.mp4", TIPO_ZONA, tolerancia)


def positivo(seg, segundo, rostros=(), nombres=frozenset()):
    return seg.actualizar(segundo, CAJA, FRAME, list(rostros), set(nombres))


def negativo(seg, segundo):
    return seg.actualizar(segundo, None, FRAME, [], set())


class TestFormatearTiempo:
    @pytest.mark.parametrize(
        "segundos, esperado",
        [
            (0, "00:00:00"),
            (59, "00:00:59"),
            (60, "00:01:00"),
            (3661, "01:01:01"),
            (86399, "23:59:59"),
            (90000, "25:00:00"),  # más de un día: no da la vuelta
        ],
    )
    def test_valores(self, segundos, esperado):
        assert formatear_tiempo(segundos) == esperado

    def test_redondea_en_vez_de_truncar(self):
        assert formatear_tiempo(1.4) == "00:00:01"
        assert formatear_tiempo(1.6) == "00:00:02"


class TestSeguidorEventos:
    def test_sin_detecciones_no_hay_evento(self):
        seg = seguidor()
        for s in range(10):
            assert negativo(seg, s) is None
        assert not seg.activo
        assert seg.cerrar() is None

    def test_deteccion_abre_evento_pero_no_lo_cierra(self):
        seg = seguidor()
        assert positivo(seg, 10.0) is None
        assert seg.activo

    def test_cierra_solo_al_superar_la_tolerancia(self):
        seg = seguidor(tolerancia=3.0)
        positivo(seg, 10.0)
        positivo(seg, 11.0)
        # Huecos de exactamente la tolerancia NO cierran: la comparación es estricta.
        assert negativo(seg, 12.0) is None
        assert negativo(seg, 14.0) is None
        cerrado = negativo(seg, 14.5)
        assert cerrado is not None
        assert cerrado.evento.inicio == 10.0
        # El fin es la última detección, no el instante en que se cerró.
        assert cerrado.evento.fin == 11.0
        assert cerrado.evento.duracion == 1.0
        assert not seg.activo

    def test_hueco_corto_no_parte_el_evento(self):
        seg = seguidor(tolerancia=3.0)
        positivo(seg, 0.0)
        negativo(seg, 1.0)
        negativo(seg, 2.0)
        positivo(seg, 3.0)  # 3 s después: dentro de la tolerancia
        assert negativo(seg, 10.0).evento.fin == 3.0

    def test_dos_eventos_separados(self):
        seg = seguidor(tolerancia=1.0)
        positivo(seg, 0.0)
        primero = negativo(seg, 5.0)
        positivo(seg, 20.0)
        segundo = negativo(seg, 30.0)
        assert (primero.evento.inicio, primero.evento.fin) == (0.0, 0.0)
        assert (segundo.evento.inicio, segundo.evento.fin) == (20.0, 20.0)

    def test_tolerancia_cero_cierra_en_el_primer_hueco(self):
        seg = seguidor(tolerancia=0.0)
        positivo(seg, 0.0)
        assert negativo(seg, 0.5) is not None

    def test_cerrar_al_final_del_video(self):
        seg = seguidor()
        positivo(seg, 100.0)
        positivo(seg, 101.0)
        cerrado = seg.cerrar()
        assert cerrado.evento.fin == 101.0
        assert seg.cerrar() is None  # ya no hay nada abierto

    def test_tipo_y_archivo_se_propagan(self):
        seg = SeguidorEventos("puerta.avi", TIPO_GENERAL, 1.0)
        positivo(seg, 0.0)
        evento = seg.cerrar().evento
        assert evento.archivo == "puerta.avi"
        assert evento.tipo == TIPO_GENERAL

    def test_miniatura_es_copia_del_primer_frame(self):
        seg = seguidor()
        frame = np.full((8, 8, 3), 7, dtype=np.uint8)
        seg.actualizar(0.0, CAJA, frame, [], set())
        frame[:] = 0  # el llamador reutiliza el buffer
        cerrado = seg.cerrar()
        assert cerrado.frame_miniatura[0, 0, 0] == 7
        assert cerrado.caja_miniatura == CAJA

    def test_conserva_el_frame_con_mas_rostros(self):
        seg = seguidor()
        f1 = np.full((8, 8, 3), 1, dtype=np.uint8)
        f2 = np.full((8, 8, 3), 2, dtype=np.uint8)
        f3 = np.full((8, 8, 3), 3, dtype=np.uint8)
        seg.actualizar(0.0, CAJA, f1, [(0, 0, 2, 2)], set())
        seg.actualizar(1.0, CAJA, f2, [(0, 0, 2, 2), (4, 4, 6, 6)], set())
        seg.actualizar(2.0, CAJA, f3, [(0, 0, 2, 2)], set())  # menos: no reemplaza
        cerrado = seg.cerrar()
        assert cerrado.evento.rostros == 2
        assert cerrado.frame_rostros[0, 0, 0] == 2
        assert len(cerrado.cajas_rostros) == 2

    def test_nombres_se_acumulan_y_ordenan(self):
        seg = seguidor()
        positivo(seg, 0.0, nombres={"maria"})
        positivo(seg, 1.0, nombres={"ana", "maria"})
        assert seg.cerrar().evento.personas == "ana, maria"

    def test_el_estado_se_limpia_entre_eventos(self):
        seg = seguidor(tolerancia=0.0)
        positivo(seg, 0.0, rostros=[(0, 0, 1, 1)], nombres={"ana"})
        seg.actualizar(0.5, CAJA, FRAME, [], set(), ids={1, 2}, simultaneas=2)
        negativo(seg, 1.0)
        positivo(seg, 5.0)
        evento = seg.cerrar().evento
        assert evento.rostros == 0
        assert evento.personas == ""
        assert evento.n_personas == 1

    def test_cuenta_personas_distintas_por_id(self):
        seg = seguidor()
        seg.actualizar(0.0, CAJA, FRAME, [], set(), ids={1}, simultaneas=1)
        seg.actualizar(1.0, CAJA, FRAME, [], set(), ids={1, 2}, simultaneas=2)
        seg.actualizar(2.0, CAJA, FRAME, [], set(), ids={3}, simultaneas=1)
        cerrado = seg.cerrar()
        assert cerrado.evento.n_personas == 3
        assert cerrado.ids == {1, 2, 3}

    def test_sin_ids_cuenta_el_maximo_simultaneo(self):
        seg = seguidor()
        seg.actualizar(0.0, CAJA, FRAME, [], set(), ids=None, simultaneas=1)
        seg.actualizar(1.0, CAJA, FRAME, [], set(), ids=set(), simultaneas=2)
        seg.actualizar(2.0, CAJA, FRAME, [], set(), ids=None, simultaneas=1)
        assert seg.cerrar().evento.n_personas == 2

    def test_el_maximo_simultaneo_gana_si_el_seguimiento_perdio_gente(self):
        # Dos personas juntas pero solo una con pista confirmada.
        seg = seguidor()
        seg.actualizar(0.0, CAJA, FRAME, [], set(), ids={1}, simultaneas=2)
        assert seg.cerrar().evento.n_personas == 2

    def test_por_defecto_una_persona(self):
        seg = seguidor()
        positivo(seg, 0.0)
        assert seg.cerrar().evento.n_personas == 1


class TestEvento:
    def test_duracion_nunca_negativa(self):
        assert Evento("a", inicio=10, fin=5).duracion == 0.0

    def test_fila_csv(self):
        ev = Evento("a.mp4", 61.0, 63.5, rostros=2, personas="ana")
        assert ev.a_fila_csv() == {
            "archivo": "a.mp4",
            "tipo": TIPO_ZONA,
            "inicio": "00:01:01",
            "fin": "00:01:04",
            "duracion_segundos": "2.50",
            "rostros": "2",
            "personas": "ana",
            "personas_distintas": "0",
            "direccion": "",
        }

    def test_las_columnas_del_csv_coinciden_con_la_fila(self):
        assert tuple(Evento("a", 0, 1).a_fila_csv()) == COLUMNAS_CSV

    def test_str_incluye_personas_y_direccion_solo_si_aportan(self):
        assert "persona(s)" not in str(Evento("a", 0, 1, n_personas=1))
        con = str(Evento("a", 0, 1, n_personas=3, direccion="2 entran, 1 sale"))
        assert "3 persona(s)" in con and "2 entran, 1 sale" in con

    def test_str_incluye_rostros_y_personas_solo_si_hay(self):
        assert "rostro" not in str(Evento("a", 0, 1))
        con = str(Evento("a", 0, 1, rostros=1, personas="ana"))
        assert "1 rostro(s)" in con and "[ana]" in con


class TestEscribirCsv:
    def test_escribe_todos_los_eventos_con_bom(self, tmp_path):
        r1 = ResultadoVideo(archivo="a.mp4")
        r1.eventos = [Evento("a.mp4", 0, 1), Evento("a.mp4", 5, 9, tipo=TIPO_GENERAL)]
        r2 = ResultadoVideo(archivo="b.mp4")
        r2.eventos = [Evento("b.mp4", 2, 3)]
        r3 = ResultadoVideo(archivo="roto.mp4", error="no abre")

        destino = tmp_path / "sub" / "eventos.csv"
        assert escribir_csv([r1, r2, r3], destino) == destino

        crudo = destino.read_bytes()
        assert crudo.startswith(b"\xef\xbb\xbf"), "Excel necesita el BOM para el UTF-8"
        with destino.open(encoding="utf-8-sig", newline="") as f:
            filas = list(csv.DictReader(f))
        assert [(f["archivo"], f["tipo"]) for f in filas] == [
            ("a.mp4", TIPO_ZONA),
            ("a.mp4", TIPO_GENERAL),
            ("b.mp4", TIPO_ZONA),
        ]


class TestEncontrarVideos:
    def test_filtra_por_extension_sin_distinguir_mayusculas(self, tmp_path):
        for nombre in ("b.MP4", "a.mkv", "c.avi", "notas.txt", "d.mov"):
            (tmp_path / nombre).write_bytes(b"")
        assert [p.name for p in encontrar_videos(tmp_path)] == ["a.mkv", "b.MP4", "c.avi"]

    def test_recursivo_por_defecto(self, tmp_path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "x.mp4").write_bytes(b"")
        (tmp_path / "y.mp4").write_bytes(b"")
        assert len(encontrar_videos(tmp_path)) == 2
        assert [p.name for p in encontrar_videos(tmp_path, recursivo=False)] == ["y.mp4"]

    def test_carpeta_inexistente(self, tmp_path):
        assert encontrar_videos(tmp_path / "nada") == []


class TestConfiguracionValidar:
    def config(self, **kw) -> ConfiguracionAnalisis:
        base = dict(zona_puerta=(0, 0, 10, 10))
        base.update(kw)
        return ConfiguracionAnalisis(**base)

    def test_valida_por_defecto(self):
        self.config().validar()

    def test_zona_como_objeto(self):
        assert self.config().zona.como_tupla == (0, 0, 10, 10)

    @pytest.mark.parametrize(
        "cambio, mensaje",
        [
            (dict(zona_puerta=(0, 0, 0, 10)), "zona"),
            (dict(zona_puerta=(0, 10, 10, 10)), "zona"),
            (dict(fps_analisis=0), "frames por segundo"),
            (dict(tolerancia_segundos=-1), "tolerancia"),
            (dict(criterio_zona="cabeza"), "Criterio"),
            (dict(min_solape=0), "solape"),
            (dict(min_solape=1.01), "solape"),
            (dict(acelerador="tpu"), "Acelerador"),
            (dict(umbral_movimiento=1.0), "umbral de movimiento"),
            (dict(umbral_movimiento=-0.1), "umbral de movimiento"),
        ],
    )
    def test_rechaza_valores_invalidos(self, cambio, mensaje):
        with pytest.raises(ValueError, match=mensaje):
            self.config(**cambio).validar()

    def test_limites_incluidos(self):
        self.config(min_solape=1.0, umbral_movimiento=0.0, tolerancia_segundos=0).validar()
