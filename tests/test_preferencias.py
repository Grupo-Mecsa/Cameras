"""Persistencia de preferencias y parseo de tiempos de la interfaz.

Importar ``decam.ui`` carga Tkinter pero no abre ninguna ventana.
"""

from __future__ import annotations

import json

import pytest

from decam import registro
from decam.ui.preferencias import RUTA_CONFIG, Preferencias
from decam.ui.utilidades import parsear_tiempo


class TestPreferencias:
    def test_ida_y_vuelta(self, tmp_path):
        destino = tmp_path / "config.json"
        prefs = Preferencias(
            carpeta_videos="D:/videos",
            fps_analisis=2.5,
            zona_puerta=[1, 2, 3, 4],
            detectar_rostros=True,
            incremental=False,
        )
        prefs.guardar(destino)
        assert Preferencias.cargar(destino) == prefs

    def test_zona_poligonal_ida_y_vuelta(self, tmp_path):
        destino = tmp_path / "config.json"
        prefs = Preferencias(zona_puerta=[[10, 20], [30, 20], [20, 40]])
        prefs.guardar(destino)
        assert Preferencias.cargar(destino).zona_puerta == [[10, 20], [30, 20], [20, 40]]

    def test_fichero_ausente_da_defaults(self, tmp_path):
        assert Preferencias.cargar(tmp_path / "nada.json") == Preferencias()

    def test_json_corrupto_da_defaults(self, tmp_path):
        malo = tmp_path / "malo.json"
        malo.write_text("{no es json", encoding="utf-8")
        assert Preferencias.cargar(malo) == Preferencias()

    def test_ignora_claves_desconocidas(self, tmp_path):
        # Un config.json de una versión más nueva, o editado a mano.
        raro = tmp_path / "raro.json"
        raro.write_text(json.dumps({"fps_analisis": 3.0, "inventada": 1}), encoding="utf-8")
        leido = Preferencias.cargar(raro)
        assert leido.fps_analisis == 3.0
        assert not hasattr(leido, "inventada")

    def test_acepta_config_incompleto(self, tmp_path):
        # Un config.json de una versión más vieja, sin las claves nuevas.
        viejo = tmp_path / "viejo.json"
        viejo.write_text(json.dumps({"carpeta_videos": "X:/"}), encoding="utf-8")
        leido = Preferencias.cargar(viejo)
        assert leido.carpeta_videos == "X:/"
        assert leido.backend_rostros == "yunet"
        assert leido.usar_tracking is True

    def test_guardar_en_carpeta_inexistente_no_revienta(self, tmp_path):
        Preferencias().guardar(tmp_path / "no" / "existe" / "config.json")

    def test_todo_campo_se_serializa(self, tmp_path):
        destino = tmp_path / "config.json"
        Preferencias().guardar(destino)
        claves = set(json.loads(destino.read_text(encoding="utf-8")))
        assert claves == set(Preferencias.__dataclass_fields__)

    def test_la_ruta_por_defecto_viene_de_registro(self):
        assert RUTA_CONFIG == registro.ruta_config()


class TestParsearTiempo:
    @pytest.mark.parametrize(
        "texto, esperado",
        [
            ("90", 90.0),
            ("1.5", 1.5),
            ("1:30", 90.0),
            ("01:01:01", 3661.0),
            ("  2:00  ", 120.0),
            ("0:0:0", 0.0),
        ],
    )
    def test_formatos_validos(self, texto, esperado):
        assert parsear_tiempo(texto) == esperado

    @pytest.mark.parametrize("texto", ["", "abc", "1:2:3:4", "1::2", "1:xx"])
    def test_formatos_invalidos(self, texto):
        with pytest.raises(ValueError):
            parsear_tiempo(texto)
