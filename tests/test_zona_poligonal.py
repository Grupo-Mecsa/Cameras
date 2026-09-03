"""Zona poligonal: geometría, especificación persistible y uso en el análisis."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest

from decam.analizador import AnalizadorPuerta
from decam.configuracion import ConfiguracionAnalisis
from decam.eventos import TIPO_ZONA
from decam.manifiesto import huella_config
from decam.zona import (
    ZonaPoligonal,
    ZonaRectangular,
    es_poligono,
    espec_a_lista,
    normalizar_espec,
    persona_en_zona,
    zona_desde,
)
from tests.test_analizador import DetectorGuionizado, video  # noqa: F401 - fixture

# Triángulo rectángulo con el ángulo recto en el origen.
TRIANGULO = ZonaPoligonal(((0, 0), (100, 0), (0, 100)))
# Un trapecio como el que dibuja una puerta vista en perspectiva.
TRAPECIO = ZonaPoligonal.desde([(40, 0), (60, 0), (100, 100), (0, 100)])


class TestGeometria:
    def test_contiene_punto(self):
        assert TRIANGULO.contiene_punto((10, 10))
        assert TRIANGULO.contiene_punto((50, 50))  # sobre la hipotenusa
        assert TRIANGULO.contiene_punto((0, 0))  # vértice
        assert not TRIANGULO.contiene_punto((60, 60))
        assert not TRIANGULO.contiene_punto((-1, 10))

    def test_fraccion_de_caja_que_cubre_el_triangulo(self):
        # El triángulo es la mitad de la caja (0,0)-(100,100).
        assert TRIANGULO.fraccion_de((0, 0, 100, 100)) == pytest.approx(0.5, abs=0.02)

    def test_fraccion_de_caja_dentro_y_fuera(self):
        assert TRIANGULO.fraccion_de((5, 5, 25, 25)) == pytest.approx(1.0, abs=0.02)
        assert TRIANGULO.fraccion_de((70, 70, 90, 90)) == 0.0
        assert TRIANGULO.fraccion_de((200, 200, 300, 300)) == 0.0

    def test_fraccion_de_caja_degenerada(self):
        assert TRIANGULO.fraccion_de((10, 10, 10, 50)) == 0.0

    def test_trapecio_estrecho_arriba_ancho_abajo(self):
        # Arriba (y=10) solo hay zona entre x≈42 y x≈58; abajo (y=90) casi todo.
        assert not TRAPECIO.contiene_punto((20, 10))
        assert TRAPECIO.contiene_punto((50, 10))
        assert TRAPECIO.contiene_punto((20, 90))

    def test_caja_envolvente(self):
        assert TRAPECIO.caja_envolvente == (0, 0, 100, 100)

    def test_validar(self):
        TRIANGULO.validar()
        with pytest.raises(ValueError, match="tres puntos"):
            ZonaPoligonal(((0, 0), (10, 10))).validar()
        with pytest.raises(ValueError, match="alineados"):
            ZonaPoligonal(((0, 0), (10, 10), (20, 20))).validar()

    def test_dibujar_y_str(self):
        imagen = np.zeros((120, 120, 3), dtype=np.uint8)
        TRIANGULO.dibujar(imagen)
        assert imagen.any()
        assert str(TRIANGULO) == "polígono de 3 puntos"

    def test_es_inmutable_y_hashable(self):
        with pytest.raises(AttributeError):
            TRIANGULO.puntos = ()  # type: ignore[misc]
        assert hash(TRIANGULO) == hash(ZonaPoligonal(((0, 0), (100, 0), (0, 100))))


class TestCriteriosConPoligono:
    def test_pies_centro_y_solape(self):
        # Persona cuyos pies pisan el triángulo pero cuya caja sobresale.
        caja = (10, -80, 50, 20)
        assert persona_en_zona(caja, TRIANGULO, "pies")  # pies en (30, 20)
        assert not persona_en_zona(caja, TRIANGULO, "centro")  # centro en (30, -30)
        # Solo la franja y∈[0,20] está dentro: 20 de 100 de alto -> 20 %.
        assert persona_en_zona(caja, TRIANGULO, "solape", min_solape=0.15)
        assert not persona_en_zona(caja, TRIANGULO, "solape", min_solape=0.25)


class TestEspecificacion:
    def test_normalizar_rectangulo(self):
        assert normalizar_espec([20, 20, 10, 10]) == (10, 10, 20, 20)
        assert not es_poligono(normalizar_espec([1, 2, 3, 4]))

    def test_normalizar_poligono_desde_json(self):
        espec = normalizar_espec([[1.4, 2.6], [10, 0], [0, 10]])
        assert espec == ((1, 3), (10, 0), (0, 10))
        assert es_poligono(espec)

    @pytest.mark.parametrize("datos", [[], [1, 2, 3], [1, 2, 3, 4, 5], [[1, 2], [3, 4]]])
    def test_normalizar_rechaza_formas_raras(self, datos):
        with pytest.raises(ValueError):
            normalizar_espec(datos)

    def test_espec_a_lista_es_json(self):
        assert espec_a_lista((1, 2, 3, 4)) == [1, 2, 3, 4]
        assert espec_a_lista(((1, 2), (3, 4), (5, 6))) == [[1, 2], [3, 4], [5, 6]]

    def test_zona_desde_elige_la_clase(self):
        assert isinstance(zona_desde([0, 0, 10, 10]), ZonaRectangular)
        assert isinstance(zona_desde([[0, 0], [10, 0], [0, 10]]), ZonaPoligonal)

    def test_configuracion_acepta_poligono(self):
        cfg = ConfiguracionAnalisis(zona_puerta=((0, 0), (100, 0), (0, 100)))
        cfg.validar()
        assert isinstance(cfg.zona, ZonaPoligonal)
        with pytest.raises(ValueError, match="alineados"):
            ConfiguracionAnalisis(zona_puerta=((0, 0), (1, 1), (2, 2))).validar()

    def test_la_huella_distingue_rectangulo_de_poligono(self):
        rect = ConfiguracionAnalisis(zona_puerta=(0, 0, 100, 100))
        poli = ConfiguracionAnalisis(zona_puerta=((0, 0), (100, 0), (100, 100), (0, 100)))
        assert huella_config(asdict(rect)) != huella_config(asdict(poli))
        # Y es estable aunque la tupla llegue como lista (tras pasar por JSON).
        poli_lista = ConfiguracionAnalisis(zona_puerta=[[0, 0], [100, 0], [100, 100], [0, 100]])  # type: ignore[arg-type]
        assert huella_config(asdict(poli)) == huella_config(asdict(poli_lista))


class TestAnalisisConPoligono:
    def test_un_poligono_con_las_esquinas_del_rectangulo_da_lo_mismo(self, video):  # noqa: F811
        from tests.test_analizador import ZONA, config

        x1, y1, x2, y2 = ZONA
        poligono = ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
        con_rect = AnalizadorPuerta(config(), DetectorGuionizado()).analizar_video(video)
        con_poli = AnalizadorPuerta(
            config(zona_puerta=poligono), DetectorGuionizado()
        ).analizar_video(video)
        zona_r = next(e for e in con_rect.eventos if e.tipo == TIPO_ZONA)
        zona_p = next(e for e in con_poli.eventos if e.tipo == TIPO_ZONA)
        assert (zona_p.inicio, zona_p.fin) == (zona_r.inicio, zona_r.fin)

    def test_un_triangulo_que_solo_cubre_el_final_del_recorrido(self, video):  # noqa: F811
        from tests.test_analizador import config

        # Los pies van de x=130 a x=255 sobre y=200. Este triángulo solo
        # contiene puntos con x ≥ 230 en esa altura: la persona entra más tarde.
        triangulo = ((230, 200), (300, 200), (300, 100))
        resultado = AnalizadorPuerta(
            config(zona_puerta=triangulo), DetectorGuionizado()
        ).analizar_video(video)
        zona = next(e for e in resultado.eventos if e.tipo == TIPO_ZONA)
        assert zona.inicio == 3.5  # llamada 7: pies en x=230
