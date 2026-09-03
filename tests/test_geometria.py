"""Geometría de cajas y zona: la base de la decisión "está en la puerta"."""

from __future__ import annotations

import pytest

from detector import (
    area_solapada,
    cajas_se_solapan,
    fraccion_dentro,
    normalizar_zona,
    persona_en_zona,
    punto_centro,
    punto_dentro,
    punto_pies,
)

ZONA = (100, 100, 200, 200)


class TestAreaSolapada:
    def test_caja_totalmente_dentro(self):
        assert area_solapada((120, 120, 140, 150), ZONA) == 20 * 30

    def test_solape_parcial(self):
        # Se mete 50 px en x y 20 px en y.
        assert area_solapada((50, 80, 150, 120), ZONA) == 50 * 20

    def test_sin_contacto(self):
        assert area_solapada((0, 0, 50, 50), ZONA) == 0.0

    def test_borde_compartido_no_cuenta(self):
        # Tocarse por el borde no es solaparse.
        assert area_solapada((0, 100, 100, 200), ZONA) == 0.0
        assert not cajas_se_solapan((0, 100, 100, 200), ZONA)

    def test_zona_dentro_de_la_caja(self):
        assert area_solapada((0, 0, 500, 500), ZONA) == 100 * 100

    def test_es_conmutativa(self):
        caja = (150, 150, 300, 260)
        assert area_solapada(caja, ZONA) == area_solapada(ZONA, caja)  # type: ignore[arg-type]


class TestNormalizarZona:
    def test_ordena_las_esquinas(self):
        # Al arrastrar de abajo-derecha a arriba-izquierda llegan invertidas.
        assert normalizar_zona((200, 200, 100, 100)) == (100, 100, 200, 200)

    def test_mezcla_de_ejes(self):
        assert normalizar_zona((200, 100, 100, 200)) == (100, 100, 200, 200)

    def test_redondea_a_entero(self):
        assert normalizar_zona((10.4, 10.6, 20.5, 20.49)) == (10, 11, 20, 20)
        assert all(isinstance(v, int) for v in normalizar_zona((1.0, 2.0, 3.0, 4.0)))


class TestFraccionDentro:
    def test_totalmente_dentro(self):
        assert fraccion_dentro((120, 120, 140, 150), ZONA) == 1.0

    def test_mitad_dentro(self):
        assert fraccion_dentro((50, 100, 150, 200), ZONA) == pytest.approx(0.5)

    def test_fuera(self):
        assert fraccion_dentro((0, 0, 50, 50), ZONA) == 0.0

    def test_caja_degenerada_no_divide_por_cero(self):
        assert fraccion_dentro((150, 150, 150, 180), ZONA) == 0.0
        assert fraccion_dentro((150, 150, 180, 150), ZONA) == 0.0


class TestPuntos:
    def test_pies_es_centro_del_borde_inferior(self):
        assert punto_pies((100, 0, 200, 300)) == (150.0, 300.0)

    def test_centro(self):
        assert punto_centro((100, 100, 200, 300)) == (150.0, 200.0)

    def test_punto_dentro_incluye_bordes(self):
        assert punto_dentro((100, 100), ZONA)
        assert punto_dentro((200, 200), ZONA)
        assert punto_dentro((150, 150), ZONA)

    def test_punto_fuera(self):
        assert not punto_dentro((99, 150), ZONA)
        assert not punto_dentro((150, 201), ZONA)


class TestPersonaEnZona:
    """El escenario del pasillo: una caja alta que roza la zona sin pisarla."""

    # Persona lejos, pero su caja alta se solapa con la esquina superior de la zona.
    CAJA_ROZA = (180, 0, 260, 120)
    # Persona pisando la zona.
    CAJA_PISA = (130, 20, 170, 180)

    def test_pies_no_se_deja_engañar_por_el_roce(self):
        assert not persona_en_zona(self.CAJA_ROZA, ZONA, "pies")
        assert persona_en_zona(self.CAJA_PISA, ZONA, "pies")

    def test_centro(self):
        assert not persona_en_zona(self.CAJA_ROZA, ZONA, "centro")
        # Centro de CAJA_PISA es (150, 100): justo en el borde superior.
        assert persona_en_zona(self.CAJA_PISA, ZONA, "centro")

    def test_solape_depende_del_umbral(self):
        # CAJA_ROZA: solape 20x20=400 de un área 80x120=9600 -> 4.2 %.
        assert persona_en_zona(self.CAJA_ROZA, ZONA, "solape", min_solape=0.04)
        assert not persona_en_zona(self.CAJA_ROZA, ZONA, "solape", min_solape=0.05)

    def test_el_criterio_por_defecto_es_pies(self):
        assert persona_en_zona(self.CAJA_PISA, ZONA) == persona_en_zona(
            self.CAJA_PISA, ZONA, "pies"
        )

    def test_criterio_desconocido(self):
        with pytest.raises(ValueError, match="desconocido"):
            persona_en_zona(self.CAJA_PISA, ZONA, "cabeza")
