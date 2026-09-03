"""Geometría de cajas y zona: la base de la decisión "está en la puerta"."""

from __future__ import annotations

import numpy as np
import pytest

from decam.zona import (
    ZonaRectangular,
    normalizar_zona,
    persona_en_zona,
    punto_centro,
    punto_pies,
)

ZONA = ZonaRectangular(100, 100, 200, 200)


class TestAreaSolapada:
    def test_caja_totalmente_dentro(self):
        assert ZONA.area_solapada((120, 120, 140, 150)) == 20 * 30

    def test_solape_parcial(self):
        # Se mete 50 px en x y 20 px en y.
        assert ZONA.area_solapada((50, 80, 150, 120)) == 50 * 20

    def test_sin_contacto(self):
        assert ZONA.area_solapada((0, 0, 50, 50)) == 0.0

    def test_borde_compartido_no_cuenta(self):
        # Tocarse por el borde no es solaparse.
        assert ZONA.area_solapada((0, 100, 100, 200)) == 0.0

    def test_zona_dentro_de_la_caja(self):
        assert ZONA.area_solapada((0, 0, 500, 500)) == 100 * 100


class TestZonaRectangular:
    def test_desde_ordena_las_esquinas(self):
        # Al arrastrar de abajo-derecha a arriba-izquierda llegan invertidas.
        assert ZonaRectangular.desde((200, 200, 100, 100)) == ZONA
        assert ZonaRectangular.desde((200, 100, 100, 200)) == ZONA

    def test_normalizar_redondea_a_entero(self):
        assert normalizar_zona((10.4, 10.6, 20.5, 20.49)) == (10, 11, 20, 20)
        assert all(isinstance(v, int) for v in normalizar_zona((1.0, 2.0, 3.0, 4.0)))

    def test_como_tupla_y_str(self):
        assert ZONA.como_tupla == (100, 100, 200, 200)
        assert str(ZONA) == "(100, 100) - (200, 200)"
        assert (ZONA.ancho, ZONA.alto) == (100, 100)

    @pytest.mark.parametrize("zona", [(0, 0, 0, 10), (0, 10, 10, 10), (5, 5, 5, 5)])
    def test_validar_rechaza_degeneradas(self, zona):
        with pytest.raises(ValueError, match="zona"):
            ZonaRectangular.desde(zona).validar()

    def test_validar_acepta_normales(self):
        ZONA.validar()

    def test_dibujar_pinta_sobre_la_imagen(self):
        imagen = np.zeros((300, 300, 3), dtype=np.uint8)
        ZONA.dibujar(imagen)
        assert imagen.any(), "algo se pintó"
        assert imagen[100, 150].tolist() == [255, 0, 0], "borde superior en azul (BGR)"

    def test_es_inmutable(self):
        with pytest.raises(AttributeError):
            ZONA.x1 = 0  # type: ignore[misc]


class TestFraccionDe:
    def test_totalmente_dentro(self):
        assert ZONA.fraccion_de((120, 120, 140, 150)) == 1.0

    def test_mitad_dentro(self):
        assert ZONA.fraccion_de((50, 100, 150, 200)) == pytest.approx(0.5)

    def test_fuera(self):
        assert ZONA.fraccion_de((0, 0, 50, 50)) == 0.0

    def test_caja_degenerada_no_divide_por_cero(self):
        assert ZONA.fraccion_de((150, 150, 150, 180)) == 0.0
        assert ZONA.fraccion_de((150, 150, 180, 150)) == 0.0


class TestPuntos:
    def test_pies_es_centro_del_borde_inferior(self):
        assert punto_pies((100, 0, 200, 300)) == (150.0, 300.0)

    def test_centro(self):
        assert punto_centro((100, 100, 200, 300)) == (150.0, 200.0)

    def test_contiene_punto_incluye_bordes(self):
        assert ZONA.contiene_punto((100, 100))
        assert ZONA.contiene_punto((200, 200))
        assert ZONA.contiene_punto((150, 150))

    def test_punto_fuera(self):
        assert not ZONA.contiene_punto((99, 150))
        assert not ZONA.contiene_punto((150, 201))


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

    def test_acepta_cualquier_zona_que_cumpla_el_protocolo(self):
        """Los criterios no saben que la zona es un rectángulo (abierto/cerrado)."""

        class ZonaMitadIzquierda:
            def contiene_punto(self, punto):
                return punto[0] < 150

            def fraccion_de(self, caja):
                return 1.0 if caja[2] <= 150 else 0.0

            def validar(self):
                pass

            def dibujar(self, imagen):
                pass

        zona = ZonaMitadIzquierda()
        assert persona_en_zona((0, 0, 100, 100), zona, "pies")
        assert not persona_en_zona((200, 0, 300, 100), zona, "pies")
        assert persona_en_zona((0, 0, 100, 100), zona, "solape", 0.5)
