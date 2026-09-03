"""Seguimiento de personas: identificadores estables y dirección respecto a la zona."""

from __future__ import annotations

import pytest

from decam.deteccion import Deteccion, DeteccionCruda
from decam.seguimiento import DIRECCIONES, RastreadorPersonas, Trayectorias


def cajas(*xyxy: tuple[int, int, int, int], conf: float = 0.9) -> list[DeteccionCruda]:
    """Las detecciones de un frame, tal como salen del detector."""
    return [DeteccionCruda(c, conf) for c in xyxy]


A = (100, 100, 200, 400)
B = (1500, 100, 1600, 400)  # lejos de A: nunca se confunden


def mover(caja, dx):
    x1, y1, x2, y2 = caja
    return (x1 + dx, y1, x2 + dx, y2)


class TestRastreadorPersonas:
    def test_misma_persona_conserva_el_id(self):
        r = RastreadorPersonas(fps_analisis=1.0, tolerancia_segundos=3.0)
        ids = [r.asignar_ids(cajas(mover(A, dx)))[0] for dx in (0, 10, 20, 30)]
        assert ids[0] is not None
        assert len(set(ids)) == 1

    def test_ninguna_deteccion_se_pierde(self):
        # Esta es la razón de no usar modelo.track(): una persona nueva sale
        # igualmente en la lista, aunque su pista aún no esté confirmada.
        r = RastreadorPersonas(1.0, 3.0)
        r.asignar_ids(cajas(A))
        r.asignar_ids(cajas(mover(A, 10)))
        ids = r.asignar_ids(cajas(mover(A, 20), B))
        assert len(ids) == 2
        assert ids[0] is not None
        assert ids[1] is None  # B acaba de aparecer: pista sin confirmar

        ids = r.asignar_ids(cajas(mover(A, 30), mover(B, 5)))
        assert ids[1] is not None and ids[1] != ids[0]

    def test_alineado_con_las_cajas_aunque_cambie_el_orden(self):
        r = RastreadorPersonas(1.0, 3.0)
        id_a, id_b = r.asignar_ids(cajas(A, B))  # frame 1: ambas activadas
        assert None not in (id_a, id_b) and id_a != id_b
        ids = r.asignar_ids(cajas(mover(B, 5), mover(A, 5)))  # orden invertido
        assert ids == [id_b, id_a]

    def test_frame_sin_personas_no_falla_y_envejece_pistas(self):
        r = RastreadorPersonas(1.0, 0.0)
        assert r.buffer == RastreadorPersonas.BUFFER_MINIMO
        id_a = r.asignar_ids(cajas(A))[0]
        for _ in range(r.buffer + 2):
            assert r.asignar_ids([]) == []
        # Tras superar el buffer la pista se descarta: reaparecer da otro id.
        r.asignar_ids(cajas(A))
        nuevo = r.asignar_ids(cajas(mover(A, 5)))[0]
        assert nuevo is not None and nuevo != id_a

    def test_pista_perdida_brevemente_recupera_el_id(self):
        r = RastreadorPersonas(1.0, 3.0)
        id_a = r.asignar_ids(cajas(A))[0]
        r.asignar_ids(cajas(mover(A, 10)))
        r.asignar_ids([])  # dos frames sin detectarla
        r.asignar_ids([])
        assert r.asignar_ids(cajas(mover(A, 40)))[0] == id_a

    def test_reiniciar_olvida_las_pistas(self):
        r = RastreadorPersonas(1.0, 3.0)
        r.asignar_ids(cajas(A))
        r.reiniciar()
        # Tras reiniciar vuelve a ser el primer frame: la pista se activa ya.
        assert r.asignar_ids(cajas(B))[0] is not None

    @pytest.mark.parametrize(
        "fps, tolerancia, esperado",
        [(1.0, 3.0, 30), (1.0, 60.0, 61), (10.0, 5.0, 51), (0.5, 3.0, 30)],
    )
    def test_buffer_cubre_al_menos_la_tolerancia(self, fps, tolerancia, esperado):
        assert RastreadorPersonas(fps, tolerancia).buffer == esperado


class TestTrayectorias:
    def test_direcciones_basicas(self):
        t = Trayectorias()
        # 1: llega de fuera y desaparece dentro -> entra
        for dentro in (False, False, True, True):
            t.observar(1, dentro)
        # 2: aparece dentro y se va -> sale
        for dentro in (True, False, False):
            t.observar(2, dentro)
        # 3: pasa por la zona -> cruza
        for dentro in (False, True, False):
            t.observar(3, dentro)
        # 4: siempre dentro -> permanece
        for dentro in (True, True):
            t.observar(4, dentro)
        assert [t.direccion(i) for i in (1, 2, 3, 4)] == [
            "entra", "sale", "cruza", "permanece",
        ]

    def test_pista_que_nunca_toca_la_zona_no_tiene_direccion(self):
        t = Trayectorias()
        for _ in range(3):
            t.observar(7, False)
        assert t.direccion(7) is None
        assert t.direccion(99) is None  # desconocida

    def test_una_sola_observacion_dentro_es_permanece(self):
        t = Trayectorias()
        t.observar(1, True)
        assert t.direccion(1) == "permanece"

    def test_resumen_ordena_y_pluraliza(self):
        t = Trayectorias()
        for i in (1, 2):
            t.observar(i, False)
            t.observar(i, True)
        t.observar(3, True)
        t.observar(3, False)
        t.observar(4, False)  # nunca dentro: no cuenta
        assert t.resumen([4, 3, 2, 1]) == "2 entran, 1 sale"

    def test_resumen_vacio(self):
        assert Trayectorias().resumen([]) == ""
        assert Trayectorias().resumen([1, 2]) == ""

    def test_tabla_de_direcciones_cubre_los_cuatro_casos(self):
        assert set(DIRECCIONES) == {(a, b) for a in (False, True) for b in (False, True)}


def test_deteccion_area():
    assert Deteccion((10, 10, 30, 50)).area == 20 * 40
    assert Deteccion((10, 10, 10, 50)).area == 0
    assert Deteccion((10, 10, 30, 50)).id is None
