"""Filtro de movimiento: decide si vale la pena invocar al modelo."""

from __future__ import annotations

import numpy as np

from detector import UMBRAL_MOVIMIENTO, DetectorMovimiento


def test_el_primer_frame_siempre_cuenta_como_movimiento(frame_negro):
    assert DetectorMovimiento().fraccion_cambiada(frame_negro) == 1.0


def test_frames_identicos_no_cambian(frame_negro):
    det = DetectorMovimiento()
    det.fraccion_cambiada(frame_negro)
    assert det.fraccion_cambiada(frame_negro.copy()) == 0.0


def test_un_bloque_grande_supera_el_umbral(frame_negro, frame_con_bloque):
    det = DetectorMovimiento()
    det.fraccion_cambiada(frame_negro)
    cambio = det.fraccion_cambiada(frame_con_bloque)
    # El bloque es un cuarto del frame; el difuminado apenas lo altera.
    assert 0.2 < cambio < 0.3
    assert cambio > UMBRAL_MOVIMIENTO


def test_el_ruido_leve_se_ignora(frame_negro):
    # Ruido de ±5 niveles: por debajo del umbral de píxel (25) y además difuminado.
    rng = np.random.default_rng(0)
    ruidoso = np.clip(
        frame_negro.astype(int) + 100 + rng.integers(-5, 6, frame_negro.shape), 0, 255
    ).astype(np.uint8)
    base = np.full_like(frame_negro, 100)
    det = DetectorMovimiento()
    det.fraccion_cambiada(base)
    assert det.fraccion_cambiada(ruidoso) == 0.0


def test_compara_siempre_con_el_frame_anterior(frame_negro, frame_con_bloque):
    det = DetectorMovimiento()
    det.fraccion_cambiada(frame_negro)
    det.fraccion_cambiada(frame_con_bloque)
    # El bloque sigue ahí: respecto al anterior ya no hay cambio.
    assert det.fraccion_cambiada(frame_con_bloque) == 0.0


def test_reiniciar_olvida_la_referencia(frame_negro):
    det = DetectorMovimiento()
    det.fraccion_cambiada(frame_negro)
    det.reiniciar()
    assert det.fraccion_cambiada(frame_negro) == 1.0


def test_acepta_resoluciones_distintas_al_ancho_de_trabajo():
    # 1080p simulado: se reduce a ``ancho`` manteniendo la proporción.
    det = DetectorMovimiento(ancho=64)
    grande = np.zeros((1080, 1920, 3), dtype=np.uint8)
    det.fraccion_cambiada(grande)
    assert det._previo.shape == (36, 64)
