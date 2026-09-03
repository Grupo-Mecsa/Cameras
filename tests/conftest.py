"""Configuración compartida de los tests.

El proyecto es un conjunto de módulos planos sin paquete, así que hay que poner
la raíz en ``sys.path`` para poder importar ``detector``, ``registro`` y ``app``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))


@pytest.fixture
def frame_negro():
    """Un frame BGR de 64x48 completamente negro."""
    return np.zeros((48, 64, 3), dtype=np.uint8)


@pytest.fixture
def frame_con_bloque(frame_negro):
    """El frame negro con un bloque blanco que ocupa un cuarto de la imagen."""
    frame = frame_negro.copy()
    frame[0:24, 0:32] = 255
    return frame
