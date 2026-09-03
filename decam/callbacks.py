"""Tipos de los callbacks con los que el análisis informa a quien lo lanza.

Son funciones sueltas y no una interfaz con varios métodos a propósito: quien
solo quiere el log no tiene que implementar nada más.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from decam.eventos import Evento

#: ``(nombre_video, porcentaje_del_video, porcentaje_total)``.
CallbackProgreso = Callable[[str, float, float], None]
#: Una línea de texto para el registro.
CallbackLog = Callable[[str], None]
#: Cada evento en cuanto se cierra.
CallbackEvento = Callable[["Evento"], None]
