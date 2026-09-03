"""Modelo de datos del resultado del análisis: eventos y resultados por video.

Solo estructuras y su serialización; no saben cómo se detectan ni cómo se
guardan.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

#: Tipos de evento que produce el análisis.
TIPO_ZONA = "zona"        # la persona cumple el criterio de la zona de la puerta
TIPO_GENERAL = "general"  # cualquier persona vista en el frame

#: Columnas del CSV de eventos, en orden. Coinciden con :meth:`Evento.a_fila_csv`.
COLUMNAS_CSV = (
    "archivo",
    "tipo",
    "inicio",
    "fin",
    "duracion_segundos",
    "rostros",
    "personas",
    "personas_distintas",
    "direccion",
)


def formatear_tiempo(segundos: float) -> str:
    """Convierte segundos a una cadena ``HH:MM:SS``."""
    total = int(round(segundos))
    horas, resto = divmod(total, 3600)
    minutos, segs = divmod(resto, 60)
    return f"{horas:02d}:{minutos:02d}:{segs:02d}"


@dataclass
class Evento:
    """Un intervalo de tiempo en el que hubo una persona en la zona de la puerta."""

    archivo: str
    inicio: float
    fin: float
    tipo: str = TIPO_ZONA
    miniatura: str = ""
    rostros: int = 0
    personas: str = ""
    #: Personas distintas vistas durante el evento (por seguimiento; si no hay
    #: seguimiento, el máximo de personas simultáneas en un frame).
    n_personas: int = 0
    #: Resumen de hacia dónde iban: ``"2 entran, 1 sale"``. Vacío sin seguimiento.
    direccion: str = ""

    @property
    def duracion(self) -> float:
        """Duración del evento en segundos."""
        return max(0.0, self.fin - self.inicio)

    def a_dict(self) -> dict[str, Any]:
        """Representación serializable (para el manifiesto)."""
        return asdict(self)

    @classmethod
    def desde_dict(cls, datos: dict[str, Any]) -> "Evento":
        """Reconstruye un evento; ignora claves desconocidas de otras versiones."""
        campos = {k: v for k, v in datos.items() if k in cls.__dataclass_fields__}
        return cls(**campos)

    def a_fila_csv(self) -> dict[str, str]:
        """Representación del evento como fila del CSV de salida."""
        return {
            "archivo": self.archivo,
            "tipo": self.tipo,
            "inicio": formatear_tiempo(self.inicio),
            "fin": formatear_tiempo(self.fin),
            "duracion_segundos": f"{self.duracion:.2f}",
            "rostros": str(self.rostros),
            "personas": self.personas,
            "personas_distintas": str(self.n_personas),
            "direccion": self.direccion,
        }

    def __str__(self) -> str:
        extra = f", {self.n_personas} persona(s)" if self.n_personas > 1 else ""
        if self.direccion:
            extra += f", {self.direccion}"
        if self.rostros:
            extra += f", {self.rostros} rostro(s)"
        if self.personas:
            extra += f" [{self.personas}]"
        return (
            f"[{self.tipo}] {self.archivo} | {formatear_tiempo(self.inicio)} -> "
            f"{formatear_tiempo(self.fin)} ({self.duracion:.1f}s{extra})"
        )


@dataclass
class ResultadoVideo:
    """Resultado del análisis de un video."""

    archivo: str
    eventos: list[Evento] = field(default_factory=list)
    error: Optional[str] = None
    frames_analizados: int = 0
    frames_omitidos: int = 0
    #: ``True`` si se recuperó del manifiesto en vez de analizarse ahora.
    reutilizado: bool = False

    def a_dict(self) -> dict[str, Any]:
        """Representación serializable (para el manifiesto)."""
        return {
            "archivo": self.archivo,
            "eventos": [e.a_dict() for e in self.eventos],
            "error": self.error,
            "frames_analizados": self.frames_analizados,
            "frames_omitidos": self.frames_omitidos,
        }

    @classmethod
    def desde_dict(cls, datos: dict[str, Any]) -> "ResultadoVideo":
        """Reconstruye un resultado guardado y lo marca como reutilizado."""
        return cls(
            archivo=str(datos.get("archivo", "")),
            eventos=[Evento.desde_dict(e) for e in datos.get("eventos", [])],
            error=datos.get("error"),
            frames_analizados=int(datos.get("frames_analizados", 0)),
            frames_omitidos=int(datos.get("frames_omitidos", 0)),
            reutilizado=True,
        )

    @property
    def frames_muestreados(self) -> int:
        """Frames que se muestrearon, se analizaran o no."""
        return self.frames_analizados + self.frames_omitidos

    @property
    def ahorro_movimiento(self) -> float:
        """Fracción de frames muestreados que el filtro de movimiento descartó."""
        total = self.frames_muestreados
        return self.frames_omitidos / total if total else 0.0
