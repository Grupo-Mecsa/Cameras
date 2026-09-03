"""Detección del hardware disponible y elección del acelerador de inferencia."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from decam.callbacks import CallbackLog

#: Aceleradores que puede usar el análisis.
#:   auto          elige el mejor disponible (CUDA > GPU Intel > CPU)
#:   cuda          GPU NVIDIA vía PyTorch
#:   openvino-gpu  GPU integrada Intel vía OpenVINO
#:   cpu           PyTorch sobre CPU
ACELERADORES = ("auto", "cuda", "openvino-gpu", "cpu")


def cuda_disponible() -> bool:
    """Indica si hay una GPU NVIDIA utilizable por PyTorch."""
    try:
        import torch  # import perezoso: viene instalado con ultralytics
    except ImportError:  # pragma: no cover - solo si torch no está instalado
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - driver roto
        return False


def openvino_gpu_disponible() -> bool:
    """Indica si OpenVINO ve una GPU Intel (integrada o dedicada)."""
    try:
        import openvino
    except ImportError:
        return False
    try:
        return "GPU" in openvino.Core().available_devices
    except Exception:  # pragma: no cover - runtime sin driver de GPU
        return False


def aceleradores_disponibles() -> list[str]:
    """Devuelve los aceleradores utilizables ahora mismo, del mejor al peor."""
    disponibles = ["auto"]
    if cuda_disponible():
        disponibles.append("cuda")
    if openvino_gpu_disponible():
        disponibles.append("openvino-gpu")
    disponibles.append("cpu")
    return disponibles


def resolver_acelerador(preferido: str = "auto") -> str:
    """Convierte la preferencia del usuario en un acelerador concreto.

    Si el preferido no está disponible se cae a la siguiente mejor opción, de
    modo que un ``config.json`` traído de otro equipo nunca rompe el análisis.

    Returns:
        Uno de ``cuda``, ``openvino-gpu`` o ``cpu``.
    """
    if preferido == "cuda" and cuda_disponible():
        return "cuda"
    if preferido == "openvino-gpu" and openvino_gpu_disponible():
        return "openvino-gpu"
    if preferido == "cpu":
        return "cpu"
    # auto, o un preferido que ya no está disponible.
    if cuda_disponible():
        return "cuda"
    if openvino_gpu_disponible():
        return "openvino-gpu"
    return "cpu"


def dispositivo_de_prediccion(acelerador: str) -> str:
    """Traduce el acelerador al valor que espera ``model.predict(device=...)``."""
    return {
        "cuda": "cuda",
        "openvino-gpu": "intel:gpu",
        "cpu": "cpu",
    }[acelerador]


def ruta_modelo_openvino(modelo: str) -> Path:
    """Carpeta donde vive (o vivirá) la versión OpenVINO de un modelo YOLO."""
    nombre = modelo[:-3] if modelo.endswith(".pt") else modelo
    return Path(f"{nombre}_openvino_model")


def exportar_a_openvino(modelo: str, on_log: Optional[CallbackLog] = None) -> Path:
    """Exporta un modelo YOLO al formato OpenVINO si aún no está exportado.

    La exportación tarda unos segundos y se hace una sola vez: el resultado
    queda en disco junto al ``.pt``.

    Returns:
        La carpeta del modelo OpenVINO.
    """
    destino = ruta_modelo_openvino(modelo)
    if destino.is_dir() and any(destino.glob("*.xml")):
        return destino

    from ultralytics import YOLO

    if on_log:
        on_log(f"Exportando {modelo} a OpenVINO (solo la primera vez)...")
    nombre = modelo if modelo.endswith(".pt") else f"{modelo}.pt"
    YOLO(nombre).export(format="openvino", imgsz=640)
    return destino
