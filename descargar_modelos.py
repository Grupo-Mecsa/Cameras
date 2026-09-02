"""Descarga los modelos ONNX de rostros desde el repositorio oficial OpenCV Zoo.

Uso:
    python descargar_modelos.py

Deja en ``models/``:
    face_detection_yunet_2023mar.onnx     detección de rostros (YuNet)
    face_recognition_sface_2021dec.onnx   identificación de rostros (SFace)

Ambos son necesarios solo si activas las funciones de rostros. Son ficheros
pequeños (~230 KB y ~37 MB) y se descargan una sola vez.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

CARPETA_MODELOS = Path(__file__).with_name("models")
BASE = "https://github.com/opencv/opencv_zoo/raw/main/models"

MODELOS: dict[str, str] = {
    "face_detection_yunet_2023mar.onnx": (
        f"{BASE}/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    ),
    "face_recognition_sface_2021dec.onnx": (
        f"{BASE}/face_recognition_sface/face_recognition_sface_2021dec.onnx"
    ),
}


def descargar(nombre: str, url: str, destino: Path) -> bool:
    """Descarga un modelo si no está ya en disco.

    Returns:
        ``True`` si el archivo está disponible al terminar.
    """
    ruta = destino / nombre
    if ruta.is_file() and ruta.stat().st_size > 0:
        print(f"  ya existe: {nombre} ({ruta.stat().st_size / 1024:.0f} KB)")
        return True
    print(f"  descargando {nombre}...")
    try:
        urllib.request.urlretrieve(url, ruta)
    except (urllib.error.URLError, OSError) as exc:
        print(f"  ERROR al descargar {nombre}: {exc}")
        ruta.unlink(missing_ok=True)
        return False
    print(f"  listo: {nombre} ({ruta.stat().st_size / 1024:.0f} KB)")
    return True


def main() -> int:
    """Descarga todos los modelos y devuelve 0 si todos están disponibles."""
    CARPETA_MODELOS.mkdir(parents=True, exist_ok=True)
    print(f"Carpeta de modelos: {CARPETA_MODELOS}")
    ok = all(
        descargar(nombre, url, CARPETA_MODELOS) for nombre, url in MODELOS.items()
    )
    print("\nTodo listo." if ok else "\nFaltan modelos; revisa tu conexión.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
