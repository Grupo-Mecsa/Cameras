"""Lógica de análisis de video: detecta personas que se acercan a una puerta.

Este módulo es independiente de la interfaz gráfica. La comunicación con la GUI
se hace mediante callbacks (progreso, log, evento) y un ``threading.Event`` para
poder cancelar el análisis.
"""

from __future__ import annotations

import csv
import math
import queue
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional, Sequence

import cv2

import manifiesto

# Extensiones de video soportadas.
EXTENSIONES_VIDEO: tuple[str, ...] = (".mp4", ".avi", ".mkv")

# En el dataset COCO la clase 0 corresponde a "person".
CLASE_PERSONA: int = 0

#: Tipos de evento que produce el análisis.
TIPO_ZONA = "zona"        # la persona cumple el criterio de la zona de la puerta
TIPO_GENERAL = "general"  # cualquier persona vista en el frame

#: Criterios para decidir si una persona "está" en la zona.
CRITERIOS_ZONA = ("pies", "centro", "solape")


# Tipos de los callbacks usados para informar a la interfaz.
CallbackProgreso = Callable[[str, float, float], None]
CallbackLog = Callable[[str], None]
CallbackEvento = Callable[["Evento"], None]


class AnalisisCancelado(Exception):
    """Se lanza internamente cuando el usuario cancela el análisis."""


def formatear_tiempo(segundos: float) -> str:
    """Convierte segundos a una cadena ``HH:MM:SS``."""
    total = int(round(segundos))
    horas, resto = divmod(total, 3600)
    minutos, segs = divmod(resto, 60)
    return f"{horas:02d}:{minutos:02d}:{segs:02d}"


def encontrar_videos(carpeta: str | Path, recursivo: bool = True) -> list[Path]:
    """Devuelve los videos soportados dentro de ``carpeta``, ordenados por nombre.

    Args:
        carpeta: carpeta donde buscar.
        recursivo: si es ``True`` (por defecto) también busca en las subcarpetas,
            como en las exportaciones de NVR que crean una carpeta por descarga.
    """
    ruta = Path(carpeta)
    if not ruta.is_dir():
        return []
    candidatos = ruta.rglob("*") if recursivo else ruta.iterdir()
    videos = [
        archivo
        for archivo in candidatos
        if archivo.is_file() and archivo.suffix.lower() in EXTENSIONES_VIDEO
    ]
    return sorted(videos, key=lambda p: str(p).lower())


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


def detectar_dispositivo() -> str:
    """Devuelve ``"cuda"`` si hay GPU NVIDIA, si no ``"cpu"``.

    Se mantiene por compatibilidad; para elegir acelerador usa
    :func:`resolver_acelerador`.
    """
    return "cuda" if cuda_disponible() else "cpu"


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


def abrir_captura(video: str | Path, hardware: bool = False) -> cv2.VideoCapture:
    """Abre un video, opcionalmente con decodificación acelerada por hardware.

    La aceleración (Quick Sync, D3D11) descomprime ~2.2x más rápido, pero el
    decodificador del driver aplica una conversión de color ligeramente distinta
    a la del decodificador por software: la imagen sale con un sesgo uniforme de
    unos 8 niveles de gris. Las cajas apenas se mueven (±2 px), pero las
    detecciones que rozaban el umbral de confianza pueden caerse. Por eso viene
    desactivada por defecto.
    """
    if not hardware:
        return cv2.VideoCapture(str(video))
    return cv2.VideoCapture(
        str(video),
        cv2.CAP_FFMPEG,
        [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY],
    )


def primer_frame(video: str | Path):
    """Lee el primer frame de un video.

    Returns:
        El frame en formato BGR (``numpy.ndarray``) o ``None`` si no se pudo leer.
    """
    captura = cv2.VideoCapture(str(video))
    try:
        if not captura.isOpened():
            return None
        ok, frame = captura.read()
        return frame if ok else None
    finally:
        captura.release()


@dataclass
class InfoVideo:
    """Metadatos básicos de un video."""

    fps: float
    total_frames: int
    ancho: int
    alto: int

    @property
    def duracion(self) -> float:
        """Duración aproximada en segundos."""
        return self.total_frames / self.fps if self.fps > 0 else 0.0


def info_video(captura: cv2.VideoCapture) -> InfoVideo:
    """Extrae los metadatos de una captura ya abierta.

    Si el contenedor no informa los FPS se asume 25, valor habitual en NVR.
    """
    fps = captura.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0:
        fps = 25.0
    return InfoVideo(
        fps=fps,
        total_frames=int(captura.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        ancho=int(captura.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        alto=int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
    )


#: Frames que se decodifican antes del objetivo para que el códec se
#: resincronice tras un salto. En H.265/HEVC, leer justo después de un
#: ``CAP_PROP_POS_FRAMES`` devuelve imágenes rotas: solo se decodifica la zona
#: en movimiento sobre un fondo gris.
#:
#: El margen necesario depende del GOP del archivo. Medido contra decodificación
#: secuencial (la referencia fiable) sobre grabaciones Hikvision 1080p H.265:
#: con margen 5 el 27 % de los píxeles sigue mal y con margen 15 aún el 9 %;
#: a partir de 30 la imagen es correcta y no mejora subiendo más. Se deja en 30.
MARGEN_RESINCRONIZACION = 30


def leer_frame(
    captura: cv2.VideoCapture,
    indice: int,
    margen: int = MARGEN_RESINCRONIZACION,
):
    """Salta al frame ``indice`` de una captura abierta y lo devuelve.

    Para evitar frames corruptos se salta ``margen`` frames antes del objetivo y
    se decodifica hacia delante hasta llegar a él.

    Returns:
        El frame en BGR, o ``None`` si no se pudo leer.
    """
    indice = max(0, indice)
    inicio = max(0, indice - margen)
    captura.set(cv2.CAP_PROP_POS_FRAMES, inicio)
    frame = None
    for _ in range(indice - inicio + 1):
        ok, leido = captura.read()
        if not ok:
            return frame
        frame = leido
    return frame


def area_solapada(
    caja: tuple[float, float, float, float],
    zona: tuple[int, int, int, int],
) -> float:
    """Devuelve el área de intersección entre la caja y la zona (0 si no se tocan)."""
    ax1, ay1, ax2, ay2 = caja
    bx1, by1, bx2, by2 = zona
    ancho = min(ax2, bx2) - max(ax1, bx1)
    alto = min(ay2, by2) - max(ay1, by1)
    if ancho <= 0 or alto <= 0:
        return 0.0
    return float(ancho * alto)


def cajas_se_solapan(
    caja: tuple[float, float, float, float],
    zona: tuple[int, int, int, int],
) -> bool:
    """Indica si dos rectángulos ``(x1, y1, x2, y2)`` se solapan."""
    return area_solapada(caja, zona) > 0.0


def normalizar_zona(zona: Sequence[float]) -> tuple[int, int, int, int]:
    """Ordena y redondea una zona a ``(x1, y1, x2, y2)`` con enteros."""
    x1, y1, x2, y2 = zona
    return (
        int(round(min(x1, x2))),
        int(round(min(y1, y2))),
        int(round(max(x1, x2))),
        int(round(max(y1, y2))),
    )


#: Fracción de píxeles que deben cambiar entre dos frames analizados para que
#: valga la pena invocar al modelo. Medido sobre grabaciones de pasillo 1080p:
#: los frames con persona no bajan de 0.009 y los vacíos no pasan de 0.0011 (p90),
#: así que 0.002 deja un margen holgado por ambos lados.
UMBRAL_MOVIMIENTO = 0.002

#: Ancho al que se reduce el frame para comparar. Reducir es lo que hace barata
#: la comparación y, de paso, diluye el reloj sobreimpreso de las cámaras.
ANCHO_MOVIMIENTO = 320


class DetectorMovimiento:
    """Mide cuánto cambia un frame respecto al anterior analizado.

    Sirve para no invocar al modelo en frames donde no ocurre nada, que en una
    cámara de vigilancia son la inmensa mayoría. El frame se pasa a gris, se
    reduce y se difumina antes de compararlo: así un cambio irrelevante —el
    reloj sobreimpreso, el ruido del sensor— no cuenta como movimiento.
    """

    def __init__(
        self,
        ancho: int = ANCHO_MOVIMIENTO,
        umbral_pixel: int = 25,
    ) -> None:
        """Prepara el detector.

        Args:
            ancho: ancho al que se reduce el frame antes de comparar.
            umbral_pixel: diferencia de gris a partir de la cual un píxel cuenta
                como cambiado.
        """
        self.ancho = ancho
        self.umbral_pixel = umbral_pixel
        self._previo = None

    def _preparar(self, frame):
        """Pasa el frame a gris reducido y difuminado."""
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        alto = max(1, int(gris.shape[0] * self.ancho / gris.shape[1]))
        return cv2.GaussianBlur(cv2.resize(gris, (self.ancho, alto)), (5, 5), 0)

    def fraccion_cambiada(self, frame) -> float:
        """Devuelve la fracción de píxeles que cambian respecto al frame previo.

        El primer frame no tiene con qué compararse, así que devuelve ``1.0``
        para que siempre se analice.
        """
        actual = self._preparar(frame)
        if self._previo is None:
            self._previo = actual
            return 1.0

        diferencia = cv2.absdiff(self._previo, actual)
        _, binaria = cv2.threshold(
            diferencia, self.umbral_pixel, 255, cv2.THRESH_BINARY
        )
        self._previo = actual
        return cv2.countNonZero(binaria) / binaria.size

    def reiniciar(self) -> None:
        """Olvida el frame de referencia; se usa al cambiar de video."""
        self._previo = None


#: Modelos ONNX de OpenCV Zoo. Se buscan en la carpeta ``models/``.
CARPETA_MODELOS = Path(__file__).with_name("models")
RUTA_MODELO_YUNET = CARPETA_MODELOS / "face_detection_yunet_2023mar.onnx"
RUTA_MODELO_SFACE = CARPETA_MODELOS / "face_recognition_sface_2021dec.onnx"

#: Umbral de similitud coseno recomendado por OpenCV para SFace: por encima de
#: este valor, dos rostros se consideran de la misma persona.
UMBRAL_SFACE = 0.363

#: Etiqueta usada cuando ningún rostro del catálogo supera el umbral.
DESCONOCIDO = "desconocido"


#: Extensiones aceptadas para las fotos de referencia de cada persona.
EXTENSIONES_IMAGEN = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def haar_disponible() -> bool:
    """Indica si esta instalación de OpenCV puede usar cascadas Haar.

    OpenCV 5 eliminó ``cv2.CascadeClassifier`` del módulo principal y ya no
    distribuye los XML de las cascadas, así que en esas versiones el backend
    ``haar`` no existe.
    """
    if not hasattr(cv2, "CascadeClassifier"):
        return False
    cascada = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    return cascada.is_file()


def backends_rostros_disponibles() -> list[str]:
    """Devuelve los backends de detección de rostros utilizables ahora mismo."""
    disponibles = []
    if RUTA_MODELO_YUNET.is_file():
        disponibles.append("yunet")
    if haar_disponible():
        disponibles.append("haar")
    return disponibles


@dataclass
class RostroDetectado:
    """Un rostro localizado dentro de la caja de una persona.

    Attributes:
        caja: ``(x1, y1, x2, y2)`` en coordenadas absolutas del frame.
        region: recorte de la imagen donde se hizo la detección.
        fila: fila cruda de YuNet (caja + 5 puntos faciales) relativa a
            ``region``, necesaria para alinear el rostro antes de identificarlo.
            Es ``None`` con el backend Haar, que no da puntos faciales.
    """

    caja: tuple[int, int, int, int]
    region: Any = None
    fila: Any = None


class DetectorRostros:
    """Detecta rostros dentro de la caja de una persona ya localizada.

    Buscar solo en la mitad superior de la caja de la persona es más rápido y
    reduce falsos positivos frente a recorrer el frame completo.

    Backends disponibles:
        ``yunet``: red YuNet (``cv2.FaceDetectorYN``). Es el recomendado: aguanta
            ángulos y rostros pequeños, y aporta los puntos faciales que hacen
            falta para identificar. Necesita el ONNX en :data:`RUTA_MODELO_YUNET`.
        ``haar``: cascada Haar frontal. Sin descargas, pero solo detecta rostros
            frontales y grandes, no sirve para identificar, y **no existe en
            OpenCV 5 o superior**.
    """

    def __init__(self, backend: str = "yunet", confianza: float = 0.6) -> None:
        """Prepara el backend indicado.

        Raises:
            FileNotFoundError: si falta el modelo o la cascada del backend.
            ValueError: si el backend no existe.
        """
        self.backend = backend
        self.confianza = confianza
        self._haar = None
        self._yunet = None

        if backend == "yunet":
            if not RUTA_MODELO_YUNET.is_file():
                raise FileNotFoundError(
                    "Falta el modelo YuNet. Ejecuta 'python descargar_modelos.py' "
                    f"o copia face_detection_yunet_2023mar.onnx en {CARPETA_MODELOS}"
                )
            self._yunet = cv2.FaceDetectorYN.create(
                str(RUTA_MODELO_YUNET), "", (320, 320), self.confianza
            )
        elif backend == "haar":
            if not haar_disponible():
                raise FileNotFoundError(
                    "Este OpenCV no incluye cascadas Haar "
                    f"(version {cv2.__version__}). Usa el backend 'yunet'."
                )
            ruta = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            self._haar = cv2.CascadeClassifier(str(ruta))
            if self._haar.empty():  # pragma: no cover - instalación corrupta
                raise FileNotFoundError(f"No se pudo cargar la cascada Haar: {ruta}")
        else:
            raise ValueError(f"Backend de rostros desconocido: {backend}")

    @property
    def permite_identificar(self) -> bool:
        """Solo YuNet aporta los puntos faciales que necesita la identificación."""
        return self._yunet is not None

    def detectar(
        self,
        frame,
        caja_persona: tuple[int, int, int, int],
        proporcion_superior: float = 0.55,
    ) -> list[RostroDetectado]:
        """Busca rostros en la parte superior de ``caja_persona``.

        Args:
            frame: imagen completa en BGR.
            caja_persona: caja ``(x1, y1, x2, y2)`` de la persona detectada.
            proporcion_superior: fracción de la altura de la caja donde buscar.

        Returns:
            Los rostros encontrados, con la caja en coordenadas del frame.
        """
        alto_frame, ancho_frame = frame.shape[:2]
        x1, y1, x2, y2 = caja_persona
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(ancho_frame, x2)
        y2 = min(alto_frame, int(y1 + (y2 - y1) * proporcion_superior))
        if x2 - x1 < 20 or y2 - y1 < 20:
            return []

        region = frame[y1:y2, x1:x2]
        rostros: list[RostroDetectado] = []
        if self._yunet is not None:
            for fila in self.filas_yunet(region):
                rx, ry, rw, rh = (int(v) for v in fila[:4])
                rostros.append(
                    RostroDetectado(
                        caja=(x1 + rx, y1 + ry, x1 + rx + rw, y1 + ry + rh),
                        region=region,
                        fila=fila,
                    )
                )
        else:
            for rx, ry, rw, rh in self._filas_haar(region):
                rostros.append(
                    RostroDetectado(
                        caja=(x1 + rx, y1 + ry, x1 + rx + rw, y1 + ry + rh),
                        region=region,
                    )
                )
        return rostros

    def filas_yunet(self, imagen) -> list:
        """Ejecuta YuNet sobre una imagen y devuelve sus filas crudas.

        Cada fila trae la caja y los cinco puntos faciales que usa ``alignCrop``.
        Devuelve una lista vacía si no hay rostros o si el backend no es YuNet.
        """
        if self._yunet is None:
            return []
        alto, ancho = imagen.shape[:2]
        if alto < 20 or ancho < 20:
            return []
        self._yunet.setInputSize((ancho, alto))
        _, resultado = self._yunet.detect(imagen)
        return [] if resultado is None else list(resultado)

    def _filas_haar(self, region) -> list[tuple[int, int, int, int]]:
        """Detección con cascada Haar; devuelve ``(x, y, w, h)`` en la región."""
        gris = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        gris = cv2.equalizeHist(gris)  # ayuda con el contraste bajo de las cámaras
        detecciones = self._haar.detectMultiScale(
            gris, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24)
        )
        return [(int(x), int(y), int(w), int(h)) for x, y, w, h in detecciones]


@dataclass
class Identidad:
    """Resultado de comparar un rostro contra el catálogo de personas."""

    nombre: str
    similitud: float

    @property
    def conocida(self) -> bool:
        """``True`` si el rostro se pudo asignar a una persona del catálogo."""
        return self.nombre != DESCONOCIDO


class IdentificadorRostros:
    """Compara rostros contra un catálogo de personas conocidas.

    Usa SFace (``cv2.FaceRecognizerSF``): convierte cada rostro alineado en un
    vector de 128 dimensiones y los compara por similitud coseno. Necesita el
    backend ``yunet``, porque la alineación depende de sus puntos faciales.

    El catálogo se construye desde una carpeta con una subcarpeta por persona::

        personas/
        ├── ana_torres/
        │   ├── 1.jpg
        │   └── 2.jpg
        └── juan_perez/
            └── frente.jpg

    El nombre de la subcarpeta es la etiqueta que aparecerá en el CSV.
    """

    def __init__(
        self,
        carpeta_personas: str | Path,
        detector: DetectorRostros,
        umbral: float = UMBRAL_SFACE,
        on_log: Optional[CallbackLog] = None,
    ) -> None:
        """Carga SFace y construye el catálogo de referencia.

        Raises:
            FileNotFoundError: si falta el ONNX de SFace o la carpeta de personas.
            ValueError: si el detector no es YuNet o el catálogo queda vacío.
        """
        if not detector.permite_identificar:
            raise ValueError(
                "La identificación necesita el backend 'yunet' de detección."
            )
        if not RUTA_MODELO_SFACE.is_file():
            raise FileNotFoundError(
                "Falta el modelo SFace. Ejecuta 'python descargar_modelos.py' o "
                f"copia face_recognition_sface_2021dec.onnx en {CARPETA_MODELOS}"
            )
        carpeta = Path(carpeta_personas)
        if not carpeta.is_dir():
            raise FileNotFoundError(f"No existe la carpeta de personas: {carpeta}")

        self.detector = detector
        self.umbral = umbral
        self._log = on_log or (lambda _m: None)
        self._sface = cv2.FaceRecognizerSF.create(str(RUTA_MODELO_SFACE), "")
        self.catalogo: dict[str, list] = {}
        self._cargar_catalogo(carpeta)

        if not self.catalogo:
            raise ValueError(
                f"No se pudo extraer ningún rostro de referencia en {carpeta}. "
                "Cada persona necesita su propia subcarpeta con fotos suyas."
            )

    def _cargar_catalogo(self, carpeta: Path) -> None:
        """Calcula los vectores de referencia de cada persona del catálogo."""
        for subcarpeta in sorted(p for p in carpeta.iterdir() if p.is_dir()):
            vectores = []
            for imagen in sorted(subcarpeta.iterdir()):
                if imagen.suffix.lower() not in EXTENSIONES_IMAGEN:
                    continue
                vector = self._vector_de_archivo(imagen)
                if vector is not None:
                    vectores.append(vector)
                else:
                    self._log(f"  Sin rostro utilizable en {imagen.name}, se ignora.")
            if vectores:
                self.catalogo[subcarpeta.name] = vectores
                self._log(f"  {subcarpeta.name}: {len(vectores)} foto(s) de referencia.")

    def _vector_de_archivo(self, ruta: Path):
        """Extrae el vector del rostro más grande de una foto de referencia."""
        imagen = cv2.imread(str(ruta))
        if imagen is None:
            return None
        filas = self.detector.filas_yunet(imagen)
        if not filas:
            return None
        # La cara más grande es casi siempre el sujeto de la foto de referencia.
        fila = max(filas, key=lambda f: float(f[2]) * float(f[3]))
        return self.vector(imagen, fila)

    def vector(self, imagen, fila):
        """Alinea el rostro descrito por ``fila`` y devuelve su vector SFace."""
        alineado = self._sface.alignCrop(imagen, fila)
        return self._sface.feature(alineado)

    def identificar(self, rostro: RostroDetectado) -> Identidad:
        """Compara un rostro detectado contra todo el catálogo.

        Returns:
            La :class:`Identidad` con mayor similitud, o ``desconocido`` si
            ninguna supera el umbral.
        """
        if rostro.fila is None or rostro.region is None:
            return Identidad(DESCONOCIDO, 0.0)
        try:
            vector = self.vector(rostro.region, rostro.fila)
        except cv2.error:  # pragma: no cover - rostro en el borde del recorte
            return Identidad(DESCONOCIDO, 0.0)

        mejor_nombre = DESCONOCIDO
        # La similitud coseno va de -1 a 1: arrancar en 0 ocultaría los negativos
        # y haría creer que la comparación falló.
        mejor_score = -1.0
        for nombre, referencias in self.catalogo.items():
            for referencia in referencias:
                score = float(
                    self._sface.match(vector, referencia, cv2.FaceRecognizerSF_FR_COSINE)
                )
                if score > mejor_score:
                    mejor_score = score
                    mejor_nombre = nombre
        if mejor_score < self.umbral:
            return Identidad(DESCONOCIDO, mejor_score)
        return Identidad(mejor_nombre, mejor_score)


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
class ConfiguracionAnalisis:
    """Parámetros con los que se ejecuta el análisis."""

    zona_puerta: tuple[int, int, int, int]
    fps_analisis: float = 1.0
    tolerancia_segundos: float = 3.0
    modelo: str = "yolov8n"
    confianza: float = 0.35
    acelerador: str = "auto"
    decodificacion_hardware: bool = False
    filtro_movimiento: bool = True
    umbral_movimiento: float = UMBRAL_MOVIMIENTO
    criterio_zona: str = "pies"
    min_solape: float = 0.25
    registrar_general: bool = True
    detectar_rostros: bool = False
    backend_rostros: str = "yunet"
    guardar_recortes_rostros: bool = True
    identificar_rostros: bool = False
    carpeta_personas: str = ""
    umbral_identificacion: float = UMBRAL_SFACE
    #: Seguir a cada persona entre frames (ByteTrack) para contar personas
    #: distintas y saber si entran o salen. Cuesta poco; solo se desactiva para
    #: comparar resultados con versiones anteriores.
    usar_tracking: bool = True

    def validar(self) -> None:
        """Verifica que los parámetros sean utilizables.

        Raises:
            ValueError: si algún parámetro es inválido.
        """
        x1, y1, x2, y2 = self.zona_puerta
        if x2 <= x1 or y2 <= y1:
            raise ValueError("La zona de la puerta no es válida (ancho o alto cero).")
        if self.fps_analisis <= 0:
            raise ValueError("Los frames por segundo deben ser mayores que cero.")
        if self.tolerancia_segundos < 0:
            raise ValueError("La tolerancia no puede ser negativa.")
        if self.criterio_zona not in CRITERIOS_ZONA:
            raise ValueError(
                f"Criterio de zona inválido: {self.criterio_zona}. "
                f"Opciones: {', '.join(CRITERIOS_ZONA)}"
            )
        if not 0 < self.min_solape <= 1:
            raise ValueError("El solape mínimo debe estar entre 0 y 1.")
        if self.acelerador not in ACELERADORES:
            raise ValueError(f"Acelerador inválido: {self.acelerador}")
        if not 0 <= self.umbral_movimiento < 1:
            raise ValueError("El umbral de movimiento debe estar entre 0 y 1.")


def fraccion_dentro(
    caja: tuple[float, float, float, float],
    zona: tuple[int, int, int, int],
) -> float:
    """Fracción del área de ``caja`` que cae dentro de ``zona`` (0 a 1)."""
    ancho = caja[2] - caja[0]
    alto = caja[3] - caja[1]
    if ancho <= 0 or alto <= 0:
        return 0.0
    return area_solapada(caja, zona) / float(ancho * alto)


def punto_pies(caja: tuple[float, float, float, float]) -> tuple[float, float]:
    """Punto de apoyo de una persona: el centro del borde inferior de su caja."""
    x1, _, x2, y2 = caja
    return ((x1 + x2) / 2.0, y2)


def punto_centro(caja: tuple[float, float, float, float]) -> tuple[float, float]:
    """Centro geométrico de la caja."""
    x1, y1, x2, y2 = caja
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def punto_dentro(
    punto: tuple[float, float], zona: tuple[int, int, int, int]
) -> bool:
    """Indica si un punto cae dentro del rectángulo ``zona``."""
    x, y = punto
    x1, y1, x2, y2 = zona
    return x1 <= x <= x2 and y1 <= y <= y2


def persona_en_zona(
    caja: tuple[float, float, float, float],
    zona: tuple[int, int, int, int],
    criterio: str = "pies",
    min_solape: float = 0.25,
) -> bool:
    """Decide si una persona detectada cuenta como "dentro de la zona".

    El criterio importa mucho en pasillos: por perspectiva, la caja de alguien
    que camina por el centro puede rozar la esquina de una puerta lejana sin que
    la persona esté ni cerca de ella. Por eso el solape simple da falsos
    positivos y no es el criterio por defecto.

    Args:
        caja: caja ``(x1, y1, x2, y2)`` de la persona.
        zona: zona de la puerta.
        criterio:
            ``pies``: el punto de apoyo (centro del borde inferior) cae en la
                zona. Es el más fiable cuando la zona se dibuja sobre el suelo o
                el vano por el que se pisa.
            ``centro``: el centro de la caja cae en la zona.
            ``solape``: al menos ``min_solape`` del área de la persona está
                dentro de la zona.
        min_solape: fracción mínima para el criterio ``solape``.

    Returns:
        ``True`` si la persona cuenta como presente en la zona.
    """
    if criterio == "pies":
        return punto_dentro(punto_pies(caja), zona)
    if criterio == "centro":
        return punto_dentro(punto_centro(caja), zona)
    if criterio == "solape":
        return fraccion_dentro(caja, zona) >= min_solape
    raise ValueError(f"Criterio de zona desconocido: {criterio}")


@dataclass
class _EventoCerrado:
    """Un evento terminado junto a las imágenes que hay que guardar de él."""

    evento: Evento
    frame_miniatura: Any = None
    caja_miniatura: Optional[tuple[int, int, int, int]] = None
    frame_rostros: Any = None
    cajas_rostros: list[tuple[int, int, int, int]] = field(default_factory=list)
    #: Identificadores de seguimiento de las personas vistas en el evento.
    ids: set[int] = field(default_factory=set)


class _SeguidorEventos:
    """Agrupa detecciones sueltas de un mismo tipo en intervalos de tiempo.

    Se le pasa cada frame analizado; devuelve un :class:`_EventoCerrado` en el
    momento en que un intervalo termina, es decir cuando pasan más de
    ``tolerancia`` segundos sin ninguna detección.
    """

    def __init__(self, archivo: str, tipo: str, tolerancia: float) -> None:
        """Crea un seguidor para un tipo de evento de un video concreto."""
        self.archivo = archivo
        self.tipo = tipo
        self.tolerancia = tolerancia
        self._evento: Optional[Evento] = None
        self._ultimo_positivo = 0.0
        self._frame_miniatura = None
        self._caja_miniatura: Optional[tuple[int, int, int, int]] = None
        self._frame_rostros = None
        self._cajas_rostros: list[tuple[int, int, int, int]] = []
        self._nombres: set[str] = set()
        self._ids: set[int] = set()
        self._max_simultaneas = 0

    @property
    def activo(self) -> bool:
        """``True`` si hay un evento abierto pendiente de cerrar."""
        return self._evento is not None

    def actualizar(
        self,
        segundo: float,
        caja: Optional[tuple[int, int, int, int]],
        frame,
        rostros: list[tuple[int, int, int, int]],
        nombres: set[str],
        ids: Optional[set[int]] = None,
        simultaneas: int = 1,
    ) -> Optional[_EventoCerrado]:
        """Incorpora el resultado de un frame.

        Args:
            segundo: instante del frame dentro del video.
            caja: caja de la persona principal detectada, o ``None`` si no hay
                ninguna.
            frame: imagen del frame (solo se usa si hay detección).
            rostros: cajas de los rostros encontrados en esa persona.
            nombres: nombres del catálogo reconocidos en ese frame.
            ids: identificadores de seguimiento de todas las personas del frame
                que cuentan para este seguidor.
            simultaneas: cuántas personas había en el frame para este seguidor;
                sirve de respaldo para contar cuando no hay seguimiento.

        Returns:
            El evento que se acaba de cerrar, o ``None`` si no se cerró ninguno.
        """
        if caja is not None:
            if self._evento is None:
                self._evento = Evento(
                    archivo=self.archivo, inicio=segundo, fin=segundo, tipo=self.tipo
                )
                self._frame_miniatura = frame.copy()
                self._caja_miniatura = caja
            else:
                self._evento.fin = segundo
            self._ultimo_positivo = segundo
            self._nombres |= nombres
            if ids:
                self._ids |= ids
            self._max_simultaneas = max(self._max_simultaneas, simultaneas)

            # Se conserva el frame con más rostros: es el mejor para los recortes.
            if len(rostros) > self._evento.rostros:
                self._evento.rostros = len(rostros)
                self._frame_rostros = frame.copy()
                self._cajas_rostros = rostros
            return None

        if self._evento is not None and segundo - self._ultimo_positivo > self.tolerancia:
            return self.cerrar()
        return None

    def cerrar(self) -> Optional[_EventoCerrado]:
        """Cierra el evento en curso, si lo hay, y reinicia el seguidor."""
        if self._evento is None:
            return None

        self._evento.personas = ", ".join(sorted(self._nombres))
        # Con seguimiento, las pistas distintas; sin él, o si el seguimiento
        # perdió a alguien, al menos las que llegaron a verse a la vez.
        self._evento.n_personas = max(len(self._ids), self._max_simultaneas)
        cerrado = _EventoCerrado(
            evento=self._evento,
            frame_miniatura=self._frame_miniatura,
            caja_miniatura=self._caja_miniatura,
            frame_rostros=self._frame_rostros,
            cajas_rostros=self._cajas_rostros,
            ids=self._ids,
        )
        self._evento = None
        self._frame_miniatura = None
        self._caja_miniatura = None
        self._frame_rostros = None
        self._cajas_rostros = []
        self._nombres = set()
        self._ids = set()
        self._max_simultaneas = 0
        return cerrado


# ------------------------------------------------------------------ seguimiento


@dataclass(frozen=True)
class Deteccion:
    """Una persona detectada en un frame, con su identificador de seguimiento."""

    caja: tuple[int, int, int, int]
    #: ``None`` si no hay seguimiento o la pista aún no se ha confirmado.
    id: Optional[int] = None

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.caja
        return max(0, x2 - x1) * max(0, y2 - y1)


class RastreadorPersonas:
    """Asigna un identificador estable a cada persona entre frames (ByteTrack).

    Se usa el tracker de ultralytics directamente, en vez de ``modelo.track()``,
    por dos motivos: ``track()`` descarta las detecciones cuya pista aún no se
    ha confirmado (una persona vista en un solo frame desaparecería), y no deja
    reiniciar el seguimiento al cambiar de video de forma explícita. Aquí todas
    las detecciones se conservan; las que todavía no tienen pista llevan
    ``id=None``.
    """

    #: Frames que ByteTrack mantiene viva una pista perdida, como mínimo.
    BUFFER_MINIMO = 30

    def __init__(self, fps_analisis: float, tolerancia_segundos: float) -> None:
        """Crea el tracker.

        Args:
            fps_analisis: frames analizados por segundo de video.
            tolerancia_segundos: tolerancia de agrupación de eventos. Una pista
                perdida se mantiene al menos ese tiempo para que la misma
                persona no reciba dos identificadores si la detección falla
                unos frames.
        """
        from types import SimpleNamespace

        from ultralytics.trackers import BYTETracker  # import perezoso

        # ByteTrack cuenta en frames analizados, no en segundos.
        self.buffer = max(
            self.BUFFER_MINIMO, math.ceil(tolerancia_segundos * fps_analisis) + 1
        )
        argumentos = SimpleNamespace(
            track_high_thresh=0.25,
            track_low_thresh=0.1,
            new_track_thresh=0.25,
            track_buffer=self.buffer,
            match_thresh=0.8,
            fuse_score=True,
        )
        self._tracker = BYTETracker(argumentos)

    def reiniciar(self) -> None:
        """Olvida todas las pistas; se llama al empezar cada video."""
        self._tracker.reset()

    def asignar_ids(self, cajas) -> list[Optional[int]]:
        """Devuelve el identificador de cada detección, alineado con ``cajas``.

        Args:
            cajas: objeto ``Boxes`` de ultralytics ya pasado a numpy. Puede estar
                vacío: entonces solo se avanza el tiempo del tracker para que
                las pistas perdidas envejezcan.
        """
        ids: list[Optional[int]] = [None] * len(cajas)
        pistas = self._tracker.update(cajas)
        # Cada fila: x1, y1, x2, y2, id, confianza, clase, índice en ``cajas``.
        for fila in pistas:
            ids[int(fila[-1])] = int(fila[4])
        return ids


#: Etiquetas de dirección, por (estaba dentro al principio, estaba dentro al final).
DIRECCIONES = {
    (False, True): "entra",
    (True, False): "sale",
    (False, False): "cruza",
    (True, True): "permanece",
}
_PLURALES = {"entra": "entran", "sale": "salen", "cruza": "cruzan", "permanece": "permanecen"}


@dataclass
class _Trayectoria:
    """Lo que se sabe del recorrido de una pista respecto a la zona."""

    primero_dentro: bool
    ultimo_dentro: bool
    estuvo_dentro: bool


class _Trayectorias:
    """Recorrido de cada pista respecto a la zona de la puerta, en un video.

    Con esto se decide la dirección de cada persona: si la primera vez que se
    la vio estaba fuera de la zona y la última dentro, "entra" (llegó a la
    puerta y desapareció por ella); al revés, "sale"; fuera-fuera, "cruza" la
    zona de paso; dentro-dentro, "permanece".
    """

    def __init__(self) -> None:
        self._pistas: dict[int, _Trayectoria] = {}

    def observar(self, id_pista: int, dentro: bool) -> None:
        """Anota dónde está la pista en este frame."""
        actual = self._pistas.get(id_pista)
        if actual is None:
            self._pistas[id_pista] = _Trayectoria(dentro, dentro, dentro)
        else:
            actual.ultimo_dentro = dentro
            actual.estuvo_dentro = actual.estuvo_dentro or dentro

    def direccion(self, id_pista: int) -> Optional[str]:
        """Dirección de una pista, o ``None`` si nunca tocó la zona."""
        pista = self._pistas.get(id_pista)
        if pista is None or not pista.estuvo_dentro:
            return None
        return DIRECCIONES[(pista.primero_dentro, pista.ultimo_dentro)]

    def resumen(self, ids: Iterable[int]) -> str:
        """Resume las direcciones de varias pistas: ``"2 entran, 1 sale"``."""
        conteo: dict[str, int] = {}
        for id_pista in ids:
            etiqueta = self.direccion(id_pista)
            if etiqueta is not None:
                conteo[etiqueta] = conteo.get(etiqueta, 0) + 1
        partes = []
        for etiqueta in DIRECCIONES.values():  # orden fijo y legible
            n = conteo.get(etiqueta, 0)
            if n == 1:
                partes.append(f"1 {etiqueta}")
            elif n > 1:
                partes.append(f"{n} {_PLURALES[etiqueta]}")
        return ", ".join(partes)


#: Frames muestreados que caben en la cola entre el lector y el analizador.
#: Cada uno ocupa ~6 MB en 1080p, así que 8 son unos 50 MB de margen.
TAM_COLA_LECTURA = 8


@dataclass
class _FrameMuestreado:
    """Un frame que toca analizar, con su posición en el video."""

    indice: int
    segundo: float
    imagen: Any


class _LectorFrames:
    """Decodifica un video en un hilo aparte y entrega los frames a analizar.

    Decodificar e inferir son trabajos distintos —CPU el primero, acelerador el
    segundo—, así que hacerlos en serie desaprovecha ambos. Este lector los
    solapa: mientras el hilo principal infiere sobre un frame, el lector ya está
    decodificando los siguientes. El tiempo total pasa de ser la suma de ambos a
    ser aproximadamente el mayor de los dos.

    La cola es pequeña a propósito: si el analizador se retrasa, el lector se
    frena en lugar de acumular cientos de megas de frames en memoria.

    Se usa como gestor de contexto, que garantiza parar el hilo antes de que el
    llamante libere la captura::

        with _LectorFrames(captura, paso, fps, cancelar) as lector:
            for muestra in lector:
                ...
    """

    def __init__(
        self,
        captura: cv2.VideoCapture,
        paso: int,
        fps: float,
        cancelar: threading.Event,
        tam_cola: int = TAM_COLA_LECTURA,
    ) -> None:
        """Prepara el lector.

        Args:
            captura: captura ya abierta; el lector no la cierra.
            paso: se entrega uno de cada ``paso`` frames.
            fps: fotogramas por segundo del video, para calcular el instante.
            cancelar: evento compartido para abortar el análisis.
            tam_cola: frames decodificados por adelantado como máximo.
        """
        self._captura = captura
        self._paso = max(1, paso)
        self._fps = fps if fps > 0 else 25.0
        self._cancelar = cancelar
        self._cola: "queue.Queue[Optional[_FrameMuestreado]]" = queue.Queue(
            maxsize=tam_cola
        )
        self._parar = threading.Event()
        self._hilo: Optional[threading.Thread] = None
        self.error: Optional[BaseException] = None

    # ------------------------------------------------------- gestor de contexto

    def __enter__(self) -> "_LectorFrames":
        """Arranca el hilo de lectura."""
        self._hilo = threading.Thread(
            target=self._producir, name="lector-frames", daemon=True
        )
        self._hilo.start()
        return self

    def __exit__(self, *_excepcion) -> None:
        """Detiene el hilo de lectura y espera a que termine."""
        self._parar.set()
        # Vaciar la cola desbloquea al lector si estaba esperando hueco.
        while True:
            try:
                self._cola.get_nowait()
            except queue.Empty:
                break
        if self._hilo is not None:
            self._hilo.join(timeout=5.0)
            self._hilo = None

    # ----------------------------------------------------------------- productor

    def _producir(self) -> None:
        """Recorre el video y encola los frames que toca analizar."""
        indice = 0
        try:
            while not self._debe_parar():
                # grab() avanza sin convertir el frame; solo se decodifica del
                # todo (retrieve) el que se va a analizar.
                if not self._captura.grab():
                    break
                if indice % self._paso == 0:
                    ok, imagen = self._captura.retrieve()
                    if not ok:
                        break
                    muestra = _FrameMuestreado(
                        indice=indice, segundo=indice / self._fps, imagen=imagen
                    )
                    if not self._encolar(muestra):
                        break
                indice += 1
        except BaseException as exc:  # noqa: BLE001 - se reenvía al consumidor
            self.error = exc
        finally:
            self._encolar(None)  # centinela de fin

    def _debe_parar(self) -> bool:
        """Indica si hay que abandonar la lectura."""
        return self._parar.is_set() or self._cancelar.is_set()

    def _encolar(self, muestra: Optional[_FrameMuestreado]) -> bool:
        """Encola un elemento sin bloquearse para siempre si hay que parar.

        Returns:
            ``True`` si se encoló, ``False`` si se abandonó la lectura.
        """
        while not self._debe_parar():
            try:
                self._cola.put(muestra, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    # ---------------------------------------------------------------- consumidor

    def __iter__(self) -> Iterator[_FrameMuestreado]:
        """Entrega los frames muestreados en orden, hasta agotar el video."""
        while True:
            try:
                muestra = self._cola.get(timeout=0.5)
            except queue.Empty:
                # Sin datos y sin lector vivo: no va a llegar nada más.
                if self._hilo is None or not self._hilo.is_alive():
                    return
                continue
            if muestra is None:
                return
            yield muestra


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


class AnalizadorPuerta:
    """Analiza videos buscando personas dentro de una zona (la puerta).

    El modelo YOLO se carga una sola vez y se reutiliza para todos los videos.
    """

    def __init__(
        self,
        config: ConfiguracionAnalisis,
        cancelar: Optional[threading.Event] = None,
        on_progreso: Optional[CallbackProgreso] = None,
        on_log: Optional[CallbackLog] = None,
        on_evento: Optional[CallbackEvento] = None,
    ) -> None:
        """Crea el analizador.

        Args:
            config: parámetros del análisis (zona, fps, modelo, tolerancia).
            cancelar: evento que, al activarse, detiene el análisis.
            on_progreso: recibe ``(nombre_video, porcentaje_video, porcentaje_total)``.
            on_log: recibe mensajes de texto para mostrar en el log.
            on_evento: recibe cada :class:`Evento` en cuanto se cierra.
        """
        config.validar()
        self.config = config
        self.cancelar = cancelar or threading.Event()
        self._on_progreso = on_progreso
        self._on_log = on_log
        self._on_evento = on_evento
        self._modelo = None
        self.acelerador = resolver_acelerador(config.acelerador)
        self.dispositivo = dispositivo_de_prediccion(self.acelerador)
        if config.acelerador not in ("auto", self.acelerador):
            self._log(
                f"El acelerador '{config.acelerador}' no está disponible; "
                f"se usará '{self.acelerador}'."
            )

        # Un fallo en rostros nunca debe impedir el análisis de personas: se
        # avisa por el log y se sigue sin esa función.
        self.movimiento: Optional[DetectorMovimiento] = (
            DetectorMovimiento() if config.filtro_movimiento else None
        )

        self.rastreador: Optional[RastreadorPersonas] = None
        if config.usar_tracking:
            try:
                self.rastreador = RastreadorPersonas(
                    config.fps_analisis, config.tolerancia_segundos
                )
            except ImportError as exc:
                self._log(f"Seguimiento de personas desactivado: {exc}")

        self.detector_rostros: Optional[DetectorRostros] = None
        if config.detectar_rostros:
            try:
                self.detector_rostros = DetectorRostros(config.backend_rostros)
            except (FileNotFoundError, ValueError) as exc:
                self._log(f"Detección de rostros desactivada: {exc}")

        self.identificador: Optional[IdentificadorRostros] = None
        if config.identificar_rostros and self.detector_rostros is not None:
            try:
                self._log("Cargando catálogo de personas conocidas...")
                self.identificador = IdentificadorRostros(
                    carpeta_personas=config.carpeta_personas,
                    detector=self.detector_rostros,
                    umbral=config.umbral_identificacion,
                    on_log=self._log,
                )
                self._log(
                    f"Catálogo listo: {len(self.identificador.catalogo)} persona(s)."
                )
            except (FileNotFoundError, ValueError) as exc:
                self._log(f"Identificación desactivada: {exc}")

    # ------------------------------------------------------------------ utilidades

    def _log(self, mensaje: str) -> None:
        """Envía un mensaje al callback de log si existe."""
        if self._on_log is not None:
            self._on_log(mensaje)

    def _progreso(self, nombre: str, pct_video: float, pct_total: float) -> None:
        """Envía el progreso al callback si existe."""
        if self._on_progreso is not None:
            self._on_progreso(nombre, pct_video, pct_total)

    def _comprobar_cancelacion(self) -> None:
        """Lanza :class:`AnalisisCancelado` si el usuario pidió cancelar."""
        if self.cancelar.is_set():
            raise AnalisisCancelado()

    def cargar_modelo(self):
        """Carga (una sola vez) el modelo YOLO según el acelerador elegido.

        Con ``openvino-gpu`` se carga la versión OpenVINO del modelo, que se
        exporta automáticamente la primera vez.
        """
        if self._modelo is not None:
            return self._modelo

        from ultralytics import YOLO  # import perezoso: tarda en cargar

        if self.acelerador == "openvino-gpu":
            carpeta = exportar_a_openvino(self.config.modelo, self._log)
            self._log(f"Cargando {carpeta.name} en GPU Intel (OpenVINO)...")
            self._modelo = YOLO(str(carpeta), task="detect")
        else:
            nombre = self.config.modelo
            if not nombre.endswith(".pt"):
                nombre = f"{nombre}.pt"
            self._log(f"Cargando modelo {nombre} en {self.acelerador.upper()}...")
            self._modelo = YOLO(nombre)
        return self._modelo

    # -------------------------------------------------------------------- análisis

    def analizar_video(
        self,
        video: str | Path,
        carpeta_miniaturas: Optional[Path] = None,
        peso_progreso: tuple[float, float] = (0.0, 1.0),
    ) -> ResultadoVideo:
        """Analiza un video y devuelve los eventos detectados.

        Se llevan dos registros en paralelo: las personas que cumplen el criterio
        de la zona (``TIPO_ZONA``) y, si está activado, todas las personas vistas
        en cualquier punto del frame (``TIPO_GENERAL``).

        Args:
            video: ruta del archivo de video.
            carpeta_miniaturas: carpeta donde guardar la miniatura de cada evento.
            peso_progreso: ``(inicio, tamaño)`` del tramo de progreso global que
                ocupa este video, ambos entre 0 y 1.

        Returns:
            Un :class:`ResultadoVideo`; si el video no se pudo abrir, ``error``
            contiene el motivo y ``eventos`` queda vacío.
        """
        ruta = Path(video)
        resultado = ResultadoVideo(archivo=ruta.name)
        captura = abrir_captura(ruta, self.config.decodificacion_hardware)

        if not captura.isOpened():
            captura.release()
            resultado.error = f"No se pudo abrir el video: {ruta.name}"
            return resultado

        try:
            info = info_video(captura)
            if self.movimiento is not None:
                self.movimiento.reiniciar()
            if self.rastreador is not None:
                self.rastreador.reiniciar()
            trayectorias = _Trayectorias()
            paso = max(1, int(round(info.fps / self.config.fps_analisis)))
            modelo = self.cargar_modelo()
            zona = self.config.zona_puerta
            base_progreso, ancho_progreso = peso_progreso

            seguidores = [
                _SeguidorEventos(ruta.name, TIPO_ZONA, self.config.tolerancia_segundos)
            ]
            if self.config.registrar_general:
                seguidores.append(
                    _SeguidorEventos(
                        ruta.name, TIPO_GENERAL, self.config.tolerancia_segundos
                    )
                )

            # La decodificación va en un hilo aparte y se solapa con la
            # inferencia, que es lo que consume el tiempo aquí.
            with _LectorFrames(captura, paso, info.fps, self.cancelar) as lector:
                for muestra in lector:
                    self._comprobar_cancelacion()
                    self._procesar_frame(
                        muestra.imagen,
                        muestra.segundo,
                        zona,
                        modelo,
                        seguidores,
                        resultado,
                        carpeta_miniaturas,
                        trayectorias,
                    )

                    pct_video = (
                        min(100.0, muestra.indice / info.total_frames * 100.0)
                        if info.total_frames > 0
                        else 0.0
                    )
                    pct_total = (
                        base_progreso + ancho_progreso * pct_video / 100.0
                    ) * 100.0
                    self._progreso(ruta.name, pct_video, pct_total)

            if lector.error is not None:
                self._log(f"  Aviso al leer {ruta.name}: {lector.error}")

            # Los eventos que siguen abiertos al acabar el video se cierran igual.
            for seguidor in seguidores:
                self._registrar(
                    seguidor.cerrar(), resultado, carpeta_miniaturas, trayectorias
                )

            self._progreso(ruta.name, 100.0, (base_progreso + ancho_progreso) * 100.0)
            return resultado
        finally:
            captura.release()

    def _procesar_frame(
        self,
        frame,
        segundo: float,
        zona: tuple[int, int, int, int],
        modelo,
        seguidores: list["_SeguidorEventos"],
        resultado: ResultadoVideo,
        carpeta_miniaturas: Optional[Path],
        trayectorias: _Trayectorias,
    ) -> None:
        """Detecta personas en un frame y alimenta cada seguidor de eventos.

        Si el filtro de movimiento está activo y el frame no ha cambiado, se
        salta la inferencia: es lo que evita ejecutar el modelo sobre horas de
        pasillo vacío. Nunca se salta mientras haya un evento abierto, para no
        perder a alguien que se queda quieto delante de la puerta.
        """
        if self._sin_movimiento(frame, seguidores):
            resultado.frames_omitidos += 1
            for seguidor in seguidores:
                self._registrar(
                    seguidor.actualizar(segundo, None, None, [], set()),
                    resultado,
                    carpeta_miniaturas,
                )
            return

        resultado.frames_analizados += 1
        personas = self._detectar_personas(modelo, frame)
        en_zona: list[Deteccion] = []
        for deteccion in personas:
            dentro = persona_en_zona(
                deteccion.caja, zona, self.config.criterio_zona, self.config.min_solape
            )
            if dentro:
                en_zona.append(deteccion)
            if deteccion.id is not None:
                trayectorias.observar(deteccion.id, dentro)

        for seguidor in seguidores:
            candidatas = en_zona if seguidor.tipo == TIPO_ZONA else personas
            if not candidatas:
                cerrado = seguidor.actualizar(segundo, None, None, [], set())
            else:
                # La persona más grande es la más cercana a la cámara y la que
                # mejor se ve en la miniatura.
                principal = max(candidatas, key=lambda d: d.area)
                rostros, nombres = self._analizar_rostros(frame, principal.caja)
                ids = {d.id for d in candidatas if d.id is not None}
                cerrado = seguidor.actualizar(
                    segundo,
                    principal.caja,
                    frame,
                    rostros,
                    nombres,
                    ids=ids,
                    simultaneas=len(candidatas),
                )
            self._registrar(cerrado, resultado, carpeta_miniaturas, trayectorias)

    def _sin_movimiento(
        self, frame, seguidores: list["_SeguidorEventos"]
    ) -> bool:
        """Indica si el frame puede saltarse por no haber cambiado nada.

        Mientras algún seguidor tenga un evento abierto se devuelve ``False``
        aunque la imagen esté quieta: una persona parada apenas genera cambios,
        y cortarle el evento la partiría en dos o la perdería.
        """
        if self.movimiento is None:
            return False
        cambio = self.movimiento.fraccion_cambiada(frame)
        if any(seguidor.activo for seguidor in seguidores):
            return False
        return cambio < self.config.umbral_movimiento

    def _analizar_rostros(
        self, frame, caja: tuple[int, int, int, int]
    ) -> tuple[list[tuple[int, int, int, int]], set[str]]:
        """Detecta e identifica los rostros dentro de la caja de una persona.

        Returns:
            Las cajas de los rostros y los nombres del catálogo reconocidos.
        """
        if self.detector_rostros is None:
            return [], set()

        rostros = self.detector_rostros.detectar(frame, caja)
        nombres: set[str] = set()
        if self.identificador is not None:
            for rostro in rostros:
                identidad = self.identificador.identificar(rostro)
                if identidad.conocida:
                    nombres.add(identidad.nombre)
        return [r.caja for r in rostros], nombres

    def _detectar_personas(self, modelo, frame) -> list[Deteccion]:
        """Devuelve todas las personas detectadas en el frame, con su pista."""
        predicciones = modelo.predict(
            frame,
            classes=[CLASE_PERSONA],
            conf=self.config.confianza,
            device=self.dispositivo,
            verbose=False,
        )
        detecciones: list[Deteccion] = []
        for prediccion in predicciones:
            cajas_pred = getattr(prediccion, "boxes", None)
            if cajas_pred is None:
                continue
            cajas_np = cajas_pred.cpu().numpy()
            ids: list[Optional[int]] = (
                self.rastreador.asignar_ids(cajas_np)
                if self.rastreador is not None
                else [None] * len(cajas_np)
            )
            for valores, id_pista in zip(cajas_np.xyxy, ids):
                x1, y1, x2, y2 = (int(v) for v in valores[:4])
                detecciones.append(Deteccion((x1, y1, x2, y2), id_pista))
        return detecciones

    def _registrar(
        self,
        cerrado: Optional["_EventoCerrado"],
        resultado: ResultadoVideo,
        carpeta_miniaturas: Optional[Path],
        trayectorias: Optional[_Trayectorias] = None,
    ) -> None:
        """Guarda las imágenes de un evento recién cerrado y lo notifica."""
        if cerrado is None:
            return
        evento = cerrado.evento
        if trayectorias is not None and cerrado.ids:
            evento.direccion = trayectorias.resumen(cerrado.ids)

        if carpeta_miniaturas is not None and cerrado.frame_miniatura is not None:
            evento.miniatura = self._guardar_miniatura(
                evento,
                cerrado.frame_miniatura,
                cerrado.caja_miniatura,
                carpeta_miniaturas / evento.tipo,
            )
        if (
            self.config.guardar_recortes_rostros
            and carpeta_miniaturas is not None
            and cerrado.frame_rostros is not None
            and cerrado.cajas_rostros
        ):
            self._guardar_recortes_rostros(
                evento,
                cerrado.frame_rostros,
                cerrado.cajas_rostros,
                carpeta_miniaturas.parent / "rostros" / evento.tipo,
            )

        resultado.eventos.append(evento)
        self._log(f"  Evento: {evento}")
        if self._on_evento is not None:
            self._on_evento(evento)

    def _guardar_miniatura(
        self,
        evento: Evento,
        frame,
        caja: Optional[tuple[int, int, int, int]],
        carpeta: Path,
    ) -> str:
        """Guarda un JPG del primer frame del evento con la zona y la caja dibujadas.

        Returns:
            La ruta del archivo generado, o cadena vacía si no se pudo guardar.
        """
        carpeta.mkdir(parents=True, exist_ok=True)
        imagen = frame.copy()
        zx1, zy1, zx2, zy2 = self.config.zona_puerta
        cv2.rectangle(imagen, (zx1, zy1), (zx2, zy2), (255, 0, 0), 2)
        cv2.putText(
            imagen,
            "puerta",
            (zx1, max(15, zy1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )
        if caja is not None:
            cx1, cy1, cx2, cy2 = caja
            cv2.rectangle(imagen, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)
            cv2.putText(
                imagen,
                "persona",
                (cx1, max(15, cy1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        marca = formatear_tiempo(evento.inicio).replace(":", "-")
        destino = carpeta / f"{Path(evento.archivo).stem}_{marca}.jpg"
        return str(destino) if cv2.imwrite(str(destino), imagen) else ""

    def _guardar_recortes_rostros(
        self,
        evento: Evento,
        frame,
        cajas: list[tuple[int, int, int, int]],
        carpeta: Path,
    ) -> list[str]:
        """Guarda un recorte JPG por rostro más el frame completo señalizado.

        Se usa el frame del evento con más rostros, que no tiene por qué ser el
        mismo de la miniatura.

        Returns:
            Las rutas de los recortes generados.
        """
        carpeta.mkdir(parents=True, exist_ok=True)
        alto, ancho = frame.shape[:2]
        marca = formatear_tiempo(evento.inicio).replace(":", "-")
        base = f"{Path(evento.archivo).stem}_{marca}"

        rutas: list[str] = []
        señalizado = frame.copy()
        for i, (x1, y1, x2, y2) in enumerate(cajas, start=1):
            cv2.rectangle(señalizado, (x1, y1), (x2, y2), (0, 200, 255), 2)
            # Margen del 30% alrededor del rostro: sirve para identificarlo mejor.
            margen_x = int((x2 - x1) * 0.3)
            margen_y = int((y2 - y1) * 0.3)
            recorte = frame[
                max(0, y1 - margen_y) : min(alto, y2 + margen_y),
                max(0, x1 - margen_x) : min(ancho, x2 + margen_x),
            ]
            if recorte.size == 0:
                continue
            destino = carpeta / f"{base}_rostro{i}.jpg"
            if cv2.imwrite(str(destino), recorte):
                rutas.append(str(destino))

        cv2.imwrite(str(carpeta / f"{base}_frame.jpg"), señalizado)
        return rutas

    def analizar_videos(
        self,
        videos: Iterable[str | Path],
        carpeta_salida: str | Path,
        incremental: bool = False,
    ) -> list[ResultadoVideo]:
        """Analiza varios videos y escribe ``eventos.csv`` y las miniaturas.

        Los videos que no se puedan abrir se saltan: su ``ResultadoVideo`` lleva
        el mensaje en ``error`` y el análisis continúa con el siguiente.

        Args:
            videos: rutas de los videos a analizar.
            carpeta_salida: carpeta donde se escriben los resultados.
            incremental: reutilizar, desde el manifiesto de la carpeta de
                salida, los videos ya analizados con esta misma configuración
                y que no hayan cambiado. Cada video analizado se apunta en el
                acto, así que una cancelación no pierde lo hecho hasta entonces.

        Returns:
            La lista de resultados, uno por video (analizado o reutilizado).
        """
        lista = [Path(v) for v in videos]
        salida = Path(carpeta_salida)
        salida.mkdir(parents=True, exist_ok=True)
        miniaturas = salida / "miniaturas"

        registro_previo: Optional[manifiesto.Manifiesto] = None
        huella = manifiesto.huella_config(asdict(self.config))
        if incremental:
            registro_previo = manifiesto.Manifiesto(
                salida / manifiesto.NOMBRE_MANIFIESTO
            )

        self._log(f"Acelerador: {self.acelerador} (device={self.dispositivo})")
        self._log(f"Videos a analizar: {len(lista)}")

        resultados: list[ResultadoVideo] = []
        reutilizados = 0
        try:
            for i, video in enumerate(lista):
                self._comprobar_cancelacion()
                peso = (i / len(lista), 1 / len(lista))

                previo = (
                    registro_previo.buscar(video, huella)
                    if registro_previo is not None
                    else None
                )
                if previo is not None:
                    resultado = ResultadoVideo.desde_dict(previo)
                    reutilizados += 1
                    self._log(
                        f"[{i + 1}/{len(lista)}] {video.name} — ya analizado, "
                        f"se reutiliza ({len(resultado.eventos)} eventos)"
                    )
                    for evento in resultado.eventos:
                        if self._on_evento is not None:
                            self._on_evento(evento)
                    self._progreso(video.name, 100.0, (peso[0] + peso[1]) * 100.0)
                    resultados.append(resultado)
                    continue

                self._log(f"[{i + 1}/{len(lista)}] {video.name}")
                resultado = self.analizar_video(video, miniaturas, peso)
                if resultado.error:
                    self._log(f"  ERROR: {resultado.error}")
                elif resultado.frames_omitidos:
                    self._log(
                        f"  {resultado.frames_analizados} frames analizados, "
                        f"{resultado.frames_omitidos} omitidos sin movimiento "
                        f"({resultado.ahorro_movimiento:.0%} de ahorro)"
                    )
                resultados.append(resultado)
                # Los videos con error no se apuntan: la próxima vez se reintentan.
                if registro_previo is not None and not resultado.error:
                    registro_previo.registrar(video, huella, resultado.a_dict())
        except AnalisisCancelado:
            self._log("Análisis cancelado por el usuario.")

        if reutilizados:
            self._log(
                f"Reutilizados {reutilizados} de {len(lista)} videos ya analizados."
            )
        self.escribir_csv(resultados, salida / "eventos.csv")
        return resultados

    @staticmethod
    def escribir_csv(resultados: Sequence[ResultadoVideo], destino: str | Path) -> Path:
        """Escribe todos los eventos en un CSV y devuelve su ruta."""
        ruta = Path(destino)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        with ruta.open("w", newline="", encoding="utf-8-sig") as archivo:
            escritor = csv.DictWriter(
                archivo,
                fieldnames=[
                    "archivo",
                    "tipo",
                    "inicio",
                    "fin",
                    "duracion_segundos",
                    "rostros",
                    "personas",
                    "personas_distintas",
                    "direccion",
                ],
            )
            escritor.writeheader()
            for resultado in resultados:
                for evento in resultado.eventos:
                    escritor.writerow(evento.a_fila_csv())
        return ruta
