"""Detección e identificación de rostros dentro de la caja de una persona."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2

from decam.callbacks import CallbackLog

#: Modelos ONNX de OpenCV Zoo. Se buscan en la carpeta ``models/`` del proyecto
#: (o del ejecutable, donde ``build_exe.py`` la copia con el mismo nombre).
CARPETA_MODELOS = Path(__file__).resolve().parent.parent / "models"
RUTA_MODELO_YUNET = CARPETA_MODELOS / "face_detection_yunet_2023mar.onnx"
RUTA_MODELO_SFACE = CARPETA_MODELOS / "face_recognition_sface_2021dec.onnx"

#: Umbral de similitud coseno recomendado por OpenCV para SFace: por encima de
#: este valor, dos rostros se consideran de la misma persona.
UMBRAL_SFACE = 0.363

#: Etiqueta usada cuando ningún rostro del catálogo supera el umbral.
DESCONOCIDO = "desconocido"

#: Extensiones aceptadas para las fotos de referencia de cada persona.
EXTENSIONES_IMAGEN = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

CajaEntera = tuple[int, int, int, int]


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

    caja: CajaEntera
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
        caja_persona: CajaEntera,
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

    def _filas_haar(self, region) -> list[CajaEntera]:
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


class AnalizadorRostros:
    """Lo que el analizador de video necesita de los rostros: cajas y nombres.

    Une un :class:`DetectorRostros` con un :class:`IdentificadorRostros`
    opcional, para que el analizador no tenga que saber que son dos cosas.
    """

    def __init__(
        self,
        detector: DetectorRostros,
        identificador: Optional[IdentificadorRostros] = None,
    ) -> None:
        self.detector = detector
        self.identificador = identificador

    def analizar(
        self, frame, caja_persona: CajaEntera
    ) -> tuple[list[CajaEntera], set[str]]:
        """Detecta e identifica los rostros dentro de la caja de una persona.

        Returns:
            Las cajas de los rostros y los nombres del catálogo reconocidos.
        """
        rostros = self.detector.detectar(frame, caja_persona)
        nombres: set[str] = set()
        if self.identificador is not None:
            for rostro in rostros:
                identidad = self.identificador.identificar(rostro)
                if identidad.conocida:
                    nombres.add(identidad.nombre)
        return [r.caja for r in rostros], nombres
