"""Interfaz gráfica (Tkinter) para analizar grabaciones de cámaras de seguridad.

Permite elegir una carpeta con videos, dibujar la zona de la puerta sobre una
vista previa y lanzar el análisis en un hilo aparte, mostrando progreso y log.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Optional

import cv2
from PIL import Image, ImageTk

from detector import (
    CRITERIOS_ZONA,
    aceleradores_disponibles,
    TIPO_GENERAL,
    TIPO_ZONA,
    AnalizadorPuerta,
    ConfiguracionAnalisis,
    Evento,
    InfoVideo,
    ResultadoVideo,
    backends_rostros_disponibles,
    encontrar_videos,
    formatear_tiempo,
    info_video,
    leer_frame,
    normalizar_zona,
)

RUTA_CONFIG = Path(__file__).with_name("config.json")
MODELOS = ("yolov8n", "yolov8s", "yolov8m")
ANCHO_PREVIA = 640
ALTO_PREVIA = 380


@dataclass
class Preferencias:
    """Configuración persistida entre ejecuciones (``config.json``)."""

    carpeta_videos: str = ""
    carpeta_salida: str = ""
    zona_puerta: Optional[list[int]] = None
    fps_analisis: float = 1.0
    tolerancia_segundos: float = 3.0
    modelo: str = "yolov8n"
    acelerador: str = "auto"
    criterio_zona: str = "pies"
    min_solape: float = 0.25
    registrar_general: bool = True
    detectar_rostros: bool = False
    backend_rostros: str = "yunet"
    guardar_recortes_rostros: bool = True
    identificar_rostros: bool = False
    carpeta_personas: str = ""

    @classmethod
    def cargar(cls, ruta: Path = RUTA_CONFIG) -> "Preferencias":
        """Lee las preferencias del disco; devuelve valores por defecto si falla."""
        try:
            datos: dict[str, Any] = json.loads(ruta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        validos = {k: v for k, v in datos.items() if k in cls.__dataclass_fields__}
        try:
            return cls(**validos)
        except TypeError:
            return cls()

    def guardar(self, ruta: Path = RUTA_CONFIG) -> None:
        """Escribe las preferencias en disco (los errores se ignoran)."""
        try:
            ruta.write_text(
                json.dumps(self.__dict__, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            pass


def abrir_carpeta(ruta: str | Path) -> None:
    """Abre una carpeta en el explorador de archivos del sistema."""
    ruta = str(ruta)
    if sys.platform.startswith("win"):
        os.startfile(ruta)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", ruta])
    else:
        subprocess.Popen(["xdg-open", ruta])


class AplicacionDeCam(ttk.Frame):
    """Ventana principal de la aplicación."""

    def __init__(self, root: tk.Tk) -> None:
        """Construye la interfaz y restaura la configuración guardada."""
        super().__init__(root, padding=10)
        self.root = root
        self.pack(fill="both", expand=True)

        self.prefs = Preferencias.cargar()
        self.videos: list[Path] = []
        self._videos_por_nombre: dict[str, Path] = {}
        self.mensajes: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.cancelar_evento = threading.Event()
        self.hilo: Optional[threading.Thread] = None

        # Estado de la vista previa.
        self.imagen_previa: Optional[ImageTk.PhotoImage] = None
        self.escala_previa: float = 1.0
        self.zona_puerta: Optional[tuple[int, int, int, int]] = (
            tuple(self.prefs.zona_puerta) if self.prefs.zona_puerta else None  # type: ignore[assignment]
        )
        self._arrastre_inicio: Optional[tuple[float, float]] = None
        self._rect_id: Optional[int] = None

        # Captura abierta del video de referencia, para navegar entre frames sin
        # reabrir el archivo en cada salto.
        self._cap_previa: Optional[cv2.VideoCapture] = None
        self._info_previa: Optional[InfoVideo] = None
        self._frame_actual: int = 0
        self._seek_pendiente: Optional[str] = None

        self._construir_interfaz()
        self._restaurar_preferencias()
        self.root.protocol("WM_DELETE_WINDOW", self._al_cerrar)
        self.root.after(100, self._procesar_mensajes)

    # ------------------------------------------------------------------ interfaz

    def _construir_interfaz(self) -> None:
        """Crea todos los widgets de la ventana."""
        # --- Carpetas ---
        marco_carpetas = ttk.LabelFrame(self, text="Carpetas", padding=8)
        marco_carpetas.pack(fill="x")

        ttk.Button(
            marco_carpetas, text="Seleccionar carpeta", command=self.elegir_carpeta_videos
        ).grid(row=0, column=0, sticky="w")
        self.var_carpeta = tk.StringVar(value="(sin seleccionar)")
        ttk.Label(marco_carpetas, textvariable=self.var_carpeta).grid(
            row=0, column=1, sticky="w", padx=8
        )

        ttk.Button(
            marco_carpetas,
            text="Seleccionar carpeta de salida",
            command=self.elegir_carpeta_salida,
        ).grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.var_salida = tk.StringVar(value="(sin seleccionar)")
        ttk.Label(marco_carpetas, textvariable=self.var_salida).grid(
            row=1, column=1, sticky="w", padx=8, pady=(6, 0)
        )
        marco_carpetas.columnconfigure(1, weight=1)

        # --- Vista previa ---
        marco_previa = ttk.LabelFrame(
            self, text="Zona de la puerta (clic y arrastrar sobre la imagen)", padding=8
        )
        marco_previa.pack(fill="both", expand=True, pady=8)

        barra = ttk.Frame(marco_previa)
        barra.pack(fill="x")
        ttk.Label(barra, text="Video de referencia:").pack(side="left")
        self.var_video_previa = tk.StringVar()
        self.combo_videos = ttk.Combobox(
            barra, textvariable=self.var_video_previa, state="readonly", width=45
        )
        self.combo_videos.pack(side="left", padx=6)
        self.combo_videos.bind("<<ComboboxSelected>>", lambda _e: self.cargar_vista_previa())
        self.var_zona = tk.StringVar(value="Zona: sin definir")
        ttk.Label(barra, textvariable=self.var_zona).pack(side="left", padx=10)
        ttk.Button(barra, text="Borrar zona", command=self.borrar_zona).pack(side="right")

        self.canvas = tk.Canvas(
            marco_previa,
            width=ANCHO_PREVIA,
            height=ALTO_PREVIA,
            background="#20242b",
            highlightthickness=1,
            highlightbackground="#888",
            cursor="cross",
        )
        self.canvas.pack(pady=6)
        self.canvas.bind("<ButtonPress-1>", self._inicio_arrastre)
        self.canvas.bind("<B1-Motion>", self._durante_arrastre)
        self.canvas.bind("<ButtonRelease-1>", self._fin_arrastre)

        # --- Selector del frame de referencia ---
        navegacion = ttk.Frame(marco_previa)
        navegacion.pack(fill="x")

        self.escala_frames = ttk.Scale(
            navegacion, from_=0, to=0, orient="horizontal", command=self._al_mover_escala
        )
        self.escala_frames.pack(fill="x", pady=(0, 4))

        fila = ttk.Frame(navegacion)
        fila.pack(fill="x")
        ttk.Button(fila, text="<<", width=4, command=lambda: self._saltar(-300)).pack(side="left")
        ttk.Button(fila, text="<", width=4, command=lambda: self._saltar(-30)).pack(side="left")
        ttk.Button(fila, text=">", width=4, command=lambda: self._saltar(30)).pack(side="left")
        ttk.Button(fila, text=">>", width=4, command=lambda: self._saltar(300)).pack(side="left")

        ttk.Label(fila, text="Frame:").pack(side="left", padx=(12, 2))
        self.var_frame = tk.StringVar(value="0")
        entrada_frame = ttk.Entry(fila, textvariable=self.var_frame, width=10)
        entrada_frame.pack(side="left")
        entrada_frame.bind("<Return>", lambda _e: self._ir_a_frame())

        ttk.Label(fila, text="Ir a (HH:MM:SS):").pack(side="left", padx=(12, 2))
        self.var_tiempo_ir = tk.StringVar(value="00:00:00")
        entrada_tiempo = ttk.Entry(fila, textvariable=self.var_tiempo_ir, width=10)
        entrada_tiempo.pack(side="left")
        entrada_tiempo.bind("<Return>", lambda _e: self._ir_a_tiempo())
        ttk.Button(fila, text="Ir", width=4, command=self._ir_a_tiempo).pack(side="left", padx=2)

        self.var_info_frame = tk.StringVar(value="")
        ttk.Label(fila, textvariable=self.var_info_frame).pack(side="left", padx=10)

        # --- Parámetros ---
        marco_params = ttk.LabelFrame(self, text="Parámetros", padding=8)
        marco_params.pack(fill="x")

        ttk.Label(marco_params, text="Frames por segundo a analizar:").grid(
            row=0, column=0, sticky="w"
        )
        self.var_fps = tk.DoubleVar(value=self.prefs.fps_analisis)
        ttk.Spinbox(
            marco_params, from_=0.1, to=30.0, increment=0.5,
            textvariable=self.var_fps, width=8,
        ).grid(row=0, column=1, sticky="w", padx=(6, 20))

        ttk.Label(marco_params, text="Tolerancia (segundos):").grid(
            row=0, column=2, sticky="w"
        )
        self.var_tolerancia = tk.DoubleVar(value=self.prefs.tolerancia_segundos)
        ttk.Spinbox(
            marco_params, from_=0.0, to=120.0, increment=1.0,
            textvariable=self.var_tolerancia, width=8,
        ).grid(row=0, column=3, sticky="w", padx=(6, 20))

        ttk.Label(marco_params, text="Modelo YOLO:").grid(row=0, column=4, sticky="w")
        self.var_modelo = tk.StringVar(value=self.prefs.modelo)
        ttk.Combobox(
            marco_params, textvariable=self.var_modelo, values=list(MODELOS),
            state="readonly", width=10,
        ).grid(row=0, column=5, sticky="w", padx=6)

        ttk.Label(marco_params, text="Acelerador:").grid(
            row=0, column=6, sticky="e", padx=(20, 0)
        )
        # Solo se ofrece lo que este equipo puede usar de verdad.
        self.aceleradores = aceleradores_disponibles()
        self.var_acelerador = tk.StringVar(
            value=self.prefs.acelerador
            if self.prefs.acelerador in self.aceleradores
            else "auto"
        )
        ttk.Combobox(
            marco_params,
            textvariable=self.var_acelerador,
            values=self.aceleradores,
            state="readonly",
            width=13,
        ).grid(row=0, column=7, sticky="w", padx=6)

        self.var_rostros = tk.BooleanVar(value=self.prefs.detectar_rostros)
        ttk.Checkbutton(
            marco_params,
            text="Detectar rostros",
            variable=self.var_rostros,
            command=self._actualizar_estado_rostros,
        ).grid(row=1, column=0, sticky="w", pady=(8, 0))

        ttk.Label(marco_params, text="Backend:").grid(row=1, column=1, sticky="e", pady=(8, 0))
        # Solo se ofrecen los backends que esta instalación puede usar de verdad.
        self.backends = backends_rostros_disponibles()
        self.var_backend_rostros = tk.StringVar(
            value=self.prefs.backend_rostros
            if self.prefs.backend_rostros in self.backends
            else (self.backends[0] if self.backends else "")
        )
        self.combo_backend = ttk.Combobox(
            marco_params,
            textvariable=self.var_backend_rostros,
            values=self.backends,
            state="readonly",
            width=10,
        )
        self.combo_backend.grid(row=1, column=2, sticky="w", padx=6, pady=(8, 0))
        self.combo_backend.bind(
            "<<ComboboxSelected>>", lambda _e: self._actualizar_estado_rostros()
        )

        self.var_recortes = tk.BooleanVar(value=self.prefs.guardar_recortes_rostros)
        self.check_recortes = ttk.Checkbutton(
            marco_params, text="Guardar recortes", variable=self.var_recortes
        )
        self.check_recortes.grid(row=1, column=3, sticky="w", padx=6, pady=(8, 0))

        # --- Identificación ---
        self.var_identificar = tk.BooleanVar(value=self.prefs.identificar_rostros)
        self.check_identificar = ttk.Checkbutton(
            marco_params,
            text="Identificar personas conocidas",
            variable=self.var_identificar,
            command=self._actualizar_estado_rostros,
        )
        self.check_identificar.grid(row=2, column=0, sticky="w", pady=(6, 0))

        self.boton_personas = ttk.Button(
            marco_params, text="Carpeta de personas", command=self.elegir_carpeta_personas
        )
        self.boton_personas.grid(row=2, column=1, columnspan=2, sticky="w", pady=(6, 0))
        self.var_personas = tk.StringVar(
            value=self.prefs.carpeta_personas or "(sin catálogo)"
        )
        ttk.Label(marco_params, textvariable=self.var_personas).grid(
            row=2, column=3, columnspan=3, sticky="w", padx=6, pady=(6, 0)
        )
        self._actualizar_estado_rostros()

        # --- Criterio de zona ---
        ttk.Label(marco_params, text="Criterio de zona:").grid(
            row=3, column=0, sticky="w", pady=(10, 0)
        )
        self.var_criterio = tk.StringVar(value=self.prefs.criterio_zona)
        self.combo_criterio = ttk.Combobox(
            marco_params,
            textvariable=self.var_criterio,
            values=list(CRITERIOS_ZONA),
            state="readonly",
            width=10,
        )
        self.combo_criterio.grid(row=3, column=1, sticky="w", padx=6, pady=(10, 0))
        self.combo_criterio.bind(
            "<<ComboboxSelected>>", lambda _e: self._actualizar_estado_criterio()
        )

        ttk.Label(marco_params, text="Solape mín.:").grid(
            row=3, column=2, sticky="e", pady=(10, 0)
        )
        self.var_min_solape = tk.DoubleVar(value=self.prefs.min_solape)
        self.spin_solape = ttk.Spinbox(
            marco_params, from_=0.05, to=1.0, increment=0.05,
            textvariable=self.var_min_solape, width=6,
        )
        self.spin_solape.grid(row=3, column=3, sticky="w", padx=6, pady=(10, 0))

        self.var_general = tk.BooleanVar(value=self.prefs.registrar_general)
        ttk.Checkbutton(
            marco_params,
            text="Registrar también detecciones generales",
            variable=self.var_general,
        ).grid(row=3, column=4, columnspan=2, sticky="w", padx=6, pady=(10, 0))

        self.var_ayuda_criterio = tk.StringVar()
        ttk.Label(
            marco_params, textvariable=self.var_ayuda_criterio, foreground="#666"
        ).grid(row=4, column=0, columnspan=6, sticky="w", pady=(2, 0))
        self._actualizar_estado_criterio()

        # --- Acciones y progreso ---
        marco_acciones = ttk.Frame(self)
        marco_acciones.pack(fill="x", pady=8)
        self.boton_iniciar = ttk.Button(
            marco_acciones, text="Iniciar análisis", command=self.iniciar_analisis
        )
        self.boton_iniciar.pack(side="left")
        self.boton_cancelar = ttk.Button(
            marco_acciones, text="Cancelar", command=self.cancelar_analisis, state="disabled"
        )
        self.boton_cancelar.pack(side="left", padx=6)
        self.boton_resultados = ttk.Button(
            marco_acciones,
            text="Abrir carpeta de resultados",
            command=self.abrir_resultados,
            state="disabled",
        )
        self.boton_resultados.pack(side="left", padx=6)

        self.progreso = ttk.Progressbar(self, mode="determinate", maximum=100.0)
        self.progreso.pack(fill="x")
        self.var_estado = tk.StringVar(value="Listo.")
        ttk.Label(self, textvariable=self.var_estado).pack(anchor="w", pady=(2, 6))

        # --- Log ---
        marco_log = ttk.LabelFrame(self, text="Registro", padding=4)
        marco_log.pack(fill="both", expand=True)
        self.texto_log = tk.Text(marco_log, height=12, wrap="none", state="disabled")
        scroll = ttk.Scrollbar(marco_log, orient="vertical", command=self.texto_log.yview)
        self.texto_log.configure(yscrollcommand=scroll.set)
        self.texto_log.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

    def _restaurar_preferencias(self) -> None:
        """Aplica la configuración guardada al abrir la aplicación."""
        if self.prefs.carpeta_salida:
            self.var_salida.set(self.prefs.carpeta_salida)
            self.boton_resultados.configure(state="normal")
        if self.prefs.carpeta_videos and Path(self.prefs.carpeta_videos).is_dir():
            self._cargar_carpeta(self.prefs.carpeta_videos)
        self._actualizar_etiqueta_zona()

    # ------------------------------------------------------------------- carpetas

    def elegir_carpeta_videos(self) -> None:
        """Pide al usuario la carpeta con los videos a analizar."""
        carpeta = filedialog.askdirectory(
            title="Carpeta con los videos", initialdir=self.prefs.carpeta_videos or None
        )
        if carpeta:
            self._cargar_carpeta(carpeta)

    def _cargar_carpeta(self, carpeta: str) -> None:
        """Busca los videos de la carpeta y refresca la vista previa."""
        self.prefs.carpeta_videos = carpeta
        self.videos = encontrar_videos(carpeta)
        subcarpetas = len({v.parent for v in self.videos})
        detalle = f" en {subcarpetas} subcarpetas" if subcarpetas > 1 else ""
        self.var_carpeta.set(f"{carpeta}  ({len(self.videos)} videos{detalle})")
        # Se muestra la ruta relativa porque los videos pueden estar en subcarpetas.
        nombres = [str(v.relative_to(carpeta)) for v in self.videos]
        self._videos_por_nombre = dict(zip(nombres, self.videos))
        self.combo_videos.configure(values=nombres)
        if nombres:
            self.var_video_previa.set(nombres[0])
            self.cargar_vista_previa()
        else:
            self.var_video_previa.set("")
            self._cerrar_captura_previa()
            self.canvas.delete("all")
            self.var_info_frame.set("")
            self.log(f"No se encontraron videos en {carpeta}")

    def elegir_carpeta_salida(self) -> None:
        """Pide al usuario la carpeta donde guardar los resultados."""
        carpeta = filedialog.askdirectory(
            title="Carpeta de salida", initialdir=self.prefs.carpeta_salida or None
        )
        if carpeta:
            self.prefs.carpeta_salida = carpeta
            self.var_salida.set(carpeta)
            self.boton_resultados.configure(state="normal")

    def abrir_resultados(self) -> None:
        """Abre la carpeta de resultados en el explorador."""
        carpeta = self.prefs.carpeta_salida
        if carpeta and Path(carpeta).is_dir():
            abrir_carpeta(carpeta)
        else:
            messagebox.showwarning(
                "Carpeta de salida", "Todavía no hay una carpeta de salida válida."
            )

    # --------------------------------------------------------------- vista previa

    def cargar_vista_previa(self) -> None:
        """Abre el video de referencia y muestra su primer frame."""
        nombre = self.var_video_previa.get()
        video = self._videos_por_nombre.get(nombre)
        if video is None:
            return

        self._cerrar_captura_previa()
        captura = cv2.VideoCapture(str(video))
        if not captura.isOpened():
            captura.release()
            messagebox.showerror("Vista previa", f"No se pudo abrir {nombre}.")
            return

        self._cap_previa = captura
        self._info_previa = info_video(captura)
        ultimo = max(0, self._info_previa.total_frames - 1)
        self.escala_frames.configure(to=ultimo)
        self._mostrar_frame(0)

    def _cerrar_captura_previa(self) -> None:
        """Libera la captura del video de referencia si hay una abierta."""
        if self._cap_previa is not None:
            self._cap_previa.release()
            self._cap_previa = None
        self._info_previa = None

    def _mostrar_frame(self, indice: int) -> None:
        """Dibuja en el canvas el frame ``indice`` del video de referencia."""
        if self._cap_previa is None or self._info_previa is None:
            return
        info = self._info_previa
        indice = max(0, min(indice, max(0, info.total_frames - 1)))

        frame = leer_frame(self._cap_previa, indice)
        if frame is None:
            self.log(f"No se pudo leer el frame {indice}.")
            return

        self._frame_actual = indice
        alto, ancho = frame.shape[:2]
        self.escala_previa = min(ANCHO_PREVIA / ancho, ALTO_PREVIA / alto, 1.0)
        nuevo = (
            max(1, int(ancho * self.escala_previa)),
            max(1, int(alto * self.escala_previa)),
        )
        imagen = cv2.cvtColor(cv2.resize(frame, nuevo), cv2.COLOR_BGR2RGB)
        self.imagen_previa = ImageTk.PhotoImage(Image.fromarray(imagen))

        self.canvas.delete("all")
        self.canvas.configure(width=nuevo[0], height=nuevo[1])
        self.canvas.create_image(0, 0, anchor="nw", image=self.imagen_previa)
        self._rect_id = None
        self._dibujar_zona_guardada()

        segundo = indice / info.fps if info.fps else 0.0
        self.var_frame.set(str(indice))
        self.var_tiempo_ir.set(formatear_tiempo(segundo))
        self.escala_frames.set(indice)
        self.var_info_frame.set(
            f"{formatear_tiempo(segundo)} / {formatear_tiempo(info.duracion)}"
            f"  ·  {ancho}x{alto} @ {info.fps:.0f} fps  ·  {info.total_frames} frames"
        )

    def _al_mover_escala(self, valor: str) -> None:
        """Programa la carga del frame tras mover el deslizador.

        El salto se retrasa 150 ms para no decodificar en cada píxel arrastrado.
        """
        if self._cap_previa is None:
            return
        indice = int(float(valor))
        if indice == self._frame_actual:
            return
        if self._seek_pendiente is not None:
            self.root.after_cancel(self._seek_pendiente)
        self._seek_pendiente = self.root.after(150, lambda: self._mostrar_frame(indice))

    def _saltar(self, delta_frames: int) -> None:
        """Avanza o retrocede un número de frames desde el actual."""
        self._mostrar_frame(self._frame_actual + delta_frames)

    def _ir_a_frame(self) -> None:
        """Salta al número de frame escrito en la caja de texto."""
        try:
            self._mostrar_frame(int(self.var_frame.get()))
        except ValueError:
            messagebox.showwarning("Frame", "Escribe un número de frame válido.")

    def _ir_a_tiempo(self) -> None:
        """Salta al instante escrito como ``HH:MM:SS``, ``MM:SS`` o segundos."""
        if self._info_previa is None:
            return
        try:
            segundos = self._parsear_tiempo(self.var_tiempo_ir.get())
        except ValueError:
            messagebox.showwarning("Tiempo", "Usa el formato HH:MM:SS, MM:SS o segundos.")
            return
        self._mostrar_frame(int(segundos * self._info_previa.fps))

    @staticmethod
    def _parsear_tiempo(texto: str) -> float:
        """Convierte ``HH:MM:SS``, ``MM:SS`` o un número de segundos a segundos.

        Raises:
            ValueError: si el texto no tiene un formato reconocible.
        """
        partes = texto.strip().split(":")
        if len(partes) > 3:
            raise ValueError(texto)
        total = 0.0
        for parte in partes:
            total = total * 60 + float(parte)
        return total

    def _dibujar_zona_guardada(self) -> None:
        """Dibuja sobre el canvas la zona de la puerta ya definida, si existe."""
        if self.zona_puerta is None or self.imagen_previa is None:
            return
        x1, y1, x2, y2 = (c * self.escala_previa for c in self.zona_puerta)
        self._rect_id = self.canvas.create_rectangle(
            x1, y1, x2, y2, outline="#ff3b3b", width=2
        )

    def _inicio_arrastre(self, evento: tk.Event) -> None:
        """Registra el punto inicial del rectángulo."""
        if self.imagen_previa is None:
            return
        self._arrastre_inicio = (self.canvas.canvasx(evento.x), self.canvas.canvasy(evento.y))
        if self._rect_id is not None:
            self.canvas.delete(self._rect_id)
        self._rect_id = self.canvas.create_rectangle(
            *self._arrastre_inicio, *self._arrastre_inicio, outline="#ff3b3b", width=2
        )

    def _durante_arrastre(self, evento: tk.Event) -> None:
        """Redimensiona el rectángulo mientras se arrastra el mouse."""
        if self._arrastre_inicio is None or self._rect_id is None:
            return
        x0, y0 = self._arrastre_inicio
        self.canvas.coords(
            self._rect_id, x0, y0, self.canvas.canvasx(evento.x), self.canvas.canvasy(evento.y)
        )

    def _fin_arrastre(self, evento: tk.Event) -> None:
        """Convierte el rectángulo del canvas a coordenadas reales del video."""
        if self._arrastre_inicio is None:
            return
        x0, y0 = self._arrastre_inicio
        x1, y1 = self.canvas.canvasx(evento.x), self.canvas.canvasy(evento.y)
        self._arrastre_inicio = None

        escala = self.escala_previa or 1.0
        zona = normalizar_zona((x0 / escala, y0 / escala, x1 / escala, y1 / escala))
        if zona[2] - zona[0] < 5 or zona[3] - zona[1] < 5:
            self.log("Zona demasiado pequeña, vuelve a dibujarla.")
            return
        self.zona_puerta = zona
        self._actualizar_etiqueta_zona()

    def _actualizar_estado_criterio(self) -> None:
        """Explica el criterio elegido y habilita el umbral solo si aplica."""
        ayudas = {
            "pies": (
                "pies: cuenta si el punto de apoyo (centro del borde inferior de "
                "la caja) cae dentro de la zona. El más fiable en pasillos."
            ),
            "centro": "centro: cuenta si el centro de la caja cae dentro de la zona.",
            "solape": (
                "solape: cuenta si al menos el porcentaje indicado del área de la "
                "persona está dentro de la zona."
            ),
        }
        criterio = self.var_criterio.get()
        self.var_ayuda_criterio.set(ayudas.get(criterio, ""))
        self.spin_solape.configure(
            state="normal" if criterio == "solape" else "disabled"
        )

    def _actualizar_estado_rostros(self) -> None:
        """Habilita o deshabilita las opciones de rostros e identificación.

        La identificación cuelga de la detección: sin rostros no hay nada que
        identificar, y además exige el backend YuNet.
        """
        activo = self.var_rostros.get()
        self.combo_backend.configure(state="readonly" if activo else "disabled")
        self.check_recortes.configure(state="normal" if activo else "disabled")

        puede_identificar = activo and self.var_backend_rostros.get() == "yunet"
        self.check_identificar.configure(
            state="normal" if puede_identificar else "disabled"
        )
        if not puede_identificar:
            self.var_identificar.set(False)
        self.boton_personas.configure(
            state="normal" if self.var_identificar.get() else "disabled"
        )

    def elegir_carpeta_personas(self) -> None:
        """Pide la carpeta con una subcarpeta de fotos por persona conocida."""
        carpeta = filedialog.askdirectory(
            title="Carpeta de personas (una subcarpeta por persona)",
            initialdir=self.prefs.carpeta_personas or None,
        )
        if not carpeta:
            return
        self.prefs.carpeta_personas = carpeta
        subcarpetas = [p for p in Path(carpeta).iterdir() if p.is_dir()]
        self.var_personas.set(f"{carpeta}  ({len(subcarpetas)} persona/s)")
        if not subcarpetas:
            messagebox.showwarning(
                "Catálogo vacío",
                "Esa carpeta no tiene subcarpetas.\n\n"
                "Cada persona necesita su propia subcarpeta con fotos suyas; el "
                "nombre de la subcarpeta es la etiqueta que saldrá en el CSV.",
            )

    def borrar_zona(self) -> None:
        """Elimina la zona de la puerta definida."""
        self.zona_puerta = None
        if self._rect_id is not None:
            self.canvas.delete(self._rect_id)
            self._rect_id = None
        self._actualizar_etiqueta_zona()

    def _actualizar_etiqueta_zona(self) -> None:
        """Refresca la etiqueta que muestra las coordenadas de la zona."""
        if self.zona_puerta is None:
            self.var_zona.set("Zona: sin definir")
        else:
            x1, y1, x2, y2 = self.zona_puerta
            self.var_zona.set(f"Zona: ({x1}, {y1}) - ({x2}, {y2})")

    # ---------------------------------------------------------------- log y cola

    def log(self, mensaje: str) -> None:
        """Escribe una línea en el área de registro (solo desde el hilo principal)."""
        self.texto_log.configure(state="normal")
        self.texto_log.insert("end", mensaje + "\n")
        self.texto_log.see("end")
        self.texto_log.configure(state="disabled")

    def _encolar(self, tipo: str, dato: Any = None) -> None:
        """Encola un mensaje del hilo de análisis para la interfaz."""
        self.mensajes.put((tipo, dato))

    def _procesar_mensajes(self) -> None:
        """Consume la cola de mensajes y actualiza la interfaz. Se repite cada 100 ms."""
        try:
            while True:
                tipo, dato = self.mensajes.get_nowait()
                if tipo == "log":
                    self.log(dato)
                elif tipo == "progreso":
                    nombre, pct_video, pct_total = dato
                    self.progreso["value"] = pct_total
                    self.var_estado.set(
                        f"Analizando {nombre} — {pct_video:.0f}% "
                        f"(total {pct_total:.0f}%)"
                    )
                elif tipo == "error":
                    messagebox.showerror("Error", dato)
                elif tipo == "fin":
                    self._al_terminar(dato)
        except queue.Empty:
            pass
        self.root.after(100, self._procesar_mensajes)

    # ------------------------------------------------------------------ análisis

    def iniciar_analisis(self) -> None:
        """Valida los parámetros y lanza el análisis en un hilo aparte."""
        if self.hilo is not None and self.hilo.is_alive():
            return
        if not self.videos:
            messagebox.showwarning("Videos", "Selecciona una carpeta con videos.")
            return
        if not self.prefs.carpeta_salida:
            messagebox.showwarning("Salida", "Selecciona la carpeta de salida.")
            return
        if self.zona_puerta is None:
            messagebox.showwarning(
                "Zona", "Dibuja la zona de la puerta sobre la vista previa."
            )
            return
        if self.var_identificar.get() and not self.prefs.carpeta_personas:
            messagebox.showwarning(
                "Personas",
                "Elige la carpeta con las fotos de las personas conocidas.",
            )
            return

        try:
            config = ConfiguracionAnalisis(
                zona_puerta=self.zona_puerta,
                fps_analisis=float(self.var_fps.get()),
                tolerancia_segundos=float(self.var_tolerancia.get()),
                modelo=self.var_modelo.get(),
                acelerador=self.var_acelerador.get(),
                criterio_zona=self.var_criterio.get(),
                min_solape=float(self.var_min_solape.get()),
                registrar_general=self.var_general.get(),
                detectar_rostros=self.var_rostros.get(),
                backend_rostros=self.var_backend_rostros.get(),
                guardar_recortes_rostros=self.var_recortes.get(),
                identificar_rostros=self.var_identificar.get(),
                carpeta_personas=self.prefs.carpeta_personas,
            )
            config.validar()
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror("Parámetros", str(exc))
            return

        self._guardar_preferencias()
        self.cancelar_evento = threading.Event()
        self.progreso["value"] = 0.0
        self.boton_iniciar.configure(state="disabled")
        self.boton_cancelar.configure(state="normal")
        self.var_estado.set("Iniciando análisis...")
        self.log("=" * 60)
        self.log("Iniciando análisis...")

        self.hilo = threading.Thread(target=self._ejecutar_analisis, args=(config,), daemon=True)
        self.hilo.start()

    def _ejecutar_analisis(self, config: ConfiguracionAnalisis) -> None:
        """Corre el análisis en segundo plano (no toca widgets directamente)."""
        try:
            analizador = AnalizadorPuerta(
                config=config,
                cancelar=self.cancelar_evento,
                on_progreso=lambda n, pv, pt: self._encolar("progreso", (n, pv, pt)),
                on_log=lambda m: self._encolar("log", m),
                on_evento=self._notificar_evento,
            )
            resultados = analizador.analizar_videos(self.videos, self.prefs.carpeta_salida)
            self._encolar("fin", resultados)
        except Exception as exc:  # noqa: BLE001 - cualquier fallo debe llegar a la GUI
            self._encolar("error", f"El análisis falló: {exc}")
            self._encolar("fin", [])

    def _notificar_evento(self, evento: Evento) -> None:
        """Callback del analizador: registra el evento detectado."""
        self._encolar(
            "log",
            f"    -> [{evento.tipo}] {evento.archivo}: "
            f"{formatear_tiempo(evento.inicio)} a "
            f"{formatear_tiempo(evento.fin)} ({evento.duracion:.1f}s)",
        )

    def cancelar_analisis(self) -> None:
        """Pide al hilo de análisis que se detenga."""
        if self.hilo is not None and self.hilo.is_alive():
            self.cancelar_evento.set()
            self.var_estado.set("Cancelando...")
            self.log("Cancelación solicitada...")

    def _al_terminar(self, resultados: list[ResultadoVideo]) -> None:
        """Muestra el resumen final y restablece los botones."""
        self.boton_iniciar.configure(state="normal")
        self.boton_cancelar.configure(state="disabled")
        self.boton_resultados.configure(state="normal")

        def contar(eventos: list[Evento], tipo: str) -> int:
            """Cuenta los eventos de un tipo concreto."""
            return sum(1 for e in eventos if e.tipo == tipo)

        self.log("-" * 60)
        self.log("RESUMEN                            en zona    generales")
        total_zona = total_general = 0
        for resultado in resultados:
            if resultado.error:
                self.log(f"  {resultado.archivo}: ERROR - {resultado.error}")
                continue
            zona = contar(resultado.eventos, TIPO_ZONA)
            general = contar(resultado.eventos, TIPO_GENERAL)
            total_zona += zona
            total_general += general
            self.log(f"  {resultado.archivo[:34]:<34} {zona:>7}   {general:>10}")

        self.log(f"  {'TOTAL':<34} {total_zona:>7}   {total_general:>10}")
        if self.prefs.carpeta_salida:
            self.log(f"Resultados en: {self.prefs.carpeta_salida}")
        self.progreso["value"] = 100.0
        self.var_estado.set(
            f"Terminado. {total_zona} eventos en la zona, "
            f"{total_general} generales."
        )

    # ------------------------------------------------------------------- cierre

    def _guardar_preferencias(self) -> None:
        """Vuelca los valores actuales de la interfaz a ``config.json``."""
        try:
            self.prefs.fps_analisis = float(self.var_fps.get())
            self.prefs.tolerancia_segundos = float(self.var_tolerancia.get())
        except tk.TclError:
            pass
        self.prefs.modelo = self.var_modelo.get()
        self.prefs.acelerador = self.var_acelerador.get()
        self.prefs.criterio_zona = self.var_criterio.get()
        self.prefs.registrar_general = self.var_general.get()
        try:
            self.prefs.min_solape = float(self.var_min_solape.get())
        except tk.TclError:
            pass
        self.prefs.detectar_rostros = self.var_rostros.get()
        self.prefs.backend_rostros = self.var_backend_rostros.get()
        self.prefs.guardar_recortes_rostros = self.var_recortes.get()
        self.prefs.identificar_rostros = self.var_identificar.get()
        self.prefs.zona_puerta = list(self.zona_puerta) if self.zona_puerta else None
        self.prefs.guardar()

    def _al_cerrar(self) -> None:
        """Guarda la configuración, libera recursos y cierra la ventana."""
        self.cancelar_evento.set()
        self._guardar_preferencias()
        self._cerrar_captura_previa()
        self.root.destroy()


def main() -> None:
    """Punto de entrada de la aplicación."""
    root = tk.Tk()
    root.title("DeCam - Detección de personas en la puerta")
    root.geometry("900x900")
    root.minsize(820, 700)
    AplicacionDeCam(root)
    root.mainloop()


if __name__ == "__main__":
    main()
