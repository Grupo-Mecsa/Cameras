"""Interfaz gráfica (Tkinter) para analizar grabaciones de cámaras de seguridad.

La ventana se organiza en tres zonas: la cabecera con las carpetas, un panel de
parámetros a la izquierda y, a la derecha, pestañas con la vista previa, la tabla
de eventos y el registro. El análisis corre en un hilo aparte y se comunica con
la interfaz mediante una cola que solo consume el hilo principal.
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

import registro
import reporte
from detector import (
    CRITERIOS_ZONA,
    TIPO_GENERAL,
    TIPO_ZONA,
    UMBRAL_MOVIMIENTO,
    AnalizadorPuerta,
    ConfiguracionAnalisis,
    Evento,
    InfoVideo,
    ResultadoVideo,
    aceleradores_disponibles,
    backends_rostros_disponibles,
    encontrar_videos,
    formatear_tiempo,
    info_video,
    leer_frame,
    normalizar_zona,
)

RUTA_CONFIG = registro.ruta_config()
MODELOS = ("yolov8n", "yolov8s", "yolov8m")
ANCHO_PREVIA = 720
ALTO_PREVIA = 405

#: Colores de fondo de la tabla, para distinguir los tipos de un vistazo.
COLOR_ZONA = "#e0f0e4"
COLOR_GENERAL = "#f2f3f5"


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
    usar_tracking: bool = True
    incremental: bool = True

    @classmethod
    def cargar(cls, ruta: Path = RUTA_CONFIG) -> "Preferencias":
        """Lee las preferencias del disco; devuelve valores por defecto si falla."""
        try:
            datos: dict[str, Any] = json.loads(ruta.read_text(encoding="utf-8"))
        except FileNotFoundError:
            registro.log.info("Sin configuración previa en %s", ruta)
            return cls()
        except (OSError, ValueError) as exc:
            registro.log.warning("No se pudo leer %s: %s", ruta, exc)
            return cls()
        validos = {k: v for k, v in datos.items() if k in cls.__dataclass_fields__}
        try:
            return cls(**validos)
        except TypeError as exc:
            registro.log.warning("Configuración inválida en %s: %s", ruta, exc)
            return cls()

    def guardar(self, ruta: Path = RUTA_CONFIG) -> None:
        """Escribe las preferencias en disco (los errores se ignoran)."""
        try:
            ruta.write_text(
                json.dumps(self.__dict__, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            registro.log.error("No se pudo guardar %s: %s", ruta, exc)


def abrir_en_sistema(ruta: str | Path) -> None:
    """Abre un archivo o carpeta con la aplicación predeterminada del sistema."""
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
        super().__init__(root, padding=(10, 8))
        self.root = root
        self.pack(fill="both", expand=True)

        self.prefs = Preferencias.cargar()
        self.videos: list[Path] = []
        self._videos_por_nombre: dict[str, Path] = {}
        self.mensajes: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.cancelar_evento = threading.Event()
        self.hilo: Optional[threading.Thread] = None
        self.informes: list[Path] = []

        # Todos los eventos detectados y los que ahora mismo hay en la tabla.
        self._eventos: list[Evento] = []
        self._filas: dict[str, Evento] = {}

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

        self._configurar_estilo()
        self._construir_interfaz()
        self._restaurar_preferencias()
        self._anunciar_rutas()
        self.root.protocol("WM_DELETE_WINDOW", self._al_cerrar)
        self.root.after(100, self._procesar_mensajes)

    # ------------------------------------------------------------------ interfaz

    def _configurar_estilo(self) -> None:
        """Ajusta detalles visuales que ttk no trae por defecto."""
        estilo = ttk.Style()
        if "vista" in estilo.theme_names():
            estilo.theme_use("vista")
        estilo.configure("Titulo.TLabel", font=("Segoe UI", 9, "bold"))
        estilo.configure("Pista.TLabel", foreground="#6a7280")
        estilo.configure("Accion.TButton", font=("Segoe UI", 9, "bold"))
        estilo.configure("Treeview", rowheight=22)

    def _construir_interfaz(self) -> None:
        """Crea todos los widgets de la ventana."""
        self._construir_cabecera()

        cuerpo = ttk.Frame(self)
        cuerpo.pack(fill="both", expand=True, pady=(8, 0))
        self._construir_panel_parametros(cuerpo)
        self._construir_pestanas(cuerpo)

        self._construir_barra_inferior()

    def _construir_cabecera(self) -> None:
        """Fila superior con las dos carpetas de trabajo."""
        marco = ttk.LabelFrame(self, text="Carpetas", padding=8)
        marco.pack(fill="x")

        ttk.Button(
            marco, text="Videos a analizar…", width=22,
            command=self.elegir_carpeta_videos,
        ).grid(row=0, column=0, sticky="w")
        self.var_carpeta = tk.StringVar(value="(sin seleccionar)")
        ttk.Label(marco, textvariable=self.var_carpeta).grid(
            row=0, column=1, sticky="w", padx=10
        )

        ttk.Button(
            marco, text="Carpeta de resultados…", width=22,
            command=self.elegir_carpeta_salida,
        ).grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.var_salida = tk.StringVar(value="(sin seleccionar)")
        ttk.Label(marco, textvariable=self.var_salida).grid(
            row=1, column=1, sticky="w", padx=10, pady=(6, 0)
        )
        marco.columnconfigure(1, weight=1)

    # -------------------------------------------------------- panel parámetros

    def _construir_panel_parametros(self, padre: ttk.Frame) -> None:
        """Columna izquierda con todos los ajustes, agrupados por tema."""
        panel = ttk.Frame(padre)
        panel.pack(side="left", fill="y", padx=(0, 10))

        self._seccion_analisis(panel)
        self._seccion_zona(panel)
        self._seccion_rendimiento(panel)
        self._seccion_rostros(panel)

        self._actualizar_estado_criterio()
        self._actualizar_estado_movimiento()
        self._actualizar_estado_rostros()

    def _seccion_analisis(self, panel: ttk.Frame) -> None:
        """Parámetros generales del análisis."""
        marco = ttk.LabelFrame(panel, text="Análisis", padding=8)
        marco.pack(fill="x")

        ttk.Label(marco, text="Frames por segundo:").grid(row=0, column=0, sticky="w")
        self.var_fps = tk.DoubleVar(value=self.prefs.fps_analisis)
        ttk.Spinbox(
            marco, from_=0.1, to=30.0, increment=0.5, textvariable=self.var_fps,
            width=8,
        ).grid(row=0, column=1, sticky="e", padx=(8, 0))

        ttk.Label(marco, text="Tolerancia (s):").grid(
            row=1, column=0, sticky="w", pady=(5, 0)
        )
        self.var_tolerancia = tk.DoubleVar(value=self.prefs.tolerancia_segundos)
        ttk.Spinbox(
            marco, from_=0.0, to=120.0, increment=1.0,
            textvariable=self.var_tolerancia, width=8,
        ).grid(row=1, column=1, sticky="e", padx=(8, 0), pady=(5, 0))

        ttk.Label(marco, text="Modelo:").grid(row=2, column=0, sticky="w", pady=(5, 0))
        self.var_modelo = tk.StringVar(value=self.prefs.modelo)
        ttk.Combobox(
            marco, textvariable=self.var_modelo, values=list(MODELOS),
            state="readonly", width=12,
        ).grid(row=2, column=1, sticky="e", padx=(8, 0), pady=(5, 0))

        ttk.Label(marco, text="Acelerador:").grid(
            row=3, column=0, sticky="w", pady=(5, 0)
        )
        # Solo se ofrece lo que este equipo puede usar de verdad.
        self.aceleradores = aceleradores_disponibles()
        self.var_acelerador = tk.StringVar(
            value=self.prefs.acelerador
            if self.prefs.acelerador in self.aceleradores
            else "auto"
        )
        ttk.Combobox(
            marco, textvariable=self.var_acelerador, values=self.aceleradores,
            state="readonly", width=12,
        ).grid(row=3, column=1, sticky="e", padx=(8, 0), pady=(5, 0))

        self.var_tracking = tk.BooleanVar(value=self.prefs.usar_tracking)
        ttk.Checkbutton(
            marco, text="Seguir a cada persona (ByteTrack)",
            variable=self.var_tracking,
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Label(
            marco,
            text="Cuenta personas distintas por evento y anota si entran o "
                 "salen. Funciona mejor con 2 fps o más.",
            style="Pista.TLabel", wraplength=225, justify="left",
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(2, 0))

        self.var_incremental = tk.BooleanVar(value=self.prefs.incremental)
        ttk.Checkbutton(
            marco, text="Reutilizar videos ya analizados",
            variable=self.var_incremental,
        ).grid(row=6, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Label(
            marco,
            text="Salta los videos que ya se analizaron en esta carpeta de "
                 "resultados con los mismos parámetros. Desmárcalo para "
                 "reanalizar todo.",
            style="Pista.TLabel", wraplength=225, justify="left",
        ).grid(row=7, column=0, columnspan=2, sticky="w", pady=(2, 0))
        marco.columnconfigure(0, weight=1)

    def _seccion_zona(self, panel: ttk.Frame) -> None:
        """Criterio de pertenencia a la zona y registro de generales."""
        marco = ttk.LabelFrame(panel, text="Zona de la puerta", padding=8)
        marco.pack(fill="x", pady=(8, 0))

        ttk.Label(marco, text="Criterio:").grid(row=0, column=0, sticky="w")
        self.var_criterio = tk.StringVar(value=self.prefs.criterio_zona)
        self.combo_criterio = ttk.Combobox(
            marco, textvariable=self.var_criterio, values=list(CRITERIOS_ZONA),
            state="readonly", width=12,
        )
        self.combo_criterio.grid(row=0, column=1, sticky="e", padx=(8, 0))
        self.combo_criterio.bind(
            "<<ComboboxSelected>>", lambda _e: self._actualizar_estado_criterio()
        )

        ttk.Label(marco, text="Solape mínimo:").grid(
            row=1, column=0, sticky="w", pady=(5, 0)
        )
        self.var_min_solape = tk.DoubleVar(value=self.prefs.min_solape)
        self.spin_solape = ttk.Spinbox(
            marco, from_=0.05, to=1.0, increment=0.05,
            textvariable=self.var_min_solape, width=8,
        )
        self.spin_solape.grid(row=1, column=1, sticky="e", padx=(8, 0), pady=(5, 0))

        self.var_ayuda_criterio = tk.StringVar()
        ttk.Label(
            marco, textvariable=self.var_ayuda_criterio, style="Pista.TLabel",
            wraplength=225, justify="left",
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))

        self.var_general = tk.BooleanVar(value=self.prefs.registrar_general)
        ttk.Checkbutton(
            marco, text="Registrar detecciones generales", variable=self.var_general,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))
        marco.columnconfigure(0, weight=1)

    def _seccion_rendimiento(self, panel: ttk.Frame) -> None:
        """Filtro de movimiento previo."""
        marco = ttk.LabelFrame(panel, text="Rendimiento", padding=8)
        marco.pack(fill="x", pady=(8, 0))

        self.var_hardware = tk.BooleanVar(value=self.prefs.decodificacion_hardware)
        ttk.Checkbutton(
            marco, text="Decodificación por hardware", variable=self.var_hardware,
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(
            marco,
            text="Descomprime el video ~2.2x más rápido usando la GPU, pero cambia "
                 "ligeramente el color y puede perder detecciones que rocen el "
                 "umbral de confianza. Compruébalo en un video antes de fiarte.",
            style="Pista.TLabel", wraplength=225, justify="left",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(2, 8))

        self.var_movimiento = tk.BooleanVar(value=self.prefs.filtro_movimiento)
        ttk.Checkbutton(
            marco, text="Filtro de movimiento previo", variable=self.var_movimiento,
            command=self._actualizar_estado_movimiento,
        ).grid(row=2, column=0, columnspan=2, sticky="w")

        ttk.Label(marco, text="Umbral:").grid(row=3, column=0, sticky="w", pady=(5, 0))
        self.var_umbral_mov = tk.DoubleVar(value=self.prefs.umbral_movimiento)
        self.spin_movimiento = ttk.Spinbox(
            marco, from_=0.0005, to=0.05, increment=0.0005, format="%.4f",
            textvariable=self.var_umbral_mov, width=8,
        )
        self.spin_movimiento.grid(row=3, column=1, sticky="e", padx=(8, 0), pady=(5, 0))

        ttk.Label(
            marco,
            text="Omite los frames sin cambios para no ejecutar el modelo sobre "
                 "imagen estática. Bajarlo es más seguro; subirlo, más rápido.",
            style="Pista.TLabel", wraplength=225, justify="left",
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(6, 0))
        marco.columnconfigure(0, weight=1)

    def _seccion_rostros(self, panel: ttk.Frame) -> None:
        """Detección e identificación de rostros."""
        marco = ttk.LabelFrame(panel, text="Rostros", padding=8)
        marco.pack(fill="x", pady=(8, 0))

        self.var_rostros = tk.BooleanVar(value=self.prefs.detectar_rostros)
        ttk.Checkbutton(
            marco, text="Detectar rostros", variable=self.var_rostros,
            command=self._actualizar_estado_rostros,
        ).grid(row=0, column=0, columnspan=2, sticky="w")

        ttk.Label(marco, text="Backend:").grid(row=1, column=0, sticky="w", pady=(5, 0))
        self.backends = backends_rostros_disponibles()
        self.var_backend_rostros = tk.StringVar(
            value=self.prefs.backend_rostros
            if self.prefs.backend_rostros in self.backends
            else (self.backends[0] if self.backends else "")
        )
        self.combo_backend = ttk.Combobox(
            marco, textvariable=self.var_backend_rostros, values=self.backends,
            state="readonly", width=12,
        )
        self.combo_backend.grid(row=1, column=1, sticky="e", padx=(8, 0), pady=(5, 0))
        self.combo_backend.bind(
            "<<ComboboxSelected>>", lambda _e: self._actualizar_estado_rostros()
        )

        self.var_recortes = tk.BooleanVar(value=self.prefs.guardar_recortes_rostros)
        self.check_recortes = ttk.Checkbutton(
            marco, text="Guardar recortes", variable=self.var_recortes
        )
        self.check_recortes.grid(row=2, column=0, columnspan=2, sticky="w", pady=(5, 0))

        self.var_identificar = tk.BooleanVar(value=self.prefs.identificar_rostros)
        self.check_identificar = ttk.Checkbutton(
            marco, text="Identificar personas conocidas",
            variable=self.var_identificar, command=self._actualizar_estado_rostros,
        )
        self.check_identificar.grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(5, 0)
        )

        self.boton_personas = ttk.Button(
            marco, text="Catálogo de personas…", command=self.elegir_carpeta_personas
        )
        self.boton_personas.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        self.var_personas = tk.StringVar(value="(sin catálogo)")
        ttk.Label(
            marco, textvariable=self.var_personas, style="Pista.TLabel",
            wraplength=225, justify="left",
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(4, 0))
        marco.columnconfigure(0, weight=1)

    # ---------------------------------------------------------------- pestañas

    def _construir_pestanas(self, padre: ttk.Frame) -> None:
        """Columna derecha: vista previa, tabla de eventos y registro."""
        self.pestanas = ttk.Notebook(padre)
        self.pestanas.pack(side="left", fill="both", expand=True)

        self._construir_pestana_previa()
        self._construir_pestana_eventos()
        self._construir_pestana_registro()

    def _construir_pestana_previa(self) -> None:
        """Pestaña con el frame de referencia y el dibujo de la zona."""
        marco = ttk.Frame(self.pestanas, padding=8)
        self.pestanas.add(marco, text="  Vista previa  ")

        barra = ttk.Frame(marco)
        barra.pack(fill="x")
        ttk.Label(barra, text="Video:").pack(side="left")
        self.var_video_previa = tk.StringVar()
        self.combo_videos = ttk.Combobox(
            barra, textvariable=self.var_video_previa, state="readonly", width=50
        )
        self.combo_videos.pack(side="left", padx=6)
        self.combo_videos.bind(
            "<<ComboboxSelected>>", lambda _e: self.cargar_vista_previa()
        )
        ttk.Button(barra, text="Borrar zona", command=self.borrar_zona).pack(side="right")
        self.var_zona = tk.StringVar(value="Zona: sin definir")
        ttk.Label(barra, textvariable=self.var_zona, style="Titulo.TLabel").pack(
            side="right", padx=10
        )

        ttk.Label(
            marco,
            text="Elige el frame de referencia y arrastra con el ratón para dibujar "
                 "la zona de la puerta.",
            style="Pista.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        self.canvas = tk.Canvas(
            marco, width=ANCHO_PREVIA, height=ALTO_PREVIA, background="#20242b",
            highlightthickness=1, highlightbackground="#9aa0a6", cursor="cross",
        )
        self.canvas.pack(pady=6)
        self.canvas.bind("<ButtonPress-1>", self._inicio_arrastre)
        self.canvas.bind("<B1-Motion>", self._durante_arrastre)
        self.canvas.bind("<ButtonRelease-1>", self._fin_arrastre)

        navegacion = ttk.Frame(marco)
        navegacion.pack(fill="x")
        self.escala_frames = ttk.Scale(
            navegacion, from_=0, to=0, orient="horizontal",
            command=self._al_mover_escala,
        )
        self.escala_frames.pack(fill="x", pady=(0, 4))

        fila = ttk.Frame(navegacion)
        fila.pack(fill="x")
        for texto, salto in (("<<", -300), ("<", -30), (">", 30), (">>", 300)):
            ttk.Button(
                fila, text=texto, width=4, command=lambda s=salto: self._saltar(s)
            ).pack(side="left")

        ttk.Label(fila, text="Frame:").pack(side="left", padx=(12, 2))
        self.var_frame = tk.StringVar(value="0")
        entrada_frame = ttk.Entry(fila, textvariable=self.var_frame, width=10)
        entrada_frame.pack(side="left")
        entrada_frame.bind("<Return>", lambda _e: self._ir_a_frame())

        ttk.Label(fila, text="Ir a:").pack(side="left", padx=(12, 2))
        self.var_tiempo_ir = tk.StringVar(value="00:00:00")
        entrada_tiempo = ttk.Entry(fila, textvariable=self.var_tiempo_ir, width=10)
        entrada_tiempo.pack(side="left")
        entrada_tiempo.bind("<Return>", lambda _e: self._ir_a_tiempo())
        ttk.Button(fila, text="Ir", width=4, command=self._ir_a_tiempo).pack(
            side="left", padx=2
        )

        self.var_info_frame = tk.StringVar()
        ttk.Label(
            navegacion, textvariable=self.var_info_frame, style="Pista.TLabel"
        ).pack(anchor="w", pady=(4, 0))

    def _construir_pestana_eventos(self) -> None:
        """Pestaña con la tabla de eventos detectados."""
        marco = ttk.Frame(self.pestanas, padding=8)
        self.pestanas.add(marco, text="  Eventos  ")

        barra = ttk.Frame(marco)
        barra.pack(fill="x", pady=(0, 6))
        ttk.Label(barra, text="Mostrar:").pack(side="left")
        self.var_filtro_tipo = tk.StringVar(value="todos")
        for texto, valor in (
            ("Todos", "todos"),
            ("Solo zona", TIPO_ZONA),
            ("Solo generales", TIPO_GENERAL),
        ):
            ttk.Radiobutton(
                barra, text=texto, value=valor, variable=self.var_filtro_tipo,
                command=self._refrescar_tabla,
            ).pack(side="left", padx=(8, 0))
        ttk.Label(
            barra, text="Doble clic en una fila abre su miniatura.",
            style="Pista.TLabel",
        ).pack(side="right")

        contenedor = ttk.Frame(marco)
        contenedor.pack(fill="both", expand=True)

        columnas = (
            "tipo", "archivo", "inicio", "fin", "duracion",
            "n_personas", "direccion", "rostros", "personas",
        )
        self.tabla = ttk.Treeview(
            contenedor, columns=columnas, show="headings", selectmode="browse"
        )
        encabezados = {
            "tipo": ("Tipo", 65, "w"),
            "archivo": ("Archivo", 200, "w"),
            "inicio": ("Inicio", 70, "e"),
            "fin": ("Fin", 70, "e"),
            "duracion": ("Duración", 70, "e"),
            "n_personas": ("Pers.", 45, "e"),
            "direccion": ("Dirección", 110, "w"),
            "rostros": ("Rostros", 55, "e"),
            "personas": ("Reconocidas", 120, "w"),
        }
        for col, (titulo, ancho, anclaje) in encabezados.items():
            self.tabla.heading(col, text=titulo)
            self.tabla.column(
                col, width=ancho, anchor=anclaje, stretch=(col == "archivo")
            )
        self.tabla.tag_configure(TIPO_ZONA, background=COLOR_ZONA)
        self.tabla.tag_configure(TIPO_GENERAL, background=COLOR_GENERAL)
        self.tabla.bind("<Double-1>", self._abrir_miniatura_seleccionada)

        scroll = ttk.Scrollbar(contenedor, orient="vertical", command=self.tabla.yview)
        self.tabla.configure(yscrollcommand=scroll.set)
        self.tabla.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

    def _construir_pestana_registro(self) -> None:
        """Pestaña con el registro de texto del análisis."""
        marco = ttk.Frame(self.pestanas, padding=8)
        self.pestanas.add(marco, text="  Registro  ")

        self.texto_log = tk.Text(
            marco, wrap="none", state="disabled", font=("Consolas", 9),
            background="#1e2127", foreground="#d8dee9", insertbackground="#d8dee9",
        )
        scroll = ttk.Scrollbar(marco, orient="vertical", command=self.texto_log.yview)
        self.texto_log.configure(yscrollcommand=scroll.set)
        self.texto_log.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

    def _construir_barra_inferior(self) -> None:
        """Progreso, estado y botones de acción."""
        marco = ttk.Frame(self)
        marco.pack(fill="x", pady=(8, 0))

        self.progreso = ttk.Progressbar(marco, mode="determinate", maximum=100.0)
        self.progreso.pack(fill="x")

        fila = ttk.Frame(marco)
        fila.pack(fill="x", pady=(6, 0))

        self.var_estado = tk.StringVar(value="Listo.")
        ttk.Label(fila, textvariable=self.var_estado).pack(side="left")

        self.boton_informe = ttk.Button(
            fila, text="Abrir informe", command=self.abrir_informe, state="disabled"
        )
        self.boton_informe.pack(side="right", padx=(6, 0))
        self.boton_resultados = ttk.Button(
            fila, text="Abrir carpeta", command=self.abrir_resultados, state="disabled"
        )
        self.boton_resultados.pack(side="right", padx=(6, 0))
        self.boton_cancelar = ttk.Button(
            fila, text="Cancelar", command=self.cancelar_analisis, state="disabled"
        )
        self.boton_cancelar.pack(side="right", padx=(6, 0))
        self.boton_iniciar = ttk.Button(
            fila, text="Iniciar análisis", style="Accion.TButton",
            command=self.iniciar_analisis,
        )
        self.boton_iniciar.pack(side="right")

    def _restaurar_preferencias(self) -> None:
        """Aplica la configuración guardada al abrir la aplicación."""
        if self.prefs.carpeta_salida:
            self.var_salida.set(self.prefs.carpeta_salida)
            self.boton_resultados.configure(state="normal")
        if self.prefs.carpeta_personas:
            self.var_personas.set(Path(self.prefs.carpeta_personas).name)
        if self.prefs.carpeta_videos and Path(self.prefs.carpeta_videos).is_dir():
            self._cargar_carpeta(self.prefs.carpeta_videos)
        self._actualizar_etiqueta_zona()

    # ------------------------------------------------------------------- estados

    def _anunciar_rutas(self) -> None:
        """Deja en el registro dónde viven la configuración y el log.

        Es la única pista que tiene el usuario para encontrar el log cuando algo
        falla en el ejecutable, donde esas rutas no son evidentes.
        """
        self.log(f"Configuración: {RUTA_CONFIG}")
        ruta_log = registro.ruta_log()
        if ruta_log is not None:
            self.log(f"Registro: {ruta_log}")
        else:
            self.log("Aviso: no se pudo abrir el fichero de registro.")

    def _actualizar_estado_criterio(self) -> None:
        """Explica el criterio elegido y habilita el umbral solo si aplica."""
        ayudas = {
            "pies": "Cuenta si el punto de apoyo (centro del borde inferior de la "
                    "caja) cae en la zona. El más fiable en pasillos.",
            "centro": "Cuenta si el centro de la caja cae en la zona.",
            "solape": "Cuenta si al menos el porcentaje indicado del área de la "
                      "persona está dentro de la zona.",
        }
        criterio = self.var_criterio.get()
        self.var_ayuda_criterio.set(ayudas.get(criterio, ""))
        self.spin_solape.configure(
            state="normal" if criterio == "solape" else "disabled"
        )

    def _actualizar_estado_movimiento(self) -> None:
        """Habilita el umbral solo si el filtro de movimiento está activo."""
        self.spin_movimiento.configure(
            state="normal" if self.var_movimiento.get() else "disabled"
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

    # ------------------------------------------------------------------ carpetas

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
            title="Carpeta de resultados", initialdir=self.prefs.carpeta_salida or None
        )
        if carpeta:
            self.prefs.carpeta_salida = carpeta
            self.var_salida.set(carpeta)
            self.boton_resultados.configure(state="normal")

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
        self.var_personas.set(f"{Path(carpeta).name}: {len(subcarpetas)} persona/s")
        if not subcarpetas:
            messagebox.showwarning(
                "Catálogo vacío",
                "Esa carpeta no tiene subcarpetas.\n\n"
                "Cada persona necesita su propia subcarpeta con fotos suyas; el "
                "nombre de la subcarpeta es la etiqueta que saldrá en el informe.",
            )

    def abrir_resultados(self) -> None:
        """Abre la carpeta de resultados en el explorador."""
        carpeta = self.prefs.carpeta_salida
        if carpeta and Path(carpeta).is_dir():
            abrir_en_sistema(carpeta)
        else:
            messagebox.showwarning(
                "Carpeta de resultados", "Todavía no hay una carpeta válida."
            )

    def abrir_informe(self) -> None:
        """Abre el informe generado; prefiere el PDF si existe."""
        if not self.informes:
            return
        pdf = next((r for r in self.informes if r.suffix == ".pdf"), None)
        abrir_en_sistema(pdf or self.informes[0])

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
        self.escala_frames.configure(to=max(0, self._info_previa.total_frames - 1))
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
            f"{formatear_tiempo(segundo)} de {formatear_tiempo(info.duracion)}   ·   "
            f"{ancho}x{alto} a {info.fps:.0f} fps   ·   {info.total_frames} frames"
        )

    def _dibujar_zona_guardada(self) -> None:
        """Dibuja sobre el canvas la zona de la puerta ya definida, si existe."""
        if self.zona_puerta is None or self.imagen_previa is None:
            return
        x1, y1, x2, y2 = (c * self.escala_previa for c in self.zona_puerta)
        self._rect_id = self.canvas.create_rectangle(
            x1, y1, x2, y2, outline="#ff3b3b", width=2
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
            messagebox.showwarning(
                "Tiempo", "Usa el formato HH:MM:SS, MM:SS o segundos."
            )
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

    def _inicio_arrastre(self, evento: tk.Event) -> None:
        """Registra el punto inicial del rectángulo."""
        if self.imagen_previa is None:
            return
        self._arrastre_inicio = (
            self.canvas.canvasx(evento.x), self.canvas.canvasy(evento.y)
        )
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
            self._rect_id, x0, y0,
            self.canvas.canvasx(evento.x), self.canvas.canvasy(evento.y),
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
            self.var_zona.set(f"Zona: ({x1}, {y1}) — ({x2}, {y2})")

    # ------------------------------------------------------------- tabla y log

    def log(self, mensaje: str) -> None:
        """Escribe una línea en el registro (solo desde el hilo principal)."""
        registro.log.info("%s", mensaje)
        self.texto_log.configure(state="normal")
        self.texto_log.insert("end", mensaje + "\n")
        self.texto_log.see("end")
        self.texto_log.configure(state="disabled")

    def _registrar_evento(self, evento: Evento) -> None:
        """Guarda un evento y lo muestra en la tabla si el filtro lo permite."""
        self._eventos.append(evento)
        self._insertar_fila(evento)

    def _insertar_fila(self, evento: Evento) -> None:
        """Inserta una fila en la tabla si encaja con el filtro activo."""
        filtro = self.var_filtro_tipo.get()
        if filtro != "todos" and evento.tipo != filtro:
            return
        iid = self.tabla.insert(
            "", "end",
            values=(
                evento.tipo,
                evento.archivo,
                formatear_tiempo(evento.inicio),
                formatear_tiempo(evento.fin),
                f"{evento.duracion:.0f} s",
                evento.n_personas or "",
                evento.direccion,
                evento.rostros or "",
                evento.personas,
            ),
            tags=(evento.tipo,),
        )
        self._filas[iid] = evento
        self.tabla.see(iid)

    def _refrescar_tabla(self) -> None:
        """Redibuja la tabla al cambiar el filtro por tipo."""
        self.tabla.delete(*self.tabla.get_children())
        self._filas.clear()
        for evento in self._eventos:
            self._insertar_fila(evento)

    def _abrir_miniatura_seleccionada(self, _evento: tk.Event) -> None:
        """Abre la miniatura del evento sobre el que se hizo doble clic."""
        seleccion = self.tabla.selection()
        if not seleccion:
            return
        evento = self._filas.get(seleccion[0])
        if evento is None or not evento.miniatura:
            messagebox.showinfo("Miniatura", "Este evento no tiene miniatura.")
            return
        if not Path(evento.miniatura).is_file():
            messagebox.showwarning("Miniatura", "El archivo ya no existe.")
            return
        abrir_en_sistema(evento.miniatura)

    def _encolar(self, tipo: str, dato: Any = None) -> None:
        """Encola un mensaje del hilo de análisis para la interfaz."""
        self.mensajes.put((tipo, dato))

    def _procesar_mensajes(self) -> None:
        """Consume la cola de mensajes y actualiza la interfaz cada 100 ms."""
        try:
            while True:
                tipo, dato = self.mensajes.get_nowait()
                if tipo == "log":
                    self.log(dato)
                elif tipo == "evento":
                    self._registrar_evento(dato)
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
            messagebox.showwarning("Salida", "Selecciona la carpeta de resultados.")
            return
        if self.zona_puerta is None:
            messagebox.showwarning(
                "Zona", "Dibuja la zona de la puerta sobre la vista previa."
            )
            return
        if self.var_identificar.get() and not self.prefs.carpeta_personas:
            messagebox.showwarning(
                "Personas", "Elige la carpeta con las fotos de las personas conocidas."
            )
            return

        try:
            config = self._construir_configuracion()
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror("Parámetros", str(exc))
            return

        self._guardar_preferencias()
        self.cancelar_evento = threading.Event()
        self.informes = []
        self._eventos.clear()
        self._filas.clear()
        self.tabla.delete(*self.tabla.get_children())
        self.progreso["value"] = 0.0
        self.boton_iniciar.configure(state="disabled")
        self.boton_cancelar.configure(state="normal")
        self.boton_informe.configure(state="disabled")
        self.var_estado.set("Iniciando análisis…")
        self.log("=" * 70)
        self.log("Iniciando análisis…")

        self.hilo = threading.Thread(
            target=self._ejecutar_analisis, args=(config,), daemon=True
        )
        self.hilo.start()

    def _construir_configuracion(self) -> ConfiguracionAnalisis:
        """Arma y valida la configuración a partir de los widgets."""
        config = ConfiguracionAnalisis(
            zona_puerta=self.zona_puerta,  # type: ignore[arg-type]
            fps_analisis=float(self.var_fps.get()),
            tolerancia_segundos=float(self.var_tolerancia.get()),
            modelo=self.var_modelo.get(),
            acelerador=self.var_acelerador.get(),
            decodificacion_hardware=self.var_hardware.get(),
            filtro_movimiento=self.var_movimiento.get(),
            umbral_movimiento=float(self.var_umbral_mov.get()),
            criterio_zona=self.var_criterio.get(),
            min_solape=float(self.var_min_solape.get()),
            registrar_general=self.var_general.get(),
            detectar_rostros=self.var_rostros.get(),
            backend_rostros=self.var_backend_rostros.get(),
            guardar_recortes_rostros=self.var_recortes.get(),
            identificar_rostros=self.var_identificar.get(),
            carpeta_personas=self.prefs.carpeta_personas,
            usar_tracking=self.var_tracking.get(),
        )
        config.validar()
        return config

    def _ejecutar_analisis(self, config: ConfiguracionAnalisis) -> None:
        """Corre el análisis en segundo plano (no toca widgets directamente)."""
        try:
            analizador = AnalizadorPuerta(
                config=config,
                cancelar=self.cancelar_evento,
                on_progreso=lambda n, pv, pt: self._encolar("progreso", (n, pv, pt)),
                on_log=lambda m: self._encolar("log", m),
                on_evento=lambda e: self._encolar("evento", e),
            )
            resultados = analizador.analizar_videos(
                self.videos, self.prefs.carpeta_salida,
                incremental=self.prefs.incremental,
            )
            informes: list[Path] = []
            try:
                informes = reporte.generar_informes(
                    resultados, config, self.prefs.carpeta_salida,
                    self.prefs.carpeta_videos,
                )
            except Exception as exc:  # noqa: BLE001 - el informe no debe tumbar todo
                registro.log.exception("Falló la generación del informe")
                self._encolar("log", f"No se pudo generar el informe: {exc}")
            self._encolar("fin", (resultados, informes))
        except Exception as exc:  # noqa: BLE001 - cualquier fallo debe llegar a la GUI
            registro.log.exception("Falló el análisis")
            self._encolar("error", f"El análisis falló: {exc}")
            self._encolar("fin", ([], []))

    def cancelar_analisis(self) -> None:
        """Pide al hilo de análisis que se detenga."""
        if self.hilo is not None and self.hilo.is_alive():
            self.cancelar_evento.set()
            self.var_estado.set("Cancelando…")
            self.log("Cancelación solicitada…")

    def _al_terminar(self, datos: tuple[list[ResultadoVideo], list[Path]]) -> None:
        """Muestra el resumen final y restablece los botones."""
        resultados, informes = datos
        self.informes = informes
        self.boton_iniciar.configure(state="normal")
        self.boton_cancelar.configure(state="disabled")
        self.boton_resultados.configure(state="normal")
        self.boton_informe.configure(state="normal" if informes else "disabled")

        def contar(eventos: list[Evento], tipo: str) -> int:
            """Cuenta los eventos de un tipo concreto."""
            return sum(1 for e in eventos if e.tipo == tipo)

        self.log("-" * 70)
        self.log(f"{'RESUMEN':<38}{'en zona':>10}{'generales':>12}")
        total_zona = total_general = 0
        omitidos = analizados = 0
        reutilizados = sum(1 for r in resultados if r.reutilizado)
        for resultado in resultados:
            omitidos += resultado.frames_omitidos
            analizados += resultado.frames_analizados
            if resultado.error:
                self.log(f"  {resultado.archivo}: ERROR — {resultado.error}")
                continue
            zona = contar(resultado.eventos, TIPO_ZONA)
            general = contar(resultado.eventos, TIPO_GENERAL)
            total_zona += zona
            total_general += general
            self.log(f"  {resultado.archivo[:34]:<36}{zona:>10}{general:>12}")

        self.log(f"  {'TOTAL':<36}{total_zona:>10}{total_general:>12}")
        if reutilizados:
            self.log(
                f"Análisis incremental: {reutilizados} de {len(resultados)} videos "
                f"reutilizados del manifiesto."
            )
        if omitidos:
            total = omitidos + analizados
            self.log(
                f"Filtro de movimiento: {omitidos} de {total} frames omitidos "
                f"({omitidos / total:.0%} menos de inferencia)."
            )
        for informe in informes:
            self.log(f"Informe: {informe}")

        self.progreso["value"] = 100.0
        self.var_estado.set(
            f"Terminado. {total_zona} eventos en la zona, {total_general} generales."
        )
        if resultados:
            self.pestanas.select(1)  # llevar al usuario a la tabla de eventos

    # -------------------------------------------------------------------- cierre

    def _guardar_preferencias(self) -> None:
        """Vuelca los valores actuales de la interfaz a ``config.json``."""
        try:
            self.prefs.fps_analisis = float(self.var_fps.get())
            self.prefs.tolerancia_segundos = float(self.var_tolerancia.get())
            self.prefs.min_solape = float(self.var_min_solape.get())
            self.prefs.umbral_movimiento = float(self.var_umbral_mov.get())
        except tk.TclError:
            pass
        self.prefs.modelo = self.var_modelo.get()
        self.prefs.acelerador = self.var_acelerador.get()
        self.prefs.decodificacion_hardware = self.var_hardware.get()
        self.prefs.filtro_movimiento = self.var_movimiento.get()
        self.prefs.criterio_zona = self.var_criterio.get()
        self.prefs.registrar_general = self.var_general.get()
        self.prefs.detectar_rostros = self.var_rostros.get()
        self.prefs.backend_rostros = self.var_backend_rostros.get()
        self.prefs.guardar_recortes_rostros = self.var_recortes.get()
        self.prefs.identificar_rostros = self.var_identificar.get()
        self.prefs.usar_tracking = self.var_tracking.get()
        self.prefs.incremental = self.var_incremental.get()
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
    registro.configurar()
    root = tk.Tk()
    # Sin esto, un fallo en un callback solo se imprime por stderr, que no
    # existe en el ejecutable compilado sin consola.
    root.report_callback_exception = registro.excepcion_de_tk
    root.title("DeCam — Detección de personas en la puerta")
    root.geometry("1280x860")
    root.minsize(1120, 760)
    AplicacionDeCam(root)
    root.mainloop()


if __name__ == "__main__":
    main()
