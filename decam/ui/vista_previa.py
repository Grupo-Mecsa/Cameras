"""Pestaña de vista previa: frame de referencia, navegación y dibujo de la zona."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Callable, Optional

import cv2
from PIL import Image, ImageTk

from decam.callbacks import CallbackLog
from decam.eventos import formatear_tiempo
from decam.ui.utilidades import parsear_tiempo
from decam.video import InfoVideo, info_video, leer_frame
from decam.zona import normalizar_zona

ANCHO_PREVIA = 720
ALTO_PREVIA = 405
COLOR_ZONA = "#ff3b3b"

Zona = tuple[int, int, int, int]


class VistaPrevia(ttk.Frame):
    """Muestra un frame del video elegido y deja dibujar la zona sobre él.

    La zona se guarda en coordenadas del video (no del canvas) y se expone en
    :attr:`zona`; cada cambio se notifica por ``on_zona_cambiada``.
    """

    def __init__(
        self,
        padre: tk.Misc,
        on_log: CallbackLog,
        zona_inicial: Optional[Zona] = None,
        on_zona_cambiada: Optional[Callable[[Optional[Zona]], None]] = None,
    ) -> None:
        super().__init__(padre, padding=8)
        self._log = on_log
        self._on_zona_cambiada = on_zona_cambiada
        self.zona: Optional[Zona] = zona_inicial

        self._videos: dict[str, Path] = {}
        self._captura: Optional[cv2.VideoCapture] = None
        self._info: Optional[InfoVideo] = None
        self._frame_actual = 0
        self._seek_pendiente: Optional[str] = None
        self._imagen: Optional[ImageTk.PhotoImage] = None
        self._escala = 1.0
        self._arrastre_inicio: Optional[tuple[float, float]] = None
        self._rect_id: Optional[int] = None

        self._construir()
        self._actualizar_etiqueta_zona()

    # ------------------------------------------------------------- construcción

    def _construir(self) -> None:
        barra = ttk.Frame(self)
        barra.pack(fill="x")
        ttk.Label(barra, text="Video:").pack(side="left")
        self.var_video = tk.StringVar()
        self.combo_videos = ttk.Combobox(
            barra, textvariable=self.var_video, state="readonly", width=50
        )
        self.combo_videos.pack(side="left", padx=6)
        self.combo_videos.bind("<<ComboboxSelected>>", lambda _e: self._cargar_elegido())
        ttk.Button(barra, text="Borrar zona", command=self.borrar_zona).pack(side="right")
        self.var_zona = tk.StringVar()
        ttk.Label(barra, textvariable=self.var_zona, style="Titulo.TLabel").pack(
            side="right", padx=10
        )

        ttk.Label(
            self,
            text="Elige el frame de referencia y arrastra con el ratón para dibujar "
                 "la zona de la puerta.",
            style="Pista.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        self.canvas = tk.Canvas(
            self, width=ANCHO_PREVIA, height=ALTO_PREVIA, background="#20242b",
            highlightthickness=1, highlightbackground="#9aa0a6", cursor="cross",
        )
        self.canvas.pack(pady=6)
        self.canvas.bind("<ButtonPress-1>", self._inicio_arrastre)
        self.canvas.bind("<B1-Motion>", self._durante_arrastre)
        self.canvas.bind("<ButtonRelease-1>", self._fin_arrastre)

        navegacion = ttk.Frame(self)
        navegacion.pack(fill="x")
        self.escala_frames = ttk.Scale(
            navegacion, from_=0, to=0, orient="horizontal", command=self._al_mover_escala,
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
        self.var_tiempo = tk.StringVar(value="00:00:00")
        entrada_tiempo = ttk.Entry(fila, textvariable=self.var_tiempo, width=10)
        entrada_tiempo.pack(side="left")
        entrada_tiempo.bind("<Return>", lambda _e: self._ir_a_tiempo())
        ttk.Button(fila, text="Ir", width=4, command=self._ir_a_tiempo).pack(
            side="left", padx=2
        )

        self.var_info = tk.StringVar()
        ttk.Label(navegacion, textvariable=self.var_info, style="Pista.TLabel").pack(
            anchor="w", pady=(4, 0)
        )

    # ------------------------------------------------------------------- videos

    def mostrar_videos(self, videos: dict[str, Path]) -> None:
        """Rellena el desplegable y carga el primer video, si lo hay.

        Args:
            videos: nombre a mostrar -> ruta del video.
        """
        self._videos = dict(videos)
        nombres = list(self._videos)
        self.combo_videos.configure(values=nombres)
        if nombres:
            self.var_video.set(nombres[0])
            self._cargar_elegido()
        else:
            self.var_video.set("")
            self.cerrar()
            self.canvas.delete("all")
            self._imagen = None
            self.var_info.set("")

    def _cargar_elegido(self) -> None:
        """Abre el video seleccionado en el desplegable y muestra su primer frame."""
        video = self._videos.get(self.var_video.get())
        if video is None:
            return
        self.cerrar()
        captura = cv2.VideoCapture(str(video))
        if not captura.isOpened():
            captura.release()
            messagebox.showerror("Vista previa", f"No se pudo abrir {video.name}.")
            return
        self._captura = captura
        self._info = info_video(captura)
        self.escala_frames.configure(to=max(0, self._info.total_frames - 1))
        self._mostrar_frame(0)

    def cerrar(self) -> None:
        """Libera la captura abierta, si la hay."""
        if self._captura is not None:
            self._captura.release()
            self._captura = None
        self._info = None

    # --------------------------------------------------------------- navegación

    def _mostrar_frame(self, indice: int) -> None:
        """Dibuja en el canvas el frame ``indice`` del video de referencia."""
        if self._captura is None or self._info is None:
            return
        info = self._info
        indice = max(0, min(indice, max(0, info.total_frames - 1)))

        frame = leer_frame(self._captura, indice)
        if frame is None:
            self._log(f"No se pudo leer el frame {indice}.")
            return

        self._frame_actual = indice
        alto, ancho = frame.shape[:2]
        self._escala = min(ANCHO_PREVIA / ancho, ALTO_PREVIA / alto, 1.0)
        nuevo = (max(1, int(ancho * self._escala)), max(1, int(alto * self._escala)))
        imagen = cv2.cvtColor(cv2.resize(frame, nuevo), cv2.COLOR_BGR2RGB)
        self._imagen = ImageTk.PhotoImage(Image.fromarray(imagen))

        self.canvas.delete("all")
        self.canvas.configure(width=nuevo[0], height=nuevo[1])
        self.canvas.create_image(0, 0, anchor="nw", image=self._imagen)
        self._rect_id = None
        self._dibujar_zona()

        segundo = indice / info.fps if info.fps else 0.0
        self.var_frame.set(str(indice))
        self.var_tiempo.set(formatear_tiempo(segundo))
        self.escala_frames.set(indice)
        self.var_info.set(
            f"{formatear_tiempo(segundo)} de {formatear_tiempo(info.duracion)}   ·   "
            f"{ancho}x{alto} a {info.fps:.0f} fps   ·   {info.total_frames} frames"
        )

    def _al_mover_escala(self, valor: str) -> None:
        """Programa la carga del frame tras mover el deslizador.

        El salto se retrasa 150 ms para no decodificar en cada píxel arrastrado.
        """
        if self._captura is None:
            return
        indice = int(float(valor))
        if indice == self._frame_actual:
            return
        if self._seek_pendiente is not None:
            self.after_cancel(self._seek_pendiente)
        self._seek_pendiente = self.after(150, lambda: self._mostrar_frame(indice))

    def _saltar(self, delta_frames: int) -> None:
        self._mostrar_frame(self._frame_actual + delta_frames)

    def _ir_a_frame(self) -> None:
        try:
            self._mostrar_frame(int(self.var_frame.get()))
        except ValueError:
            messagebox.showwarning("Frame", "Escribe un número de frame válido.")

    def _ir_a_tiempo(self) -> None:
        if self._info is None:
            return
        try:
            segundos = parsear_tiempo(self.var_tiempo.get())
        except ValueError:
            messagebox.showwarning("Tiempo", "Usa el formato HH:MM:SS, MM:SS o segundos.")
            return
        self._mostrar_frame(int(segundos * self._info.fps))

    # --------------------------------------------------------------------- zona

    def _dibujar_zona(self) -> None:
        """Dibuja sobre el canvas la zona ya definida, si existe."""
        if self.zona is None or self._imagen is None:
            return
        x1, y1, x2, y2 = (c * self._escala for c in self.zona)
        self._rect_id = self.canvas.create_rectangle(
            x1, y1, x2, y2, outline=COLOR_ZONA, width=2
        )

    def _inicio_arrastre(self, evento: tk.Event) -> None:
        if self._imagen is None:
            return
        self._arrastre_inicio = (self.canvas.canvasx(evento.x), self.canvas.canvasy(evento.y))
        if self._rect_id is not None:
            self.canvas.delete(self._rect_id)
        self._rect_id = self.canvas.create_rectangle(
            *self._arrastre_inicio, *self._arrastre_inicio, outline=COLOR_ZONA, width=2
        )

    def _durante_arrastre(self, evento: tk.Event) -> None:
        if self._arrastre_inicio is None or self._rect_id is None:
            return
        x0, y0 = self._arrastre_inicio
        self.canvas.coords(
            self._rect_id, x0, y0, self.canvas.canvasx(evento.x), self.canvas.canvasy(evento.y),
        )

    def _fin_arrastre(self, evento: tk.Event) -> None:
        """Convierte el rectángulo del canvas a coordenadas reales del video."""
        if self._arrastre_inicio is None:
            return
        x0, y0 = self._arrastre_inicio
        x1, y1 = self.canvas.canvasx(evento.x), self.canvas.canvasy(evento.y)
        self._arrastre_inicio = None

        escala = self._escala or 1.0
        zona = normalizar_zona((x0 / escala, y0 / escala, x1 / escala, y1 / escala))
        if zona[2] - zona[0] < 5 or zona[3] - zona[1] < 5:
            self._log("Zona demasiado pequeña, vuelve a dibujarla.")
            return
        self._establecer_zona(zona)

    def borrar_zona(self) -> None:
        """Elimina la zona de la puerta definida."""
        if self._rect_id is not None:
            self.canvas.delete(self._rect_id)
            self._rect_id = None
        self._establecer_zona(None)

    def _establecer_zona(self, zona: Optional[Zona]) -> None:
        self.zona = zona
        self._actualizar_etiqueta_zona()
        if self._on_zona_cambiada is not None:
            self._on_zona_cambiada(zona)

    def _actualizar_etiqueta_zona(self) -> None:
        if self.zona is None:
            self.var_zona.set("Zona: sin definir")
        else:
            x1, y1, x2, y2 = self.zona
            self.var_zona.set(f"Zona: ({x1}, {y1}) — ({x2}, {y2})")
