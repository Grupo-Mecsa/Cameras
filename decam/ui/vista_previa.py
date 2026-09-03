"""Pestaña de vista previa: frame de referencia, navegación y dibujo de la zona.

La zona se dibuja de dos maneras: arrastrando un rectángulo, o marcando los
vértices de un polígono clic a clic (doble clic o clic derecho para cerrarlo,
Esc para empezar de nuevo). En ambos casos se guarda en coordenadas del video,
no del canvas, como una especificación de :mod:`decam.zona`.
"""

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
from decam.zona import EspecZona, ZonaPoligonal, es_poligono, normalizar_zona

ANCHO_PREVIA = 720
ALTO_PREVIA = 405
COLOR_ZONA = "#ff3b3b"

MODO_RECTANGULO = "rectangulo"
MODO_POLIGONO = "poligono"
AYUDAS = {
    MODO_RECTANGULO: "Elige el frame de referencia y arrastra con el ratón para "
                     "dibujar el rectángulo de la puerta.",
    MODO_POLIGONO: "Haz clic en cada esquina de la zona; doble clic o clic derecho "
                   "para cerrarla, Esc para empezar de nuevo.",
}
#: Tamaño mínimo de un rectángulo, en píxeles del video.
LADO_MINIMO = 5
#: Dos vértices más cerca que esto (en el canvas) se consideran el mismo: es lo
#: que deja el doble clic con el que se cierra el polígono.
RADIO_VERTICE = 4
ETIQUETA_EN_CURSO = "en_curso"


class VistaPrevia(ttk.Frame):
    """Muestra un frame del video elegido y deja dibujar la zona sobre él.

    La zona actual está en :attr:`zona` y cada cambio se notifica por
    ``on_zona_cambiada``.
    """

    def __init__(
        self,
        padre: tk.Misc,
        on_log: CallbackLog,
        zona_inicial: Optional[EspecZona] = None,
        on_zona_cambiada: Optional[Callable[[Optional[EspecZona]], None]] = None,
    ) -> None:
        super().__init__(padre, padding=8)
        self._log = on_log
        self._on_zona_cambiada = on_zona_cambiada
        self.zona: Optional[EspecZona] = zona_inicial

        self._videos: dict[str, Path] = {}
        self._captura: Optional[cv2.VideoCapture] = None
        self._info: Optional[InfoVideo] = None
        self._frame_actual = 0
        self._seek_pendiente: Optional[str] = None
        self._imagen: Optional[ImageTk.PhotoImage] = None
        self._escala = 1.0
        self._zona_id: Optional[int] = None
        # Rectángulo en curso.
        self._arrastre_inicio: Optional[tuple[float, float]] = None
        # Polígono en curso, en coordenadas del canvas.
        self._vertices: list[tuple[float, float]] = []

        self._construir()
        self._actualizar_etiqueta_zona()

    # ------------------------------------------------------------- construcción

    def _construir(self) -> None:
        barra = ttk.Frame(self)
        barra.pack(fill="x")
        ttk.Label(barra, text="Video:").pack(side="left")
        self.var_video = tk.StringVar()
        self.combo_videos = ttk.Combobox(
            barra, textvariable=self.var_video, state="readonly", width=42
        )
        self.combo_videos.pack(side="left", padx=6)
        self.combo_videos.bind("<<ComboboxSelected>>", lambda _e: self._cargar_elegido())

        ttk.Label(barra, text="Dibujar:").pack(side="left", padx=(10, 0))
        self.var_modo = tk.StringVar(
            value=MODO_POLIGONO if self.zona and es_poligono(self.zona) else MODO_RECTANGULO
        )
        for texto, valor in (("Rectángulo", MODO_RECTANGULO), ("Polígono", MODO_POLIGONO)):
            ttk.Radiobutton(
                barra, text=texto, value=valor, variable=self.var_modo,
                command=self._cambiar_modo,
            ).pack(side="left", padx=(6, 0))

        ttk.Button(barra, text="Borrar zona", command=self.borrar_zona).pack(side="right")
        self.var_zona = tk.StringVar()
        ttk.Label(barra, textvariable=self.var_zona, style="Titulo.TLabel").pack(
            side="right", padx=10
        )

        self.var_ayuda = tk.StringVar(value=AYUDAS[self.var_modo.get()])
        ttk.Label(self, textvariable=self.var_ayuda, style="Pista.TLabel").pack(
            anchor="w", pady=(4, 0)
        )

        self.canvas = tk.Canvas(
            self, width=ANCHO_PREVIA, height=ALTO_PREVIA, background="#20242b",
            highlightthickness=1, highlightbackground="#9aa0a6", cursor="cross",
        )
        self.canvas.pack(pady=6)
        self.canvas.bind("<ButtonPress-1>", self._clic)
        self.canvas.bind("<B1-Motion>", self._durante_arrastre)
        self.canvas.bind("<ButtonRelease-1>", self._fin_arrastre)
        self.canvas.bind("<Double-1>", lambda _e: self._cerrar_poligono())
        self.canvas.bind("<ButtonPress-3>", lambda _e: self._cerrar_poligono())
        self.canvas.bind("<Escape>", lambda _e: self._cancelar_poligono())

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
            self._vertices = []
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
        self._vertices = []  # un polígono a medias no sobrevive al cambio de frame
        self.canvas.configure(width=nuevo[0], height=nuevo[1])
        self.canvas.create_image(0, 0, anchor="nw", image=self._imagen)
        self._zona_id = None
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
        if es_poligono(self.zona):
            coords = [c * self._escala for punto in self.zona for c in punto]
            self._zona_id = self.canvas.create_polygon(
                *coords, outline=COLOR_ZONA, fill="", width=2
            )
        else:
            x1, y1, x2, y2 = (c * self._escala for c in self.zona)
            self._zona_id = self.canvas.create_rectangle(
                x1, y1, x2, y2, outline=COLOR_ZONA, width=2
            )

    def _borrar_dibujo_zona(self) -> None:
        if self._zona_id is not None:
            self.canvas.delete(self._zona_id)
            self._zona_id = None

    def _cambiar_modo(self) -> None:
        self.var_ayuda.set(AYUDAS[self.var_modo.get()])
        self._cancelar_poligono(avisar=False)

    def _punto_canvas(self, evento: tk.Event) -> tuple[float, float]:
        return (self.canvas.canvasx(evento.x), self.canvas.canvasy(evento.y))

    # --- rectángulo

    def _clic(self, evento: tk.Event) -> None:
        if self._imagen is None:
            return
        self.canvas.focus_set()  # para que llegue Esc
        if self.var_modo.get() == MODO_POLIGONO:
            self._anadir_vertice(self._punto_canvas(evento))
            return
        self._arrastre_inicio = self._punto_canvas(evento)
        self._borrar_dibujo_zona()
        self._zona_id = self.canvas.create_rectangle(
            *self._arrastre_inicio, *self._arrastre_inicio, outline=COLOR_ZONA, width=2
        )

    def _durante_arrastre(self, evento: tk.Event) -> None:
        if self._arrastre_inicio is None or self._zona_id is None:
            return
        x0, y0 = self._arrastre_inicio
        self.canvas.coords(self._zona_id, x0, y0, *self._punto_canvas(evento))

    def _fin_arrastre(self, evento: tk.Event) -> None:
        """Convierte el rectángulo del canvas a coordenadas reales del video."""
        if self._arrastre_inicio is None:
            return
        x0, y0 = self._arrastre_inicio
        x1, y1 = self._punto_canvas(evento)
        self._arrastre_inicio = None

        escala = self._escala or 1.0
        zona = normalizar_zona((x0 / escala, y0 / escala, x1 / escala, y1 / escala))
        if zona[2] - zona[0] < LADO_MINIMO or zona[3] - zona[1] < LADO_MINIMO:
            self._log("Zona demasiado pequeña, vuelve a dibujarla.")
            self._borrar_dibujo_zona()
            self._dibujar_zona()
            return
        self._establecer_zona(zona)

    # --- polígono

    def _anadir_vertice(self, punto: tuple[float, float]) -> None:
        if not self._vertices:
            self._borrar_dibujo_zona()  # se empieza una zona nueva
        else:
            self.canvas.create_line(
                *self._vertices[-1], *punto, fill=COLOR_ZONA, width=2,
                tags=ETIQUETA_EN_CURSO,
            )
        x, y = punto
        self.canvas.create_oval(
            x - RADIO_VERTICE, y - RADIO_VERTICE, x + RADIO_VERTICE, y + RADIO_VERTICE,
            outline=COLOR_ZONA, fill=COLOR_ZONA, tags=ETIQUETA_EN_CURSO,
        )
        self._vertices.append(punto)

    def _cerrar_poligono(self) -> None:
        """Termina el polígono en curso y lo convierte en la zona."""
        if self.var_modo.get() != MODO_POLIGONO or not self._vertices:
            return
        # El doble clic con el que se cierra deja un vértice repetido al final.
        vertices = list(self._vertices)
        while len(vertices) >= 2 and _cerca(vertices[-1], vertices[-2]):
            vertices.pop()

        escala = self._escala or 1.0
        zona = ZonaPoligonal.desde([(x / escala, y / escala) for x, y in vertices])
        try:
            zona.validar()
        except ValueError as exc:
            self._log(f"{exc} Vuelve a dibujarla.")
            self._cancelar_poligono(avisar=False)
            return
        self.canvas.delete(ETIQUETA_EN_CURSO)
        self._vertices = []
        self._establecer_zona(zona.como_tupla)

    def _cancelar_poligono(self, avisar: bool = True) -> None:
        if not self._vertices:
            return
        self.canvas.delete(ETIQUETA_EN_CURSO)
        self._vertices = []
        self._dibujar_zona()  # se recupera la zona que había
        if avisar:
            self._log("Polígono descartado.")

    # --- común

    def borrar_zona(self) -> None:
        """Elimina la zona de la puerta definida."""
        self._cancelar_poligono(avisar=False)
        self._borrar_dibujo_zona()
        self._establecer_zona(None)

    def _establecer_zona(self, zona: Optional[EspecZona]) -> None:
        self.zona = zona
        self._borrar_dibujo_zona()
        self._dibujar_zona()
        self._actualizar_etiqueta_zona()
        if self._on_zona_cambiada is not None:
            self._on_zona_cambiada(zona)

    def _actualizar_etiqueta_zona(self) -> None:
        if self.zona is None:
            self.var_zona.set("Zona: sin definir")
        elif es_poligono(self.zona):
            self.var_zona.set(f"Zona: polígono de {len(self.zona)} puntos")
        else:
            x1, y1, x2, y2 = self.zona
            self.var_zona.set(f"Zona: ({x1}, {y1}) — ({x2}, {y2})")


def _cerca(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return abs(a[0] - b[0]) <= RADIO_VERTICE and abs(a[1] - b[1]) <= RADIO_VERTICE
