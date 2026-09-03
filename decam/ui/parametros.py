"""Panel izquierdo con todos los ajustes del análisis."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable

from decam.aceleradores import aceleradores_disponibles
from decam.configuracion import ConfiguracionAnalisis
from decam.rostros import backends_rostros_disponibles
from decam.ui.preferencias import Preferencias
from decam.zona import CRITERIOS_ZONA, EspecZona

MODELOS = ("yolov8n", "yolov8s", "yolov8m")

#: Ancho de las etiquetas de ayuda, que van debajo de cada control.
ANCHO_AYUDA = 225


class PanelParametros(ttk.Frame):
    """Los ajustes agrupados por tema, y su traducción a configuración.

    Es el único sitio que sabe qué widget corresponde a qué parámetro: hacia
    fuera ofrece :meth:`configuracion` (para lanzar el análisis) y
    :meth:`volcar_en` (para persistir).
    """

    def __init__(
        self,
        padre: tk.Misc,
        prefs: Preferencias,
        on_elegir_personas: Callable[[], None],
    ) -> None:
        """Construye el panel a partir de las preferencias guardadas.

        Args:
            padre: widget contenedor.
            prefs: valores iniciales.
            on_elegir_personas: se llama al pulsar "Catálogo de personas…"; la
                elección de carpeta es cosa de la ventana principal.
        """
        super().__init__(padre)
        self._on_elegir_personas = on_elegir_personas
        self._seccion_analisis(prefs)
        self._seccion_zona(prefs)
        self._seccion_rendimiento(prefs)
        self._seccion_rostros(prefs)
        self._actualizar_estado_criterio()
        self._actualizar_estado_movimiento()
        self._actualizar_estado_rostros()

    # --------------------------------------------------------------- secciones

    def _ayuda(self, marco: ttk.Frame, texto: str, fila: int, pady=(6, 0)) -> None:
        ttk.Label(
            marco, text=texto, style="Pista.TLabel", wraplength=ANCHO_AYUDA,
            justify="left",
        ).grid(row=fila, column=0, columnspan=2, sticky="w", pady=pady)

    def _seccion_analisis(self, prefs: Preferencias) -> None:
        """Parámetros generales del análisis."""
        marco = ttk.LabelFrame(self, text="Análisis", padding=8)
        marco.pack(fill="x")

        ttk.Label(marco, text="Frames por segundo:").grid(row=0, column=0, sticky="w")
        self.var_fps = tk.DoubleVar(value=prefs.fps_analisis)
        ttk.Spinbox(
            marco, from_=0.1, to=30.0, increment=0.5, textvariable=self.var_fps,
            width=8,
        ).grid(row=0, column=1, sticky="e", padx=(8, 0))

        ttk.Label(marco, text="Tolerancia (s):").grid(
            row=1, column=0, sticky="w", pady=(5, 0)
        )
        self.var_tolerancia = tk.DoubleVar(value=prefs.tolerancia_segundos)
        ttk.Spinbox(
            marco, from_=0.0, to=120.0, increment=1.0,
            textvariable=self.var_tolerancia, width=8,
        ).grid(row=1, column=1, sticky="e", padx=(8, 0), pady=(5, 0))

        ttk.Label(marco, text="Modelo:").grid(row=2, column=0, sticky="w", pady=(5, 0))
        self.var_modelo = tk.StringVar(value=prefs.modelo)
        ttk.Combobox(
            marco, textvariable=self.var_modelo, values=list(MODELOS),
            state="readonly", width=12,
        ).grid(row=2, column=1, sticky="e", padx=(8, 0), pady=(5, 0))

        ttk.Label(marco, text="Acelerador:").grid(
            row=3, column=0, sticky="w", pady=(5, 0)
        )
        # Solo se ofrece lo que este equipo puede usar de verdad.
        aceleradores = aceleradores_disponibles()
        self.var_acelerador = tk.StringVar(
            value=prefs.acelerador if prefs.acelerador in aceleradores else "auto"
        )
        ttk.Combobox(
            marco, textvariable=self.var_acelerador, values=aceleradores,
            state="readonly", width=12,
        ).grid(row=3, column=1, sticky="e", padx=(8, 0), pady=(5, 0))

        self.var_tracking = tk.BooleanVar(value=prefs.usar_tracking)
        ttk.Checkbutton(
            marco, text="Seguir a cada persona (ByteTrack)", variable=self.var_tracking,
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self._ayuda(
            marco,
            "Cuenta personas distintas por evento y anota si entran o salen. "
            "Funciona mejor con 2 fps o más.",
            5, pady=(2, 0),
        )

        self.var_incremental = tk.BooleanVar(value=prefs.incremental)
        ttk.Checkbutton(
            marco, text="Reutilizar videos ya analizados", variable=self.var_incremental,
        ).grid(row=6, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self._ayuda(
            marco,
            "Salta los videos que ya se analizaron en esta carpeta de resultados "
            "con los mismos parámetros. Desmárcalo para reanalizar todo.",
            7, pady=(2, 0),
        )
        marco.columnconfigure(0, weight=1)

    def _seccion_zona(self, prefs: Preferencias) -> None:
        """Criterio con el que una persona cuenta como "en la zona"."""
        marco = ttk.LabelFrame(self, text="Zona de la puerta", padding=8)
        marco.pack(fill="x", pady=(8, 0))

        ttk.Label(marco, text="Criterio:").grid(row=0, column=0, sticky="w")
        self.var_criterio = tk.StringVar(value=prefs.criterio_zona)
        combo = ttk.Combobox(
            marco, textvariable=self.var_criterio, values=list(CRITERIOS_ZONA),
            state="readonly", width=12,
        )
        combo.grid(row=0, column=1, sticky="e", padx=(8, 0))
        combo.bind("<<ComboboxSelected>>", lambda _e: self._actualizar_estado_criterio())

        ttk.Label(marco, text="Solape mínimo:").grid(
            row=1, column=0, sticky="w", pady=(5, 0)
        )
        self.var_min_solape = tk.DoubleVar(value=prefs.min_solape)
        self.spin_solape = ttk.Spinbox(
            marco, from_=0.05, to=1.0, increment=0.05,
            textvariable=self.var_min_solape, width=8,
        )
        self.spin_solape.grid(row=1, column=1, sticky="e", padx=(8, 0), pady=(5, 0))

        self.var_ayuda_criterio = tk.StringVar()
        ttk.Label(
            marco, textvariable=self.var_ayuda_criterio, style="Pista.TLabel",
            wraplength=ANCHO_AYUDA, justify="left",
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))

        self.var_general = tk.BooleanVar(value=prefs.registrar_general)
        ttk.Checkbutton(
            marco, text="Registrar detecciones generales", variable=self.var_general,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))
        marco.columnconfigure(0, weight=1)

    def _seccion_rendimiento(self, prefs: Preferencias) -> None:
        """Decodificación por hardware y filtro de movimiento."""
        marco = ttk.LabelFrame(self, text="Rendimiento", padding=8)
        marco.pack(fill="x", pady=(8, 0))

        self.var_hardware = tk.BooleanVar(value=prefs.decodificacion_hardware)
        ttk.Checkbutton(
            marco, text="Decodificación por hardware", variable=self.var_hardware,
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        self._ayuda(
            marco,
            "Descomprime el video ~2.2x más rápido usando la GPU, pero cambia "
            "ligeramente el color y puede perder detecciones que rocen el umbral "
            "de confianza. Compruébalo en un video antes de fiarte.",
            1, pady=(2, 8),
        )

        self.var_movimiento = tk.BooleanVar(value=prefs.filtro_movimiento)
        ttk.Checkbutton(
            marco, text="Filtro de movimiento previo", variable=self.var_movimiento,
            command=self._actualizar_estado_movimiento,
        ).grid(row=2, column=0, columnspan=2, sticky="w")

        ttk.Label(marco, text="Umbral:").grid(row=3, column=0, sticky="w", pady=(5, 0))
        self.var_umbral_mov = tk.DoubleVar(value=prefs.umbral_movimiento)
        self.spin_movimiento = ttk.Spinbox(
            marco, from_=0.0005, to=0.05, increment=0.0005, format="%.4f",
            textvariable=self.var_umbral_mov, width=8,
        )
        self.spin_movimiento.grid(row=3, column=1, sticky="e", padx=(8, 0), pady=(5, 0))
        self._ayuda(
            marco,
            "Omite los frames sin cambios para no ejecutar el modelo sobre imagen "
            "estática. Bajarlo es más seguro; subirlo, más rápido.",
            4,
        )
        marco.columnconfigure(0, weight=1)

    def _seccion_rostros(self, prefs: Preferencias) -> None:
        """Detección e identificación de rostros."""
        marco = ttk.LabelFrame(self, text="Rostros", padding=8)
        marco.pack(fill="x", pady=(8, 0))

        self.var_rostros = tk.BooleanVar(value=prefs.detectar_rostros)
        ttk.Checkbutton(
            marco, text="Detectar rostros", variable=self.var_rostros,
            command=self._actualizar_estado_rostros,
        ).grid(row=0, column=0, columnspan=2, sticky="w")

        ttk.Label(marco, text="Backend:").grid(row=1, column=0, sticky="w", pady=(5, 0))
        backends = backends_rostros_disponibles()
        self.var_backend_rostros = tk.StringVar(
            value=prefs.backend_rostros
            if prefs.backend_rostros in backends
            else (backends[0] if backends else "")
        )
        self.combo_backend = ttk.Combobox(
            marco, textvariable=self.var_backend_rostros, values=backends,
            state="readonly", width=12,
        )
        self.combo_backend.grid(row=1, column=1, sticky="e", padx=(8, 0), pady=(5, 0))
        self.combo_backend.bind(
            "<<ComboboxSelected>>", lambda _e: self._actualizar_estado_rostros()
        )

        self.var_recortes = tk.BooleanVar(value=prefs.guardar_recortes_rostros)
        self.check_recortes = ttk.Checkbutton(
            marco, text="Guardar recortes", variable=self.var_recortes
        )
        self.check_recortes.grid(row=2, column=0, columnspan=2, sticky="w", pady=(5, 0))

        self.var_identificar = tk.BooleanVar(value=prefs.identificar_rostros)
        self.check_identificar = ttk.Checkbutton(
            marco, text="Identificar personas conocidas",
            variable=self.var_identificar, command=self._actualizar_estado_rostros,
        )
        self.check_identificar.grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(5, 0)
        )

        self.boton_personas = ttk.Button(
            marco, text="Catálogo de personas…", command=self._on_elegir_personas
        )
        self.boton_personas.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        self.var_personas = tk.StringVar(value="(sin catálogo)")
        ttk.Label(
            marco, textvariable=self.var_personas, style="Pista.TLabel",
            wraplength=ANCHO_AYUDA, justify="left",
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(4, 0))
        marco.columnconfigure(0, weight=1)

    # ----------------------------------------------------------------- estados

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

    # ---------------------------------------------------------------- interfaz

    @property
    def identificar_activo(self) -> bool:
        return bool(self.var_identificar.get())

    @property
    def incremental(self) -> bool:
        return bool(self.var_incremental.get())

    def mostrar_catalogo(self, texto: str) -> None:
        """Muestra bajo el botón qué catálogo de personas está elegido."""
        self.var_personas.set(texto)

    def configuracion(self, zona: EspecZona, carpeta_personas: str) -> ConfiguracionAnalisis:
        """Arma y valida la configuración a partir de los widgets.

        Raises:
            ValueError: si algún parámetro no es válido.
            tkinter.TclError: si un campo numérico tiene texto no numérico.
        """
        config = ConfiguracionAnalisis(
            zona_puerta=zona,
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
            carpeta_personas=carpeta_personas,
            usar_tracking=self.var_tracking.get(),
        )
        config.validar()
        return config

    def volcar_en(self, prefs: Preferencias) -> None:
        """Copia los valores actuales de los widgets a las preferencias."""
        try:
            prefs.fps_analisis = float(self.var_fps.get())
            prefs.tolerancia_segundos = float(self.var_tolerancia.get())
            prefs.min_solape = float(self.var_min_solape.get())
            prefs.umbral_movimiento = float(self.var_umbral_mov.get())
        except tk.TclError:
            pass  # un campo numérico a medio escribir: se conserva el anterior
        prefs.modelo = self.var_modelo.get()
        prefs.acelerador = self.var_acelerador.get()
        prefs.decodificacion_hardware = self.var_hardware.get()
        prefs.filtro_movimiento = self.var_movimiento.get()
        prefs.criterio_zona = self.var_criterio.get()
        prefs.registrar_general = self.var_general.get()
        prefs.detectar_rostros = self.var_rostros.get()
        prefs.backend_rostros = self.var_backend_rostros.get()
        prefs.guardar_recortes_rostros = self.var_recortes.get()
        prefs.identificar_rostros = self.var_identificar.get()
        prefs.usar_tracking = self.var_tracking.get()
        prefs.incremental = self.var_incremental.get()
