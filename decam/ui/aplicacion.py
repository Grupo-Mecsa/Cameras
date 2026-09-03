"""Ventana principal: une los componentes y lanza el análisis en un hilo.

La ventana se organiza en tres zonas: la cabecera con las carpetas, el panel de
parámetros a la izquierda y, a la derecha, pestañas con la vista previa, la
tabla de eventos y el registro. El análisis corre en un hilo aparte y se
comunica con la interfaz mediante una cola que solo consume el hilo principal.
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Optional

from decam import manifiesto, registro, reporte
from decam.analizador import crear_analizador
from decam.configuracion import ConfiguracionAnalisis
from decam.eventos import TIPO_GENERAL, TIPO_ZONA, Evento, ResultadoVideo
from decam.salida import EscritorSalida
from decam.ui.parametros import PanelParametros
from decam.ui.preferencias import RUTA_CONFIG, Preferencias
from decam.ui.tabla_eventos import TablaEventos
from decam.ui.utilidades import abrir_en_sistema
from decam.ui.vista_previa import VistaPrevia
from decam.video import encontrar_videos
from decam.zona import espec_a_lista, normalizar_espec


class AplicacionDeCam(ttk.Frame):
    """Ventana principal de la aplicación."""

    def __init__(self, root: tk.Tk) -> None:
        """Construye la interfaz y restaura la configuración guardada."""
        super().__init__(root, padding=(10, 8))
        self.root = root
        self.pack(fill="both", expand=True)

        self.prefs = Preferencias.cargar()
        self.videos: list[Path] = []
        self.mensajes: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.cancelar_evento = threading.Event()
        self.hilo: Optional[threading.Thread] = None
        self.informes: list[Path] = []

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
        self._construir_cabecera()

        cuerpo = ttk.Frame(self)
        cuerpo.pack(fill="both", expand=True, pady=(8, 0))

        self.panel = PanelParametros(cuerpo, self.prefs, self.elegir_carpeta_personas)
        self.panel.pack(side="left", fill="y", padx=(0, 10))

        self.pestanas = ttk.Notebook(cuerpo)
        self.pestanas.pack(side="left", fill="both", expand=True)
        zona_inicial = None
        if self.prefs.zona_puerta:
            try:
                zona_inicial = normalizar_espec(self.prefs.zona_puerta)
            except ValueError as exc:
                registro.log.warning("Zona guardada inválida, se descarta: %s", exc)
        self.vista = VistaPrevia(self.pestanas, on_log=self.log, zona_inicial=zona_inicial)
        self.pestanas.add(self.vista, text="  Vista previa  ")
        self.tabla = TablaEventos(self.pestanas)
        self.pestanas.add(self.tabla, text="  Eventos  ")
        self._construir_pestana_registro()

        self._construir_barra_inferior()

    def _construir_cabecera(self) -> None:
        """Botones y etiquetas de las carpetas de entrada y salida."""
        marco = ttk.Frame(self)
        marco.pack(fill="x")

        ttk.Button(
            marco, text="Videos a analizar…", width=22, command=self.elegir_carpeta_videos,
        ).grid(row=0, column=0, sticky="w")
        self.var_carpeta = tk.StringVar(value="(sin seleccionar)")
        ttk.Label(marco, textvariable=self.var_carpeta).grid(
            row=0, column=1, sticky="w", padx=10
        )

        ttk.Button(
            marco, text="Carpeta de resultados…", width=22, command=self.elegir_carpeta_salida,
        ).grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.var_salida = tk.StringVar(value="(sin seleccionar)")
        ttk.Label(marco, textvariable=self.var_salida).grid(
            row=1, column=1, sticky="w", padx=10, pady=(6, 0)
        )
        marco.columnconfigure(1, weight=1)

    def _construir_pestana_registro(self) -> None:
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
            self.panel.mostrar_catalogo(Path(self.prefs.carpeta_personas).name)
        if self.prefs.carpeta_videos and Path(self.prefs.carpeta_videos).is_dir():
            self._cargar_carpeta(self.prefs.carpeta_videos)

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

    # ------------------------------------------------------------------ carpetas

    def elegir_carpeta_videos(self) -> None:
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
        self.vista.mostrar_videos(
            {str(v.relative_to(carpeta)): v for v in self.videos}
        )
        if not self.videos:
            self.log(f"No se encontraron videos en {carpeta}")

    def elegir_carpeta_salida(self) -> None:
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
        self.panel.mostrar_catalogo(f"{Path(carpeta).name}: {len(subcarpetas)} persona/s")
        if not subcarpetas:
            messagebox.showwarning(
                "Catálogo vacío",
                "Esa carpeta no tiene subcarpetas.\n\n"
                "Cada persona necesita su propia subcarpeta con fotos suyas; el "
                "nombre de la subcarpeta es la etiqueta que saldrá en el informe.",
            )

    def abrir_resultados(self) -> None:
        carpeta = self.prefs.carpeta_salida
        if carpeta and Path(carpeta).is_dir():
            abrir_en_sistema(carpeta)
        else:
            messagebox.showwarning("Carpeta de resultados", "Todavía no hay una carpeta válida.")

    def abrir_informe(self) -> None:
        """Abre el informe generado; prefiere el PDF si existe."""
        if not self.informes:
            return
        pdf = next((r for r in self.informes if r.suffix == ".pdf"), None)
        abrir_en_sistema(pdf or self.informes[0])

    # ---------------------------------------------------------------------- log

    def log(self, mensaje: str) -> None:
        """Escribe una línea en el registro (solo desde el hilo principal)."""
        registro.log.info("%s", mensaje)
        self.texto_log.configure(state="normal")
        self.texto_log.insert("end", mensaje + "\n")
        self.texto_log.see("end")
        self.texto_log.configure(state="disabled")

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
                    self.tabla.agregar(dato)
                elif tipo == "progreso":
                    nombre, pct_video, pct_total = dato
                    self.progreso["value"] = pct_total
                    self.var_estado.set(
                        f"Analizando {nombre} — {pct_video:.0f}% (total {pct_total:.0f}%)"
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
        if self.vista.zona is None:
            messagebox.showwarning("Zona", "Dibuja la zona de la puerta sobre la vista previa.")
            return
        if self.panel.identificar_activo and not self.prefs.carpeta_personas:
            messagebox.showwarning(
                "Personas", "Elige la carpeta con las fotos de las personas conocidas."
            )
            return

        try:
            config = self.panel.configuracion(self.vista.zona, self.prefs.carpeta_personas)
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror("Parámetros", str(exc))
            return

        self._guardar_preferencias()
        self.cancelar_evento = threading.Event()
        self.informes = []
        self.tabla.limpiar()
        self.progreso["value"] = 0.0
        self.boton_iniciar.configure(state="disabled")
        self.boton_cancelar.configure(state="normal")
        self.boton_informe.configure(state="disabled")
        self.var_estado.set("Iniciando análisis…")
        self.log("=" * 70)
        self.log("Iniciando análisis…")

        self.hilo = threading.Thread(
            target=self._ejecutar_analisis, args=(config, self.prefs.incremental),
            daemon=True,
        )
        self.hilo.start()

    def _ejecutar_analisis(self, config: ConfiguracionAnalisis, incremental: bool) -> None:
        """Corre el análisis en segundo plano (no toca widgets directamente)."""
        try:
            analizador = crear_analizador(
                config,
                cancelar=self.cancelar_evento,
                on_progreso=lambda n, pv, pt: self._encolar("progreso", (n, pv, pt)),
                on_log=lambda m: self._encolar("log", m),
                on_evento=lambda e: self._encolar("evento", e),
            )
            salida = EscritorSalida(
                self.prefs.carpeta_salida, config.zona, config.guardar_recortes_rostros
            )
            registro_previo = (
                manifiesto.Manifiesto(salida.carpeta / manifiesto.NOMBRE_MANIFIESTO)
                if incremental
                else None
            )
            resultados = analizador.analizar_videos(self.videos, salida, registro_previo)

            informes: list[Path] = []
            try:
                informes = reporte.generar_informes(
                    resultados, config, salida.carpeta, self.prefs.carpeta_videos,
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
        self.panel.volcar_en(self.prefs)
        self.prefs.zona_puerta = espec_a_lista(self.vista.zona) if self.vista.zona else None
        self.prefs.guardar()

    def _al_cerrar(self) -> None:
        """Guarda la configuración, libera recursos y cierra la ventana."""
        self.cancelar_evento.set()
        self._guardar_preferencias()
        self.vista.cerrar()
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
