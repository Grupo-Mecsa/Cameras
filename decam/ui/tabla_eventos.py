"""Pestaña con la tabla de eventos detectados y su filtro por tipo."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from decam.eventos import TIPO_GENERAL, TIPO_ZONA, Evento, formatear_tiempo
from decam.ui.utilidades import abrir_en_sistema

#: Colores de fondo de la tabla, para distinguir los tipos de un vistazo.
COLOR_ZONA = "#e0f0e4"
COLOR_GENERAL = "#f2f3f5"

COLUMNAS = {
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


class TablaEventos(ttk.Frame):
    """Lista los eventos según llegan; doble clic abre la miniatura."""

    def __init__(self, padre: tk.Misc) -> None:
        super().__init__(padre, padding=8)
        self._eventos: list[Evento] = []
        self._filas: dict[str, Evento] = {}
        self._construir()

    def _construir(self) -> None:
        barra = ttk.Frame(self)
        barra.pack(fill="x", pady=(0, 6))
        ttk.Label(barra, text="Mostrar:").pack(side="left")
        self.var_filtro = tk.StringVar(value="todos")
        for texto, valor in (
            ("Todos", "todos"),
            ("Solo zona", TIPO_ZONA),
            ("Solo generales", TIPO_GENERAL),
        ):
            ttk.Radiobutton(
                barra, text=texto, value=valor, variable=self.var_filtro,
                command=self._refrescar,
            ).pack(side="left", padx=(8, 0))
        ttk.Label(
            barra, text="Doble clic en una fila abre su miniatura.", style="Pista.TLabel",
        ).pack(side="right")

        contenedor = ttk.Frame(self)
        contenedor.pack(fill="both", expand=True)

        self.tabla = ttk.Treeview(
            contenedor, columns=tuple(COLUMNAS), show="headings", selectmode="browse"
        )
        for col, (titulo, ancho, anclaje) in COLUMNAS.items():
            self.tabla.heading(col, text=titulo)
            self.tabla.column(col, width=ancho, anchor=anclaje, stretch=(col == "archivo"))
        self.tabla.tag_configure(TIPO_ZONA, background=COLOR_ZONA)
        self.tabla.tag_configure(TIPO_GENERAL, background=COLOR_GENERAL)
        self.tabla.bind("<Double-1>", self._abrir_miniatura_seleccionada)

        scroll = ttk.Scrollbar(contenedor, orient="vertical", command=self.tabla.yview)
        self.tabla.configure(yscrollcommand=scroll.set)
        self.tabla.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

    # ---------------------------------------------------------------- interfaz

    @property
    def columnas(self) -> tuple[str, ...]:
        return tuple(COLUMNAS)

    def agregar(self, evento: Evento) -> None:
        """Guarda un evento y lo muestra si el filtro lo permite."""
        self._eventos.append(evento)
        self._insertar_fila(evento)

    def limpiar(self) -> None:
        self._eventos.clear()
        self._filas.clear()
        self.tabla.delete(*self.tabla.get_children())

    # ----------------------------------------------------------------- interno

    def _insertar_fila(self, evento: Evento) -> None:
        filtro = self.var_filtro.get()
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

    def _refrescar(self) -> None:
        """Redibuja la tabla al cambiar el filtro por tipo."""
        self.tabla.delete(*self.tabla.get_children())
        self._filas.clear()
        for evento in self._eventos:
            self._insertar_fila(evento)

    def _abrir_miniatura_seleccionada(self, _evento: tk.Event) -> None:
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
