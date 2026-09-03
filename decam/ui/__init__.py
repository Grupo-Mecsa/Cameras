"""Interfaz gráfica (Tkinter).

Cada pieza de la ventana es un componente con una responsabilidad:

* :class:`~decam.ui.parametros.PanelParametros`: los ajustes y su traducción a
  :class:`~decam.configuracion.ConfiguracionAnalisis`.
* :class:`~decam.ui.vista_previa.VistaPrevia`: el frame de referencia, la
  navegación por el video y el dibujo de la zona.
* :class:`~decam.ui.tabla_eventos.TablaEventos`: la tabla de eventos y su filtro.
* :class:`~decam.ui.aplicacion.AplicacionDeCam`: los une, elige carpetas y
  lanza el análisis en un hilo.
"""
