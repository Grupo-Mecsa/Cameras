"""DeCam: detección de personas en la puerta a partir de grabaciones.

Organización del paquete, de abajo arriba (ningún módulo importa a otro que
esté por encima de él):

* ``eventos``, ``zona``, ``aceleradores``, ``movimiento``, ``video``,
  ``rostros``: piezas independientes, sin conocimiento del análisis.
* ``configuracion``: los parámetros del análisis y su validación.
* ``deteccion`` y ``seguimiento``: las dos abstracciones que el analizador
  consume (:class:`~decam.deteccion.DetectorPersonas` y
  :class:`~decam.seguimiento.Rastreador`) con sus implementaciones actuales
  (YOLO y ByteTrack).
* ``salida`` y ``manifiesto``: lo que se escribe a disco.
* ``analizador``: orquesta todo lo anterior; ``crear_analizador`` es la única
  función que sabe construir las implementaciones concretas a partir de la
  configuración.
* ``reporte``, ``registro``: informes y log.
* ``ui``: la interfaz Tkinter, que solo habla con ``analizador`` mediante
  callbacks.
"""
