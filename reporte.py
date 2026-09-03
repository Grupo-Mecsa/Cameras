"""Generación de informes de resultados en HTML y PDF.

El HTML no necesita dependencias extra y lleva las miniaturas incrustadas, así
que es un único archivo que se puede enviar por correo. El PDF requiere
``reportlab``; si no está instalado, se genera solo el HTML.
"""

from __future__ import annotations

import base64
import html
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

from detector import (
    TIPO_GENERAL,
    TIPO_ZONA,
    ConfiguracionAnalisis,
    Evento,
    ResultadoVideo,
    formatear_tiempo,
)

#: Ancho al que se incrustan las miniaturas en el informe.
ANCHO_MINIATURA = 320


@dataclass
class ResumenInforme:
    """Cifras agregadas que encabezan el informe."""

    videos: int
    videos_con_error: int
    eventos_zona: int
    eventos_general: int
    duracion_zona: float
    eventos_con_rostro: int
    personas: list[str]
    #: Suma de personas distintas de los eventos en la zona.
    personas_zona: int = 0

    @classmethod
    def desde(cls, resultados: Sequence[ResultadoVideo]) -> "ResumenInforme":
        """Calcula el resumen a partir de los resultados del análisis."""
        eventos = [e for r in resultados for e in r.eventos]
        zona = [e for e in eventos if e.tipo == TIPO_ZONA]
        personas: set[str] = set()
        for evento in eventos:
            personas.update(p.strip() for p in evento.personas.split(",") if p.strip())
        return cls(
            videos=len(resultados),
            videos_con_error=sum(1 for r in resultados if r.error),
            eventos_zona=len(zona),
            eventos_general=sum(1 for e in eventos if e.tipo == TIPO_GENERAL),
            duracion_zona=sum(e.duracion for e in zona),
            eventos_con_rostro=sum(1 for e in eventos if e.rostros > 0),
            personas=sorted(personas),
            personas_zona=sum(e.n_personas for e in zona),
        )


def _duracion_larga(segundos: float) -> str:
    """Formatea una duración en un texto legible del tipo ``2 min 05 s``."""
    if segundos < 60:
        return f"{segundos:.0f} s"
    minutos, segs = divmod(int(round(segundos)), 60)
    if minutos < 60:
        return f"{minutos} min {segs:02d} s"
    horas, minutos = divmod(minutos, 60)
    return f"{horas} h {minutos:02d} min"


def _parametros(config: ConfiguracionAnalisis) -> list[tuple[str, str]]:
    """Describe la configuración usada, para dejar constancia en el informe."""
    x1, y1, x2, y2 = config.zona_puerta
    filas = [
        ("Zona de la puerta", f"({x1}, {y1}) - ({x2}, {y2})"),
        ("Criterio de zona", config.criterio_zona),
        ("Frames por segundo analizados", f"{config.fps_analisis:g}"),
        ("Tolerancia de agrupación", f"{config.tolerancia_segundos:g} s"),
        ("Modelo", config.modelo),
        ("Confianza mínima", f"{config.confianza:g}"),
        (
            "Seguimiento de personas",
            "ByteTrack" if config.usar_tracking else "desactivado",
        ),
    ]
    if config.criterio_zona == "solape":
        filas.insert(2, ("Solape mínimo", f"{config.min_solape:.0%}"))
    if config.detectar_rostros:
        filas.append(("Detección de rostros", config.backend_rostros))
    if config.identificar_rostros:
        filas.append(("Identificación", config.carpeta_personas or "—"))
    return filas


# --------------------------------------------------------------------- HTML


def _imagen_incrustada(ruta: str) -> str:
    """Devuelve una miniatura como ``data:`` URI, o cadena vacía si no se puede."""
    if not ruta:
        return ""
    archivo = Path(ruta)
    if not archivo.is_file():
        return ""
    try:
        datos = base64.b64encode(archivo.read_bytes()).decode("ascii")
    except OSError:
        return ""
    return f"data:image/jpeg;base64,{datos}"


_ESTILO = """
:root { color-scheme: light; }
body { font-family: "Segoe UI", system-ui, sans-serif; margin: 0; padding: 32px;
       background: #f5f6f8; color: #1d2129; }
.hoja { max-width: 1100px; margin: 0 auto; background: #fff; padding: 40px;
        border-radius: 10px; box-shadow: 0 1px 4px rgba(0,0,0,.10); }
h1 { margin: 0 0 4px; font-size: 26px; }
h2 { margin: 36px 0 12px; font-size: 18px; border-bottom: 2px solid #e6e8eb;
     padding-bottom: 6px; }
.sub { color: #6a7280; margin: 0 0 24px; }
.tarjetas { display: flex; flex-wrap: wrap; gap: 14px; margin: 20px 0; }
.tarjeta { flex: 1 1 150px; background: #f0f4f9; border-radius: 8px; padding: 14px 16px; }
.tarjeta .n { font-size: 26px; font-weight: 600; }
.tarjeta .e { font-size: 12px; color: #6a7280; text-transform: uppercase;
              letter-spacing: .04em; }
table { border-collapse: collapse; width: 100%; font-size: 14px; }
th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid #e6e8eb; }
th { background: #f0f4f9; font-weight: 600; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.etq { display: inline-block; padding: 2px 9px; border-radius: 20px; font-size: 12px;
       font-weight: 600; }
.etq-zona { background: #e0f0e4; color: #1d6b33; }
.etq-general { background: #e8eaee; color: #4a5160; }
.galeria { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
           gap: 18px; }
.galeria figure { margin: 0; }
.galeria img { width: 100%; border-radius: 6px; border: 1px solid #dcdfe4; }
.galeria figcaption { font-size: 12px; color: #6a7280; margin-top: 6px; }
.vacio { color: #6a7280; font-style: italic; }
.error { color: #a4262c; }
footer { margin-top: 40px; font-size: 12px; color: #8a919c; }
@media print { body { background: #fff; padding: 0; }
                .hoja { box-shadow: none; padding: 0; max-width: none; } }
"""


def _tabla_eventos_html(eventos: Sequence[Evento], con_personas: bool) -> str:
    """Construye la tabla HTML de eventos."""
    if not eventos:
        return '<p class="vacio">Sin eventos.</p>'
    cabeceras = [
        "Archivo", "Tipo", "Inicio", "Fin", "Duración", "Personas", "Dirección", "Rostros",
    ]
    if con_personas:
        cabeceras.append("Reconocidas")
    numericas = ("Inicio", "Fin", "Duración", "Personas", "Rostros")
    filas = []
    for e in eventos:
        celdas = [
            f"<td>{html.escape(e.archivo)}</td>",
            f'<td><span class="etq etq-{e.tipo}">{e.tipo}</span></td>',
            f"<td class='num'>{formatear_tiempo(e.inicio)}</td>",
            f"<td class='num'>{formatear_tiempo(e.fin)}</td>",
            f"<td class='num'>{e.duracion:.1f} s</td>",
            f"<td class='num'>{e.n_personas or ''}</td>",
            f"<td>{html.escape(e.direccion)}</td>",
            f"<td class='num'>{e.rostros or ''}</td>",
        ]
        if con_personas:
            celdas.append(f"<td>{html.escape(e.personas)}</td>")
        filas.append("<tr>" + "".join(celdas) + "</tr>")
    cab = "".join(
        f"<th class='num'>{c}</th>" if c in numericas else f"<th>{c}</th>"
        for c in cabeceras
    )
    return f"<table><thead><tr>{cab}</tr></thead><tbody>{''.join(filas)}</tbody></table>"


def generar_html(
    resultados: Sequence[ResultadoVideo],
    config: ConfiguracionAnalisis,
    destino: str | Path,
    carpeta_analizada: str = "",
) -> Path:
    """Escribe un informe HTML autocontenido con las miniaturas incrustadas.

    Args:
        resultados: resultados del análisis, uno por video.
        config: parámetros con los que se ejecutó.
        destino: ruta del archivo ``.html`` a escribir.
        carpeta_analizada: carpeta de origen, solo informativa.

    Returns:
        La ruta del archivo generado.
    """
    ruta = Path(destino)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    resumen = ResumenInforme.desde(resultados)
    eventos = [e for r in resultados for e in r.eventos]
    con_personas = any(e.personas for e in eventos)
    ahora = datetime.now().strftime("%d/%m/%Y %H:%M")

    tarjetas = [
        ("Eventos en la zona", str(resumen.eventos_zona)),
        ("Personas en la zona", str(resumen.personas_zona)),
        ("Detecciones generales", str(resumen.eventos_general)),
        ("Tiempo total en la zona", _duracion_larga(resumen.duracion_zona)),
        ("Videos analizados", str(resumen.videos)),
    ]
    if config.detectar_rostros:
        tarjetas.append(("Eventos con rostro", str(resumen.eventos_con_rostro)))
    if resumen.personas:
        tarjetas.append(("Personas reconocidas", str(len(resumen.personas))))

    html_tarjetas = "".join(
        f'<div class="tarjeta"><div class="n">{html.escape(v)}</div>'
        f'<div class="e">{html.escape(k)}</div></div>'
        for k, v in tarjetas
    )

    filas_videos = []
    for r in resultados:
        if r.error:
            filas_videos.append(
                f"<tr><td>{html.escape(r.archivo)}</td>"
                f'<td colspan="2" class="error">{html.escape(r.error)}</td></tr>'
            )
            continue
        z = sum(1 for e in r.eventos if e.tipo == TIPO_ZONA)
        g = sum(1 for e in r.eventos if e.tipo == TIPO_GENERAL)
        filas_videos.append(
            f"<tr><td>{html.escape(r.archivo)}</td>"
            f"<td class='num'>{z}</td><td class='num'>{g}</td></tr>"
        )

    filas_params = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{html.escape(v)}</td></tr>"
        for k, v in _parametros(config)
    )

    eventos_zona = [e for e in eventos if e.tipo == TIPO_ZONA]
    figuras = []
    for e in eventos_zona:
        src = _imagen_incrustada(e.miniatura)
        if not src:
            continue
        pie = f"{html.escape(e.archivo)} · {formatear_tiempo(e.inicio)}"
        if e.personas:
            pie += f" · {html.escape(e.personas)}"
        figuras.append(f'<figure><img src="{src}" alt=""><figcaption>{pie}</figcaption></figure>')
    galeria = (
        f'<div class="galeria">{"".join(figuras)}</div>'
        if figuras
        else '<p class="vacio">No hay miniaturas de eventos en la zona.</p>'
    )

    documento = f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<title>Informe DeCam — {ahora}</title>
<style>{_ESTILO}</style></head><body><div class="hoja">
<h1>Informe de detecciones</h1>
<p class="sub">Generado el {ahora}{
    " · " + html.escape(carpeta_analizada) if carpeta_analizada else ""}</p>
<div class="tarjetas">{html_tarjetas}</div>

<h2>Resumen por video</h2>
<table><thead><tr><th>Archivo</th><th class="num">En la zona</th>
<th class="num">Generales</th></tr></thead><tbody>{''.join(filas_videos)}</tbody></table>

<h2>Eventos en la zona de la puerta</h2>
{_tabla_eventos_html(eventos_zona, con_personas)}

<h2>Detecciones generales</h2>
{_tabla_eventos_html([e for e in eventos if e.tipo == TIPO_GENERAL], con_personas)}

<h2>Miniaturas</h2>
{galeria}

<h2>Parámetros del análisis</h2>
<table><tbody>{filas_params}</tbody></table>

<footer>DeCam · Los tiempos son relativos al inicio de cada video.</footer>
</div></body></html>"""
    ruta.write_text(documento, encoding="utf-8")
    return ruta


# ---------------------------------------------------------------------- PDF


def pdf_disponible() -> bool:
    """Indica si ``reportlab`` está instalado y se puede generar el PDF."""
    try:
        import reportlab  # noqa: F401
    except ImportError:
        return False
    return True


def generar_pdf(
    resultados: Sequence[ResultadoVideo],
    config: ConfiguracionAnalisis,
    destino: str | Path,
    carpeta_analizada: str = "",
) -> Optional[Path]:
    """Escribe el informe en PDF usando ``reportlab``.

    Returns:
        La ruta del PDF, o ``None`` si ``reportlab`` no está instalado.
    """
    if not pdf_disponible():
        return None

    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Image,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    ruta = Path(destino)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    resumen = ResumenInforme.desde(resultados)
    eventos = [e for r in resultados for e in r.eventos]
    con_personas = any(e.personas for e in eventos)
    ahora = datetime.now().strftime("%d/%m/%Y %H:%M")

    estilos = getSampleStyleSheet()
    estilo_titulo = ParagraphStyle(
        "TituloDeCam", parent=estilos["Title"], fontSize=20, alignment=TA_LEFT,
        spaceAfter=2,
    )
    estilo_sub = ParagraphStyle(
        "SubDeCam", parent=estilos["Normal"], fontSize=9,
        textColor=colors.HexColor("#6a7280"), spaceAfter=14,
    )
    estilo_h2 = ParagraphStyle(
        "H2DeCam", parent=estilos["Heading2"], fontSize=13, spaceBefore=16,
        spaceAfter=6,
    )
    estilo_celda = ParagraphStyle(
        "CeldaDeCam", parent=estilos["Normal"], fontSize=8, leading=10
    )

    def tabla(datos: list[list], anchos: list[float]) -> Table:
        """Crea una tabla con el estilo del informe."""
        t = Table(datos, colWidths=anchos, repeatRows=1)
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f4f9")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1d2129")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#dcdfe4")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                     [colors.white, colors.HexColor("#fafbfc")]),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        return t

    hist: list = [
        Paragraph("Informe de detecciones", estilo_titulo),
        Paragraph(
            f"Generado el {ahora}"
            + (f" &middot; {html.escape(carpeta_analizada)}" if carpeta_analizada else ""),
            estilo_sub,
        ),
    ]

    # Resumen
    resumen_datos = [
        ["Eventos en la zona", str(resumen.eventos_zona)],
        ["Detecciones generales", str(resumen.eventos_general)],
        ["Tiempo total en la zona", _duracion_larga(resumen.duracion_zona)],
        ["Personas en la zona", str(resumen.personas_zona)],
        ["Videos analizados", str(resumen.videos)],
    ]
    if resumen.videos_con_error:
        resumen_datos.append(["Videos con error", str(resumen.videos_con_error)])
    if config.detectar_rostros:
        resumen_datos.append(["Eventos con rostro", str(resumen.eventos_con_rostro)])
    if resumen.personas:
        resumen_datos.append(["Personas reconocidas", ", ".join(resumen.personas)])
    hist += [
        Paragraph("Resumen", estilo_h2),
        tabla([["Concepto", "Valor"]] + resumen_datos, [90 * mm, 75 * mm]),
    ]

    # Por video
    filas = [["Archivo", "En la zona", "Generales"]]
    for r in resultados:
        if r.error:
            filas.append([Paragraph(r.archivo, estilo_celda), "error", "—"])
            continue
        filas.append([
            Paragraph(r.archivo, estilo_celda),
            str(sum(1 for e in r.eventos if e.tipo == TIPO_ZONA)),
            str(sum(1 for e in r.eventos if e.tipo == TIPO_GENERAL)),
        ])
    hist += [
        Paragraph("Resumen por video", estilo_h2),
        tabla(filas, [105 * mm, 30 * mm, 30 * mm]),
    ]

    # Eventos
    for titulo, tipo in (
        ("Eventos en la zona de la puerta", TIPO_ZONA),
        ("Detecciones generales", TIPO_GENERAL),
    ):
        del_tipo = [e for e in eventos if e.tipo == tipo]
        hist.append(Paragraph(titulo, estilo_h2))
        if not del_tipo:
            hist.append(Paragraph("Sin eventos.", estilo_celda))
            continue
        cab = ["Archivo", "Inicio", "Fin", "Dur.", "Pers.", "Dirección", "Rostros"]
        anchos = [48 * mm, 19 * mm, 19 * mm, 15 * mm, 13 * mm, 26 * mm, 14 * mm]
        if con_personas:
            cab.append("Reconocidas")
            anchos = [
                38 * mm, 18 * mm, 18 * mm, 13 * mm, 12 * mm, 22 * mm, 13 * mm, 20 * mm,
            ]
        filas = [cab]
        for e in del_tipo:
            fila = [
                Paragraph(e.archivo, estilo_celda),
                formatear_tiempo(e.inicio),
                formatear_tiempo(e.fin),
                f"{e.duracion:.0f} s",
                str(e.n_personas or ""),
                Paragraph(e.direccion, estilo_celda),
                str(e.rostros or ""),
            ]
            if con_personas:
                fila.append(Paragraph(e.personas, estilo_celda))
            filas.append(fila)
        hist.append(tabla(filas, anchos))

    # Miniaturas de los eventos de la zona
    miniaturas = [e for e in eventos if e.tipo == TIPO_ZONA and Path(e.miniatura or "").is_file()]
    if miniaturas:
        hist += [PageBreak(), Paragraph("Miniaturas", estilo_h2)]
        for e in miniaturas[:40]:  # un tope razonable para no generar un PDF enorme
            hist += [
                Image(e.miniatura, width=150 * mm, height=150 * mm * 9 / 16),
                Paragraph(
                    f"{html.escape(e.archivo)} &middot; {formatear_tiempo(e.inicio)}"
                    + (f" &middot; {html.escape(e.personas)}" if e.personas else ""),
                    estilo_sub,
                ),
                Spacer(1, 4 * mm),
            ]

    hist += [
        Paragraph("Parámetros del análisis", estilo_h2),
        tabla(
            [["Parámetro", "Valor"]] + [list(f) for f in _parametros(config)],
            [70 * mm, 95 * mm],
        ),
    ]

    SimpleDocTemplate(
        str(ruta),
        pagesize=A4,
        title="Informe DeCam",
        author="DeCam",
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
    ).build(hist)
    return ruta


def generar_informes(
    resultados: Sequence[ResultadoVideo],
    config: ConfiguracionAnalisis,
    carpeta_salida: str | Path,
    carpeta_analizada: str = "",
) -> list[Path]:
    """Genera el informe HTML y, si se puede, también el PDF.

    Returns:
        Las rutas de los informes generados.
    """
    salida = Path(carpeta_salida)
    marca = datetime.now().strftime("%Y%m%d_%H%M")
    generados = [
        generar_html(
            resultados, config, salida / f"informe_{marca}.html", carpeta_analizada
        )
    ]
    pdf = generar_pdf(
        resultados, config, salida / f"informe_{marca}.pdf", carpeta_analizada
    )
    if pdf is not None:
        generados.append(pdf)
    return generados
