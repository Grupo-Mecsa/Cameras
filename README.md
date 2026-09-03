# DeCam — Detección de personas en la puerta

Aplicación de escritorio (Python + Tkinter) que analiza grabaciones de cámaras de
seguridad con YOLOv8 y reporta **en qué momentos una persona se acerca a una puerta**.

## Características

- Selección de una carpeta con videos (`.mp4`, `.avi`, `.mkv`), **incluyendo
  subcarpetas** (las exportaciones de NVR crean una carpeta por descarga).
- Vista previa **navegable frame a frame** (deslizador, salto por número de frame o
  por tiempo) para elegir la imagen de referencia y **dibujar la zona de la puerta
  con el mouse**.
- Detección de personas (clase `person` de COCO) con YOLOv8 (`n` / `s` / `m`).
- **Dos clasificaciones en paralelo**: detecciones dentro de la zona de la puerta
  y detecciones generales en cualquier punto del frame.
- **Detección de rostros opcional** dentro de cada persona detectada, con recortes
  guardados aparte.
- **Identificación opcional** de personas conocidas con SFace, a partir de un
  catálogo de fotos de referencia.
- **Aceleración automática**: GPU NVIDIA (CUDA), GPU Intel integrada
  (OpenVINO) o CPU, lo mejor que haya en el equipo.
- Agrupación de detecciones consecutivas en intervalos con tolerancia configurable.
- Análisis en un hilo aparte: la ventana no se congela, con barra de progreso y log.
- Resultados: `eventos.csv` + miniaturas JPG con la persona y la zona dibujadas.
- La configuración se guarda en `config.json` y se restaura al abrir.

## Requisitos

- Python 3.9 o superior (recomendado 3.10+).
- Tkinter (incluido en el instalador oficial de Python para Windows).

## Instalación

```bash
cd DeCam
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS
pip install -r requirements.txt
```

La primera vez que se ejecute un análisis, `ultralytics` descarga
automáticamente los pesos del modelo elegido (`yolov8n.pt`, etc.).

## Aceleración por GPU

El desplegable **Acelerador** solo ofrece lo que el equipo puede usar de verdad.
El valor `auto` elige la mejor opción disponible por este orden:

| Acelerador | Hardware | Cómo se habilita |
|------------|----------|------------------|
| `cuda` | GPU NVIDIA | instalar PyTorch con CUDA (ver abajo) |
| `openvino-gpu` | **GPU Intel integrada** (Iris Xe, UHD, Arc) | ya incluido: `openvino` está en `requirements.txt` |
| `cpu` | cualquiera | siempre disponible |

### GPU Intel integrada (OpenVINO)

Es la vía útil en portátiles sin tarjeta dedicada, que es la mayoría. No hay nada
que configurar: si OpenVINO detecta la GPU, `auto` la usa. La primera vez, la app
exporta el modelo a formato OpenVINO (unos segundos) y guarda el resultado en
`yolov8n_openvino_model/` para las siguientes.

Medido en un i5-1145G7 con Iris Xe, yolov8n a 640 px:

| | ms/frame | Proyección para 65 h de video a 1 fps |
|---|---|---|
| PyTorch CPU | 68 | 4.4 h |
| OpenVINO CPU | 116 | 7.5 h |
| **OpenVINO GPU (Iris Xe)** | **24** | **1.5 h** |

Un análisis completo de un clip de 90 s pasó de 12.3 s a 5.2 s (**2.4x**), con
resultados idénticos. Nótese que *OpenVINO CPU* es **más lento** que PyTorch CPU:
por eso no se ofrece esa combinación.

### GPU NVIDIA (CUDA)

Con solo `pip install -r requirements.txt` se instala PyTorch en versión CPU.
Para usar una GPU NVIDIA, instala la build con CUDA (ver <https://pytorch.org>):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Aparecerá `cuda` en el desplegable y `auto` la preferirá sobre la GPU Intel.

### Dónde se va el tiempo

Medido sobre grabaciones 1080p H.265 a 30 fps analizadas a 1 fps: la
decodificación va a unos 700 frames/s y supone ~2.8 h por cada 65 h de material;
el resto es inferencia. Es decir, **el cuello de botella es el modelo, no leer el
video**: bajar de `yolov8m` a `yolov8n` o activar la GPU se nota; analizar menos
frames por segundo, también.

## Uso

```bash
python app.py
```

1. **Seleccionar carpeta**: elige la carpeta con las grabaciones. Se muestra la ruta
   y cuántos videos se encontraron.
2. **Seleccionar carpeta de salida**: dónde se escribirán los resultados.
3. En **Video de referencia** elige de qué video tomar la vista previa.
4. **Elige el frame concreto** que quieras usar de referencia:
   - arrastra el **deslizador** bajo la imagen (el salto se aplica al soltar);
   - usa `<<` / `<` / `>` / `>>` para moverte 300 o 30 frames;
   - escribe un **número de frame** y pulsa Enter;
   - o escribe un tiempo en **`HH:MM:SS`** (también valen `MM:SS` o segundos sueltos)
     y pulsa **Ir**.

   A la derecha se muestran el tiempo actual, la duración, la resolución y los FPS.
5. Sobre la imagen, dibuja la zona de la puerta. Hay dos modos, junto al desplegable:
   - **Rectángulo**: haz clic y arrastra.
   - **Polígono**: haz clic en cada esquina y cierra con doble clic o clic derecho
     (Esc descarta el que está a medias). Es lo que conviene cuando la cámara ve
     la puerta en perspectiva y el vano o el suelo delante no son un rectángulo
     en la imagen; el criterio `pies` funciona igual con un trapecio dibujado
     sobre el suelo.
   Las coordenadas se convierten automáticamente a la resolución real del video.
6. Ajusta los parámetros:
   - **Frames por segundo a analizar** (por defecto `1`): más FPS = más precisión y
     más tiempo de proceso.
   - **Tolerancia (segundos)** (por defecto `3`): cuánto silencio se permite entre
     dos detecciones antes de cerrar el evento.
   - **Modelo YOLO**: `yolov8n` (rápido) → `yolov8m` (más preciso, más lento).
   - **Acelerador**: `auto` salvo que quieras forzar CPU o una GPU concreta.
   - **Criterio de zona** y **Solape mín.**: ver la sección siguiente.
   - **Registrar también detecciones generales**: añade al CSV las personas
     vistas fuera de la zona.
   - **Detectar rostros** (opcional): ver la sección siguiente.
7. Pulsa **Iniciar análisis**. Puedes **Cancelar** en cualquier momento.
8. Al terminar, usa **Abrir carpeta de resultados**.

## Clasificación de las detecciones

Cada detección se clasifica en uno de dos tipos, que conviven en el mismo CSV en
la columna `tipo`:

| Tipo | Qué es |
|------|--------|
| `zona` | La persona cumple el criterio de la zona de la puerta. |
| `general` | Cualquier persona vista en el frame, esté donde esté. |

Las detecciones generales se registran si está marcada la casilla *Registrar
también detecciones generales*. Sirven de control: si un tramo tiene muchos
eventos `general` y ninguno `zona`, es que pasa gente pero no se acerca a la
puerta.

### Criterio de zona (esto es lo que evita los falsos positivos)

En un pasillo, la caja de alguien que camina por el centro puede **rozar la
esquina** de una puerta lejana por pura perspectiva, sin que la persona esté ni
cerca de ella. Por eso el solape simple da falsos positivos, y no es el criterio
por defecto:

| Criterio | Cuenta como "en la zona" cuando... | Cuándo usarlo |
|----------|------------------------------------|---------------|
| **`pies`** (por defecto) | el punto de apoyo (centro del borde inferior de la caja) cae dentro de la zona | zona dibujada sobre el suelo o el vano que se pisa. El más fiable |
| `centro` | el centro de la caja cae dentro de la zona | zonas altas, como una ventanilla o un mostrador |
| `solape` | al menos *Solape mín.* del área de la persona está dentro de la zona | zonas grandes donde la persona entra de lado |

Ejemplo medido sobre una captura real de pasillo: la caja de una persona a media
distancia solapaba la zona de la puerta del fondo en apenas un **1.7 % de su
área**. El criterio antiguo (cualquier solape) lo daba por bueno; los tres
criterios actuales lo descartan.

### Seguimiento de personas (cuántas y hacia dónde)

Con **Seguir a cada persona (ByteTrack)** activado (por defecto), cada persona
recibe un identificador estable entre frames. Eso añade dos columnas a cada
evento, en la tabla, el CSV y los informes:

- **Personas**: cuántas personas *distintas* pasaron durante el evento. Sin
  seguimiento, dos personas que cruzan seguidas son un solo evento sin más
  información; con él, el evento dice "2".
- **Dirección**: hacia dónde iba cada una respecto a la zona de la puerta:
  - `entra`: apareció fuera de la zona y la última vez se la vio dentro (llegó a
    la puerta y desapareció por ella).
  - `sale`: apareció dentro de la zona y se alejó.
  - `cruza`: pasó por la zona sin quedarse (fuera → dentro → fuera).
  - `permanece`: siempre dentro de la zona.

  Se resume por evento: `2 entran, 1 sale`.

Si el seguimiento pierde a alguien (a 1 fps una persona rápida puede aparecer
en un solo frame), la detección **no se pierde**: cuenta igualmente en el
evento, y el conteo nunca baja del máximo de personas vistas a la vez. Para
que las pistas sean fiables conviene analizar a **2 fps o más**; a 1 fps la
dirección sale bien en pasillos largos y peor en puertas que se cruzan en dos
segundos.

## Detección de rostros

Es **opcional** y se activa con la casilla *Detectar rostros*. Solo se busca dentro
de la **mitad superior de la caja de cada persona ya detectada**, lo que la hace
rápida y reduce falsos positivos.

Dos backends:

| Backend | Requiere | Calidad |
|---------|----------|---------|
| `haar`  | nada, viene con `opencv-python` | solo rostros bastante frontales y grandes |
| `yunet` | el ONNX (ver abajo) | bastante mejor con ángulos y rostros pequeños |

Los modelos se descargan con:

```powershell
python descargar_modelos.py
```

Si faltan, la app avisa en el log y sigue analizando personas sin rostros. El
desplegable *Backend* solo ofrece los backends que tu instalación puede usar
realmente.

> **OpenCV 5 eliminó `cv2.CascadeClassifier`** y ya no distribuye las cascadas
> Haar, así que en esas versiones el único backend disponible es `yunet`.

## Identificación de personas

Sobre la detección de rostros, la casilla *Identificar personas conocidas* compara
cada rostro contra un catálogo usando **SFace** (`cv2.FaceRecognizerSF`): convierte
el rostro alineado en un vector de 128 dimensiones y lo compara por similitud
coseno. Requiere el backend `yunet`, porque la alineación usa sus puntos faciales.

El catálogo es una carpeta con **una subcarpeta por persona**; el nombre de la
subcarpeta es la etiqueta que sale en el CSV:

```
personas/
├── ana_torres/
│   ├── 1.jpg
│   └── 2.jpg
└── juan_perez/
    └── frente.jpg
```

Cuantas más fotos por persona (distintos ángulos e iluminaciones), mejor. Se toma
la cara más grande de cada foto. Si ninguna referencia supera el umbral de
similitud (`0.363`, el que recomienda OpenCV), el rostro queda como
`desconocido` y no aparece en la columna `personas`.

> **Requisitos reales para que funcione.** Medido sobre grabaciones de pasillo a
> 1080p: con el rostro a **115x176 px** el reconocimiento acierta, pero a
> **99x113 px o menos** la similitud se desploma por debajo del umbral aunque sea
> la misma persona, por desenfoque de movimiento y poca luz. En la práctica hace
> falta la cara **frontal, nítida y de al menos ~120 px de alto**, lo que implica
> una cámara cerca de la puerta y a la altura de la cabeza. En una cámara de
> pasillo montada en alto, la mayoría de eventos quedarán como `desconocido`.

Los datos biométricos están regulados (RGPD y equivalentes locales): usar esta
función sobre personas identificables suele exigir base legal, información previa
y un plazo de conservación definido.

## Salida

```
<carpeta de salida>/
├── eventos.csv
├── miniaturas/
│   ├── zona/                       eventos dentro de la zona de la puerta
│   │   └── camara1_00-01-24.jpg
│   └── general/                    resto de detecciones de personas
│       └── camara1_00-03-02.jpg
└── rostros/                        (solo si se activa la detección de rostros)
    ├── zona/
    └── general/
        ├── camara1_00-03-02_frame.jpg   frame completo con los rostros marcados
        └── camara1_00-03-02_rostro1.jpg
```

`eventos.csv` (UTF-8 con BOM, se abre bien en Excel):

| archivo      | tipo    | inicio   | fin      | duracion_segundos | rostros | personas   |
|--------------|---------|----------|----------|-------------------|---------|------------|
| camara1.mp4  | zona    | 00:01:24 | 00:01:39 | 15.00             | 1       | ana_torres |
| camara1.mp4  | general | 00:03:02 | 00:03:11 | 9.00              | 1       |            |

Cada miniatura es el primer frame del evento con la **zona de la puerta** en azul y
la **caja de la persona** en verde. La columna `rostros` es el **máximo de rostros
vistos a la vez** durante el evento, y los recortes salen del frame del evento donde
más rostros se detectaron (que no tiene por qué ser el de la miniatura). La
columna `personas` lista las personas del catálogo reconocidas durante el evento.

## Análisis incremental

Con **Reutilizar videos ya analizados** activado (por defecto), la app apunta en
`analizados.json`, dentro de la carpeta de resultados, cada video que termina
de analizar: tamaño, fecha, los parámetros usados y sus eventos. Al volver a
lanzar el análisis sobre la misma carpeta:

- los videos que ya están en el manifiesto **con los mismos parámetros** y que
  no han cambiado se saltan, y sus eventos se recuperan al instante (aparecen
  en la tabla y en el informe como si se hubieran analizado ahora);
- solo se procesan los videos nuevos o modificados;
- el informe y el CSV se regeneran con todo, reutilizado o no.

Cambiar la zona, el criterio, los fps, la tolerancia, el modelo o cualquier
otro parámetro que altere los eventos invalida lo guardado y se reanaliza.
Cambiar de acelerador o activar la decodificación por hardware **no**: dan los
mismos eventos.

Se escribe tras cada video, no al final, así que si cancelas o se corta a la
mitad, lo ya hecho se conserva y al relanzar continúa donde estaba. Los videos
que dieron error no se apuntan: se reintentan la próxima vez. Para reanalizar
todo desde cero, desmarca la casilla o borra `analizados.json`.

## Generar el ejecutable (.exe)

```powershell
pip install -r requirements-dev.txt
python build_exe.py
```

Queda en `dist\DeCam\DeCam.exe` (carpeta distribuible completa). Verificado: el
build ocupa **1.1 GB**, el `.exe` son 47 MB y arranca en pocos segundos con unos
280 MB de RAM. Con `--onefile` se genera un único archivo, pero **no es
recomendable**: se descomprime en una carpeta temporal en cada arranque.

Notas:

- Si PyInstaller no está en el intérprete que ejecuta el script, este aborta con
  un mensaje indicando cuál es ese intérprete. Ojo al lanzarlo desde el depurador
  de VS Code: el diálogo de `SystemExit: 1` tapa el mensaje real, que está en la
  consola justo encima.
- El build pesa **1–3 GB** según si tienes PyTorch CPU o CUDA. Es normal.
- Compila en la misma arquitectura de destino: un `.exe` solo corre en Windows.
- Si ya descargaste los pesos (`yolov8n.pt`) o el ONNX de YuNet en `models/`, el
  script los empaqueta y el ejecutable funciona **sin internet**. Si no, la primera
  ejecución intentará descargar los pesos.
- Windows Defender / SmartScreen suele marcar ejecutables de PyInstaller sin firmar.
  Para distribuirlo fuera de tu equipo necesitarías un certificado de firma de código.
- Para depurar un `.exe` que se cierra al instante, recompila con `--console` y
  lánzalo desde una terminal para ver el error.

### Releases automáticos (GitHub Actions)

Cada push a `main` ejecuta [`.github/workflows/release.yml`](.github/workflows/release.yml):

1. instala las dependencias y corre los tests (si fallan, no hay release);
2. descarga los modelos (YuNet/SFace, `yolov8n.pt`) y exporta YOLO a OpenVINO,
   para que el equipo destino no tenga que descargar nada;
3. compila con `build_exe.py` y genera el instalador con Inno Setup
   ([`instalador.iss`](instalador.iss));
4. publica un release con la etiqueta `v<VERSION>.<número de ejecución>` y el
   `DeCam-Setup-*.exe` adjunto, con notas generadas a partir de los commits.

La versión base vive en el fichero [`VERSION`](VERSION) (`1.0.0`); el cuarto
número lo pone GitHub, así cada build es única sin tocar nada. Para subir de
versión, edita `VERSION`. El instalador es **por usuario** (no pide
administrador) e instala en `%LOCALAPPDATA%\Programs\DeCam`. Pesa varios
cientos de MB porque lleva PyTorch y OpenVINO dentro; el build tarda del orden
de 20–30 minutos.

## Estructura del proyecto

Todo el código vive en el paquete `decam/`; `app.py` solo lanza la interfaz.

| Ruta | Qué hace |
|---|---|
| `app.py` | Punto de entrada: `python app.py`. |
| `decam/analizador.py` | `AnalizadorPuerta` orquesta el análisis; `crear_analizador` monta las piezas reales. |
| `decam/deteccion.py` | `DetectorPersonas` (protocolo) y `DetectorYOLO`. |
| `decam/seguimiento.py` | `Rastreador` (protocolo) y ByteTrack; trayectorias y dirección; agrupación en eventos. |
| `decam/zona.py` | `Zona` (protocolo), `ZonaRectangular` y los criterios pies/centro/solape. |
| `decam/eventos.py` | `Evento`, `ResultadoVideo`, columnas del CSV. |
| `decam/configuracion.py` | `ConfiguracionAnalisis` y su validación. |
| `decam/movimiento.py` | Filtro de movimiento. |
| `decam/video.py` | Búsqueda de videos, metadatos, saltos y lectura en hilo. |
| `decam/rostros.py` | Detección (YuNet/Haar) e identificación (SFace) de rostros. |
| `decam/aceleradores.py` | CUDA / OpenVINO / CPU y exportación a OpenVINO. |
| `decam/salida.py` | `EscritorSalida`: miniaturas, recortes y CSV. |
| `decam/manifiesto.py` | Manifiesto del análisis incremental. |
| `decam/reporte.py` | Informes HTML y PDF. |
| `decam/registro.py` | Carpeta de datos del usuario y log en fichero. |
| `decam/ui/` | Tkinter: `PanelParametros`, `VistaPrevia`, `TablaEventos`, `AplicacionDeCam`. |
| `tests/` | Tests unitarios (`python -m pytest`). |
| `build_exe.py`, `instalador.iss`, `VERSION`, `.github/workflows/` | Ejecutable, instalador y releases. |
| `descargar_modelos.py` | Baja los ONNX de rostros de OpenCV Zoo a `models/`. |
| `requirements*.txt` | Dependencias de ejecución y de desarrollo. |
| `config.json` | Se genera solo: última carpeta, zona y parámetros. |
| `analizados.json` | En la carpeta de resultados: manifiesto del análisis incremental. |

### Arquitectura

`AnalizadorPuerta` no construye nada: recibe un `DetectorPersonas`, un
`Rastreador`, el filtro de movimiento y el analizador de rostros ya hechos, y
escribe a través de un `EscritorSalida` opcional. `crear_analizador(config)` es
la única función que conoce YOLO y ByteTrack. Consecuencias prácticas:

- Los tests recorren `analizar_video` entero con un detector guionizado y sin
  modelo ([`tests/test_analizador.py`](tests/test_analizador.py)).
- Cambiar de modelo es implementar `DetectorPersonas.detectar(frame)`; una zona
  poligonal es implementar `Zona`. Ni los criterios ni el analizador cambian.
- La interfaz solo conoce `crear_analizador`, `EscritorSalida` y `Manifiesto`,
  y habla con el análisis por callbacks (`on_log`, `on_progreso`, `on_evento`).

### Dónde se guardan configuración y registro

Al arrancar, la app escribe ambas rutas en la pestaña **Registro**.

- **Ejecutando el código**: `config.json` y `decam.log` quedan junto a `app.py`.
- **Ejecutable (`.exe`)**: en `%LOCALAPPDATA%\DeCam\`. No pueden ir junto al
  `.exe` porque en modo `--onefile` esa carpeta es temporal y se borra al salir,
  y en `--onedir` puede ser de solo lectura.
- **Instalación portable**: si colocas un `config.json` (aunque sea `{}`) junto
  al `.exe`, se usa esa carpeta para todo.

`decam.log` rota al llegar a 1 MB y conserva 3 copias. Recoge el registro de la
interfaz y, además, el traceback completo de cualquier fallo: es lo primero que
hay que mirar si el `.exe` falla, porque al compilarse sin consola no hay otra
forma de verlo.

## Notas y ajustes

- Un video que no se pueda abrir se salta: el error queda en el log y en el resumen,
  y el análisis continúa con el siguiente.
- Los tiempos del CSV son **relativos al inicio de cada video**, no la hora del reloj.
  En exportaciones tipo `..._20260825050744_20260825070843_427095.mp4` la hora real
  del evento es el timestamp inicial del nombre + el tiempo del CSV.
- Los `.jpg` que algunos NVR dejan junto a cada `.mp4` se ignoran: solo se leen
  las extensiones de video.
- El umbral de confianza (`0.35`) está en `ConfiguracionAnalisis.confianza`
  en [detector.py](detector.py); súbelo si hay falsos positivos, bájalo si se
  escapan personas.
- La zona guardada corresponde a la resolución de la cámara con la que se dibujó.
  Si mezclas cámaras con resoluciones distintas, redibuja la zona para cada tanda.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

Cubren la lógica que decide qué se registra y qué no, sin necesitar videos ni
modelos: la geometría caja/zona y los tres criterios, la agrupación de detecciones
en eventos (tolerancia, cierre al final del video, miniatura y rostros), el filtro
de movimiento, el seguimiento (identificadores y dirección), el análisis
incremental, el recorrido completo de un video con un detector falso, el CSV,
las preferencias y las rutas de datos en el `.exe`. Duran
unos segundos. En VS Code aparecen en el panel de pruebas y en la tarea
**DeCam: ejecutar tests**.

## Depurar en VS Code

La carpeta `.vscode/` ya viene configurada. Abre la carpeta del proyecto en VS Code
y comprueba abajo a la derecha que el intérprete es `.venv` (si no,
`Ctrl+Shift+P` → *Python: Select Interpreter*).

En **Ejecutar y depurar** (`Ctrl+Shift+D`) tienes:

| Configuración | Para qué |
|---------------|----------|
| **DeCam: interfaz gráfica** | El día a día. `F5` y a poner breakpoints en `app.py` / `detector.py`. |
| **DeCam: GUI + entrar en librerías** | Igual pero con `justMyCode: false`, para meterse dentro de `ultralytics`, `torch` o `cv2`. |
| **DeCam: archivo actual** | Ejecuta el archivo abierto. |
| **DeCam: compilar .exe** | Lanza `build_exe.py` bajo el depurador. |
| **DeCam: adjuntar a proceso** | Se engancha a un análisis ya en marcha; requiere añadir `debugpy.listen(5678)` en el código. |

Tareas (`Ctrl+Shift+P` → *Tasks: Run Task*): instalar dependencias, descargar los
modelos de rostros, compilar el `.exe` y ejecutar sin depurar.

Un par de detalles al depurar esta app:

- El análisis corre en un hilo aparte. Un breakpoint dentro de `AnalizadorPuerta`
  detiene **ese** hilo; la ventana sigue respondiendo pero el progreso se congela.
- `Records/`, `output/`, `dist/` y `.venv/` están excluidos de la búsqueda y del
  watcher en `settings.json`: con 20 GB de video, indexarlos ralentiza el editor.
