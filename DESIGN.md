# DESIGN: Arquitectura e Implementación de XY-Cut++ (Rust Port)

Este apartado amplía el quickstart con decisiones de arquitectura y estado actual de implementación en el paquete `xycutppy`.

## 🧱 Arquitectura actual por backends

El wrapper Python expone dos motores en la misma librería, seleccionables en runtime:

- `paper`: backend original (GPL), basado en `src/paper/xycut_plus_plus`.
- `datalab`: backend Rust del sorter de OpenDataLoader (Apache-2.0), en `src/datalab/xycut_plus_plus_sorter`.

### Aislamiento entre módulos

`paper` y `datalab` están desacoplados:

1. `datalab` no depende de traits ni tipos de `paper`.
2. `datalab` define su propio `Element` y opera como crate independiente.
3. `src/lib.rs` del wrapper solo convierte `PyElement -> datalab::Element` cuando se llama el backend `datalab`.

Esto permite evolución independiente de ambos algoritmos y simplifica compliance de licencias.

## 🏗️ Arquitectura del algoritmo (Datalab Rust)

La implementación `datalab` conserva la estructura de 4 fases del flujo Java:

### Fase 1: detección de `Cross-Layout`

Detecta bloques muy anchos y con solape horizontal suficiente:

- Regla base: `width >= beta * max_width`.
- Confirmación de contexto: solape con al menos `MIN_OVERLAP_COUNT` elementos.
- Resultado: separación entre `cross_layout` y `remaining`.

### Fase 2: densidad y preferencia de eje

Calcula `content_area / region_area`:

- Si supera `DEFAULT_DENSITY_THRESHOLD`, activa `prefer_horizontal_first`.
- Esta preferencia se usa como desempate cuando ambos cortes son viables con gaps equivalentes.

### Fase 3: segmentación recursiva

`recursive_segment` busca mejor corte horizontal y vertical, aplica umbral de gap y divide en subgrupos:

- Usa `MIN_GAP_THRESHOLD` para evitar cortes por ruido.
- Si no hay corte válido, fallback a orden geométrico `Y desc, X asc`.
- Filtra outliers estrechos al buscar corte vertical (`NARROW_ELEMENT_WIDTH_RATIO`).

### Fase 4: merge final

Reinserta `cross_layout` comparando `top_y` contra el flujo principal:

- Merge estable y puramente geométrico.
- Sin pesos semánticos del paper original.

## 🦀 Decisiones de ingeniería aplicadas en Rust

1. **Orden robusto de `f32`**: se usa `f32::total_cmp` para evitar ambigüedades con `NaN`.
2. **Remoción segura por índice**: `cross_layout` se separa por índice/máscara booleana, no por `id`.
3. **Uso real de `prefer_horizontal_first`**: ya no es parámetro fantasma; actúa como desempate de corte.
4. **Clonación controlada y legibilidad**: se mantiene enfoque con `Vec`/`clone` por claridad; se preasigna capacidad en rutas críticas (`Vec::with_capacity`).

## 📌 Estado de labels semánticos en código

En la API Rust/Python expuesta por `paper`, el enum actual incluye:

- `CrossLayout`
- `HorizontalTitle`
- `VerticalTitle`
- `Vision`
- `Regular`

En `datalab` actual, el algoritmo es geométrico y no consume labels en el núcleo de ordenación.

## 🔧 Notas de evolución

Si en producción aparecen cuellos de botella con páginas extremas (decenas de miles de cajas), el siguiente salto de optimización recomendado es migrar la recursión a particionado in-place con `&mut [Element]`. Mientras tanto, la implementación actual prioriza mantenibilidad y trazabilidad del algoritmo.

# Entendiendo los `SemanticLabel` en XY-Cut++ (Rust Port)

Este documento explica el propósito del enumerador `SemanticLabel` dentro del trait `BoundingBox`, su correspondencia con la estructura de un PDF tradicional, y las decisiones de diseño arquitectónico tomadas para este *port* en Rust.

## 📖 Contexto Histórico: El Paper vs. La Realidad

El algoritmo XY-Cut++ original (basado en el paper *arXiv:2504.10258*) define un *pipeline* de 4 fases. La **Fase 4 (Cross-Modal Matching)** utiliza etiquetas semánticas (`SemanticLabel`) para decidir la prioridad con la que los elementos "extraídos" (como encabezados o imágenes) se vuelven a insertar en el flujo de texto principal.

**⚠️ Nota de Implementación:** La implementación oficial de OpenDataLoader en Java (en la que se basa este *port* de Rust) **omitió la Fase 4 y el uso de etiquetas semánticas**. Los mantenedores descubrieron que, al leer directamente de la estructura del PDF en lugar de usar un modelo de visión artificial (OCR), basarse en la geometría pura producía resultados más estables y rápidos. 

Por lo tanto, en nuestra implementación base, el `SemanticLabel` se expone en el trait para compatibilidad y futuras extensiones, pero el algoritmo principal toma decisiones basadas puramente en coordenadas X/Y y densidades.

---

## 🏷️ Tipos de `SemanticLabel` y su equivalencia en PDF

Si decides implementar la re-inserción semántica (Stage 4 del paper) o usar etiquetas para filtrar antes de ordenar, aquí tienes la correspondencia de los tipos semánticos sugeridos con el estándar PDF (Tagged PDF / PDF/UA):

| SemanticLabel (XY-Cut++) | Equivalencia en PDF (Tagged PDF) | Descripción y Función en el Algoritmo |
| :--- | :--- | :--- |
| `CrossLayout` | `Artifact` (Pagination), `Header`, `Footer` | Elementos que cruzan toda la página (encabezados, pies de página, números de página). Tienen la **máxima prioridad** de reinserción en los extremos del documento. |
| `Title` / `Heading` | `H1`, `H2`, `H3`, `H4`... | Títulos que a menudo abarcan varias columnas. El algoritmo original del paper los usa para anclar el inicio de nuevas secciones o columnas. |
| `BodyText` / `Regular` | `P` (Paragraph), `Span` | El texto normal. Constituye el "ruido de fondo" y es lo que la Fase 3 de segmentación recursiva (XY-Cut) ordena activamente buscando los espacios en blanco. |
| `Vision` / `Figure` | `Figure`, `Formula` | Imágenes, gráficos o ecuaciones matemáticas complejas. En el paper, tienen prioridad media y suelen usarse como "bloqueadores" alrededor de los cuales el texto debe fluir. |
| `Table` | `Table`, `TR`, `TD` | Tablas de datos. Al igual que las imágenes, son bloques sólidos indivisibles. El algoritmo no debe intentar cortar una tabla por la mitad. |

---

## 🚀 Cómo usar `SemanticLabel` en Python

Para ordenar elementos con `xycutppy` solo necesitas construir una lista de diccionarios con los campos `id`, `x1`, `y1`, `x2`, `y2` y `label`. El campo `label` acepta cualquier valor de `SemanticLabel`.

### Opción A: Modo Geométrico Puro (Recomendado - Estilo OpenDataLoader)

Si solo quieres ordenar por geometría sin semántica, asigna `SemanticLabel.Regular` a todos los elementos. El algoritmo ignorará etiquetas y operará puramente sobre coordenadas.

```python
from xycutppy import compute_order, SemanticLabel

elements = [
    {'id': 0, 'x1': 10.0, 'y1': 10.0, 'x2': 790.0, 'y2': 30.0,  'label': SemanticLabel.Regular},
    {'id': 1, 'x1': 10.0, 'y1': 50.0, 'x2': 390.0, 'y2': 100.0, 'label': SemanticLabel.Regular},
    {'id': 2, 'x1': 410.0,'y1': 50.0, 'x2': 790.0, 'y2': 100.0, 'label': SemanticLabel.Regular},
    {'id': 3, 'x1': 10.0, 'y1': 120.0,'x2': 790.0, 'y2': 200.0, 'label': SemanticLabel.Regular},
]

page_bounds = (0.0, 0.0, 800.0, 1200.0)
ordered_ids = compute_order(elements, page_bounds)
print(ordered_ids)  # e.g. [0, 1, 2, 3]
```

### Opción B: Mapeo desde etiquetas de un modelo YOLO / Layout Detection

Si usas un modelo de detección como *PP-DocLayout*, *DocLayNet* o cualquier YOLO entrenado en documentos, recibirás IDs de clase numéricos. Puedes convertirlos a `SemanticLabel` con un diccionario de mapeo:

```python
from xycutppy import compute_order, SemanticLabel

# Etiquetas típicas de un modelo YOLO de análisis de documentos
YOLO_LABEL_MAP = {
    0: 'Caption',
    1: 'Footnote',
    2: 'Formula',
    3: 'List-item',
    4: 'Page-footer',
    5: 'Page-header',
    6: 'Picture',
    7: 'Section-header',
    8: 'Table',
    9: 'Text',
    10: 'Title',
}

# Conversión de nombre de clase YOLO -> SemanticLabel de xycutppy
YOLO_TO_SEMANTIC = {
    'Caption':        SemanticLabel.Regular,
    'Footnote':       SemanticLabel.Regular,
    'Formula':        SemanticLabel.Vision,
    'List-item':      SemanticLabel.Regular,
    'Page-footer':    SemanticLabel.CrossLayout,
    'Page-header':    SemanticLabel.CrossLayout,
    'Picture':        SemanticLabel.Vision,
    'Section-header': SemanticLabel.HorizontalTitle,
    'Table':          SemanticLabel.Vision,
    'Text':           SemanticLabel.Regular,
    'Title':          SemanticLabel.HorizontalTitle,
}

def yolo_class_to_semantic(class_id: int) -> SemanticLabel:
    label_name = YOLO_LABEL_MAP.get(class_id, 'Text')
    return YOLO_TO_SEMANTIC.get(label_name, SemanticLabel.Regular)


# Salida típica de un modelo YOLO: lista de (class_id, x1, y1, x2, y2)
yolo_detections = [
    (5,  10.0,  0.0, 790.0,  20.0),   # Page-header  -> CrossLayout
    (10, 10.0,  30.0, 790.0,  60.0),  # Title        -> HorizontalTitle
    (9,  10.0,  70.0, 390.0, 200.0),  # Text (col 1) -> Regular
    (9,  410.0, 70.0, 790.0, 200.0),  # Text (col 2) -> Regular
    (8,  10.0, 210.0, 790.0, 400.0),  # Table        -> Vision
    (1,  10.0, 410.0, 790.0, 430.0),  # Footnote     -> Regular
    (4,  10.0, 440.0, 790.0, 460.0),  # Page-footer  -> CrossLayout
]

elements = [
    {
        'id': idx,
        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
        'label': yolo_class_to_semantic(class_id),
    }
    for idx, (class_id, x1, y1, x2, y2) in enumerate(yolo_detections)
]

page_bounds = (0.0, 0.0, 800.0, 1200.0)
ordered_ids = compute_order(elements, page_bounds)
print("Orden de lectura:", ordered_ids)
# Esperado: header (0) -> title (1) -> col-izq (2) -> col-der (3) -> table (4) -> footnote (5) -> footer (6)
```

### Opción C: Mapeo desde etiquetas PDF nativas (Tagged PDF)

Si extraes el documento desde una librería como `pdfplumber` o `pymupdf` y tienes acceso a los tags estructurales:

```python
from xycutppy import compute_order, SemanticLabel

PDF_TAG_TO_SEMANTIC = {
    'H1': SemanticLabel.HorizontalTitle,
    'H2': SemanticLabel.HorizontalTitle,
    'H3': SemanticLabel.HorizontalTitle,
    'P':  SemanticLabel.Regular,
    'Figure': SemanticLabel.Vision,
    'Formula': SemanticLabel.Vision,
    'Table': SemanticLabel.Vision,
    'Header': SemanticLabel.CrossLayout,
    'Footer': SemanticLabel.CrossLayout,
}

def pdf_tag_to_semantic(tag: str) -> SemanticLabel:
    return PDF_TAG_TO_SEMANTIC.get(tag, SemanticLabel.Regular)

# Bloques extraídos de pymupdf / pdfplumber con su tag estructural
pdf_blocks = [
    {'id': 0, 'tag': 'Header', 'x1': 0.0,   'y1': 0.0,   'x2': 595.0, 'y2': 30.0},
    {'id': 1, 'tag': 'H1',     'x1': 50.0,  'y1': 40.0,  'x2': 545.0, 'y2': 70.0},
    {'id': 2, 'tag': 'P',      'x1': 50.0,  'y1': 80.0,  'x2': 280.0, 'y2': 300.0},
    {'id': 3, 'tag': 'Figure', 'x1': 300.0, 'y1': 80.0,  'x2': 545.0, 'y2': 300.0},
    {'id': 4, 'tag': 'P',      'x1': 50.0,  'y1': 310.0, 'x2': 545.0, 'y2': 450.0},
    {'id': 5, 'tag': 'Footer', 'x1': 0.0,   'y1': 810.0, 'x2': 595.0, 'y2': 842.0},
]

elements = [
    {**b, 'label': pdf_tag_to_semantic(b['tag'])}
    for b in pdf_blocks
]

page_bounds = (0.0, 0.0, 595.0, 842.0)  # A4 en puntos
ordered_ids = compute_order(elements, page_bounds)
print("Orden de lectura:", ordered_ids)
```

## 🧠 Preguntas Frecuentes

**¿Por qué mi ordenamiento falla si un encabezado tiene la etiqueta `Regular`?**
En la implementación actual (puramente geométrica), el encabezado será detectado correctamente por la Fase 1 si su anchura supera el `DEFAULT_BETA` (por defecto 2.0 veces la anchura máxima) independientemente de su etiqueta. Si el encabezado es estrecho, bajará su coeficiente `beta`.

**¿Debo intentar implementar la Fase 4 (Cross-Modal Matching) del paper?**
Como aconsejan los autores originales de DataLab: *No lo hagas a menos que uses modelos de visión artificial*. Si lees el PDF nativo, las coordenadas espaciales ya son lo suficientemente precisas. Mezclar detección geométrica con pesos semánticos sin afinar suele amplificar los errores.
