"""
Batería de Pruebas Visuales y Casos de Uso para XYCut++
======================================================
Este script simula layouts complejos de documentos (Doble columna, títulos cruzados,
secciones mixtas) y genera imágenes PNG anotadas con el orden de lectura calculado
por el algoritmo, usando mapas de colores basados en datasets tipo YOLO.
"""

import logging
import os
from typing import Any, Dict, List, Tuple
from PIL import Image, ImageDraw, ImageFont

# Importamos las herramientas de nuestro paquete unificado en Rust/Python
from xycutppy import compute_order, SemanticLabel, XYCutConfig, configure_logging, available_backends

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Configuración del mapeo de etiquetas y colores (Estilo YOLO Layout)
# ----------------------------------------------------------------------
LABEL_MAP = {
    0: 'Caption', 1: 'Footnote', 2: 'Formula', 3: 'List-item',
    4: 'Page-footer', 5: 'Page-header', 6: 'Picture',
    7: 'Section-header', 8: 'Table', 9: 'Text', 10: 'Title'
}

# Asignamos un color RGB pastel/distinguible a cada tipo de etiqueta para la visualización
COLOR_MAP = {
    'Caption': (255, 182, 193),  # Rosa claro
    'Footnote': (220, 220, 220),  # Gris claro
    'Formula': (255, 218, 185),  # Melocotón
    'List-item': (204, 255, 204),  # Verde pastel
    'Page-footer': (176, 224, 230),  # Azul polvo
    'Page-header': (176, 224, 230),  # Azul polvo
    'Picture': (255, 255, 204),  # Amarillo claro
    'Section-header': (230, 230, 250),  # Lavanda
    'Table': (255, 204, 153),  # Naranja suave
    'Text': (204, 229, 255),  # Azul cielo claro
    'Title': (255, 153, 153),  # Rojo coral blando
}

# Mapeo intermedio para transformar etiquetas YOLO/Strings a los enums que acepta nuestro core en Rust
SEMANTIC_MAPPING = {
    'Title': SemanticLabel.HorizontalTitle,
    'Section-header': SemanticLabel.HorizontalTitle,
    'Page-header': SemanticLabel.CrossLayout,
    'Page-footer': SemanticLabel.CrossLayout,
    'Picture': SemanticLabel.Vision,
    'Table': SemanticLabel.Vision,
    'Formula': SemanticLabel.Vision,
    'Text': SemanticLabel.Regular,
    'List-item': SemanticLabel.Regular,
    'Caption': SemanticLabel.Regular,
    'Footnote': SemanticLabel.Regular,
}


# ----------------------------------------------------------------------
# Función de Renderizado Visual (Generación del PNG)
# ----------------------------------------------------------------------
def render_page_layout(
        filename: str,
        elements: List[Dict[str, Any]],
        ordered_ids: List[int],
        page_bounds: Tuple[float, float, float, float]
):
    """
    Dibuja una representación visual del layout de la página en un PNG,
    pintando cada caja de su color semántico y añadiendo el índice del orden
    de lectura en el centro de la caja.
    """
    _, _, width, height = page_bounds
    # Creamos un lienzo en blanco (con un extra de margen para ver los bordes)
    image = Image.new("RGB", (int(width) + 20, int(height) + 20), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    # Intentar cargar una fuente por defecto del sistema, si falla usamos la básica integrada
    try:
        font_text = ImageFont.load_default()
    except Exception:
        font_text = None

    # Mapeamos los IDs a su índice de lectura (1º, 2º, 3º...) para pintarlo con claridad
    order_position = {element_id: index + 1 for index, element_id in enumerate(ordered_ids)}

    for elem in elements:
        # Desplazamos +10 píxeles para que no se pegue al borde físico de la imagen
        x1, y1, x2, y2 = elem['x1'] + 10, elem['y1'] + 10, elem['x2'] + 10, elem['y2'] + 10
        yolo_label = elem['yolo_label']
        elem_id = elem['id']

        # Obtener color según el mapa
        color = COLOR_MAP.get(yolo_label, (200, 200, 200))

        # 1. Dibujar el rectángulo relleno semi-transparente (haciendo un patrón o borde grueso)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        # Pintar un fondo interno un poco más claro simulando opacidad
        draw.rectangle([x1 + 2, y1 + 2, x2 - 2, y2 - 2], fill=(*color, 50))

        # 2. Dibujar la etiqueta pequeña del tipo de elemento en la esquina superior izquierda
        tag_str = f"{yolo_label} (ID:{elem_id})"
        draw.text((x1 + 4, y1 + 4), tag_str, fill=(50, 50, 50), font=font_text)

        # 3. Dibujar el número del ORDEN DE LECTURA grande y destacado en el centro de la caja
        if elem_id in order_position:
            idx_lectura = str(order_position[elem_id])
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            # Dibujar un pequeño círculo blanco de fondo para que el número resalte
            draw.ellipse([cx - 12, cy - 12, cx + 12, cy + 12], fill=(255, 255, 255), outline=(0, 0, 0))
            # Escribir el número de orden en negro
            draw.text((cx - 4, cy - 6), idx_lectura, fill=(0, 0, 0), font=font_text)

    # Crear directorio si no existe y guardar
    os.makedirs("output_examples", exist_ok=True)
    full_path = os.path.join("output_examples", filename)
    image.save(full_path)
    logger.info(f"  [Visual] Imagen guardada con éxito en: {full_path}")


# ----------------------------------------------------------------------
# Inicialización de Entorno y Demostración de Loggers (Requisito 1)
# ----------------------------------------------------------------------
logger.info("======================================================================")
logger.info(" DEMOSTRACIÓN DE BACKENDS Y CONFIGURACIÓN DE LOGGING (XYCut++) ")
logger.info("======================================================================\n")

# Configuración básica del logger nativo de Python
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

backends_disponibles = available_backends()
logger.info(f"Backends detectados e instalados en este sistema: {backends_disponibles}")

# Probamos a ejecutar una llamada cambiando niveles de loggers para ver cómo interactúa con Rust
for b in backends_disponibles:
    logger.info(f"\n--> Probando comportamiento de logs con Backend: '{b.upper()}'")

    # Caso A: Sin logs (Nivel por defecto o Warning)
    logger.info("  A) Cambiando configure_logging a nivel WARNING (Silencioso)...")
    configure_logging(logging.WARNING)
    # Ejecutamos una ordenación trivial para inicializar el backend interno
    compute_order([{'id': 0, 'x1': 0., 'y1': 0., 'x2': 10., 'y2': 10., 'label': SemanticLabel.Regular}],
                  (0., 0., 50., 50.), backend=b)

    # Caso B: Forzando Debug (Veremos trazas matemáticas y de gaps en consola si Rust tiene logs activados)
    logger.info("  B) Cambiando configure_logging a nivel DEBUG (Modo Verbose)...")
    configure_logging(logging.DEBUG)
    compute_order([{'id': 0, 'x1': 0., 'y1': 0., 'x2': 10., 'y2': 10., 'label': SemanticLabel.Regular}],
                  (0., 0., 50., 50.), backend=b)

# Reseteamos a INFO para continuar limpios con los escenarios estructurales
configure_logging(logging.INFO)

# ----------------------------------------------------------------------
# Definición de Escenarios de Layout de Documentos (Requisito 2 y 3)
# ----------------------------------------------------------------------
page_dimensions = (0.0, 0.0, 800.0, 1000.0)

# ESCENARIO 1: Artículo Científico Estándar (Título completo + Doble Columna + Pie de Página)
elementos_paper = [
    {'id': 10, 'x1': 50., 'y1': 40., 'x2': 750., 'y2': 60., 'yolo_label': 'Page-header'},
    {'id': 1, 'x1': 100., 'y1': 100., 'x2': 700., 'y2': 150., 'yolo_label': 'Title'},

    # Columna Izquierda (Bloque superior, una sección intermedia y párrafo)
    {'id': 2, 'x1': 50., 'y1': 200., 'x2': 380., 'y2': 230., 'yolo_label': 'Section-header'},
    {'id': 3, 'x1': 50., 'y1': 240., 'x2': 380., 'y2': 450., 'yolo_label': 'Text'},
    {'id': 4, 'x1': 50., 'y1': 470., 'x2': 380., 'y2': 600., 'yolo_label': 'Text'},

    # Columna Derecha (Párrafos paralelos que deben leerse DESPUÉS de toda la columna izquierda)
    {'id': 5, 'x1': 420., 'y1': 200., 'x2': 750., 'y2': 380., 'yolo_label': 'Text'},
    {'id': 6, 'x1': 420., 'y1': 400., 'x2': 750., 'y2': 600., 'yolo_label': 'Text'},

    # Elemento cruzado inferior
    {'id': 11, 'x1': 50., 'y1': 920., 'x2': 750., 'y2': 940., 'yolo_label': 'Page-footer'},
]

# ESCENARIO 2: Layout Complejo de Periódico / Revista (Layout denso con imágenes cruzadas y tablas)
elementos_news = [
    {'id': 1, 'x1': 50., 'y1': 50., 'x2': 750., 'y2': 120., 'yolo_label': 'Title'},

    # Sección A: Título de columna izquierda que abre paso a un texto
    {'id': 2, 'x1': 50., 'y1': 150., 'x2': 380., 'y2': 180., 'yolo_label': 'Section-header'},
    {'id': 3, 'x1': 50., 'y1': 190., 'x2': 380., 'y2': 400., 'yolo_label': 'Text'},

    # Una gran imagen en el centro-derecha que rompe la simetría
    {'id': 4, 'x1': 420., 'y1': 150., 'x2': 750., 'y2': 400., 'yolo_label': 'Picture'},
    {'id': 5, 'x1': 420., 'y1': 410., 'x2': 750., 'y2': 440., 'yolo_label': 'Caption'},

    # Sección B Inferior: Cruza horizontalmente con un bloque de tabla de datos completa
    {'id': 6, 'x1': 50., 'y1': 480., 'x2': 750., 'y2': 680., 'yolo_label': 'Table'},

    # Debajo de la tabla volvemos a un sistema de 3 columnas estrechas
    {'id': 7, 'x1': 50., 'y1': 710., 'x2': 260., 'y2': 900., 'yolo_label': 'Text'},
    {'id': 8, 'x1': 290., 'y1': 710., 'x2': 510., 'y2': 900., 'yolo_label': 'Text'},
    {'id': 9, 'x1': 540., 'y1': 710., 'x2': 750., 'y2': 900., 'yolo_label': 'Text'},
]

escenarios = [
    ("layout_academic_paper.png", elementos_paper),
    ("layout_dense_newspaper.png", elementos_news)
]

# Configuración personalizada de tolerancias matemáticas para procesar los escenarios
config_optimizada = XYCutConfig(
    min_cut_threshold=6.0,
    histogram_resolution_scale=0.5,
    same_row_tolerance=10.0
)

logger.info("\n======================================================================")
logger.info(" PROCESANDO ESCENARIOS ESTRUCTURALES Y GENERANDO MAPAS VISUALES ")
logger.info("======================================================================")

for b in backends_disponibles:
    logger.info(f"\nEjecutando suite de layouts sobre el backend: '{b.upper()}'")

    for filename_base, raw_elements in escenarios:
        # Preparamos los elementos inyectando el Enum SemanticLabel correspondiente mapeado desde YOLO
        processed_elements = []
        for e in raw_elements:
            yolo_lbl = e['yolo_label']
            enum_label = SEMANTIC_MAPPING.get(yolo_lbl, SemanticLabel.Regular)

            processed_elements.append({
                'id': e['id'],
                'x1': e['x1'],
                'y1': e['y1'],
                'x2': e['x2'],
                'y2': e['y2'],
                'label': enum_label
            })

        logger.info(f" -> Ordenando {filename_base}...")

        # Invocamos el algoritmo empaquetado en Rust
        ids_ordenados = compute_order(
            processed_elements,
            page_dimensions,
            config=config_optimizada,
            backend=b
        )

        # Generamos el nombre del archivo final incluyendo qué backend lo ha resuelto
        final_filename = f"{b}_{filename_base}"

        # Dibujamos y guardamos el PNG
        render_page_layout(final_filename, raw_elements, ids_ordenados, page_dimensions)

logger.info("\n======================================================================")
logger.info(" ¡PROCESO COMPLETADO! Revisa la carpeta 'output_examples/' para ver los PNGs ")
logger.info("======================================================================")