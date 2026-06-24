# 1. Importar los componentes necesarios desde el paquete xycutppy
#    Esto es el equivalente al `use xycut_plus_plus::{...}` de Rust.
import logging
import sys

from xycutppy import compute_order, SemanticLabel, XYCutConfig, configure_logging

# Configura el `basicConfig` de Python para que los logs se muestren en la consola.
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s]\t{%(name)s}\t%(message)s', stream=sys.stdout)

logger = logging.getLogger(__name__)

# --- CONFIGURAR LOGS DE RUST ---

# Opción 1: Ver solo los logs de DEBUG o superior del crate `xycut_plus_plus`.
# El resto de los crates seguirán en el nivel por defecto ("warn").
logger.info("--- Configurando logs a nivel DEBUG ---")
configure_logging(logging.DEBUG)

logger.info("--- Ejecutando ejemplo de xycutppy ---")

# 2. Crear la lista de elementos.
#    En lugar de una `struct` de Rust, usamos una lista de diccionarios,
#    que es una estructura de datos muy común y cómoda en Python.
#    Nuestro wrapper en Rust convierte automáticamente estos diccionarios
#    a la struct `PyElement` que necesita.
elements = [
    {
        'id': 0,
        'x1': 10.0, 'y1': 10.0, 'x2': 200.0, 'y2': 30.0,
        'label': SemanticLabel.HorizontalTitle
    },
    {
        'id': 1,
        'x1': 10.0, 'y1': 50.0, 'x2': 400.0, 'y2': 100.0,
        'label': SemanticLabel.Regular
    },
    # --- Añadamos más elementos para un ejemplo más interesante ---
    { # Un elemento a la derecha del elemento 1
        'id': 2,
        'x1': 450.0, 'y1': 50.0, 'x2': 750.0, 'y2': 100.0,
        'label': SemanticLabel.Regular
    },
    { # Un elemento debajo de los anteriores
        'id': 3,
        'x1': 10.0, 'y1': 120.0, 'x2': 750.0, 'y2': 200.0,
        'label': SemanticLabel.Regular
    }
]

logger.info(f"Definidos {len(elements)} elementos para ordenar.")

# 3. Definir los límites de la página.
#    Esto es idéntico a Rust: una tupla (x_min, y_min, x_max, y_max).
page_bounds = (0.0, 0.0, 800.0, 1200.0)
logger.info(f"Límites de la página: {page_bounds}")

# 4. Calcular el orden de lectura usando la configuración por defecto.
#    No necesitamos crear una instancia de XYCutPlusPlus. La función `compute_order`
#    se encarga de todo. Al no pasarle el argumento `config`, el código Rust
#    usará `XYCutConfig::default()` internamente.
logger.info("\nCalculando orden con la configuración por defecto...")
ordered_ids_default = compute_order(elements, page_bounds)

# 5. Imprimir los resultados.
#    El resultado es una lista de Python con los IDs ordenados.
logger.info("Orden de lectura de los IDs (default config):")
for element_id in ordered_ids_default:
    logger.info(f"  -> Leer elemento {element_id}")

# 6. (Opcional) Calcular el orden con una configuración personalizada.
#    Creamos una instancia de la clase `XYCutConfig` que expusimos desde Rust.
logger.info("\nCalculando orden con una configuración personalizada...")
custom_config = XYCutConfig(
    min_cut_threshold=5.0,        # Umbral de corte más pequeño
    histogram_resolution_scale=1.0, # Mayor resolución del histograma
    same_row_tolerance=8.0
)

ordered_ids_custom = compute_order(elements, page_bounds, config=custom_config, backend="datalab")

logger.info("Orden de lectura de los IDs (custom config):")
for element_id in ordered_ids_custom:
    logger.info(f"  -> Leer elemento {element_id}")

