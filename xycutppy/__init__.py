"""
XYCut++ Python Wrapper
======================

Un paquete de Python que proporciona bindings para el crate de Rust `xycut-plus-plus`,
permitiendo calcular el orden de lectura de elementos en una página.
"""
import logging

from xycutppy._xycutpp_core import (
    compute_order as compute_order_core,
    SemanticLabel,
    XYCutConfig,
    Element,
)
from typing import List, Dict, Any, Tuple, Optional

logging.getLogger("xycut_plus_plus").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

def configure_logging(log_level: int):
    logging.getLogger("xycut_plus_plus").setLevel(log_level)
    logger.setLevel(log_level)

# ------------------------------------

# Re-exportamos las clases principales para que el usuario pueda acceder a ellas desde el nivel superior
# del paquete, por ejemplo: from xycutppy import SemanticLabel
__all__ = ["compute_order", "SemanticLabel", "XYCutConfig", "Element", "configure_logging"]


def compute_order(
        elements: List[Dict[str, Any]],
        page_bounds: Tuple[float, float, float, float],
        config: Optional[XYCutConfig] = None,
) -> List[int]:
    """
    Calcula el orden de lectura para una lista de elementos en una página.

    Args:
        elements:
            Una lista de elementos. Cada elemento debe ser un diccionario
            con las claves 'id' (int), 'x1', 'y1', 'x2', 'y2' (float),
            y 'label' (SemanticLabel).
        page_bounds:
            Una tupla (x_min, y_min, x_max, y_max) que define los límites
            de la página.
        config:
            (Opcional) Un objeto XYCutConfig con los parámetros del algoritmo.
            Si no se proporciona, se usarán los valores por defecto.

    Returns:
        Una lista de los 'id' de los elementos en el orden de lectura calculado.

    Example:
        >>> from xycutppy import compute_order, SemanticLabel, XYCutConfig
        >>> elements = [
        ...     {'id': 0, 'x1': 10.0, 'y1': 10.0, 'x2': 200.0, 'y2': 30.0, 'label': SemanticLabel.HorizontalTitle},
        ...     {'id': 1, 'x1': 10.0, 'y1': 50.0, 'x2': 400.0, 'y2': 100.0, 'label': SemanticLabel.Regular},
        ... ]
        >>> page_bounds = (0.0, 0.0, 800.0, 1200.0)
        >>> ordered_ids = compute_order(elements, page_bounds)
        >>> print(ordered_ids)
        [0, 1]
    """
    # Aquí podrías añadir validaciones en Python antes de llamar al núcleo de Rust
    if not isinstance(elements, list):
        raise TypeError("'elements' debe ser una lista de diccionarios.")
    if not elements:
        return []

    # Convertimos la lista de diccionarios en una lista de objetos `Element`.
    # Usamos una "list comprehension" para hacerlo de forma concisa y eficiente.
    # El `**d` desempaqueta el diccionario `d` en argumentos para el constructor de `Element`.
    # Por ejemplo, `Element(**{'id': 0, 'x1': 10.0, ...})` se convierte en
    # `Element(id=0, x1=10.0, ...)`.
    try:
        element_objects = [Element(**d) for d in elements]
    except TypeError as e:
        # Esto dará un error claro si a un diccionario le falta una clave
        # o tiene una clave extra que `Element.__init__` no espera.
        raise ValueError(
            "Un diccionario en la lista 'elements' tiene claves incorrectas."
        ) from e

    # Llamamos a la función de Rust, que hemos renombrado a compute_order_core
    return compute_order_core(element_objects, page_bounds, config)


