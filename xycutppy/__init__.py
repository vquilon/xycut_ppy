"""
XYCut++ Python Wrapper
======================

Un paquete de Python que proporciona bindings para el crate de Rust `xycut-plus-plus`,
permitiendo calcular el orden de lectura de elementos en una página.
Ademas permite utilizar una implementación nueva por parte de `datalab` que es de licencia Apache 2.0.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

# Exportamos las clases principales del core de Rust
from xycutppy._xycutpp_core import (
    Element,
    SemanticLabel,
    XYCutConfig,
    reset_log_cache,
)

# Importamos nuestros submódulos de backend
from xycutppy.backends import datalab, paper

# --- REGISTRO DINÁMICO DE BACKENDS ---
_BACKENDS = {
    "datalab": datalab,
    "paper": paper,
}

# Filtramos los backends que realmente se compilaron y están disponibles
_AVAILABLE_BACKENDS = {
    name: mod for name, mod in _BACKENDS.items() if mod.is_available()
}

_DEFAULT_BACKEND = os.environ.get("XYCUTPPY_BACKEND", "paper").strip().lower()

# Fallback de seguridad: si piden 'paper' pero instalaron la versión opensource (datalab),
# o si el valor de entorno es inválido, forzamos al primer backend disponible.
if _DEFAULT_BACKEND not in _AVAILABLE_BACKENDS:
    _DEFAULT_BACKEND = next(iter(_AVAILABLE_BACKENDS.keys()), "datalab")

logging.getLogger("xycutppy").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def configure_logging(log_level: int, backend: Optional[str] = None) -> None:
    """Configura el nivel de logging para xycutppy y sus backends de Rust.

    Establece el nivel en toda la jerarquía ``xycutppy.*`` y resetea la caché
    interna de pyo3_log para que los cambios sean efectivos inmediatamente en
    el código Rust, independientemente de cuándo se llame esta función.

    La jerarquía de loggers es::

        xycutppy              ← raíz (entrada/salida de cada pyfunction)
        xycutppy.datalab      ← algoritmo datalab (Apache 2.0)
        xycutppy.paper        ← algoritmo paper (GPL3, si está instalado)

    Args:
        log_level: Nivel estándar de Python (ej. ``logging.DEBUG``, ``logging.INFO``).
        backend:   Si se especifica (``"datalab"`` o ``"paper"``), solo configura
                   ese sub-logger. Si es ``None``, configura toda la jerarquía.

    Example::

        import logging, xycutppy
        xycutppy.configure_logging(logging.DEBUG)           # todos los backends
        xycutppy.configure_logging(logging.DEBUG, "paper")  # solo paper
    """
    if backend is None:
        logging.getLogger("xycutppy").setLevel(log_level)
        logging.getLogger("xycutppy.datalab").setLevel(log_level)
        logging.getLogger("xycutppy.paper").setLevel(log_level)
    else:
        logging.getLogger(f"xycutppy.{backend.strip().lower()}").setLevel(log_level)

    # Invalida la caché de niveles de pyo3_log para que Rust lea el nuevo nivel
    reset_log_cache()


def available_backends() -> List[str]:
    """Devuelve una lista con los nombres de los backends instalados y listos para usar."""
    return list(_AVAILABLE_BACKENDS.keys())


def set_backend(backend: str) -> None:
    """Configura el backend global por defecto."""
    global _DEFAULT_BACKEND
    backend_name = backend.strip().lower()

    if backend_name not in _AVAILABLE_BACKENDS:
        # Mensaje de error amigable si intentan usar la versión GPL3 en el paquete Apache
        if backend_name == "paper" and "paper" not in _AVAILABLE_BACKENDS:
            raise ValueError(
                "El backend 'paper' requiere la licencia GPLv3 y no está en esta instalación. "
                "Para usarlo, desinstala la base `pip uninstall xycutppy` e instala: `pip install xycutppy-paper`."
            )
        raise ValueError(f"Backend invalido: {backend}. Use uno de: {list(_AVAILABLE_BACKENDS.keys())}.")

    _DEFAULT_BACKEND = backend_name


def get_backend() -> str:
    """Obtiene el nombre del backend global configurado actualmente."""
    return _DEFAULT_BACKEND


def _to_element_objects(elements: List[Dict[str, Any]]) -> List[Element]:
    # Convertimos la lista de diccionarios en una lista de objetos `Element`.
    # Usamos una "list comprehension" para hacerlo de forma concisa y eficiente.
    # El `**d` desempaqueta el diccionario `d` en argumentos para el constructor de `Element`.
    # Por ejemplo, `Element(**{'id': '0', 'x1': 10.0, ...})` se convierte en
    # `Element(id='0', x1=10.0, ...)`.

    if not isinstance(elements, list):
        raise TypeError("'elements' debe ser una lista de diccionarios.")
    if not elements:
        return []
    try:
        return [Element(**d) for d in elements]
    except TypeError as e:
        # Esto dará un error claro si a un diccionario le falta una clave
        # o tiene una clave extra que `Element.__init__` no espera.
        raise ValueError("Un diccionario en la lista 'elements' tiene claves incorrectas.") from e


def _compute_order_paper(
    elements: List[Dict[str, Any]],
    page_bounds: Tuple[float, float, float, float],
    config: Optional[XYCutConfig] = None,
) -> List[int]:
    """Llama directamente a la implementación original del paper (Requiere versión GPLv3)."""
    if "paper" not in _AVAILABLE_BACKENDS:
        raise NotImplementedError(
            "El backend 'paper' requiere la licencia GPLv3. Instala el paquete principal para usarlo."
        )
    return paper.compute_order(_to_element_objects(elements), page_bounds, config)


def _compute_order_datalab(
    elements: List[Dict[str, Any]],
    page_bounds: Tuple[float, float, float, float],
    config: Optional[XYCutConfig] = None,
) -> List[int]:
    """Llama directamente a la implementación geométrica de Datalab (Licencia Apache 2.0)."""
    if "datalab" not in _AVAILABLE_BACKENDS:
        raise NotImplementedError("El backend 'datalab' no está disponible en esta instalación.")
    return datalab.compute_order(_to_element_objects(elements), page_bounds, config)


def compute_order(
    elements: List[Dict[str, Any]],
    page_bounds: Tuple[float, float, float, float],
    config: Optional[XYCutConfig] = None,
    backend: Optional[str] = None,
) -> List[int]:
    """
    Calcula el orden de lectura para una lista de elementos en una página.

    Args:
        elements:
            Una lista de elementos. Cada elemento debe ser un diccionario
            con las claves 'id' (str), 'x1', 'y1', 'x2', 'y2' (float),
            y 'label' (SemanticLabel).
        page_bounds:
            Una tupla (x_min, y_min, x_max, y_max) que define los límites
            de la página.
        config:
            (Opcional) Un objeto XYCutConfig con los parámetros del algoritmo.
            Si no se proporciona, se usarán los valores por defecto.
        backend:
            (Opcional) El motor a utilizar ('datalab' o 'paper'). Si no se
            proporciona, se usará el backend por defecto configurado.

    Returns:
        Una lista de los 'id' de los elementos en el orden de lectura calculado.

    Example:
        >>> from xycutppy import compute_order, SemanticLabel, XYCutConfig
        >>> elements = [
        ...     {'id': '0', 'x1': 10.0, 'y1': 10.0, 'x2': 200.0, 'y2': 30.0, 'label': SemanticLabel.HorizontalTitle},
        ...     {'id': '1', 'x1': 10.0, 'y1': 50.0, 'x2': 400.0, 'y2': 100.0, 'label': SemanticLabel.Regular},
        ... ]
        >>> page_bounds = (0.0, 0.0, 800.0, 1200.0)
        >>> ordered_ids = compute_order(elements, page_bounds)
        >>> print(ordered_ids)
        ['0', '1']
    """

    selected_backend = (_DEFAULT_BACKEND if backend is None else backend).strip().lower()

    if selected_backend not in _AVAILABLE_BACKENDS:
        if selected_backend == "paper":
            raise ValueError(
                "El backend 'paper' requiere la licencia GPLv3 y no está instalado por defecto. "
                "Para usarlo, desinstala la base `pip uninstall xycutppy` e instala: `pip install xycutppy-paper`."
            )
        raise ValueError(f"Backend invalido: {selected_backend}. Use uno de: {list(_AVAILABLE_BACKENDS.keys())}.")

    # Parseamos los diccionarios a objetos Rust de forma centralizada
    rust_elements = _to_element_objects(elements)

    # Delegamos la ejecución al backend seleccionado
    backend_module = _BACKENDS[selected_backend]
    return backend_module.compute_order(rust_elements, page_bounds, config)

# ------------------------------------

# Re-exportamos las clases principales para que el usuario pueda acceder a ellas desde el nivel superior
# del paquete, por ejemplo: from xycutppy import SemanticLabel
__all__ = [
    "compute_order",
    "set_backend",
    "get_backend",
    "available_backends",
    "configure_logging",
    "reset_log_cache",
    "SemanticLabel",
    "XYCutConfig",
    "Element",
]