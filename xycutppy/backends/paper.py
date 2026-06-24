from typing import List, Optional, Tuple
from xycutppy._xycutpp_core import Element, XYCutConfig

# Intentamos importar la función C/Rust. Si falla, el backend no está disponible.
try:
    from xycutppy._xycutpp_core import compute_order_paper
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

def is_available() -> bool:
    """Retorna True si el código GPLv3 fue compilado y está disponible."""
    return _AVAILABLE

def compute_order(
    elements: List[Element],
    page_bounds: Tuple[float, float, float, float],
    config: Optional[XYCutConfig] = None,
) -> List[int]:
    if not _AVAILABLE:
        raise NotImplementedError(
            "El backend 'paper' utiliza el algoritmo original y requiere la licencia GPLv3. "
            "Para usarlo, desinstala este paquete e instala la versión completa:\n"
            "  pip uninstall xycutppy-datalab\n"
            "  pip install xycutppy"
        )
    return compute_order_paper(elements, page_bounds, config)