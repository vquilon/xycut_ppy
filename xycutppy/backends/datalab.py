from typing import List, Optional, Tuple
from xycutppy._xycutpp_core import Element, XYCutConfig, compute_order_datalab

def is_available() -> bool:
    """Retorna True si este backend puede ser ejecutado."""
    return True

def compute_order(
    elements: List[Element],
    page_bounds: Tuple[float, float, float, float],
    config: Optional[XYCutConfig] = None,
) -> List[int]:
    return compute_order_datalab(elements, page_bounds, config)