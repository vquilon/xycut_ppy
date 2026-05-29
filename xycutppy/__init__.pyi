from __future__ import annotations

import typing
from enum import Enum
from typing import List, Dict, Any, Tuple, Optional, Final


# Stub para la clase XYCutConfig
# La definimos como una clase con su constructor y atributos públicos.
# Los tipos `float` se corresponden con los `f32` de Rust.
class XYCutConfig:
    min_cut_threshold: float
    histogram_resolution_scale: float
    same_row_tolerance: float

    # El constructor `__init__` se define con los mismos argumentos y valores
    # por defecto que la macro `#[new]` en el código de Rust.
    def __init__(
        self,
        min_cut_threshold: float = 15.0,
        histogram_resolution_scale: float = 0.5,
        same_row_tolerance: float = 10.0,
    ) -> None: ...


# Stub para la enumeración SemanticLabel
# La forma más precisa de representar una enumeración de Rust en Python
# es usando `enum.Enum` del a librería estándar.
# Usamos `Final` para indicar que los miembros no pueden ser reasignados.
class SemanticLabel(Enum):
    CrossLayout: Final[SemanticLabel]
    HorizontalTitle: Final[SemanticLabel]
    VerticalTitle: Final[SemanticLabel]
    Vision: Final[SemanticLabel]
    Regular: Final[SemanticLabel]

@typing.final
class Element:
    id: str
    x1: float
    y1: float
    x2: float
    y2: float
    label: SemanticLabel

    # El constructor `__init__` se define con los mismos argumentos y valores
    # por defecto que la macro `#[new]` en el código de Rust.
    def __init__(
        self,
        id: str,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        label: SemanticLabel,
    ) -> None: ...


# 3. Stub para la función principal compute_order
# Esta firma debe coincidir con la de la función "fachada" en tu __init__.py.
def compute_order(
    elements: List[Dict[str, Any]],
    page_bounds: Tuple[float, float, float, float],
    config: Optional[XYCutConfig] = None,
) -> List[int]: ...

