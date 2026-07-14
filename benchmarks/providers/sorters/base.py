from abc import abstractmethod, ABC
from typing import List, Dict, Tuple, Any, Protocol


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

YOLO_ELEMENTS_MAPPING = {
    'Title',
    'Section-header',
    'Page-header',
    'Page-footer',
    'Picture',
    'Table',
    'Formula',
    'Text',
    'List-item',
    'Caption',
    'Footnote',
}


class ReadingOrderWrapper(Protocol):
    def __call__(self, elements: List[Dict[str, Any]], page_bounds: Tuple[float, float, float, float], **kwargs: Any) -> List[int]:
        ...

class ReadingOrderAlgorithmBase(ABC):
    __KIND: str = None

    def __init__(self, name: str):
        self.name = name
        self.kind = self.__KIND

    def prepare_inputs(
            self,
            elements: List[Dict[str, Any]],
            page_bounds: Tuple[float, float, float, float],
            **kwargs: Any
    ) -> Tuple[tuple, dict]:
        return (elements, page_bounds), kwargs

    @abstractmethod
    def main(self, inputs: Tuple[tuple, dict]):
        raise NotImplemented

    def format_output(
            self,
            elements: List[Dict[str, Any]],
            page_bounds: Tuple[float, float, float, float],
            results: Any,
    ) -> List[int]:
        return results
