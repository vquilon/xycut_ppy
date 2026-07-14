from typing import List, Dict, Tuple, Any
from typing import Optional

# Importamos las herramientas de nuestro paquete unificado en Rust/Python
from xycutppy import SemanticLabel, compute_order, XYCutConfig

from ..base import ReadingOrderAlgorithmBase

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


class XYCutPPYReadingOrderAdapter(ReadingOrderAlgorithmBase):
    __KIND = "xycutppy"

    def __init__(self, name: str, config: Optional[XYCutConfig] = None, backend: Optional[str] = "datalab"):
        super().__init__(name)
        self.config = config
        self.backend = backend

    def prepare_inputs(
            self,
            elements: List[Dict[str, Any]],
            page_bounds: Tuple[float, float, float, float],
            **kwargs: Any
    ):
        for elem in elements:
            if elem['label'] in SEMANTIC_MAPPING:
                elem['label'] = SEMANTIC_MAPPING[elem['label']]

            if elem['label'] not in SEMANTIC_MAPPING.values():
                elem['label'] = SemanticLabel.Regular

        return (elements, page_bounds), {'config': self.config, 'backend': self.backend}

    def main(self, inputs: Tuple[tuple, dict]):
        (elements, page_bounds), kwargs = inputs

        return compute_order(elements, page_bounds, **kwargs)
