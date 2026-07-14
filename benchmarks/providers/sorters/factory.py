from typing import List, Dict, Tuple, Any

from .base import ReadingOrderAlgorithmBase, ReadingOrderWrapper


class ReadingOrderSorterFactory:
    """
    Factory para registrar, configurar y adaptar diferentes algoritmos de Reading Order.
    """

    def __init__(self, sort_algorithm: ReadingOrderAlgorithmBase):
        self.name = sort_algorithm.name
        self.kind = sort_algorithm.kind
        self.algorithm = self.create_sorter(sort_algorithm)

    @staticmethod
    def create_sorter(
        sort_algorithm: ReadingOrderAlgorithmBase,
    ) -> ReadingOrderWrapper:
        """
        Inyecta hiperparámetros específicos (como umbrales para XY-Cut) envolviendo la función.
        """
        def sorter_wrapper(elements: List[Dict[str, Any]], page_bounds: Tuple[float, float, float, float], **kwargs) -> List[int]:
            prepared_inputs = sort_algorithm.prepare_inputs(elements, page_bounds, **kwargs)
            results = sort_algorithm.main(prepared_inputs)
            return sort_algorithm.format_output(elements, page_bounds, results)

        return sorter_wrapper