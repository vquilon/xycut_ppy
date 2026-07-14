from typing import List, Dict, Tuple, Any, Optional, cast

import lightgbm as lgb
import numpy as np

from ..base import ReadingOrderAlgorithmBase


class HuriDocsCandidateSelectorAdapter(ReadingOrderAlgorithmBase):
    """
    Adaptador bbox-only inspirado en el selector de candidatos de HURIDOCS.
    """
    __KIND = "huridocs/pdf-reading-order"

    def __init__(self, name: str):
        super().__init__(name)
        self._model: Optional[lgb.Booster] = None

    @staticmethod
    def _bbox_features(current_bbox: List[float], candidate_bbox: List[float]) -> List[float]:
        return [
            current_bbox[1],
            current_bbox[0],
            current_bbox[2],
            current_bbox[3],
            candidate_bbox[1],
            candidate_bbox[0],
            candidate_bbox[2],
            candidate_bbox[3],
            current_bbox[3] - candidate_bbox[1],
        ]

    def _ensure_model(self) -> lgb.Booster:
        if self._model is not None:
            return self._model

        import lightgbm as lgb
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download(
            repo_id="HURIDOCS/pdf-reading-order",
            filename="candidate_selector_model.model"
        )
        self._model = lgb.Booster(model_file=model_path)
        if self._model is None:
            raise RuntimeError()

        return self._model

    def prepare_inputs(
            self,
            elements: List[Dict[str, Any]],
            page_bounds: Tuple[float, float, float, float],
            **kwargs: Any,
    ) -> Tuple[tuple, dict]:
        _ = page_bounds
        _ = kwargs
        if not elements:
            return tuple(), {}

        model = self._ensure_model()
        assert model is not None

        bboxes = [
            {
                "id": elem["id"],
                "bbox": [float(elem["x1"]), float(elem["y1"]), float(elem["x2"]), float(elem["y2"])],
            }
            for elem in elements
        ]

        return (bboxes,), {}

    def main(self, inputs: Tuple[tuple, dict]):
        model = self._ensure_model()
        assert model is not None

        tuple_inputs, prepared_inputs = inputs
        bboxes = tuple_inputs[0]

        remaining = bboxes.copy()

        current_bbox = [0.0, 0.0, 0.0, 0.0]
        ordered_ids: List[int] = []

        while remaining:
            feature_matrix = np.array(
                [self._bbox_features(current_bbox, candidate["bbox"]) for candidate in remaining], dtype=float
            )
            prediction_scores = model.predict(feature_matrix)
            if np.ndim(prediction_scores) == 2 and prediction_scores.shape[1] > 1:
                positive_scores = prediction_scores[:, 1]
            else:
                positive_scores = np.asarray(prediction_scores).reshape(-1)
            best_idx = int(np.argmax(positive_scores))
            selected = remaining.pop(best_idx)
            ordered_ids.append(cast(int, selected["id"]))
            current_bbox = selected["bbox"]

        return ordered_ids
