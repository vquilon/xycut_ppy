from typing import List, Dict, Tuple, Any, Literal

from transformers import PreTrainedModel

from ..base import ReadingOrderAlgorithmBase


class HantianLayoutReaderAdapter(ReadingOrderAlgorithmBase):
    """
    Adaptador del modelo Hantian/LayoutReader al contrato ReadingOrderWrapper.
    """
    __KIND = "hantian/layoutreader"

    def __init__(self, name: str, model_id: str = "hantian/layoutreader", device: Literal["cpu", "cuda", "mps", "auto"] = "auto"):
        super().__init__(name)
        self.model_id = model_id
        self._device: str = device
        self._model: PreTrainedModel | None = None

    @staticmethod
    def _boxes_to_inputs(boxes: List[List[int]]) -> Dict[str, Any]:
        import torch

        max_len = 510
        cls_token_id = 0
        unk_token_id = 3
        eos_token_id = 2

        sliced = boxes[:max_len]
        bbox = [[0, 0, 0, 0]] + sliced + [[0, 0, 0, 0]]
        input_ids = [cls_token_id] + [unk_token_id] * len(sliced) + [eos_token_id]
        attention_mask = [1] + [1] * len(sliced) + [1]
        return {
            "bbox": torch.tensor([bbox]),
            "attention_mask": torch.tensor([attention_mask]),
            "input_ids": torch.tensor([input_ids]),
        }

    @staticmethod
    def _parse_logits(logits: Any, length: int) -> List[int]:
        logits = logits[1: length + 1, :length]
        orders = logits.argsort(descending=False).tolist()
        predicted = [order_candidates.pop() for order_candidates in orders]
        while True:
            order_to_indexes: Dict[int, List[int]] = {}
            for idx, order in enumerate(predicted):
                order_to_indexes.setdefault(order, []).append(idx)
            duplicates = {order: indexes for order, indexes in order_to_indexes.items() if len(indexes) > 1}
            if not duplicates:
                break
            for order, indexes in duplicates.items():
                index_to_logit = {idx: logits[idx, order] for idx in indexes}
                sorted_conflicts = sorted(index_to_logit.items(), key=lambda item: item[1], reverse=True)
                for idx, _ in sorted_conflicts[1:]:
                    predicted[idx] = orders[idx].pop()
        return predicted

    @staticmethod
    def _prepare_model_inputs(inputs: Dict[str, Any], model: PreTrainedModel) -> Dict[str, Any]:
        import torch

        prepared_inputs: Dict[str, Any] = {}
        for key, value in inputs.items():
            value = value.to(model.device)
            if torch.is_floating_point(value):
                value = value.to(model.dtype)
            prepared_inputs[key] = value
        return prepared_inputs

    def _infer_reading_order_with_layoutreader(
            self, model: PreTrainedModel, bboxes: List[List[int]]
    ) -> List[int]:
        if not bboxes:
            return []

        normalized_bboxes = [[int(coord) for coord in bbox] for bbox in bboxes]
        inputs = self._boxes_to_inputs(normalized_bboxes)
        prepared_inputs = self._prepare_model_inputs(inputs, model)
        logits = model(**prepared_inputs).logits.cpu().squeeze(0)
        return self._parse_logits(logits, len(normalized_bboxes))

    @staticmethod
    def _normalize_layoutreader_bbox(
            bbox: Tuple[float, float, float, float],
            page_bounds: Tuple[float, float, float, float],
    ) -> List[int]:
        page_x1, page_y1, page_x2, page_y2 = [float(value) for value in page_bounds]
        page_width = page_x2 - page_x1
        page_height = page_y2 - page_y1

        x1, y1, x2, y2 = bbox
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))

        if page_width <= 0 or page_height <= 0:
            return [
                max(0, min(1000, int(round(x1)))),
                max(0, min(1000, int(round(y1)))),
                max(0, min(1000, int(round(x2)))),
                max(0, min(1000, int(round(y2)))),
            ]

        x1 = max(page_x1, min(page_x2, x1))
        x2 = max(page_x1, min(page_x2, x2))
        y1 = max(page_y1, min(page_y2, y1))
        y2 = max(page_y1, min(page_y2, y2))

        norm_x1 = max(0, min(1000, int(round(((x1 - page_x1) / page_width) * 1000))))
        norm_x2 = max(0, min(1000, int(round(((x2 - page_x1) / page_width) * 1000))))
        norm_y1 = max(0, min(1000, int(round(((y1 - page_y1) / page_height) * 1000))))
        norm_y2 = max(0, min(1000, int(round(((y2 - page_y1) / page_height) * 1000))))
        return [norm_x1, norm_y1, norm_x2, norm_y2]

    def _ensure_model(self) -> Tuple[PreTrainedModel, str]:
        if self._model is not None and self._device is not None:
            return self._model, self._device

        import torch
        from transformers import LayoutLMv3ForTokenClassification

        device = self._device
        if device == "auto":
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"

        self._device = device
        self._model = (
            LayoutLMv3ForTokenClassification.from_pretrained(self.model_id)
            .bfloat16()
            .to(device)
            .eval()
        )
        if self._model is None or self._device is None:
            raise RuntimeError()

        return self._model, self._device

    def prepare_inputs(
            self,
            elements: List[Dict[str, Any]],
            page_bounds: Tuple[float, float, float, float],
            **kwargs: Any,
    ) -> Tuple[tuple, dict]:
        if not elements:
            raise RuntimeError()

        bboxes = []
        for elem in elements:
            bbox = (
                float(elem["x1"]),
                float(elem["y1"]),
                float(elem["x2"]),
                float(elem["y2"]),
            )
            bboxes.append(self._normalize_layoutreader_bbox(bbox, page_bounds))
        return (bboxes,), {}

    def main(self, inputs: Tuple[tuple, dict]):
        model, device = self._ensure_model()
        assert model is not None
        assert device is not None

        tuple_args, _ = inputs
        bboxes = tuple_args[0]
        return self._infer_reading_order_with_layoutreader(model, bboxes)

    def format_output(
            self,
            elements: List[Dict[str, Any]],
            page_bounds: Tuple[float, float, float, float],
            results: Any,
    ) -> List[int]:
        order_indexes = results
        return [elements[idx]["id"] for idx in order_indexes if 0 <= idx < len(elements)]

