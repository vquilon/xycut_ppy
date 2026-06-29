import json
import os
import shutil
import subprocess
import tempfile
import time
import traceback
from abc import abstractmethod, ABC
from pathlib import Path
from typing import List, Dict, Tuple, Any, Callable, Protocol, cast, Literal, Optional

import lightgbm as lgb
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from scipy.stats import kendalltau  # NUEVA DEPENDENCIA: pip install scipy
from tqdm import tqdm
from transformers import PreTrainedModel
# Importamos las herramientas de nuestro paquete unificado en Rust/Python
from xycutppy import SemanticLabel, compute_order, XYCutConfig

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


# ----------------------------------------------------------------------
# Función de Renderizado Visual (Generación del PNG)
# ----------------------------------------------------------------------
def render_page_layout(
        filename: str,
        elements: List[Dict[str, Any]],
        ordered_ids: List[int],
        page_bounds: Tuple[float, float, float, float]
):
    """
    Dibuja una representación visual del layout de la página en un PNG,
    pintando cada caja de su color semántico y añadiendo el índice del orden
    de lectura en el centro de la caja.
    """
    _, _, width, height = page_bounds
    # Creamos un lienzo en blanco (con un extra de margen para ver los bordes)
    image = Image.new("RGB", (int(width) + 20, int(height) + 20), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    # Intentar cargar una fuente por defecto del sistema, si falla usamos la básica integrada
    try:
        font_text = ImageFont.load_default()
    except Exception:
        font_text = None

    # Mapeamos los IDs a su índice de lectura (1º, 2º, 3º...) para pintarlo con claridad
    order_position = {element_id: index + 1 for index, element_id in enumerate(ordered_ids)}

    for elem in elements:
        # Desplazamos +10 píxeles para que no se pegue al borde físico de la imagen
        x1, y1, x2, y2 = elem['x1'] + 10, elem['y1'] + 10, elem['x2'] + 10, elem['y2'] + 10
        yolo_label = elem['yolo_label']
        elem_id = elem['id']

        # Obtener color según el mapa
        color = COLOR_MAP.get(yolo_label, (200, 200, 200))

        # 1. Dibujar el rectángulo relleno semi-transparente (haciendo un patrón o borde grueso)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        # Pintar un fondo interno un poco más claro simulando opacidad
        draw.rectangle([x1 + 2, y1 + 2, x2 - 2, y2 - 2], fill=(*color, 50))

        # 2. Dibujar la etiqueta pequeña del tipo de elemento en la esquina superior izquierda
        tag_str = f"{yolo_label} (ID:{elem_id})"
        draw.text((x1 + 4, y1 + 4), tag_str, fill=(50, 50, 50), font=font_text)

        # 3. Dibujar el número del ORDEN DE LECTURA grande y destacado en el centro de la caja
        if elem_id in order_position:
            idx_lectura = str(order_position[elem_id])
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            # Dibujar un pequeño círculo blanco de fondo para que el número resalte
            draw.ellipse([cx - 12, cy - 12, cx + 12, cy + 12], fill=(255, 255, 255), outline=(0, 0, 0))
            # Escribir el número de orden en negro
            draw.text((cx - 4, cy - 6), idx_lectura, fill=(0, 0, 0), font=font_text)

    # Crear directorio si no existe y guardar
    os.makedirs("output_examples", exist_ok=True)
    full_path = os.path.join("output_examples", filename)
    image.save(full_path)
    print(f"  [Visual] Imagen guardada con éxito en: {full_path}")


# =====================================================================
# 2. SECCIÓN FACTORY PARA LAS FUNCIONES DE ORDENACIÓN (CALLABLE TEMPLATE)
# =====================================================================
ReadingOrderProvider = Callable[[List[Dict[str, Any]], Tuple[float, float, float, float], ...], List[int]]
ReadingOrderAlgorithms = Literal["xycutppy", "hantian", "huridocs"]

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


class XYCutPPY(ReadingOrderAlgorithmBase):
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


# =====================================================================
# 3. MOTOR DEL BENCHMARK MULTI-DATASET
# =====================================================================
class ReadingOrderBenchmark:
    """
    Framework principal encargado de descargar, normalizar y evaluar
    los datasets de Reading Order (ReadingBank y OmniDocBench).
    """

    def __init__(self, cache_dir: str = "./data_cache"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.smoother = SmoothingFunction().method1

    def _download_omnidocbench_json(self) -> str:
        """
        Descarga el JSON de OmniDocBench directamente desde Hugging Face si no existe en caché.
        """
        target_path = os.path.join(self.cache_dir, "OmniDocBench.json")
        if not os.path.exists(target_path):
            url = "https://huggingface.co/datasets/opendatalab/OmniDocBench/resolve/main/OmniDocBench.json"
            # print(f"📥 Descargando OmniDocBench.json desde Hugging Face ({url})...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(target_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            # print("✅ Descarga completada.")
        return target_path

    def _parse_polygon_to_bbox(self, poly: List[float]) -> Tuple[float, float, float, float]:
        """
        Convierte una lista de coordenadas [x1, y1, x2, y2, ...] en una caja delimitadora (x1, y1, x2, y2).
        """
        xs = poly[0::2]
        ys = poly[1::2]
        return min(xs), min(ys), max(xs), max(ys)

    def _normalize_reading_bank(self, raw_sample: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[int], Tuple[float, float, float, float]]:
        """
        Transforma el formato de ReadingBank al estándar unificado del framework.
        """
        normalized_elements = []
        # 'src_layout' contiene las bounding boxes originales [x1, y1, x2, y2]
        src_layout = raw_sample.get("src_layout", [])

        # Generar elementos estándar
        for idx, bbox in enumerate(src_layout):
            normalized_elements.append({
                "id": idx,
                "x1": float(bbox[0]),
                "y1": float(bbox[1]),
                "x2": float(bbox[2]),
                "y2": float(bbox[3]),
                "label": 'Text'
            })

        # El Ground Truth exacto nos lo da 'tgt_index', el cual dice la posición real que debe ocupar cada caja
        tgt_index = raw_sample.get("tgt_index", [])
        ground_truth_order = [idx for idx in tgt_index]

        # Calcular los límites de la página basados en las coordenadas máximas detectadas
        if normalized_elements:
            x_max = max([e["x2"] for e in normalized_elements])
            y_max = max([e["y2"] for e in normalized_elements])
            page_bounds = (0.0, 0.0, x_max, y_max)
        else:
            page_bounds = (0.0, 0.0, 1000.0, 1000.0)

        return normalized_elements, ground_truth_order, page_bounds

    def _format_omni_doc_bench_to_yolo_labels(self, omnidoc_bench_label: str) -> str:
        OMNIDOC_BENCH_MAPPING_CATEGORY = {
            'abandon': 'Text',
            'algorithm_mask': 'Picture',
            'chart_mask': 'Picture',
            'code_txt': 'Text',
            'code_txt_caption': 'Caption',
            'equation_caption': 'Caption',
            'equation_explanation': 'Formula',
            'equation_isolated': 'Formula',
            'figure': 'Picture',
            'figure_caption': 'Caption',
            'figure_footnote': 'Footnote',
            'footer': 'Footnote',
            'header': 'Section-header',
            'organic_chemical_formula_mask': 'Picture',
            'page_footnote': 'Page-footer',
            'page_number': 'Page-footer',
            'reference': 'Text',
            'table': 'Table',
            'table_caption': 'Caption',
            'table_footnote': 'Footnote',
            'table_mask': 'Text',
            'text_block': 'Text',
            'title': 'Title',
            'unknown_mask': 'Text'
        }

        return OMNIDOC_BENCH_MAPPING_CATEGORY.get(omnidoc_bench_label, 'Text')

    @staticmethod
    def _normalize_parsebench_key(value: str) -> str:
        return (
            value.strip().lower()
            .replace(" ", "_")
            .replace("\u2010", "-")
            .replace("\u2011", "-")
            .replace("\u2012", "-")
            .replace("\u2013", "-")
            .replace("\u2014", "-")
            .replace("\u2212", "-")
        )

    @staticmethod
    def _parse_parsebench_bbox(raw_bbox: Any) -> Tuple[float, float, float, float] | None:
        if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
            return None
        try:
            x0, y0, width, height = [float(v) for v in raw_bbox]
        except (TypeError, ValueError):
            return None

        if width <= 0 or height <= 0:
            return None

        x1 = x0
        y1 = y0
        x2 = x0 + width
        y2 = y0 + height
        return x1, y1, x2, y2

    def _format_parse_bench_to_yolo_labels(
            self,
            source_label: str | None = None,
            canonical_class: str | None = None,
            content_type: str | None = None,
            attributes: Dict[str, Any] | None = None,
    ) -> str:
        parse_bench_source_mapping = {
            "paragraph_title": "Title",
            "text": "Text",
            "key-value-region": "Text",
            "image": "Picture",
            "caption": "Caption",
            "header": "Section-header",
            "page_footer": "Page-footer",
            "form": "Text",
            "footer": "Footnote",
            "page-footer": "Page-footer",
            "page-header": "Page-header",
            "section-header": "Section-header",
            "chart": "Picture",
            "table": "Table",
            "formula": "Formula",
            "footnote": "Footnote",
            "picture": "Picture",
            "title": "Title",
            "list_item": "List-item",
        }

        parse_bench_canonical_mapping = {
            "section": "Section-header",
            "text": "Text",
            "picture": "Picture",
            "page-footer": "Page-footer",
            "page-header": "Page-header",
            "table": "Table",
        }

        parse_bench_content_type_mapping = {
            "text": "Text",
            "table": "Table",
        }
        yolo_labels = set(SEMANTIC_MAPPING.keys()) | set(COLOR_MAP.keys())
        yolo_label_by_normalized = {
            self._normalize_parsebench_key(label): label
            for label in yolo_labels
        }

        if source_label:
            normalized_source = self._normalize_parsebench_key(source_label)
            mapped_source = parse_bench_source_mapping.get(normalized_source)
            if mapped_source is not None:
                return mapped_source

        if canonical_class:
            normalized_canonical = self._normalize_parsebench_key(canonical_class)
            canonical_yolo = yolo_label_by_normalized.get(normalized_canonical)
            if canonical_yolo is not None:
                return canonical_yolo
            mapped_canonical = parse_bench_canonical_mapping.get(normalized_canonical)
            if mapped_canonical is not None:
                return mapped_canonical

        if attributes and isinstance(attributes, dict):
            title_level = attributes.get("title_level")
            if isinstance(title_level, str):
                normalized_level = self._normalize_parsebench_key(title_level)
                if normalized_level in {"title", "paragraph_title"}:
                    return "Title"
                if normalized_level in {"section-header", "header"}:
                    return "Section-header"

            text_role = attributes.get("text_role")
            if isinstance(text_role, str):
                normalized_role = self._normalize_parsebench_key(text_role)
                if normalized_role in {"key-value", "key_value", "key-value-region"}:
                    return "Form"
                if normalized_role in {"list_item", "list-item"}:
                    return "List-item"

        if content_type:
            normalized_content_type = self._normalize_parsebench_key(content_type)
            mapped_content = parse_bench_content_type_mapping.get(normalized_content_type)
            if mapped_content is not None:
                return mapped_content

        return "Text"


    def _normalize_omni_doc_bench(self, json_item: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[int], Tuple[float, float, float, float]]:
        """
        Transforma el formato JSON jerárquico de OmniDocBench al estándar unificado.
        """
        normalized_elements = []
        layout_dets = json_item.get("layout_dets", [])

        # Filtramos elementos válidos y extraemos su orden original (Ground Truth)
        valid_boxes = [box for box in layout_dets if not box.get("ignore", False) and isinstance(box.get('order'), int)]
        # Generamos el indetificador posicional no de ordenacion
        for idx, box in enumerate(valid_boxes):
            box["idx"] = idx

        # Ordenamos las cajas según el campo 'order' oficial para construir el Ground Truth de lectura
        valid_boxes_sorted = sorted(valid_boxes, key=lambda x: x["order"])
        ground_truth_order = [box['idx'] for box in valid_boxes_sorted]

        # Mapeamos a nuestro diccionario estandarizado
        for box in valid_boxes:
            poly = box.get("poly", [])
            x1, y1, x2, y2 = self._parse_polygon_to_bbox(poly)

            # Mapeo semántico simple
            cat = box.get("category_type", "text_block")
            label = self._format_omni_doc_bench_to_yolo_labels(cat)

            normalized_elements.append({
                "id": box["idx"],
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "label": label
            })

        if 'page_info' in json_item:
            page_info = json_item["page_info"]
            page_bounds = (0.0, 0.0, float(page_info.get('width', 1000.0)), float(page_info.get('height', 1000.0)))
        else:
            x_max = max([e["x2"] for e in normalized_elements])
            y_max = max([e["y2"] for e in normalized_elements])
            page_bounds = (0.0, 0.0, x_max, y_max)

        return normalized_elements, ground_truth_order, page_bounds

    def _normalize_parsebench(self, raw_rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[int], Tuple[float, float, float, float]]:
        """
        Transforma el formato de ParseBench al estándar unificado del framework.
        """
        valid_elements: List[Dict[str, Any]] = []

        for row in raw_rows:
            raw_rule = row.get("rule", {})
            if isinstance(raw_rule, str):
                try:
                    rule = json.loads(raw_rule)
                except json.JSONDecodeError:
                    continue
            else:
                rule = raw_rule

            if not isinstance(rule, dict):
                continue

            bbox = self._parse_parsebench_bbox(rule.get("bbox"))
            if bbox is None:
                continue

            ro_index_raw = rule.get("ro_index")
            try:
                ro_index = int(str(ro_index_raw))
            except (TypeError, ValueError):
                continue

            content_obj = rule.get("content", {})
            content_type = None
            if isinstance(content_obj, dict):
                raw_type = content_obj.get("type")
                if isinstance(raw_type, str):
                    content_type = raw_type

            attributes = rule.get("attributes")
            if not isinstance(attributes, dict):
                attributes = {}

            label = self._format_parse_bench_to_yolo_labels(
                source_label=rule.get("source_label"),
                canonical_class=rule.get("canonical_class"),
                content_type=content_type,
                attributes=attributes,
            )

            x1, y1, x2, y2 = bbox
            valid_elements.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "label": label,
                "ro_index": ro_index,
            })

        if not valid_elements:
            return [], [], (0.0, 0.0, 1.0, 1.0)

        normalized_elements = []
        for idx, element in enumerate(valid_elements):
            normalized_elements.append({
                "id": idx,
                "x1": element["x1"],
                "y1": element["y1"],
                "x2": element["x2"],
                "y2": element["y2"],
                "label": element["label"],
            })
            element["id"] = idx

        ground_truth_order = [
            cast(int, e["id"])
            for e in sorted(valid_elements, key=lambda item: (cast(int, item["ro_index"]), cast(float, item["y1"]), cast(float, item["x1"])))
        ]

        x_max = max([e["x2"] for e in normalized_elements], default=1.0)
        y_max = max([e["y2"] for e in normalized_elements], default=1.0)
        page_bounds = (0.0, 0.0, float(x_max), float(y_max))

        return normalized_elements, ground_truth_order, page_bounds



    def _calculate_sequence_bleu(self, reference: List[int], hypothesis: List[int]) -> float:
        """
        Calcula la calidad del orden tratando la secuencia de IDs de cajas como tokens.
        Aísla el orden de lectura puro sin interferencia del texto OCR.
        """
        if not reference or not hypothesis:
            return 0.0
        # Sentence BLEU acepta listas de tokens (en este caso, IDs de las bboxes)
        return sentence_bleu([reference], hypothesis, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=self.smoother)

    def _calculate_kendall_tau(self, reference: List[int], hypothesis: List[int]) -> float:
        """Kendall's Tau: Mide la correlación de rangos entre GT y Predicción"""
        if not reference or not hypothesis or len(reference) < 2:
            return 1.0 if reference == hypothesis else 0.0

        hyp_rank_map = {idx: rank for rank, idx in enumerate(hypothesis)}
        ref_ranks = list(range(len(reference)))
        # Si un ID falta en la predicción, se asume el peor rango (al final)
        hyp_ranks = [hyp_rank_map.get(idx, len(hypothesis)) for idx in reference]

        tau, _ = kendalltau(ref_ranks, hyp_ranks)
        return float(tau) if not np.isnan(tau) else 0.0

    def _calculate_ard(self, reference: List[int], hypothesis: List[int]) -> float:
        """Absolute Relative Difference (ARD): Media del error relativo de posición."""
        if not reference or not hypothesis:
            return 0.0

        hyp_rank_map = {idx: rank + 1 for rank, idx in enumerate(hypothesis)}  # 1-based rank
        ard_sum = 0.0

        for ref_rank_idx, item_id in enumerate(reference):
            actual_rank = ref_rank_idx + 1
            pred_rank = hyp_rank_map.get(item_id, len(hypothesis) + 1)
            ard_sum += abs(pred_rank - actual_rank) / actual_rank

        return float(ard_sum / len(reference))

    def evaluate_dataset(
            self,
            dataset_id: str,
            sorter_callable: ReadingOrderWrapper,
            max_samples: int | None = None,
            on_progress: Callable[[int, int], None] | None = None,
    ) -> Dict[str, Any]:
        """
        Ejecuta la batería de pruebas calculando promedios de palabras/cajas y la matriz de distribución.
        """
        # print(f"\n🔄 Evaluando Reading Order en: {dataset_id}")

        # Carga dinámica dependiendo del tipo de dataset
        samples_to_process: List[Tuple[List[Dict[str, Any]], List[int], Tuple[float, float, float, float]]] = []
        if dataset_id == "zilongwang/ReadingBank":
            try:
                hf_dataset = load_dataset(dataset_id, split="test")
                if max_samples is not None and max_samples < len(hf_dataset):
                    hf_dataset = hf_dataset.select(range(max_samples))
                for item in hf_dataset:
                    samples_to_process.append(self._normalize_reading_bank(item))
            except Exception as e:
                traceback.print_exc()
                return {}
        elif dataset_id == "opendatalab/OmniDocBench":
            json_file = self._download_omnidocbench_json()
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if max_samples and max_samples < len(data):
                data = data[:max_samples]

            for item in data:
                samples_to_process.append(self._normalize_omni_doc_bench(item))

        elif dataset_id == "llamaindex/ParseBench":
            hf_dataset = load_dataset(dataset_id, split="layout")
            grouped_rows: Dict[str, List[Dict[str, Any]]] = {}
            for item in hf_dataset:
                category = item.get("type")
                if not isinstance(category, str) or category.strip().lower() != "layout":
                    continue
                group_key = f"{item.get('pdf', 'unknown')}::{item.get('page', '0')}"
                grouped_rows.setdefault(group_key, []).append(item)

            grouped_samples = list(grouped_rows.values())
            if max_samples is not None and max_samples < len(grouped_samples):
                grouped_samples = grouped_samples[:max_samples]

            for page_rows in grouped_samples:
                samples_to_process.append(self._normalize_parsebench(page_rows))
        else:
            return {}

        total_dataset_samples = len(samples_to_process)
        if on_progress:
            on_progress(0, total_dataset_samples)

        elements_lengths = []
        bleu_scores = []
        tau_scores = []
        ard_scores = []
        b1, b2, b3, b4 = 0, 0, 0, 0
        total_time_inference = 0.0

        for sample_index, (elements, gt_order, page_bounds) in enumerate(samples_to_process, start=1):
            if not gt_order:
                if on_progress:
                    on_progress(sample_index, total_dataset_samples)
                continue

            # Medimos FPS / Tiempo de inferencia (Solo engloba el módulo de ordenación)
            start_t = time.perf_counter()
            predicted_order = sorter_callable(elements, page_bounds)
            elapsed_t = time.perf_counter() - start_t
            total_time_inference += elapsed_t

            # Cálculo de Métricas
            score_bleu = self._calculate_sequence_bleu(gt_order, predicted_order)
            score_tau = self._calculate_kendall_tau(gt_order, predicted_order)
            score_ard = self._calculate_ard(gt_order, predicted_order)

            bleu_scores.append(score_bleu)
            tau_scores.append(score_tau)
            ard_scores.append(score_ard)
            elements_lengths.append(len(elements))

            if 0.0 <= score_bleu <= 0.25:
                b1 += 1
            elif 0.25 < score_bleu <= 0.50:
                b2 += 1
            elif 0.50 < score_bleu <= 0.75:
                b3 += 1
            elif 0.75 < score_bleu <= 1.00:
                b4 += 1

            if on_progress:
                on_progress(sample_index, total_dataset_samples)

        total_samples = len(bleu_scores) if bleu_scores else 1

        # Generamos la fila de resultados para el Dataset evaluado
        return {
            "avg_elements": np.mean(elements_lengths) if elements_lengths else 0.0,
            "avg_bleu": np.mean(bleu_scores) if bleu_scores else 0.0,
            "avg_tau": np.mean(tau_scores) if tau_scores else 0.0,
            "avg_ard": np.mean(ard_scores) if ard_scores else 0.0,
            "fps": total_samples / max(total_time_inference, 1e-6) if total_samples > 0 else 0.0,
            "b1": f"{b1:,}",
            "b2": f"{b2:,}",
            "b3": f"{b3:,}",
            "b4": f"{b4:,}"
        }


# =====================================================================
# 4. ORQUESTACIÓN Y GENERACIÓN DEL REPORTE FINAL EN MARKDOWN
# =====================================================================
def print_final_benchmark_table(metrics_report: Dict[str, Dict[str, Dict[str, Any]]]):
    """
    Imprime la matriz comparativa final transpuestas: Algoritmos como filas y Datasets como columnas.
    Destaca en **negrita** (Top 1) y en <u>subrayado</u> (Top 2) para cada métrica individual por dataset.
    """
    # 1. Extraer algoritmos y datasets dinámicamente de los resultados
    algorithms = list(metrics_report.keys())
    datasets = []
    for algo_data in metrics_report.values():
        for ds in algo_data.keys():
            if ds not in datasets:
                datasets.append(ds)

    # 2. Definir las sub-columnas de métricas para CADA dataset
    sub_cols_names = [
        '#Box Avg.', 'BLEU-4 ↑', 'ARD ↓', 'Tau ↑', 'FPS ↑',
        'BLEU(0,.25] ↓', 'BLEU(.25,.5] ↓', 'BLEU(.5,.75] ↑', 'BLEU(.75,1] ↑'
    ]
    n_metrics = len(sub_cols_names)

    # 3. Construir los headers expandidos simulando ColSpan
    expanded_upper_cols = ['Algorithm']
    expanded_sub_cols = ['']

    for ds in datasets:
        short_ds = ds.split("/")[-1][:15]
        # Centrar el nombre del dataset entre sus métricas rellenando con guiones '_'
        pad_middle = n_metrics - 2

        expanded_upper_cols.append(short_ds)
        expanded_upper_cols.extend(["_"] * pad_middle)
        expanded_upper_cols.append(short_ds)

        expanded_sub_cols.extend(sub_cols_names)

    # Configuraciones de ancho y rendering en consola
    col_w = 16  # Ancho fijo generoso para tolerar asteriscos Markdown
    total_width = (len(expanded_upper_cols) * (col_w + 3)) + 1

    print("\n" + "=" * total_width)
    print("📊 MATRIZ COMPARATIVA DE BENCHMARKS (READING ORDER)")
    print("=" * total_width)

    def pad_str(s, w=col_w):
        return f"{str(s):<{w}}"

    # Imprimir Headers
    print(f"| {' | '.join(pad_str(c) for c in expanded_upper_cols)} |")
    print("|" + "|".join(["---"] * len(expanded_upper_cols)) + "|")
    print(f"| {' | '.join(pad_str(c) for c in expanded_sub_cols)} |")
    print("|" + "|".join(["<span></span>"] * len(expanded_upper_cols)) + "|")

    # 4. Funciones auxiliares de parseo
    def parse_bucket(raw_val: str) -> int:
        return int(str(raw_val).replace(',', '').split()[0])

    def format_bucket_str(raw_val: str, total: int) -> str:
        v = parse_bucket(raw_val)
        pct = v / total if total > 0 else 0
        return f"{v:,} ({pct:.2%})"

    def format_cell(val_str: str, algo: str, top1: str, top2: str) -> str:
        if algo == top1:
            return f"**{val_str}**"
        elif algo == top2:
            return f"<u>{val_str}</u>"
        return val_str

    # 5. Pre-calcular Rankings (Top 1 y Top 2) por Dataset y por Métrica
    rankings = {}
    for ds in datasets:
        # Extraer todos los registros válidos de este dataset para cruzar a todos los algoritmos
        recs = [(algo, metrics_report[algo][ds]) for algo in algorithms if
                ds in metrics_report[algo] and metrics_report[algo][ds]]

        def get_top_2(key_ext, reverse: bool):
            sorted_recs = sorted(recs, key=lambda x: key_ext(x[1]), reverse=reverse)
            t1 = sorted_recs[0][0] if len(sorted_recs) > 0 else None
            t2 = sorted_recs[1][0] if len(sorted_recs) > 1 else None
            return t1, t2

        rankings[ds] = {}
        if recs:
            rankings[ds]['bleu'] = get_top_2(lambda x: x['avg_bleu'], reverse=True)
            rankings[ds]['ard'] = get_top_2(lambda x: x['avg_ard'], reverse=False)
            rankings[ds]['tau'] = get_top_2(lambda x: x['avg_tau'], reverse=True)
            rankings[ds]['fps'] = get_top_2(lambda x: x['fps'], reverse=True)
            rankings[ds]['b1'] = get_top_2(lambda x: parse_bucket(x['b1']), reverse=False)
            rankings[ds]['b2'] = get_top_2(lambda x: parse_bucket(x['b2']), reverse=False)
            rankings[ds]['b3'] = get_top_2(lambda x: parse_bucket(x['b3']), reverse=True)
            rankings[ds]['b4'] = get_top_2(lambda x: parse_bucket(x['b4']), reverse=True)

    # 6. Procesar e Imprimir Filas (Iteramos algoritmos en Y, iteramos datasets y métricas en X)
    for algo in algorithms:
        row_data = [algo]

        for ds in datasets:
            data = metrics_report[algo].get(ds)
            if not data:
                # Si el algoritmo falló en este dataset, rellenar columnas con N/A
                row_data.extend(["N/A"] * n_metrics)
                continue

            r = rankings[ds]
            total_boxes = sum(parse_bucket(data[k]) for k in ['b1', 'b2', 'b3', 'b4'])

            box_str = f"{data['avg_elements']:.2f}"
            bleu_str = format_cell(f"{data['avg_bleu']:.4f}", algo, r['bleu'][0], r['bleu'][1])
            ard_str = format_cell(f"{data['avg_ard']:.4f}", algo, r['ard'][0], r['ard'][1])
            tau_str = format_cell(f"{data['avg_tau']:.4f}", algo, r['tau'][0], r['tau'][1])
            fps_str = format_cell(f"{data['fps']:.2f}", algo, r['fps'][0], r['fps'][1])

            b1_str = format_cell(format_bucket_str(data['b1'], total_boxes), algo, r['b1'][0], r['b1'][1])
            b2_str = format_cell(format_bucket_str(data['b2'], total_boxes), algo, r['b2'][0], r['b2'][1])
            b3_str = format_cell(format_bucket_str(data['b3'], total_boxes), algo, r['b3'][0], r['b3'][1])
            b4_str = format_cell(format_bucket_str(data['b4'], total_boxes), algo, r['b4'][0], r['b4'][1])

            row_data.extend([box_str, bleu_str, ard_str, tau_str, fps_str, b1_str, b2_str, b3_str, b4_str])

        print(f"| {' | '.join(pad_str(c) for c in row_data)} |")


def print_benchmark_console_dashboard(metrics_report: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
    """
    Render legible console output (no Markdown) con ranking visual por dataset.
    """
    if not metrics_report:
        print("\nNo hay resultados para mostrar.")
        return

    def build_line(widths: List[int], left: str, mid: str, right: str, fill: str = "─") -> str:
        return left + mid.join(fill * (w + 2) for w in widths) + right

    def format_cell(value: Any, width: int, align: str = "<") -> str:
        return f" {str(value):{align}{width}} "

    def print_table(
        headers: List[str],
        rows: List[List[Any]],
        aligns: List[str] | None = None,
        min_widths: List[int] | None = None,
    ) -> None:
        if aligns is None:
            aligns = ["<"] * len(headers)
        if min_widths is None:
            min_widths = [0] * len(headers)

        widths = [len(h) for h in headers]
        for row in rows:
            for idx, value in enumerate(row):
                widths[idx] = max(widths[idx], len(str(value)))
        for idx, min_width in enumerate(min_widths):
            widths[idx] = max(widths[idx], min_width)

        print(build_line(widths, "┌", "┬", "┐"))
        print("│" + "│".join(format_cell(headers[i], widths[i], "<") for i in range(len(headers))) + "│")
        print(build_line(widths, "├", "┼", "┤"))
        for row in rows:
            print("│" + "│".join(format_cell(row[i], widths[i], aligns[i]) for i in range(len(headers))) + "│")
        print(build_line(widths, "└", "┴", "┘"))

    def metric_rank(records: List[Tuple[str, Dict[str, Any]]], key: str, higher_is_better: bool) -> Dict[str, str]:
        sorted_records = sorted(records, key=lambda item: item[1][key], reverse=higher_is_better)
        marks: Dict[str, str] = {}
        if sorted_records:
            marks[sorted_records[0][0]] = "*"
        if len(sorted_records) > 1:
            marks[sorted_records[1][0]] = "."
        return marks

    def numeric_col_width(values: List[float], decimals: int, include_rank_prefix: bool = False, extra_margin: int = 1) -> int:
        max_value_len = max((len(f"{value:.{decimals}f}") for value in values), default=0)
        rank_prefix = 2 if include_rank_prefix else 0  # marcador + espacio
        return max_value_len + rank_prefix + extra_margin

    print("\n" + "═" * 108)
    print("BENCHMARK READING ORDER · RESUMEN CONSOLA")
    print("Leyenda ranking por métrica: * mejor  . segundo")
    print("═" * 108)

    for dataset_name in sorted({ds for algo_data in metrics_report.values() for ds in algo_data.keys()}):
        dataset_records = []
        for algo_name, datasets_report in metrics_report.items():
            data = datasets_report.get(dataset_name)
            if data:
                dataset_records.append((algo_name, data))

        if not dataset_records:
            continue

        bleu_marks = metric_rank(dataset_records, "avg_bleu", higher_is_better=True)
        tau_marks = metric_rank(dataset_records, "avg_tau", higher_is_better=True)
        ard_marks = metric_rank(dataset_records, "avg_ard", higher_is_better=False)
        fps_marks = metric_rank(dataset_records, "fps", higher_is_better=True)

        headers = ["Algoritmo", "Boxes", "BLEU-4", "Tau", "ARD", "FPS", "B(0,.25]", "B(.25,.5]", "B(.5,.75]", "B(.75,1]"]
        aligns = ["<", ">", ">", ">", ">", ">", ">", ">", ">", ">"]
        min_widths = [
            max(len(algo_name) for algo_name, _ in dataset_records) + 1,
            max(len(headers[1]), numeric_col_width([data["avg_elements"] for _, data in dataset_records], decimals=2)),
            max(len(headers[2]), numeric_col_width([data["avg_bleu"] for _, data in dataset_records], decimals=4, include_rank_prefix=True)),
            max(len(headers[3]), numeric_col_width([data["avg_tau"] for _, data in dataset_records], decimals=4, include_rank_prefix=True)),
            max(len(headers[4]), numeric_col_width([data["avg_ard"] for _, data in dataset_records], decimals=4, include_rank_prefix=True)),
            max(len(headers[5]), numeric_col_width([data["fps"] for _, data in dataset_records], decimals=2, include_rank_prefix=True)),
            max(len(headers[6]), max((len(str(data["b1"])) for _, data in dataset_records), default=0) + 1),
            max(len(headers[7]), max((len(str(data["b2"])) for _, data in dataset_records), default=0) + 1),
            max(len(headers[8]), max((len(str(data["b3"])) for _, data in dataset_records), default=0) + 1),
            max(len(headers[9]), max((len(str(data["b4"])) for _, data in dataset_records), default=0) + 1),
        ]
        rows: List[List[Any]] = []
        for algo_name, data in dataset_records:
            bleu_flag = bleu_marks.get(algo_name, " ")
            tau_flag = tau_marks.get(algo_name, " ")
            ard_flag = ard_marks.get(algo_name, " ")
            fps_flag = fps_marks.get(algo_name, " ")
            rows.append([
                algo_name,
                f"{data['avg_elements']:.2f}",
                f"{bleu_flag} {data['avg_bleu']:.4f}",
                f"{tau_flag} {data['avg_tau']:.4f}",
                f"{ard_flag} {data['avg_ard']:.4f}",
                f"{fps_flag} {data['fps']:.2f}",
                data["b1"],
                data["b2"],
                data["b3"],
                data["b4"],
            ])

        short_name = dataset_name.split("/")[-1]
        print(f"\nDataset: {short_name} ({dataset_name})")
        print_table(headers, rows, aligns, min_widths=min_widths)


def print_and_render_benchmark_latex_table(metrics_report: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
    """
    Genera una versión LaTeX de la tabla comparativa, la imprime en consola y
    renderiza la primera página del PDF en PNG (si las herramientas del sistema están disponibles).
    """
    if not metrics_report:
        print("\nNo hay resultados para exportar a LaTeX.")
        return

    def latex_escape(raw: str) -> str:
        escaped = raw
        replacements = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }
        for old, new in replacements.items():
            escaped = escaped.replace(old, new)
        return escaped

    def dataset_title(dataset_id: str) -> str:
        return f"{latex_escape(dataset_id.split('/')[-1])} (Mean)"

    def method_metadata(method_name: str) -> Tuple[str, str]:
        lower = method_name.lower()
        if lower == "xycutppy-paper":
            return r"\makecell{Custom\\{[this work]}}", r"$\checkmark$"
        if lower.startswith("xycutppy"):
            return r"\makecell{Custom\\{[this work]}}", r"\boldmath{$\times$}"
        return r"\makecell{Custom\\{[this work]}}", r"\boldmath{$\times$}"

    def metric_rank(
        records: List[Tuple[str, Dict[str, Any]]], key: str, higher_is_better: bool
    ) -> Dict[str, str]:
        sorted_records = sorted(records, key=lambda item: item[1][key], reverse=higher_is_better)
        marks: Dict[str, str] = {}
        if sorted_records:
            marks[sorted_records[0][0]] = "best"
        if len(sorted_records) > 1:
            marks[sorted_records[1][0]] = "second"
        return marks

    def style_value(value: float, decimals: int, rank_mark: str | None) -> str:
        formatted = f"{value:.{decimals}f}"
        if rank_mark == "best":
            return rf"\textbf{{{formatted}}}"
        if rank_mark == "second":
            return rf"\underline{{{formatted}}}"
        return formatted

    datasets = sorted({ds for algo_data in metrics_report.values() for ds in algo_data.keys()})
    methods = list(metrics_report.keys())

    rankings: Dict[str, Dict[str, Dict[str, str]]] = {}
    for dataset_name in datasets:
        dataset_records = [
            (method_name, metrics_report[method_name][dataset_name])
            for method_name in methods
            if dataset_name in metrics_report[method_name] and metrics_report[method_name][dataset_name]
        ]
        rankings[dataset_name] = {
            "fps": metric_rank(dataset_records, "fps", higher_is_better=True),
            "avg_bleu": metric_rank(dataset_records, "avg_bleu", higher_is_better=True),
            "avg_ard": metric_rank(dataset_records, "avg_ard", higher_is_better=False),
            "avg_tau": metric_rank(dataset_records, "avg_tau", higher_is_better=True),
        }

    group_start = 4
    cmidrules = []
    for idx in range(len(datasets)):
        start = group_start + (idx * 4)
        end = start + 3
        cmidrules.append(rf"\cmidrule(lr){{{start}-{end}}}")

    header_groups = " & ".join(
        rf"\multicolumn{{4}}{{c}}{{\textbf{{{dataset_title(ds)}}}}}" for ds in datasets
    )
    header_metrics = " & ".join(
        [r"FPS $\uparrow$", r"BLEU-4 $\uparrow$", r"ARD $\downarrow$", r"Tau $\uparrow$"] * len(datasets)
    )

    row_lines: List[str] = []
    for method_name in methods:
        source_cell, semantic_cell = method_metadata(method_name)
        row_cells = [latex_escape(method_name), source_cell, semantic_cell]
        for dataset_name in datasets:
            data = metrics_report[method_name].get(dataset_name)
            if not data:
                row_cells.extend(["--", "--", "--", "--"])
                continue
            dataset_rank = rankings[dataset_name]
            row_cells.extend(
                [
                    style_value(data["fps"], 2, dataset_rank["fps"].get(method_name)),
                    style_value(data["avg_bleu"], 4, dataset_rank["avg_bleu"].get(method_name)),
                    style_value(data["avg_ard"], 4, dataset_rank["avg_ard"].get(method_name)),
                    style_value(data["avg_tau"], 4, dataset_rank["avg_tau"].get(method_name)),
                ]
            )
        row_lines.append(" & ".join(row_cells) + r" \\")

    col_spec = "llc" + ("cccc" * len(datasets))
    latex_doc = f"""\\documentclass[varwidth,border=2pt]{{standalone}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{multirow}}
\\usepackage{{makecell}}
\\usepackage{{amssymb}}

\\begin{{document}}

\\centering
\\setlength{{\\tabcolsep}}{{3pt}}
\\renewcommand{{\\arraystretch}}{{1.2}}
\\newsavebox{{\\benchmarktablebox}}
\\sbox{{\\benchmarktablebox}}{{%
\\begin{{tabular}}{{{col_spec}}}
\\toprule
\\multirow{{2}}{{*}}{{\\textbf{{Method}}}} & \\multirow{{2}}{{*}}{{\\textbf{{Source}}}} & \\multirow{{2}}{{*}}{{\\textbf{{\\makecell{{Semantic\\\\Info}}}}}} & {header_groups} \\\\
{' '.join(cmidrules)}
& & & {header_metrics} \\\\
\\midrule
{chr(10).join(row_lines)}
\\bottomrule
\\end{{tabular}}
}}
\\usebox{{\\benchmarktablebox}}

\\vspace{{0.3cm}}
\\makebox[\\wd\\benchmarktablebox][c]{{%
\\parbox{{0.98\\wd\\benchmarktablebox}}{{%
\\centering\\small
Benchmark Reading Order summary automatically generated from run\\_benchmarks.py. Metric key: FPS $\\uparrow$ / BLEU-4 $\\uparrow$ / ARD $\\downarrow$ / Tau $\\uparrow$. Best results are in \\textbf{{bold}}; second-best are \\underline{{underlined}}.
}}%
}}

\\end{{document}}
"""

    print("\n" + "=" * 108)
    print("LATEX EQUIVALENTE")
    print("=" * 108)
    print(latex_doc)

    output_path = Path(__file__).parent / "output_examples"
    output_path.mkdir(parents=True, exist_ok=True)
    tex_path = output_path / "benchmark_table.tex"
    svg_path = output_path / "benchmark_table.svg"
    png_path = output_path / "benchmark_table.png"
    with open(tex_path, "w", encoding="utf-8", newline="\n") as tex_file:
        tex_file.write(latex_doc)

    with tempfile.TemporaryDirectory() as td:
        latex_cmd = [
            "latex",
            "-jobname=file",
            f"-output-directory={td}",
            str(tex_path),
        ]
        try:
            print(latex_cmd)
            fp = subprocess.run(latex_cmd, timeout=15, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            dvi_path = os.path.join(td, 'file.dvi')
            with open(dvi_path, 'rb') as f:
                dvi_bytes = f.read()
            with open(os.path.join(td, 'file.log'), 'rb') as f:
                latex_log = f.read()
        except Exception as e:
            traceback.print_exc()

        if fp.returncode != 0:
            print(f"[LATEX] Falló latex (exit {fp.returncode}).")
            print(latex_log[-1000:])
            return
        if not dvi_bytes:
            print("[LATEX] latex no devolvió contenido.")
            return
        dvisvgm_bin = shutil.which("dvisvgm")
        if dvisvgm_bin:
            svg_name = svg_path.name
            render_cmd = [dvisvgm_bin, "-o", svg_name, "file.dvi"]
            print(render_cmd)
            render_result = subprocess.run(
                render_cmd,
                timeout=15,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=td,
            )
            if render_result.returncode == 0:
                shutil.move(os.path.join(td, svg_name), svg_path)
                print(f"[LATEX] SVG renderizado en: {svg_path}")
            else:
                print(f"[LATEX] Falló dvisvgm (exit {render_result.returncode}).")
                print(render_result.stdout[-1000:])
                print(render_result.stderr[-1000:])
        dvipng_bin = shutil.which("dvipng")
        if dvipng_bin:
            png_name = png_path.name
            render_cmd = [dvipng_bin, "-T", "tight", "-o", png_name, "file.dvi"]
            print(render_cmd)
            render_result = subprocess.run(
                render_cmd,
                timeout=15,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=td,
            )
            if render_result.returncode == 0:
                shutil.move(os.path.join(td, png_name), png_path)
                print(f"[LATEX] PNG renderizado en: {png_path}")
                return
            print(f"[LATEX] Falló dvipng (exit {render_result.returncode}).")
            print(render_result.stdout[-1000:])
            print(render_result.stderr[-1000:])
        if svg_path.exists():
            print(f"[LATEX] Imagen generada en: {svg_path}")
            return
        print("[LATEX] No se pudo generar la imagen.")





if __name__ == "__main__":
    # Inicializar el entorno del framework
    benchmark_engine = ReadingOrderBenchmark()

    # Usamos la Factory para parametrizar de forma dinámica nuestra función objetivo
    # Si mañana tu solución (ej: xycut) requiere parámetros adicionales, los pasas por aquí.
    xycutppy_sorter = ReadingOrderSorterFactory(
        XYCutPPY(name="xycutppy-datalab", backend="datalab")
    )
    xycutppy_paper_sorter = ReadingOrderSorterFactory(
        XYCutPPY(name="xycutppy-paper", backend="paper")
    )
    hantian_cpu_sorter = ReadingOrderSorterFactory(
        HantianLayoutReaderAdapter(name="LayoutReader [CPU]", device="cpu")
    )
    hantian_gpu_sorter = ReadingOrderSorterFactory(
        HantianLayoutReaderAdapter(name="LayoutReader [GPU]", device="cuda")
    )
    huridocs_sorter = ReadingOrderSorterFactory(
        HuriDocsCandidateSelectorAdapter(name="huridocs")
    )
    sorter_solutions = [
        xycutppy_sorter,
        xycutppy_paper_sorter,
        # hantian_cpu_sorter,
        hantian_gpu_sorter,
        huridocs_sorter,
    ]

    # Diccionario para acumular los reportes de rendimiento
    final_report = {sorter.name: {} for sorter in sorter_solutions}

    # Lista de Datasets configurados para evaluación secuencial
    target_datasets = [
        "zilongwang/ReadingBank",
        "opendatalab/OmniDocBench",
        "llamaindex/ParseBench"
    ]

    # Límite de muestras por dataset para pruebas ágiles de desarrollo (Sustituir por None en producción)
    MAX_SAMPLES_TO_TEST = None

    total_algorithms = len(sorter_solutions)
    total_datasets = len(target_datasets)

    # Barra única global: progreso manual en porcentaje (0-100) ponderado por algoritmo->dataset->muestra.
    with tqdm(total=100.0, desc="Benchmark global", position=0, leave=True, dynamic_ncols=True) as global_progress_bar:
        for algorithm_index, sorter_algorithm in enumerate(sorter_solutions):
            for dataset_index, target_ds in enumerate(target_datasets):
                short_dataset_name = target_ds.split("/")[-1]

                def _on_progress(completed_samples: int, total_samples: int) -> None:
                    safe_total = max(total_samples, 1)
                    sample_fraction = min(completed_samples / safe_total, 1.0)
                    global_fraction = (
                        (algorithm_index / total_algorithms) +
                        ((dataset_index + sample_fraction) / (total_algorithms * total_datasets))
                    )
                    progress_percent = min(global_fraction * 100.0, 100.0)

                    global_progress_bar.n = progress_percent
                    global_progress_bar.set_description_str(
                        f"A:{algorithm_index + 1}/{total_algorithms} "
                        f"D:{dataset_index + 1}/{total_datasets} "
                        f"S:{min(completed_samples, safe_total)}/{safe_total} "
                        f"{sorter_algorithm.name}::{short_dataset_name}"
                    )
                    global_progress_bar.refresh()

                metrics = benchmark_engine.evaluate_dataset(
                    dataset_id=target_ds,
                    sorter_callable=sorter_algorithm.algorithm,
                    max_samples=MAX_SAMPLES_TO_TEST,
                    on_progress=_on_progress,
                )
                final_report[sorter_algorithm.name][target_ds] = metrics

        global_progress_bar.n = 100.0
        global_progress_bar.set_description_str("Global 100% completado")
        global_progress_bar.refresh()

    # Renderizado final en formato de consola (sin Markdown)
    print_benchmark_console_dashboard(final_report)
    print_final_benchmark_table(final_report)
    print_and_render_benchmark_latex_table(final_report)