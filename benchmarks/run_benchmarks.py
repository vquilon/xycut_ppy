import csv
import json
import os
import time
import traceback
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Any, Callable, cast, Literal, Optional

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from scipy.stats import kendalltau
from tqdm import tqdm

from providers.sorters.base import COLOR_MAP, YOLO_ELEMENTS_MAPPING, ReadingOrderWrapper
from providers.sorters.factory import ReadingOrderSorterFactory
from providers.sorters.hantian import HantianLayoutReaderAdapter
from providers.sorters.huridocs import HuriDocsCandidateSelectorAdapter
from providers.sorters.xycutppy import XYCutPPYReadingOrderAdapter


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

    def _normalize_reading_bank(self, raw_sample: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[int], Tuple[float, float, float, float], Optional[str]]:
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

        # ReadingBank no expone categorías de tipo de documento en su esquema público
        category: Optional[str] = None

        return normalized_elements, ground_truth_order, page_bounds, category

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
        yolo_labels = YOLO_ELEMENTS_MAPPING | set(COLOR_MAP.keys())
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


    def _normalize_omni_doc_bench(self, json_item: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[int], Tuple[float, float, float, float], Optional[str]]:
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

        # OmniDocBench categoriza cada página por tipo de documento:
        # academic_literature, book, colorful_textbook, exam_paper, historical_document,
        # magazine, newspaper, note, PPT2PDF, research_report
        category: Optional[str] = (
            (json_item.get("page_info") or {})
            .get("page_attribute", {})
            .get("data_source")
        ) or None

        return normalized_elements, ground_truth_order, page_bounds, category

    def _normalize_parsebench(self, raw_rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[int], Tuple[float, float, float, float], Optional[str]]:
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
            return [], [], (0.0, 0.0, 1.0, 1.0), None

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

        # ParseBench (split layout) clasifica cada página con dificultad via tags: "easy" / "hard"
        category: Optional[str] = None
        _difficulty_tags = {"easy", "hard"}
        for _row in raw_rows:
            _tags = _row.get("tags") or []
            if isinstance(_tags, str):
                try:
                    import json as _json
                    _tags = _json.loads(_tags)
                except Exception:
                    _tags = []
            for _tag in _tags:
                if isinstance(_tag, str) and _tag.lower() in _difficulty_tags:
                    category = _tag.lower()
                    break
            if category:
                break

        return normalized_elements, ground_truth_order, page_bounds, category



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
            top_n_worst: int = 10,
    ) -> Dict[str, Any]:
        """
        Ejecuta la batería de pruebas calculando promedios de palabras/cajas y la matriz de distribución.
        """
        # print(f"\n🔄 Evaluando Reading Order en: {dataset_id}")

        # Carga dinámica dependiendo del tipo de dataset
        samples_to_process: List[Tuple[List[Dict[str, Any]], List[int], Tuple[float, float, float, float], Optional[str]]] = []
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

        # Acumuladores por categoría
        cat_bleu: Dict[str, List[float]] = defaultdict(list)
        cat_tau: Dict[str, List[float]] = defaultdict(list)
        cat_ard: Dict[str, List[float]] = defaultdict(list)
        cat_elements: Dict[str, List[int]] = defaultdict(list)
        cat_b1: Dict[str, int] = defaultdict(int)
        cat_b2: Dict[str, int] = defaultdict(int)
        cat_b3: Dict[str, int] = defaultdict(int)
        cat_b4: Dict[str, int] = defaultdict(int)

        # Todas las muestras para los peores casos
        all_sample_scores: List[Dict[str, Any]] = []

        for sample_index, (elements, gt_order, page_bounds, category) in enumerate(samples_to_process, start=1):
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

            # Acumular por categoría
            cat_key = category
            if cat_key:
                cat_bleu[cat_key].append(score_bleu)
                cat_tau[cat_key].append(score_tau)
                cat_ard[cat_key].append(score_ard)
                cat_elements[cat_key].append(len(elements))
                if 0.0 <= score_bleu <= 0.25:
                    cat_b1[cat_key] += 1
                elif 0.25 < score_bleu <= 0.50:
                    cat_b2[cat_key] += 1
                elif 0.50 < score_bleu <= 0.75:
                    cat_b3[cat_key] += 1
                else:
                    cat_b4[cat_key] += 1

            # Registrar para peores casos
            all_sample_scores.append({
                "sample_index": sample_index,
                "category": category or "",
                "n_elements": len(elements),
                "bleu": score_bleu,
                "tau": score_tau,
                "ard": score_ard,
            })

            if on_progress:
                on_progress(sample_index, total_dataset_samples)

        total_samples = len(bleu_scores) if bleu_scores else 1

        general_metrics = {
            "avg_elements": np.mean(elements_lengths) if elements_lengths else 0.0,
            "avg_bleu": np.mean(bleu_scores) if bleu_scores else 0.0,
            "avg_tau": np.mean(tau_scores) if tau_scores else 0.0,
            "avg_ard": np.mean(ard_scores) if ard_scores else 0.0,
            "fps": total_samples / max(total_time_inference, 1e-6) if total_samples > 0 else 0.0,
            "b1": f"{b1:,}",
            "b2": f"{b2:,}",
            "b3": f"{b3:,}",
            "b4": f"{b4:,}",
        }

        # Métricas por categoría
        categories_metrics: Dict[str, Dict[str, Any]] = {}
        for ck in cat_bleu:
            n = len(cat_bleu[ck])
            categories_metrics[ck] = {
                "n_samples": n,
                "avg_elements": float(np.mean(cat_elements[ck])) if cat_elements[ck] else 0.0,
                "avg_bleu": float(np.mean(cat_bleu[ck])),
                "avg_tau": float(np.mean(cat_tau[ck])),
                "avg_ard": float(np.mean(cat_ard[ck])),
                "b1": cat_b1[ck],
                "b2": cat_b2[ck],
                "b3": cat_b3[ck],
                "b4": cat_b4[ck],
            }

        # Peores N muestras (ordenar por BLEU ascendente)
        worst_cases = sorted(all_sample_scores, key=lambda x: x["bleu"])[:top_n_worst]

        return {
            "general": general_metrics,
            "categories": categories_metrics,
            "worst_cases": worst_cases,
        }


# =====================================================================
# 4. ORQUESTACIÓN Y PERSISTENCIA DE RESULTADOS
# =====================================================================
_CSV_FIELDNAMES = [
    "run_timestamp", "algorithm", "dataset",
    "avg_elements", "avg_bleu", "avg_tau", "avg_ard", "fps",
    "b1", "b2", "b3", "b4",
]

TOP_N_WORST_SAMPLES = 10  # Parametrizable en main

_CSV_CATEGORY_FIELDNAMES = [
    "run_timestamp", "algorithm", "dataset", "category",
    "n_samples", "avg_elements", "avg_bleu", "avg_tau", "avg_ard",
    "b1", "b2", "b3", "b4",
]

_CSV_WORST_CASES_FIELDNAMES = [
    "run_timestamp", "algorithm", "dataset",
    "sample_index", "category", "n_elements",
    "bleu", "tau", "ard",
]


def save_results_to_csv(
        final_report: Dict[str, Dict[str, Dict[str, Any]]],
        run_timestamp: str,
        results_base_dir: Path,
) -> Path:
    run_dir = results_base_dir / run_timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / f"{run_timestamp}.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDNAMES)
        writer.writeheader()
        for algo_name, datasets_metrics in final_report.items():
            for dataset_id, metrics in datasets_metrics.items():
                if not metrics:
                    continue
                writer.writerow({
                    "run_timestamp": run_timestamp,
                    "algorithm": algo_name,
                    "dataset": dataset_id,
                    "avg_elements": metrics["avg_elements"],
                    "avg_bleu": metrics["avg_bleu"],
                    "avg_tau": metrics["avg_tau"],
                    "avg_ard": metrics["avg_ard"],
                    "fps": metrics["fps"],
                    "b1": int(str(metrics["b1"]).replace(",", "")),
                    "b2": int(str(metrics["b2"]).replace(",", "")),
                    "b3": int(str(metrics["b3"]).replace(",", "")),
                    "b4": int(str(metrics["b4"]).replace(",", "")),
                })

    return csv_path


def save_category_results_to_csv(
        category_report: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]],
        run_timestamp: str,
        results_base_dir: Path,
) -> Path:
    run_dir = results_base_dir / run_timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / f"{run_timestamp}_categories.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_CATEGORY_FIELDNAMES)
        writer.writeheader()
        for algo_name, datasets in category_report.items():
            for dataset_id, categories in datasets.items():
                for category, metrics in categories.items():
                    writer.writerow({
                        "run_timestamp": run_timestamp,
                        "algorithm": algo_name,
                        "dataset": dataset_id,
                        "category": category,
                        "n_samples": metrics["n_samples"],
                        "avg_elements": metrics["avg_elements"],
                        "avg_bleu": metrics["avg_bleu"],
                        "avg_tau": metrics["avg_tau"],
                        "avg_ard": metrics["avg_ard"],
                        "b1": metrics["b1"],
                        "b2": metrics["b2"],
                        "b3": metrics["b3"],
                        "b4": metrics["b4"],
                    })
    return csv_path


def save_worst_cases_to_csv(
        worst_cases_report: Dict[str, Dict[str, List[Dict[str, Any]]]],
        run_timestamp: str,
        results_base_dir: Path,
) -> Path:
    run_dir = results_base_dir / run_timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / f"{run_timestamp}_worst_cases.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_WORST_CASES_FIELDNAMES)
        writer.writeheader()
        for algo_name, datasets in worst_cases_report.items():
            for dataset_id, cases in datasets.items():
                for case in cases:
                    writer.writerow({
                        "run_timestamp": run_timestamp,
                        "algorithm": algo_name,
                        "dataset": dataset_id,
                        "sample_index": case["sample_index"],
                        "category": case["category"],
                        "n_elements": case["n_elements"],
                        "bleu": case["bleu"],
                        "tau": case["tau"],
                        "ard": case["ard"],
                    })
    return csv_path


if __name__ == "__main__":
    # Inicializar el entorno del framework
    benchmark_engine = ReadingOrderBenchmark()

    # Usamos la Factory para parametrizar de forma dinámica nuestra función objetivo
    # Si mañana tu solución (ej: xycut) requiere parámetros adicionales, los pasas por aquí.
    xycutppy_sorter = ReadingOrderSorterFactory(
        XYCutPPYReadingOrderAdapter(name="xycutppy-datalab", backend="datalab")
    )
    xycutppy_paper_sorter = ReadingOrderSorterFactory(
        XYCutPPYReadingOrderAdapter(name="xycutppy-paper", backend="paper")
    )
    hantian_cpu_sorter = ReadingOrderSorterFactory(
        HantianLayoutReaderAdapter(name="LayoutReader [CPU]", device="cpu")
    )
    hantian_gpu_sorter = ReadingOrderSorterFactory(
        HantianLayoutReaderAdapter(name="LayoutReader [GPU]", device="cuda")
    )
    huridocs_sorter = ReadingOrderSorterFactory(
        HuriDocsCandidateSelectorAdapter(name="huridocs/CandidateSelector")
    )
    sorter_solutions = [
        xycutppy_sorter,
        # xycutppy_paper_sorter,
        # hantian_cpu_sorter,
        # hantian_gpu_sorter,
        # huridocs_sorter,
    ]

    # Diccionario para acumular los reportes de rendimiento
    final_report = {sorter.name: {} for sorter in sorter_solutions}
    category_report: Dict[str, Dict[str, Any]] = {}
    worst_cases_report: Dict[str, Dict[str, Any]] = {}

    # Lista de Datasets configurados para evaluación secuencial
    target_datasets = [
        "zilongwang/ReadingBank",
        "opendatalab/OmniDocBench",
        "llamaindex/ParseBench"
    ]

    # Límite de muestras por dataset para pruebas ágiles de desarrollo (Sustituir por None en producción)
    MAX_SAMPLES_TO_TEST = 10
    MAX_WORST_CASES = TOP_N_WORST_SAMPLES

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

                result = benchmark_engine.evaluate_dataset(
                    dataset_id=target_ds,
                    sorter_callable=sorter_algorithm.algorithm,
                    max_samples=MAX_SAMPLES_TO_TEST,
                    on_progress=_on_progress,
                    top_n_worst=MAX_WORST_CASES,
                )
                # general_report mantiene backward compat
                final_report[sorter_algorithm.name][target_ds] = result.get("general", {})
                # nuevas estructuras
                if result.get("categories"):
                    category_report.setdefault(sorter_algorithm.name, {})[target_ds] = result["categories"]
                if result.get("worst_cases"):
                    worst_cases_report.setdefault(sorter_algorithm.name, {})[target_ds] = result["worst_cases"]

        global_progress_bar.n = 100.0
        global_progress_bar.set_description_str("Global 100% completado")
        global_progress_bar.refresh()

    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = Path(__file__).parent / "results"
    csv_path = save_results_to_csv(final_report, run_timestamp, results_dir)
    print(f"\n✅ Resultados guardados en: {csv_path}")
    if category_report:
        cat_csv_path = save_category_results_to_csv(category_report, run_timestamp, results_dir)
        print(f"   Métricas por categoría: {cat_csv_path}")
    if worst_cases_report:
        worst_csv_path = save_worst_cases_to_csv(worst_cases_report, run_timestamp, results_dir)
        print(f"   Peores muestras: {worst_csv_path}")
    print("   Ejecuta 'python report_benchmarks.py' para ver las métricas agregadas.")