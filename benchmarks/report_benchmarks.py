import csv
import os
import shutil
import subprocess
import tempfile
import traceback
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Any

import numpy as np


# =====================================================================
# FUNCIONES DE REPORTE (movidas desde run_benchmarks.py)
# =====================================================================

def print_final_benchmark_table(metrics_report: Dict[str, Dict[str, Dict[str, Any]]]):
    """
    Imprime la matriz comparativa final transpuestas: Algoritmos como filas y Datasets como columnas.
    Destaca en **negrita** (Top 1) y en <u>subrayado</u> (Top 2) para cada métrica individual por dataset.
    """
    algorithms = list(metrics_report.keys())
    datasets = []
    for algo_data in metrics_report.values():
        for ds in algo_data.keys():
            if ds not in datasets:
                datasets.append(ds)

    sub_cols_names = [
        '#Box Avg.', 'BLEU-4 ↑', 'ARD ↓', 'Tau ↑', 'FPS ↑',
        'BLEU(0,.25] ↓', 'BLEU(.25,.5] ↓', 'BLEU(.5,.75] ↑', 'BLEU(.75,1] ↑'
    ]
    n_metrics = len(sub_cols_names)

    expanded_upper_cols = ['Algorithm']
    expanded_sub_cols = ['']

    for ds in datasets:
        short_ds = ds.split("/")[-1][:15]
        pad_middle = n_metrics - 2

        expanded_upper_cols.append(short_ds)
        expanded_upper_cols.extend(["_"] * pad_middle)
        expanded_upper_cols.append(short_ds)

        expanded_sub_cols.extend(sub_cols_names)

    col_w = 16
    total_width = (len(expanded_upper_cols) * (col_w + 3)) + 1

    print("\n" + "=" * total_width)
    print("📊 MATRIZ COMPARATIVA DE BENCHMARKS (READING ORDER)")
    print("=" * total_width)

    def pad_str(s, w=col_w):
        return f"{str(s):<{w}}"

    print(f"| {' | '.join(pad_str(c) for c in expanded_upper_cols)} |")
    print("|" + "|".join(["---"] * len(expanded_upper_cols)) + "|")
    print(f"| {' | '.join(pad_str(c) for c in expanded_sub_cols)} |")
    print("|" + "|".join(["<span></span>"] * len(expanded_upper_cols)) + "|")

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

    rankings = {}
    for ds in datasets:
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

    for algo in algorithms:
        row_data = [algo]

        for ds in datasets:
            data = metrics_report[algo].get(ds)
            if not data:
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
        rank_prefix = 2 if include_rank_prefix else 0
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
            return

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


# =====================================================================
# CARGA Y AGREGACIÓN DE RESULTADOS
# =====================================================================
_CSV_FIELDNAMES = [
    "run_timestamp", "algorithm", "dataset",
    "avg_elements", "avg_bleu", "avg_tau", "avg_ard", "fps",
    "b1", "b2", "b3", "b4",
]

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


def load_all_results(results_dir: Path) -> List[Dict[str, Any]]:
    """Carga todos los CSVs de ejecuciones desde el directorio de resultados."""
    rows = []
    for csv_path in sorted(results_dir.glob("*/*.csv")):
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    return rows


def aggregate_results(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Agrupa por (algorithm, dataset) y calcula la media de todas las ejecuciones.
    Los contadores b1-b4 se promedian y redondean al entero más cercano.
    """
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["algorithm"], row["dataset"])
        grouped[key].append(row)

    aggregated: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for (algo, dataset), run_rows in grouped.items():
        if algo not in aggregated:
            aggregated[algo] = {}

        def mean_float(field: str) -> float:
            return float(np.mean([float(r[field]) for r in run_rows]))

        def mean_int(field: str) -> int:
            return int(round(float(np.mean([int(r[field]) for r in run_rows]))))

        aggregated[algo][dataset] = {
            "avg_elements": mean_float("avg_elements"),
            "avg_bleu": mean_float("avg_bleu"),
            "avg_tau": mean_float("avg_tau"),
            "avg_ard": mean_float("avg_ard"),
            "fps": mean_float("fps"),
            "b1": f"{mean_int('b1'):,}",
            "b2": f"{mean_int('b2'):,}",
            "b3": f"{mean_int('b3'):,}",
            "b4": f"{mean_int('b4'):,}",
        }

    return aggregated


def load_all_category_results(results_dir: Path) -> List[Dict[str, Any]]:
    """Carga todos los CSVs de métricas por categoría."""
    rows = []
    for csv_path in sorted(results_dir.glob("*/*_categories.csv")):
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    return rows


def load_all_worst_cases(results_dir: Path) -> List[Dict[str, Any]]:
    """Carga todos los CSVs de peores muestras."""
    rows = []
    for csv_path in sorted(results_dir.glob("*/*_worst_cases.csv")):
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    return rows


def aggregate_category_results(
    rows: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]:
    """
    Agrupa por (algorithm, dataset, category) y calcula la media de todas las ejecuciones.
    Retorna: algo -> dataset -> category -> metrics
    """
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["algorithm"], row["dataset"], row["category"])
        grouped[key].append(row)

    aggregated: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}
    for (algo, dataset, category), run_rows in grouped.items():
        aggregated.setdefault(algo, {}).setdefault(dataset, {})[category] = {
            "n_samples": int(round(float(np.mean([int(r["n_samples"]) for r in run_rows])))),
            "avg_elements": float(np.mean([float(r["avg_elements"]) for r in run_rows])),
            "avg_bleu": float(np.mean([float(r["avg_bleu"]) for r in run_rows])),
            "avg_tau": float(np.mean([float(r["avg_tau"]) for r in run_rows])),
            "avg_ard": float(np.mean([float(r["avg_ard"]) for r in run_rows])),
            "b1": int(round(float(np.mean([int(r["b1"]) for r in run_rows])))),
            "b2": int(round(float(np.mean([int(r["b2"]) for r in run_rows])))),
            "b3": int(round(float(np.mean([int(r["b3"]) for r in run_rows])))),
            "b4": int(round(float(np.mean([int(r["b4"]) for r in run_rows])))),
        }
    return aggregated


def print_category_tables_per_dataset(
    category_report: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]
) -> None:
    """
    Para cada dataset con categorías, imprime una tabla con métricas por categoría.
    Una tabla por dataset, filas = categorías, columnas = métricas por algoritmo.
    """
    if not category_report:
        print("\nNo hay métricas categóricas disponibles.")
        return

    datasets_with_categories: set = set()
    for algo_data in category_report.values():
        for ds in algo_data:
            datasets_with_categories.add(ds)

    algorithms = list(category_report.keys())

    def build_line(widths, left="┌", mid="┬", right="┐", fill="─"):
        return left + mid.join(fill * (w + 2) for w in widths) + right

    def fc(value, width, align="<"):
        return f" {str(value):{align}{width}} "

    def pt(headers, rows, aligns=None, min_widths=None):
        if aligns is None:
            aligns = ["<"] * len(headers)
        if min_widths is None:
            min_widths = [0] * len(headers)
        widths = [max(len(h), min_widths[i] if i < len(min_widths) else 0) for i, h in enumerate(headers)]
        for row in rows:
            for i, v in enumerate(row):
                widths[i] = max(widths[i], len(str(v)))
        print(build_line(widths, "┌", "┬", "┐"))
        print("│" + "│".join(fc(headers[i], widths[i]) for i in range(len(headers))) + "│")
        print(build_line(widths, "├", "┼", "┤"))
        for row in rows:
            print("│" + "│".join(fc(row[i], widths[i], aligns[i] if i < len(aligns) else "<") for i in range(len(headers))) + "│")
        print(build_line(widths, "└", "┴", "┘"))

    print("\n" + "═" * 108)
    print("BENCHMARK READING ORDER · MÉTRICAS POR CATEGORÍA")
    print("═" * 108)

    for dataset_name in sorted(datasets_with_categories):
        short_ds = dataset_name.split("/")[-1]

        all_categories: set = set()
        for algo in algorithms:
            cats = category_report.get(algo, {}).get(dataset_name, {})
            all_categories.update(cats.keys())

        if not all_categories:
            continue

        print(f"\nDataset: {short_ds} ({dataset_name})")

        for algo in algorithms:
            cats = category_report.get(algo, {}).get(dataset_name, {})
            if not cats:
                continue

            print(f"  Algoritmo: {algo}")
            headers = ["Categoría", "N", "Boxes", "BLEU-4 ↑", "Tau ↑", "ARD ↓", "B(0,.25]", "B(.25,.5]", "B(.5,.75]", "B(.75,1]"]
            aligns = ["<", ">", ">", ">", ">", ">", ">", ">", ">", ">"]
            rows = []
            sorted_cats = sorted(cats.items(), key=lambda x: x[1]["avg_bleu"], reverse=True)
            for cat_name, m in sorted_cats:
                total = m["b1"] + m["b2"] + m["b3"] + m["b4"]

                def pct(v, t=total): return f"{v} ({v/t:.1%})" if t > 0 else str(v)
                rows.append([
                    cat_name[:30],
                    m["n_samples"],
                    f"{m['avg_elements']:.2f}",
                    f"{m['avg_bleu']:.4f}",
                    f"{m['avg_tau']:.4f}",
                    f"{m['avg_ard']:.4f}",
                    pct(m["b1"]),
                    pct(m["b2"]),
                    pct(m["b3"]),
                    pct(m["b4"]),
                ])
            pt(headers, rows, aligns)


def print_worst_cases_summary(worst_cases_rows: List[Dict[str, Any]], top_n: int = 10) -> None:
    """
    Muestra un resumen de los peores casos por algoritmo y dataset.
    """
    if not worst_cases_rows:
        print("\nNo hay datos de peores muestras disponibles.")
        return

    print("\n" + "═" * 108)
    print(f"BENCHMARK READING ORDER · PEORES {top_n} MUESTRAS POR ALGORITMO/DATASET")
    print("═" * 108)

    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in worst_cases_rows:
        grouped[(row["algorithm"], row["dataset"])].append(row)

    def build_line(widths, left="┌", mid="┬", right="┐", fill="─"):
        return left + mid.join(fill * (w + 2) for w in widths) + right

    def fc(value, width, align="<"):
        return f" {str(value):{align}{width}} "

    for (algo, dataset), cases in sorted(grouped.items()):
        short_ds = dataset.split("/")[-1]
        print(f"\n  {algo} :: {short_ds}")

        sorted_cases = sorted(cases, key=lambda x: float(x["bleu"]))[:top_n]

        headers = ["#", "Categoría", "N Elem.", "BLEU ↑", "Tau ↑", "ARD ↓"]
        aligns = [">", "<", ">", ">", ">", ">"]
        widths = [max(len(h), 4) for h in headers]
        for i, case in enumerate(sorted_cases):
            vals = [
                str(i + 1),
                (case["category"] or "N/A")[:30],
                case["n_elements"],
                f"{float(case['bleu']):.4f}",
                f"{float(case['tau']):.4f}",
                f"{float(case['ard']):.4f}",
            ]
            for j, v in enumerate(vals):
                widths[j] = max(widths[j], len(str(v)))

        print("  " + build_line(widths, "┌", "┬", "┐"))
        print("  │" + "│".join(fc(headers[i], widths[i]) for i in range(len(headers))) + "│")
        print("  " + build_line(widths, "├", "┼", "┤"))
        for i, case in enumerate(sorted_cases):
            row = [
                str(i + 1),
                (case["category"] or "N/A")[:30],
                case["n_elements"],
                f"{float(case['bleu']):.4f}",
                f"{float(case['tau']):.4f}",
                f"{float(case['ard']):.4f}",
            ]
            print("  │" + "│".join(fc(row[j], widths[j], aligns[j]) for j in range(len(headers))) + "│")
        print("  " + build_line(widths, "└", "┴", "┘"))


def generate_category_charts(
    category_report: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]],
    output_dir: Path,
) -> None:
    """
    Genera gráficos de barras comparando categorías por dataset.
    Un PNG por dataset que tenga categorías.
    - Subgráfico 1: BLEU-4 por categoría (agrupado por algoritmo)
    - Subgráfico 2: Tau por categoría
    - Subgráfico 3: ARD por categoría
    - Subgráfico 4: Distribución de buckets (stacked bar)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Charts] matplotlib no disponible, omitiendo gráficos.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    algorithms = list(category_report.keys())

    datasets_with_categories: set = set()
    for algo_data in category_report.values():
        for ds in algo_data:
            datasets_with_categories.add(ds)

    colors = plt.cm.tab10.colors

    for dataset_name in sorted(datasets_with_categories):
        short_ds = dataset_name.split("/")[-1]

        all_categories: set = set()
        for algo in algorithms:
            cats = category_report.get(algo, {}).get(dataset_name, {})
            all_categories.update(cats.keys())

        if not all_categories:
            continue

        sorted_cats = sorted(all_categories)
        n_cats = len(sorted_cats)
        n_algos = len(algorithms)

        fig, axes = plt.subplots(2, 2, figsize=(max(14, n_cats * 1.5 * n_algos * 0.5 + 4), 10))
        fig.suptitle(f"Métricas por Categoría — {short_ds}", fontsize=14, fontweight="bold")

        bar_width = 0.8 / max(n_algos, 1)
        x = np.arange(n_cats)

        metrics_config = [
            (axes[0, 0], "avg_bleu", "BLEU-4 ↑", True),
            (axes[0, 1], "avg_tau", "Kendall Tau ↑", True),
            (axes[1, 0], "avg_ard", "ARD ↓", False),
        ]

        for ax, metric_key, metric_label, higher_is_better in metrics_config:
            for algo_idx, algo in enumerate(algorithms):
                cats = category_report.get(algo, {}).get(dataset_name, {})
                values = [cats.get(cat, {}).get(metric_key, 0.0) for cat in sorted_cats]
                offset = (algo_idx - n_algos / 2 + 0.5) * bar_width
                ax.bar(x + offset, values, bar_width * 0.9, label=algo, color=colors[algo_idx % len(colors)], alpha=0.85)

            ax.set_title(metric_label, fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels([c[:15] for c in sorted_cats], rotation=35, ha="right", fontsize=8)
            ax.set_ylabel(metric_key)
            ax.legend(fontsize=7)
            ax.grid(axis="y", alpha=0.3)

        # Subgráfico 4: Distribución de buckets (stacked bar) — solo primer algoritmo disponible
        ax_bucket = axes[1, 1]
        first_algo = algorithms[0] if algorithms else None
        if first_algo:
            cats_data = category_report.get(first_algo, {}).get(dataset_name, {})
            b1_vals = [cats_data.get(cat, {}).get("b1", 0) for cat in sorted_cats]
            b2_vals = [cats_data.get(cat, {}).get("b2", 0) for cat in sorted_cats]
            b3_vals = [cats_data.get(cat, {}).get("b3", 0) for cat in sorted_cats]
            b4_vals = [cats_data.get(cat, {}).get("b4", 0) for cat in sorted_cats]

            ax_bucket.bar(x, b1_vals, label="BLEU (0,.25]", color="#d62728", alpha=0.85)
            ax_bucket.bar(x, b2_vals, bottom=b1_vals, label="BLEU (.25,.5]", color="#ff7f0e", alpha=0.85)
            ax_bucket.bar(x, b3_vals, bottom=[b1_vals[i] + b2_vals[i] for i in range(n_cats)], label="BLEU (.5,.75]", color="#2ca02c", alpha=0.85)
            ax_bucket.bar(x, b4_vals, bottom=[b1_vals[i] + b2_vals[i] + b3_vals[i] for i in range(n_cats)], label="BLEU (.75,1]", color="#1f77b4", alpha=0.85)
            ax_bucket.set_title(f"Distribución BLEU ({first_algo})", fontsize=11)
            ax_bucket.set_xticks(x)
            ax_bucket.set_xticklabels([c[:15] for c in sorted_cats], rotation=35, ha="right", fontsize=8)
            ax_bucket.legend(fontsize=7)
            ax_bucket.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        out_path = output_dir / f"categories_{short_ds}.png"
        plt.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Charts] Gráfico guardado: {out_path}")


if __name__ == "__main__":
    results_dir = Path(__file__).parent / "results"

    if not results_dir.exists() or not any(results_dir.glob("*/*.csv")):
        print("No se encontraron resultados. Ejecuta primero 'python run_benchmarks.py'.")
        raise SystemExit(1)

    rows = load_all_results(results_dir)
    if not rows:
        print("No hay datos en los CSV encontrados.")
        raise SystemExit(1)

    run_timestamps = sorted({r["run_timestamp"] for r in rows})
    print(f"📂 Ejecuciones encontradas ({len(run_timestamps)}):")
    for ts in run_timestamps:
        print(f"  • {ts}")

    aggregated_report = aggregate_results(rows)

    print_benchmark_console_dashboard(aggregated_report)
    print_final_benchmark_table(aggregated_report)
    print_and_render_benchmark_latex_table(aggregated_report)

    # Métricas por categoría
    category_rows = load_all_category_results(results_dir)
    if category_rows:
        aggregated_categories = aggregate_category_results(category_rows)
        print_category_tables_per_dataset(aggregated_categories)
        output_dir = Path(__file__).parent / "output_examples"
        generate_category_charts(aggregated_categories, output_dir)
    else:
        print("\nℹ️  No se encontraron CSVs de métricas por categoría (*_categories.csv).")

    # Peores muestras
    worst_rows = load_all_worst_cases(results_dir)
    if worst_rows:
        print_worst_cases_summary(worst_rows, top_n=10)
    else:
        print("\nℹ️  No se encontraron CSVs de peores muestras (*_worst_cases.csv).")