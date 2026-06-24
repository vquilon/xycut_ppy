# DESIGN: XY-Cut++ Architecture and Implementation (Rust Port)

This section expands on the quickstart with architecture decisions and current implementation status in the `xycutppy` package.

## 🧱 Current backend architecture

The Python wrapper exposes two engines in the same library, selectable at runtime:

- `paper`: original backend (GPL), based on `src/paper/xycut_plus_plus`.
- `datalab`: Rust backend of the OpenDataLoader sorter (Apache-2.0), in `src/datalab/xycut_plus_plus_sorter`.

### Module isolation

`paper` and `datalab` are decoupled:

1. `datalab` does not depend on traits or types from `paper`.
2. `datalab` defines its own `Element` and runs as an independent crate.
3. Wrapper `src/lib.rs` only converts `PyElement -> datalab::Element` when `datalab` is called.

This allows independent evolution of both algorithms and simplifies license compliance.

## 🏗️ Algorithm architecture (Datalab Rust)

The `datalab` implementation keeps the Java flow's 4-stage structure:

### Stage 1: `Cross-Layout` detection

Detects very wide blocks with enough horizontal overlap:

- Base rule: `width >= beta * max_width`.
- Context confirmation: overlap with at least `MIN_OVERLAP_COUNT` elements.
- Output: split into `cross_layout` and `remaining`.

### Stage 2: density and axis preference

Computes `content_area / region_area`:

- If above `DEFAULT_DENSITY_THRESHOLD`, enables `prefer_horizontal_first`.
- This preference is used as tie-breaker when both cuts are valid with equivalent gaps.

### Stage 3: recursive segmentation

`recursive_segment` finds best horizontal and vertical cuts, applies gap threshold, and splits into subgroups:

- Uses `MIN_GAP_THRESHOLD` to avoid noise cuts.
- If no valid cut exists, falls back to geometric order `Y desc, X asc`.
- Filters narrow outliers when searching vertical cuts (`NARROW_ELEMENT_WIDTH_RATIO`).

### Stage 4: final merge

Reinserts `cross_layout` by comparing `top_y` against the main flow:

- Stable, purely geometric merge.
- No semantic weighting from the original paper.

## 🦀 Engineering decisions applied in Rust

1. **Robust `f32` ordering**: uses `f32::total_cmp` to avoid `NaN` ambiguities.
2. **Safe index-based removal**: separates `cross_layout` by index/boolean mask, not by `id`.
3. **Real use of `prefer_horizontal_first`**: no longer a ghost parameter; now a cut tie-breaker.
4. **Controlled cloning and readability**: keeps `Vec`/`clone` approach for clarity; pre-allocates capacity in critical paths (`Vec::with_capacity`).

## 📌 Semantic label status in code

In the Rust/Python API exposed by `paper`, the current enum includes:

- `CrossLayout`
- `HorizontalTitle`
- `VerticalTitle`
- `Vision`
- `Regular`

In current `datalab`, the algorithm is geometric and does not consume labels in the sorting core.

## 🔧 Evolution notes

If production shows bottlenecks with extreme pages (tens of thousands of boxes), the recommended next optimization is moving recursion to in-place partitioning with `&mut [Element]`. For now, the current implementation prioritizes maintainability and algorithm traceability.

# Understanding `SemanticLabel` in XY-Cut++ (Rust Port)

This document explains the purpose of `SemanticLabel` in the `BoundingBox` trait, its mapping to traditional PDF structure, and the architecture decisions taken in this Rust port.

## 📖 Historical context: paper vs reality

The original XY-Cut++ algorithm (based on paper *arXiv:2504.10258*) defines a 4-stage pipeline. **Stage 4 (Cross-Modal Matching)** uses semantic labels (`SemanticLabel`) to decide reinsertion priority for extracted elements (like headers or images) into the main text flow.

**⚠️ Implementation note:** The official OpenDataLoader Java implementation (on which this Rust port is based) **omitted Stage 4 and semantic label usage**. Maintainers found that reading directly from PDF structure rather than OCR/vision models gave more stable and faster results.

Therefore, in this base implementation, `SemanticLabel` is exposed in the trait for compatibility and future extensions, but the main algorithm uses only X/Y coordinates and density.

---

## 🏷️ `SemanticLabel` types and PDF equivalents

If you choose to implement semantic reinsertion (paper Stage 4) or use labels for pre-sort filtering, here is the suggested mapping to PDF standards (Tagged PDF / PDF/UA):

| SemanticLabel (XY-Cut++) | PDF equivalent (Tagged PDF) | Description and role in algorithm |
| :--- | :--- | :--- |
| `CrossLayout` | `Artifact` (Pagination), `Header`, `Footer` | Elements spanning the full page (headers, footers, page numbers). They have the **highest reinsertion priority** at document edges. |
| `Title` / `Heading` | `H1`, `H2`, `H3`, `H4`... | Titles that often span multiple columns. The original paper uses them to anchor starts of new sections/columns. |
| `BodyText` / `Regular` | `P` (Paragraph), `Span` | Normal text. This is background content actively sorted by Stage 3 recursive segmentation (XY-Cut) by finding whitespace. |
| `Vision` / `Figure` | `Figure`, `Formula` | Images, charts, or complex equations. In the paper, these have medium priority and often act as blockers around which text flows. |
| `Table` | `Table`, `TR`, `TD` | Data tables. Like images, they are indivisible solid blocks. The algorithm should not cut a table in half. |

---

## 🚀 How to use `SemanticLabel` in Python

To sort elements with `xycutppy`, build a list of dictionaries with fields `id`, `x1`, `y1`, `x2`, `y2`, and `label`. The `label` field accepts any `SemanticLabel` value.

### Option A: pure geometric mode (recommended - OpenDataLoader style)

If you only want geometric ordering without semantics, assign `SemanticLabel.Regular` to all elements. The algorithm will ignore labels and operate purely on coordinates.

```python
from xycutppy import compute_order, SemanticLabel

elements = [
    {'id': 0, 'x1': 10.0, 'y1': 10.0, 'x2': 790.0, 'y2': 30.0,  'label': SemanticLabel.Regular},
    {'id': 1, 'x1': 10.0, 'y1': 50.0, 'x2': 390.0, 'y2': 100.0, 'label': SemanticLabel.Regular},
    {'id': 2, 'x1': 410.0,'y1': 50.0, 'x2': 790.0, 'y2': 100.0, 'label': SemanticLabel.Regular},
    {'id': 3, 'x1': 10.0, 'y1': 120.0,'x2': 790.0, 'y2': 200.0, 'label': SemanticLabel.Regular},
]

page_bounds = (0.0, 0.0, 800.0, 1200.0)
ordered_ids = compute_order(elements, page_bounds)
print(ordered_ids)  # e.g. [0, 1, 2, 3]
```

### Option B: mapping from YOLO / layout detection labels

If you use a detection model such as *PP-DocLayout*, *DocLayNet*, or any YOLO model trained on documents, you will get numeric class IDs. You can map them to `SemanticLabel` with a dictionary:

```python
from xycutppy import compute_order, SemanticLabel

# Typical labels from a YOLO document analysis model
YOLO_LABEL_MAP = {
    0: 'Caption',
    1: 'Footnote',
    2: 'Formula',
    3: 'List-item',
    4: 'Page-footer',
    5: 'Page-header',
    6: 'Picture',
    7: 'Section-header',
    8: 'Table',
    9: 'Text',
    10: 'Title',
}

# YOLO class name -> xycutppy SemanticLabel
YOLO_TO_SEMANTIC = {
    'Caption':        SemanticLabel.Regular,
    'Footnote':       SemanticLabel.Regular,
    'Formula':        SemanticLabel.Vision,
    'List-item':      SemanticLabel.Regular,
    'Page-footer':    SemanticLabel.CrossLayout,
    'Page-header':    SemanticLabel.CrossLayout,
    'Picture':        SemanticLabel.Vision,
    'Section-header': SemanticLabel.HorizontalTitle,
    'Table':          SemanticLabel.Vision,
    'Text':           SemanticLabel.Regular,
    'Title':          SemanticLabel.HorizontalTitle,
}

def yolo_class_to_semantic(class_id: int) -> SemanticLabel:
    label_name = YOLO_LABEL_MAP.get(class_id, 'Text')
    return YOLO_TO_SEMANTIC.get(label_name, SemanticLabel.Regular)


# Typical YOLO output: list of (class_id, x1, y1, x2, y2)
yolo_detections = [
    (5,  10.0,  0.0, 790.0,  20.0),   # Page-header  -> CrossLayout
    (10, 10.0,  30.0, 790.0,  60.0),  # Title        -> HorizontalTitle
    (9,  10.0,  70.0, 390.0, 200.0),  # Text (col 1) -> Regular
    (9,  410.0, 70.0, 790.0, 200.0),  # Text (col 2) -> Regular
    (8,  10.0, 210.0, 790.0, 400.0),  # Table        -> Vision
    (1,  10.0, 410.0, 790.0, 430.0),  # Footnote     -> Regular
    (4,  10.0, 440.0, 790.0, 460.0),  # Page-footer  -> CrossLayout
]

elements = [
    {
        'id': idx,
        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
        'label': yolo_class_to_semantic(class_id),
    }
    for idx, (class_id, x1, y1, x2, y2) in enumerate(yolo_detections)
]

page_bounds = (0.0, 0.0, 800.0, 1200.0)
ordered_ids = compute_order(elements, page_bounds)
print("Reading order:", ordered_ids)
# Expected: header (0) -> title (1) -> left-col (2) -> right-col (3) -> table (4) -> footnote (5) -> footer (6)
```

### Option C: mapping from native PDF labels (Tagged PDF)

If you extract documents from a library such as `pdfplumber` or `pymupdf` and have structural tags:

```python
from xycutppy import compute_order, SemanticLabel

PDF_TAG_TO_SEMANTIC = {
    'H1': SemanticLabel.HorizontalTitle,
    'H2': SemanticLabel.HorizontalTitle,
    'H3': SemanticLabel.HorizontalTitle,
    'P':  SemanticLabel.Regular,
    'Figure': SemanticLabel.Vision,
    'Formula': SemanticLabel.Vision,
    'Table': SemanticLabel.Vision,
    'Header': SemanticLabel.CrossLayout,
    'Footer': SemanticLabel.CrossLayout,
}

def pdf_tag_to_semantic(tag: str) -> SemanticLabel:
    return PDF_TAG_TO_SEMANTIC.get(tag, SemanticLabel.Regular)

# Blocks extracted from pymupdf / pdfplumber with structural tag
pdf_blocks = [
    {'id': 0, 'tag': 'Header', 'x1': 0.0,   'y1': 0.0,   'x2': 595.0, 'y2': 30.0},
    {'id': 1, 'tag': 'H1',     'x1': 50.0,  'y1': 40.0,  'x2': 545.0, 'y2': 70.0},
    {'id': 2, 'tag': 'P',      'x1': 50.0,  'y1': 80.0,  'x2': 280.0, 'y2': 300.0},
    {'id': 3, 'tag': 'Figure', 'x1': 300.0, 'y1': 80.0,  'x2': 545.0, 'y2': 300.0},
    {'id': 4, 'tag': 'P',      'x1': 50.0,  'y1': 310.0, 'x2': 545.0, 'y2': 450.0},
    {'id': 5, 'tag': 'Footer', 'x1': 0.0,   'y1': 810.0, 'x2': 595.0, 'y2': 842.0},
]

elements = [
    {**b, 'label': pdf_tag_to_semantic(b['tag'])}
    for b in pdf_blocks
]

page_bounds = (0.0, 0.0, 595.0, 842.0)  # A4 in points
ordered_ids = compute_order(elements, page_bounds)
print("Reading order:", ordered_ids)
```

## 🧠 Frequently asked questions

**Why does ordering fail if a header has `Regular` label?**
In the current implementation (purely geometric), the header is still detected in Stage 1 if its width exceeds `DEFAULT_BETA` (default 2.0 times max width), regardless of label. If the header is narrow, lower its `beta` coefficient.

**Should I implement paper Stage 4 (Cross-Modal Matching)?**
As advised by DataLab authors: *Do not do it unless you use vision models*. If you read native PDF, spatial coordinates are already precise enough. Mixing geometric detection with semantic weights without tuning often amplifies errors.
