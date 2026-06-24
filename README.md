# XYCut++ Python Wrapper (xycutppy)

[![PyPI version](https://img.shields.io/pypi/v/xycutppy.svg)](https://pypi.org/project/xycutppy/)
[![Publish xycutppy to PyPI](https://github.com/vquilon/xycut_ppy/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/vquilon/xycut_ppy/actions/workflows/ci-cd.yml)
[![Publish xycutppy-paper to PyPI](https://github.com/vquilon/xycut_ppy/actions/workflows/ci-cd-paper.yml/badge.svg)](https://github.com/vquilon/xycut_ppy/actions/workflows/ci-cd-paper.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0%20OR%20GPL--3.0--or--later-blue.svg)](LICENSE)

This project provides a Python wrapper with two reading backends: `paper` (original GPL core) and `datalab` (native Rust port of Datalab's Apache-2.0 module). The wrapper lets you choose the backend at runtime.

## ✨ Features

- **High performance**: The algorithm core is implemented in Rust for maximum efficiency.
- **Pythonic API**: A clean, well-documented Python layer wrapping the Rust logic.
- **Configurable**: Tune XYCut parameters through a dedicated configuration class.
- **Type-safe labels**: Uses enums for semantic labels to avoid string-based errors.
- **Easy build and distribution**: Uses `maturin` for seamless Rust/Python integration.
- **Backend selection**: Choose `paper` or `datalab` per call or as a global default.
- **Multi-license structure**: Code is split by module to simplify license compliance.

## ⚖️ Module licensing (Dual License)

This repository uses:

`Apache-2.0 OR GPL-3.0-or-later`

How this applies in practice:

- **Root pip package (`xycutppy`)**: distributed under **Apache-2.0**.
- **`xycutppy-paper` backend/package** (code in `src/paper/xycut_plus_plus`): distributed under **GPL-3.0-or-later**.

Reference files:

- `LICENSE` (root): formal dual-license declaration (`Apache-2.0 OR GPL-3.0-or-later`).
- `src/datalab/xycut_plus_plus_sorter/LICENSE`: **Apache-2.0** text.
- `src/paper/xycut_plus_plus/LICENSE`: **GPL-3.0** text.

## 📦 Installation

Once published to PyPI, the package can be installed with `pip`:

```bash
pip install xycutppy
```

## 🚀 Quickstart

Below is a basic example showing how to use the package to determine the reading order of a set of elements.

```python
from xycutppy import compute_order, SemanticLabel, XYCutConfig, set_backend

# 1. Define the elements to sort.
# Each element is a dictionary with id, coordinates (x1, y1, x2, y2), and a label.
elements = [
    {'id': 0, 'x1': 10.0, 'y1': 10.0, 'x2': 200.0, 'y2': 30.0, 'label': SemanticLabel.HorizontalTitle},
    {'id': 1, 'x1': 10.0, 'y1': 50.0, 'x2': 400.0, 'y2': 100.0, 'label': SemanticLabel.Regular},
    {'id': 2, 'x1': 450.0, 'y1': 50.0, 'x2': 600.0, 'y2': 100.0, 'label': SemanticLabel.Regular},
    {'id': 3, 'x1': 10.0, 'y1': 120.0, 'x2': 600.0, 'y2': 200.0, 'label': SemanticLabel.Regular},
]

# 2. Define page bounds (x_min, y_min, x_max, y_max)
page_bounds = (0.0, 0.0, 800.0, 1200.0)

# 3. (Optional) Customize algorithm configuration.
custom_config = XYCutConfig(
    min_cut_threshold=10.0,
    same_row_tolerance=5.0
)

# 4. Global backend selection (optional).
set_backend("paper")  # or "datalab"

# 5. Compute reading order.
# If `config` is not passed, defaults are used.
ordered_ids = compute_order(elements, page_bounds, config=custom_config)

# 6. Or explicit per-call backend selection.
ordered_ids_datalab = compute_order(elements, page_bounds, backend="datalab")

print(f"Reading order IDs: {ordered_ids}")
# Expected output: Reading order IDs:
```

## 🧪 Visual scenarios tests (`tests/scenarios.py`)

Run the visual scenario suite to generate annotated PNG outputs for every available backend:

```bash
cd tests
python scenarios.py
```

Generated images are saved in:

```text
tests/output_examples/
```

For a gallery with all example outputs rendered inline, see:

- [tests/README.md](tests/README.md)

---

## 🛠️ Developer Guide: Build the Wheel from Source

This section describes how to compile the Python package (`.whl`) from source code.

### Prerequisites

Make sure you have the following installed:

1. **Python** (version 3.8+).
2. **Rust**: Rust toolchain (including `rustc` and `cargo`). Install from [rustup.rs](https://rustup.rs/).
3. **Maturin**: Tool to build and publish Rust-based Python packages.
   ```bash
   pip install maturin
   ```

### Project structure

The project follows a backend/license-separated structure:

```
xycut_project/
├── pyproject.toml      # Project and maturin configuration
├── README.md
├── Cargo.toml
├── LICENSE             # Dual-license declaration: Apache-2.0 OR GPL-3.0-or-later
├── src/
│   ├── lib.rs          # Native Python module (PyO3)
│   ├── paper/xycut_plus_plus/  # GPL Rust core
│   └── datalab/
│       ├── LICENSE     # Apache-2.0
│       └── java/       # Original Java source kept as reference/license
└── xycutppy/           # Python package source code
    └── __init__.py
```

### Steps to generate the wheel

1. **Clone the repository** (if applicable) and move to the project root (`xycut_project/`).

2. **Create and activate a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # venv\Scripts\activate   # Windows
   ```

3. **Build the project and generate the wheel**:
   Run the following `maturin` command from the project root. It compiles the Rust crate in `release` mode and packages it with the Python wrapper into a `.whl` file.

   ```bash
   maturin build --release
   ```

   The generated wheel is stored in `target/wheels/`.

   > **Tip**: During development, you can run `maturin develop` to build and install into the active virtual environment. It is faster because it skips packaging and lets you test changes immediately.

4. **Install the wheel locally for testing**:
   Install the `.whl` you just created with `pip`:

   ```bash
   # Exact filename may vary by OS and Python version
   pip install target/wheels/xycutppy-0.0.2-*.whl
   ```

That's it. You now have a locally installed Python package ready for testing or distribution.
