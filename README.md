# XYCut++ Python Wrapper (xycutppy)

[![PyPI version](https://img.shields.io/pypi/v/xycutppy.svg)](https://pypi.org/project/xycutppy/)
[![Build Status](https://img.shields.io/github/actions/workflow/status/vquilonr/xycut_ppy/ci-cd.yml)](https://github.com/vquilonr/xycut_ppy/actions)
[![License](https://img.shields.io/badge/license-Apache--2.0%20OR%20GPL--3.0--or--later-blue.svg)](LICENSE)

Este proyecto proporciona un wrapper de Python con dos backends de lectura: `paper` (núcleo original GPL) y `datalab` (port nativo en Rust del módulo Apache-2.0 de Datalab). El wrapper permite seleccionar backend en tiempo de ejecución.

## ✨ Características

- **Alto rendimiento**: El núcleo del algoritmo está implementado en Rust para una máxima eficiencia.
- **API Pythonic**: Una capa de Python limpia y bien documentada que envuelve la lógica de Rust.
- **Configurable**: Permite ajustar los parámetros del algoritmo XYCut a través de una clase de configuración dedicada.
- **Seguridad de tipos**: Utiliza enumeraciones para las etiquetas semánticas, evitando errores por el uso de strings.
- **Fácil de construir y distribuir**: Utiliza `maturin` para una integración perfecta entre Rust y Python.
- **Selección de backend**: Puedes elegir `paper` o `datalab` por llamada o como backend por defecto.
- **Estructura multi-licencia**: El código queda separado por módulo para facilitar el cumplimiento de licencia.

## ⚖️ Licencias por módulo (Dual License)

Este repositorio usa el esquema:

`Apache-2.0 OR GPL-3.0-or-later`

Cómo aplica en la práctica:

- **Paquete pip raíz (`xycutppy`)**: se distribuye bajo **Apache-2.0**.
- **Backend/paquete `xycutppy-paper`** (código en `src/paper/xycut_plus_plus`): se distribuye bajo **GPL-3.0-or-later**.

Archivos de referencia:

- `LICENSE` (raíz): declaración formal dual-license (`Apache-2.0 OR GPL-3.0-or-later`).
- `src/datalab/xycut_plus_plus_sorter/LICENSE`: texto de **Apache-2.0**.
- `src/paper/xycut_plus_plus/LICENSE`: texto de **GPL-3.0**.

## 📦 Instalación

Una vez publicado en PyPI, el paquete se puede instalar fácilmente con `pip`:

```bash
pip install xycutppy
```

## 🚀 Uso Rápido (Quickstart)

A continuación se muestra un ejemplo básico de cómo utilizar el paquete para determinar el orden de lectura de un conjunto de elementos.

```python
from xycutppy import compute_order, SemanticLabel, XYCutConfig, set_backend

# 1. Define los elementos a ordenar.
# Cada elemento es un diccionario con id, coordenadas (x1, y1, x2, y2) y una etiqueta.
elements = [
    {'id': 0, 'x1': 10.0, 'y1': 10.0, 'x2': 200.0, 'y2': 30.0, 'label': SemanticLabel.HorizontalTitle},
    {'id': 1, 'x1': 10.0, 'y1': 50.0, 'x2': 400.0, 'y2': 100.0, 'label': SemanticLabel.Regular},
    {'id': 2, 'x1': 450.0, 'y1': 50.0, 'x2': 600.0, 'y2': 100.0, 'label': SemanticLabel.Regular},
    {'id': 3, 'x1': 10.0, 'y1': 120.0, 'x2': 600.0, 'y2': 200.0, 'label': SemanticLabel.Regular},
]

# 2. Define los límites de la página (x_min, y_min, x_max, y_max)
page_bounds = (0.0, 0.0, 800.0, 1200.0)

# 3. (Opcional) Personaliza la configuración del algoritmo.
custom_config = XYCutConfig(
    min_cut_threshold=10.0,
    same_row_tolerance=5.0
)

# 4. Selección de backend global (opcional).
set_backend("paper")  # o "datalab"

# 5. Calcula el orden de lectura.
# Si no se pasa `config`, se usarán los valores por defecto.
ordered_ids = compute_order(elements, page_bounds, config=custom_config)

# 6. O selección explícita por llamada.
ordered_ids_datalab = compute_order(elements, page_bounds, backend="datalab")

print(f"El orden de lectura de los IDs es: {ordered_ids}")
# Salida esperada: El orden de lectura de los IDs es:
```

---

## 🛠️ Guía para Desarrolladores: Compilar el Wheel desde el Código Fuente

Esta sección describe los pasos necesarios para compilar el paquete de Python (`.whl`) a partir del código fuente.

### Prerrequisitos

Asegúrate de tener instalado el siguiente software:

1.  **Python** (versión 3.8 o superior).
2.  **Rust**: La cadena de herramientas de Rust (incluyendo `rustc` y `cargo`). Puedes instalarla desde [rustup.rs](https://rustup.rs/).
3.  **Maturin**: La herramienta para construir y publicar paquetes de Python escritos en Rust.
    ```bash
    pip install maturin
    ```

### Estructura del Proyecto

El proyecto sigue una estructura con separación por backend/licencia:

```
xycut_project/
├── pyproject.toml      # Configuración del proyecto y de maturin
├── README.md
├── Cargo.toml
├── LICENSE             # Declaración dual-license: Apache-2.0 OR GPL-3.0-or-later
├── src/
│   ├── lib.rs          # Módulo Python nativo (PyO3)
│   ├── paper/xycut_plus_plus/  # Núcleo Rust GPL
│   └── datalab/
│       ├── LICENSE     # Apache-2.0
│       └── java/       # Fuente Java original mantenida como referencia/licencia
└── xycutppy/           # Código fuente del paquete Python
    └── __init__.py
```

### Pasos para Generar el Wheel

1.  **Clona el repositorio** (si aplica) y navega a la carpeta raíz del proyecto (`xycut_project/`).

2.  **Crea y activa un entorno virtual** (recomendado):
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Linux/macOS
    # venv\Scripts\activate   # En Windows
    ```

3.  **Compila el proyecto y genera el wheel**:
    Ejecuta el siguiente comando de `maturin` desde la raíz del proyecto. Este comando compilará el crate de Rust en modo `release` (optimizado) y lo empaquetará junto con el wrapper de Python en un fichero `.whl`.

    ```bash
    maturin build --release
    ```
    
    El wheel generado se guardará en la carpeta `target/wheels/`.

    > **Consejo**: Durante el desarrollo, puedes usar `maturin develop` para compilar e instalar el paquete en tu entorno virtual actual. Esto es más rápido ya que evita el paso de empaquetado y permite probar los cambios inmediatamente.

4.  **Instala el wheel localmente para probarlo**:
    Puedes instalar el fichero `.whl` que acabas de crear usando `pip`:

    ```bash
    # El nombre exacto del fichero puede variar según tu sistema operativo y versión de Python
    pip install target/wheels/xycutppy-0.0.2-*.whl
    ```

¡Y eso es todo! Ahora tienes un paquete de Python instalado localmente, listo para ser probado o distribuido.
