use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use std::sync::OnceLock;

static LOG_RESET_HANDLE: OnceLock<pyo3_log::ResetHandle> = OnceLock::new();

// Paper (GPL3) — solo disponible con --features paper
#[cfg(feature = "paper")]
use xycut_plus_plus::core::{XYCutConfig, XYCutPlusPlus};
#[cfg(feature = "paper")]
use xycut_plus_plus::traits::{BoundingBox, SemanticLabel};

use xycut_plus_plus_sorter as datalab;

// --- MÓDULO PRINCIPAL CON INICIALIZACIÓN AUTOMÁTICA ---
#[pymodule]
#[pyo3(name = "_xycutpp_core")]
fn xycut_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Inicializa el puente log→Python y guarda el handle para poder resetear la caché
    // cuando el usuario reconfigure el logging de Python en tiempo de ejecución.
    let handle = pyo3_log::init();
    let _ = LOG_RESET_HANDLE.set(handle);

    m.add_function(wrap_pyfunction!(reset_log_cache, m)?)?;
    m.add_function(wrap_pyfunction!(compute_order_datalab, m)?)?;
    #[cfg(feature = "paper")]
    m.add_function(wrap_pyfunction!(compute_order, m)?)?;
    #[cfg(feature = "paper")]
    m.add_function(wrap_pyfunction!(compute_order_paper, m)?)?;
    m.add_class::<PyXYCutConfig>()?;
    m.add_class::<PySemanticLabel>()?;
    m.add_class::<PyElement>()?;

    Ok(())
}

// -------------------
// 1. DEFINICIONES DE TIPOS PARA PYTHON
// -------------------

#[pyclass(name = "SemanticLabel", from_py_object)]
#[derive(Clone)]
enum PySemanticLabel {
    CrossLayout,
    HorizontalTitle,
    VerticalTitle,
    Vision,
    Regular,
}

#[cfg(feature = "paper")]
impl From<PySemanticLabel> for SemanticLabel {
    fn from(label: PySemanticLabel) -> Self {
        match label {
            PySemanticLabel::CrossLayout => SemanticLabel::CrossLayout,
            PySemanticLabel::HorizontalTitle => SemanticLabel::HorizontalTitle,
            PySemanticLabel::VerticalTitle => SemanticLabel::VerticalTitle,
            PySemanticLabel::Vision => SemanticLabel::Vision,
            PySemanticLabel::Regular => SemanticLabel::Regular,
        }
    }
}

#[pyclass(name = "XYCutConfig", from_py_object)]
#[derive(Clone)]
struct PyXYCutConfig {
    #[pyo3(get, set)]
    min_cut_threshold: f32,
    #[pyo3(get, set)]
    histogram_resolution_scale: f32,
    #[pyo3(get, set)]
    same_row_tolerance: f32,
}

#[pymethods]
impl PyXYCutConfig {
    #[new]
    #[pyo3(signature = (min_cut_threshold=15.0, histogram_resolution_scale=0.5, same_row_tolerance=10.0))]
    fn new(
        min_cut_threshold: f32,
        histogram_resolution_scale: f32,
        same_row_tolerance: f32,
    ) -> Self {
        Self {
            min_cut_threshold,
            histogram_resolution_scale,
            same_row_tolerance,
        }
    }
}

#[pyclass(name = "Element", from_py_object)]
#[derive(Clone)]
struct PyElement {
    #[pyo3(get)]
    id: usize,
    #[pyo3(get)]
    x1: f32,
    #[pyo3(get)]
    y1: f32,
    #[pyo3(get)]
    x2: f32,
    #[pyo3(get)]
    y2: f32,
    #[pyo3(get)]
    label: PySemanticLabel,
}

#[pymethods]
impl PyElement {
    #[new]
    fn new(id: usize, x1: f32, y1: f32, x2: f32, y2: f32, label: PySemanticLabel) -> Self {
        Self {
            id,
            x1,
            y1,
            x2,
            y2,
            label,
        }
    }
}

// -------------------
// 2. IMPLEMENTACIÓN DEL TRAIT `BoundingBox` (solo con feature paper)
// -------------------

#[cfg(feature = "paper")]
impl BoundingBox for PyElement {
    fn id(&self) -> usize {
        self.id
    }

    fn center(&self) -> (f32, f32) {
        ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)
    }

    fn bounds(&self) -> (f32, f32, f32, f32) {
        (self.x1, self.y1, self.x2, self.y2)
    }

    fn iou(&self, other: &Self) -> f32 {
        let x_overlap = (self.x2.min(other.x2) - self.x1.max(other.x1)).max(0.0);
        let y_overlap = (self.y2.min(other.y2) - self.y1.max(other.y1)).max(0.0);
        let intersection = x_overlap * y_overlap;
        let union = (self.x2 - self.x1) * (self.y2 - self.y1)
            + (other.x2 - other.x1) * (other.y2 - other.y1)
            - intersection;
        if union > 0.0 {
            intersection / union
        } else {
            0.0
        }
    }

    fn should_mask(&self) -> bool {
        matches!(
            self.label,
            PySemanticLabel::HorizontalTitle
                | PySemanticLabel::VerticalTitle
                | PySemanticLabel::Vision
        )
    }

    fn semantic_label(&self) -> SemanticLabel {
        self.label.clone().into()
    }
}

// -------------------
// 3. FUNCIONES EXPUESTAS A PYTHON
// -------------------

// compute_order_internal y sus funciones solo existen con la feature paper (GPL3)
#[cfg(feature = "paper")]
fn compute_order_internal(
    elements: Vec<PyElement>,
    page_bounds: (f32, f32, f32, f32),
    config: Option<PyXYCutConfig>,
) -> Vec<usize> {
    let rust_config = match config {
        Some(c) => XYCutConfig {
            min_cut_threshold: c.min_cut_threshold,
            histogram_resolution_scale: c.histogram_resolution_scale,
            same_row_tolerance: c.same_row_tolerance,
        },
        None => XYCutConfig::default(),
    };
    let xycut = XYCutPlusPlus::new(rust_config);
    xycut.compute_order(
        &elements,
        page_bounds.0,
        page_bounds.1,
        page_bounds.2,
        page_bounds.3,
    )
}

/// Resetea la caché de niveles de log de pyo3_log.
///
/// pyo3_log cachea los niveles de Python para evitar overhead en cada llamada.
/// Llama a esta función después de reconfigurar `logging` en Python para que
/// los cambios sean visibles en el código Rust.
#[pyfunction]
fn reset_log_cache() {
    if let Some(handle) = LOG_RESET_HANDLE.get() {
        handle.reset();
    }
}

#[cfg(feature = "paper")]
#[pyfunction]
#[pyo3(signature = (elements, page_bounds, config = None))]
fn compute_order(
    elements: Vec<PyElement>,
    page_bounds: (f32, f32, f32, f32),
    config: Option<PyXYCutConfig>,
) -> PyResult<Vec<usize>> {
    log::debug!(target: "xycutppy", "compute_order (paper): {} elements", elements.len());
    Ok(compute_order_internal(elements, page_bounds, config))
}

#[cfg(feature = "paper")]
#[pyfunction]
#[pyo3(signature = (elements, page_bounds, config = None))]
fn compute_order_paper(
    elements: Vec<PyElement>,
    page_bounds: (f32, f32, f32, f32),
    config: Option<PyXYCutConfig>,
) -> PyResult<Vec<usize>> {
    log::debug!(target: "xycutppy", "compute_order_paper: {} elements", elements.len());
    Ok(compute_order_internal(elements, page_bounds, config))
}

#[pyfunction]
#[pyo3(signature = (elements, page_bounds, config = None))]
fn compute_order_datalab(
    elements: Vec<PyElement>,
    page_bounds: (f32, f32, f32, f32),
    config: Option<PyXYCutConfig>,
) -> PyResult<Vec<usize>> {
    log::debug!(target: "xycutppy", "compute_order_datalab: {} elements", elements.len());
    let _ = page_bounds;
    let _ = config;
    let datalab_elements: Vec<datalab::Element> = elements
        .into_iter()
        .map(|e| datalab::Element {
            id: e.id,
            x1: e.x1,
            y1: e.y1,
            x2: e.x2,
            y2: e.y2,
        })
        .collect();
    Ok(datalab::sort_ids_default(&datalab_elements))
}
