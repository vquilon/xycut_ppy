use pyo3::prelude::*;

use xycut_plus_plus::core::{XYCutConfig, XYCutPlusPlus};
use xycut_plus_plus::traits::{BoundingBox, SemanticLabel};



// --- MÓDULO PRINCIPAL CON INICIALIZACIÓN AUTOMÁTICA ---
#[pymodule]
#[pyo3(name = "_xycutpp_core")]
fn xycut_core(m: &Bound<'_, PyModule>) -> PyResult<()> {

    // --- ¡LA SOLUCIÓN! ---
    // Esta línea se ejecuta una sola vez cuando el módulo se importa en Python.
    // Establece el puente entre el `log` de Rust y el `logging` de Python de forma automática.
    pyo3_log::init();
    // -------------------

    // Añadimos las funciones y clases que el usuario sí necesita.
    m.add_function(wrap_pyfunction!(compute_order, m)?)?;
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
    HorizontalTitle,
    VerticalTitle,
    Vision,
    Regular,
}

impl From<PySemanticLabel> for SemanticLabel {
    fn from(label: PySemanticLabel) -> Self {
        match label {
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
    fn new(min_cut_threshold: f32, histogram_resolution_scale: f32, same_row_tolerance: f32) -> Self {
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
    // Usamos `#[pyo3(get)]` para que los atributos sean legibles desde Python si es necesario.
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
        Self { id, x1, y1, x2, y2, label }
    }
}

// -------------------
// 2. IMPLEMENTACIÓN DEL TRAIT `BoundingBox`
// -------------------

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
            PySemanticLabel::HorizontalTitle | PySemanticLabel::VerticalTitle | PySemanticLabel::Vision
        )
    }

    fn semantic_label(&self) -> SemanticLabel {
        self.label.clone().into()
    }
}

// -------------------
// 3. FUNCIÓN PRINCIPAL EXPUESTA A PYTHON
// -------------------

#[pyfunction]
#[pyo3(signature = (elements, page_bounds, config = None))]
fn compute_order(elements: Vec<PyElement>, page_bounds: (f32, f32, f32, f32), config: Option<PyXYCutConfig>) -> PyResult<Vec<usize>> {
    let rust_config = match config {
        Some(c) => XYCutConfig {
            min_cut_threshold: c.min_cut_threshold,
            histogram_resolution_scale: c.histogram_resolution_scale,
            same_row_tolerance: c.same_row_tolerance,
        },
        None => XYCutConfig::default(),
    };
    let xycut = XYCutPlusPlus::new(rust_config);
    let ordered_ids = xycut.compute_order(&elements, page_bounds.0, page_bounds.1, page_bounds.2, page_bounds.3);
    Ok(ordered_ids)
}

