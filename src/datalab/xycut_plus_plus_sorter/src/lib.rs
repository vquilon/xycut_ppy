const DEFAULT_BETA: f32 = 2.0;
const DEFAULT_DENSITY_THRESHOLD: f32 = 0.9;
const OVERLAP_THRESHOLD: f32 = 0.1;
const MIN_OVERLAP_COUNT: usize = 2;
const MIN_GAP_THRESHOLD: f32 = 5.0;
const NARROW_ELEMENT_WIDTH_RATIO: f32 = 0.1;

use log::{debug, warn};

#[derive(Debug, Clone)]
pub struct Element {
    pub id: usize,
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

impl Element {
    fn center(&self) -> (f32, f32) {
        ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)
    }
}

#[derive(Debug, Clone, Copy)]
struct CutInfo {
    position: f32,
    gap: f32,
}

#[derive(Debug, Clone, Copy)]
struct Region {
    left: f32,
    right: f32,
    bottom: f32,
    top: f32,
}

impl Region {
    fn width(self) -> f32 {
        (self.right - self.left).max(0.0)
    }
    fn area(self) -> f32 {
        self.width() * (self.top - self.bottom).max(0.0)
    }
}

#[derive(Debug, Clone, Copy)]
struct Bounds {
    left: f32,
    right: f32,
    bottom: f32,
    top: f32,
}

impl Bounds {
    fn from_element(element: &Element) -> Self {
        Self {
            left: element.x1.min(element.x2),
            right: element.x1.max(element.x2),
            // INVERSIÓN: En imágenes, el Y menor es "Arriba" y el Y mayor es "Abajo"
            top: element.y1.min(element.y2),
            bottom: element.y1.max(element.y2),
        }
    }
    fn width(self) -> f32 {
        (self.right - self.left).max(0.0)
    }
    fn area(self) -> f32 {
        self.width() * (self.top - self.bottom).max(0.0)
    }
}

pub fn sort_default(objects: &[Element]) -> Vec<Element> {
    sort(objects, DEFAULT_BETA, DEFAULT_DENSITY_THRESHOLD)
}

pub fn sort_ids_default(objects: &[Element]) -> Vec<usize> {
    sort_default(objects).into_iter().map(|e| e.id).collect()
}

pub fn sort(objects: &[Element], beta: f32, density_threshold: f32) -> Vec<Element> {
    if objects.len() <= 1 {
        return objects.to_vec();
    }

    debug!(
        target: "xycutppy::datalab",
        "[sort] {} elements (beta={}, density_threshold={})",
        objects.len(), beta, density_threshold
    );

    let valid_objects = objects.to_vec();
    let cross_layout_indices = identify_cross_layout_indices(&valid_objects, beta);
    let mut is_cross_layout = vec![false; valid_objects.len()];
    for &idx in &cross_layout_indices {
        is_cross_layout[idx] = true;
    }

    let mut cross_layout = Vec::with_capacity(cross_layout_indices.len());
    let mut remaining = Vec::with_capacity(valid_objects.len() - cross_layout_indices.len());
    for (idx, obj) in valid_objects.iter().enumerate() {
        if is_cross_layout[idx] {
            cross_layout.push(obj.clone());
        } else {
            remaining.push(obj.clone());
        }
    }

    debug!(
        target: "xycutppy::datalab",
        "[sort] cross-layout: {}, remaining: {}",
        cross_layout.len(), remaining.len()
    );

    if remaining.is_empty() {
        warn!(
            target: "xycutppy::datalab",
            "[sort] all elements are cross-layout, falling back to y-then-x sort"
        );
        return sort_by_y_then_x(&valid_objects);
    }

    let density_ratio = compute_density_ratio(&remaining);
    let prefer_horizontal_first = density_ratio > density_threshold;
    debug!(
        target: "xycutppy::datalab",
        "[sort] density_ratio={:.3}, prefer_horizontal_first={}",
        density_ratio, prefer_horizontal_first
    );
    let sorted_main = recursive_segment(&remaining, prefer_horizontal_first);
    merge_cross_layout_elements(&sorted_main, &cross_layout)
}

fn identify_cross_layout_indices(objects: &[Element], beta: f32) -> Vec<usize> {
    if objects.len() < 3 {
        return Vec::new();
    }
    let max_width = objects
        .iter()
        .map(|o| Bounds::from_element(o).width())
        .fold(0.0_f32, f32::max);
    let threshold = beta * max_width;

    objects
        .iter()
        .enumerate()
        .filter(|(idx, obj)| {
            let width = Bounds::from_element(obj).width();
            width >= threshold && has_minimum_overlaps_at(*idx, objects, MIN_OVERLAP_COUNT)
        })
        .map(|(idx, _)| idx)
        .collect()
}

fn has_minimum_overlaps_at(index: usize, objects: &[Element], min_count: usize) -> bool {
    let element = &objects[index];
    let element_bounds = Bounds::from_element(element);
    let mut overlap_count = 0usize;
    for (other_idx, other) in objects.iter().enumerate() {
        if other_idx == index {
            continue;
        }
        let overlap_ratio =
            calculate_horizontal_overlap_ratio(element_bounds, Bounds::from_element(other));
        if overlap_ratio >= OVERLAP_THRESHOLD {
            overlap_count += 1;
            if overlap_count >= min_count {
                return true;
            }
        }
    }
    false
}

fn calculate_horizontal_overlap_ratio(box1: Bounds, box2: Bounds) -> f32 {
    let overlap_left = box1.left.max(box2.left);
    let overlap_right = box1.right.min(box2.right);
    let overlap_width = (overlap_right - overlap_left).max(0.0);
    if overlap_width <= 0.0 {
        return 0.0;
    }
    let smaller_width = box1.width().min(box2.width());
    if smaller_width > 0.0 {
        overlap_width / smaller_width
    } else {
        0.0
    }
}

fn compute_density_ratio(objects: &[Element]) -> f32 {
    if objects.is_empty() {
        return 1.0;
    }
    let Some(region) = calculate_bounding_region(objects) else {
        return 1.0;
    };
    let region_area = region.area();
    if region_area <= 0.0 {
        return 1.0;
    }
    let content_area: f32 = objects
        .iter()
        .map(|obj| Bounds::from_element(obj).area())
        .sum();
    (content_area / region_area).min(1.0)
}

fn calculate_bounding_region(objects: &[Element]) -> Option<Region> {
    let first = objects.first()?;
    let first_bounds = Bounds::from_element(first);
    let mut left = first_bounds.left;
    let mut right = first_bounds.right;
    let mut bottom = first_bounds.bottom;
    let mut top = first_bounds.top;

    for obj in objects.iter().skip(1) {
        let b = Bounds::from_element(obj);
        left = left.min(b.left);
        right = right.max(b.right);
        bottom = bottom.min(b.bottom);
        top = top.max(b.top);
    }

    Some(Region {
        left,
        right,
        bottom,
        top,
    })
}

fn recursive_segment(objects: &[Element], prefer_horizontal_first: bool) -> Vec<Element> {
    if objects.len() <= 1 {
        return objects.to_vec();
    }

    let horizontal_cut = find_best_horizontal_cut_with_projection(objects);
    let vertical_cut = find_best_vertical_cut_with_projection(objects);

    let has_valid_horizontal_cut = horizontal_cut.gap >= MIN_GAP_THRESHOLD;
    let has_valid_vertical_cut = vertical_cut.gap >= MIN_GAP_THRESHOLD;
    let use_horizontal_cut = if has_valid_horizontal_cut && has_valid_vertical_cut {
        if (horizontal_cut.gap - vertical_cut.gap).abs() < f32::EPSILON {
            prefer_horizontal_first
        } else {
            horizontal_cut.gap > vertical_cut.gap
        }
    } else if has_valid_horizontal_cut {
        true
    } else if has_valid_vertical_cut {
        false
    } else {
        debug!(
            target: "xycutppy::datalab",
            "[segment] no valid cuts for {} elements, falling back to y-then-x sort",
            objects.len()
        );
        return sort_by_y_then_x(objects);
    };

    let groups = if use_horizontal_cut {
        debug!(
            target: "xycutppy::datalab",
            "[segment] horizontal cut at y={:.1} (gap={:.1}) on {} elements",
            horizontal_cut.position, horizontal_cut.gap, objects.len()
        );
        split_by_horizontal_cut(objects, horizontal_cut.position)
    } else {
        debug!(
            target: "xycutppy::datalab",
            "[segment] vertical cut at x={:.1} (gap={:.1}) on {} elements",
            vertical_cut.position, vertical_cut.gap, objects.len()
        );
        split_by_vertical_cut(objects, vertical_cut.position)
    };

    if groups.len() <= 1 {
        warn!(
            target: "xycutppy::datalab",
            "[segment] cut produced only 1 group for {} elements, falling back to y-then-x sort",
            objects.len()
        );
        return sort_by_y_then_x(objects);
    }

    debug!(
        target: "xycutppy::datalab",
        "[segment] split into {} groups: {}",
        groups.len(),
        groups.iter().map(|g| g.len().to_string()).collect::<Vec<_>>().join(", ")
    );

    groups
        .into_iter()
        .flat_map(|group| recursive_segment(&group, prefer_horizontal_first))
        .collect()
}

fn find_best_vertical_cut_with_projection(objects: &[Element]) -> CutInfo {
    if objects.len() < 2 {
        return CutInfo {
            position: 0.0,
            gap: 0.0,
        };
    }
    let edge_cut = find_vertical_cut_by_edges(objects);
    if edge_cut.gap >= MIN_GAP_THRESHOLD {
        return edge_cut;
    }

    if objects.len() >= 3 {
        if let Some(region) = calculate_bounding_region(objects) {
            let narrow_threshold = region.width() * NARROW_ELEMENT_WIDTH_RATIO;
            let filtered: Vec<Element> = objects
                .iter()
                .filter(|obj| Bounds::from_element(obj).width() >= narrow_threshold)
                .cloned()
                .collect();

            if filtered.len() >= 2 && filtered.len() < objects.len() {
                let filtered_cut = find_vertical_cut_by_edges(&filtered);
                if filtered_cut.gap > edge_cut.gap && filtered_cut.gap >= MIN_GAP_THRESHOLD {
                    return filtered_cut;
                }
            }
        }
    }
    edge_cut
}

fn find_vertical_cut_by_edges(objects: &[Element]) -> CutInfo {
    let mut sorted: Vec<&Element> = objects.iter().collect();
    sorted.sort_by(|a, b| {
        let ab = Bounds::from_element(*a);
        let bb = Bounds::from_element(*b);
        ab.left
            .total_cmp(&bb.left)
            .then_with(|| ab.right.total_cmp(&bb.right))
    });

    let mut largest_gap = 0.0_f32;
    let mut cut_position = 0.0_f32;
    let mut prev_right: Option<f32> = None;

    for obj in &sorted {
        let bounds = Bounds::from_element(*obj);
        if let Some(prev) = prev_right {
            if bounds.left > prev {
                let gap = bounds.left - prev;
                if gap > largest_gap {
                    largest_gap = gap;
                    cut_position = (prev + bounds.left) / 2.0;
                }
            }
            prev_right = Some(prev.max(bounds.right));
        } else {
            prev_right = Some(bounds.right);
        }
    }

    CutInfo {
        position: cut_position,
        gap: largest_gap,
    }
}

fn find_best_horizontal_cut_with_projection(objects: &[Element]) -> CutInfo {
    if objects.len() < 2 {
        return CutInfo {
            position: 0.0,
            gap: 0.0,
        };
    }
    let mut sorted: Vec<&Element> = objects.iter().collect();
    sorted.sort_by(|a, b| {
        let ab = Bounds::from_element(*a);
        let bb = Bounds::from_element(*b);
        // Orden natural ascendente
        ab.top.total_cmp(&bb.top).then_with(|| ab.bottom.total_cmp(&bb.bottom))
    });

    let mut largest_gap = 0.0_f32;
    let mut cut_position = 0.0_f32;
    let mut prev_bottom: Option<f32> = None;

    for obj in &sorted {
        let bounds = Bounds::from_element(*obj);
        if let Some(prev) = prev_bottom {
            // Si el inicio de la siguiente caja es mayor que el fin de la anterior, hay hueco
            if bounds.top > prev {
                let gap = bounds.top - prev;
                if gap > largest_gap {
                    largest_gap = gap;
                    cut_position = (prev + bounds.top) / 2.0;
                }
            }
            // Acumulamos el valor Y más profundo (más abajo) que hemos visto
            prev_bottom = Some(prev.max(bounds.bottom));
        } else {
            prev_bottom = Some(bounds.bottom);
        }
    }

    CutInfo {
        position: cut_position,
        gap: largest_gap,
    }
}

fn split_by_horizontal_cut(objects: &[Element], cut_y: f32) -> Vec<Vec<Element>> {
    let expected_group_size = objects.len() / 2;
    let mut above = Vec::with_capacity(expected_group_size);
    let mut below = Vec::with_capacity(expected_group_size);

    for obj in objects {
        let (_, center_y) = obj.center();
        // INVERSIÓN: Si Y es MENOR que el corte, está visualmente por encima (above)
        if center_y < cut_y {
            above.push(obj.clone());
        } else {
            below.push(obj.clone());
        }
    }

    let mut groups = Vec::new();
    if !above.is_empty() {
        groups.push(above);
    }
    if !below.is_empty() {
        groups.push(below);
    }
    groups
}

fn split_by_vertical_cut(objects: &[Element], cut_x: f32) -> Vec<Vec<Element>> {
    let expected_group_size = objects.len() / 2;
    let mut left = Vec::with_capacity(expected_group_size);
    let mut right = Vec::with_capacity(expected_group_size);

    for obj in objects {
        let (center_x, _) = obj.center();
        if center_x < cut_x {
            left.push(obj.clone());
        } else {
            right.push(obj.clone());
        }
    }

    let mut groups = Vec::new();
    if !left.is_empty() {
        groups.push(left);
    }
    if !right.is_empty() {
        groups.push(right);
    }
    groups
}

fn merge_cross_layout_elements(sorted_main: &[Element], cross_layout_elements: &[Element]) -> Vec<Element> {
    if cross_layout_elements.is_empty() {
        return sorted_main.to_vec();
    }
    if sorted_main.is_empty() {
        return sort_by_y_then_x(cross_layout_elements);
    }

    debug!(
        target: "xycutppy::datalab",
        "[merge] merging {} cross-layout elements into {} main elements",
        cross_layout_elements.len(), sorted_main.len()
    );

    let sorted_cross_layout = sort_by_y_then_x(cross_layout_elements);
    let mut result = Vec::with_capacity(sorted_main.len() + sorted_cross_layout.len());
    let mut main_index = 0usize;
    let mut cross_index = 0usize;

    while main_index < sorted_main.len() || cross_index < sorted_cross_layout.len() {
        if cross_index >= sorted_cross_layout.len() {
            result.push(sorted_main[main_index].clone());
            main_index += 1;
        } else if main_index >= sorted_main.len() {
            result.push(sorted_cross_layout[cross_index].clone());
            cross_index += 1;
        } else {
            let main_top = Bounds::from_element(&sorted_main[main_index]).top;
            let cross_top = Bounds::from_element(&sorted_cross_layout[cross_index]).top;

            // INVERSIÓN: <= significa que el elemento cruzado está más arriba en la imagen
            if cross_top <= main_top {
                result.push(sorted_cross_layout[cross_index].clone());
                cross_index += 1;
            } else {
                result.push(sorted_main[main_index].clone());
                main_index += 1;
            }
        }
    }
    debug!(
        target: "xycutppy::datalab",
        "[merge] result: {} elements total",
        result.len()
    );
    result
}

fn sort_by_y_then_x(objects: &[Element]) -> Vec<Element> {
    let mut sorted = objects.to_vec();
    sorted.sort_by(|a, b| {
        let ab = Bounds::from_element(a);
        let bb = Bounds::from_element(b);
        // Quitamos los signos negativos. Ordenamos Y menor a mayor, luego X menor a mayor
        ab.top
            .total_cmp(&bb.top)
            .then_with(|| ab.left.total_cmp(&bb.left))
    });
    sorted
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sorts_simple_two_columns() {
        let objects = vec![
            Element {
                id: 1,
                x1: 0.0,
                y1: 80.0,
                x2: 40.0,
                y2: 100.0,
            },
            Element {
                id: 2,
                x1: 60.0,
                y1: 80.0,
                x2: 100.0,
                y2: 100.0,
            },
            Element {
                id: 3,
                x1: 0.0,
                y1: 40.0,
                x2: 40.0,
                y2: 60.0,
            },
            Element {
                id: 4,
                x1: 60.0,
                y1: 40.0,
                x2: 100.0,
                y2: 60.0,
            },
        ];

        let ids = sort_ids_default(&objects);
        assert_eq!(ids, vec![1, 3, 2, 4]);
    }
}
