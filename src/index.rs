//! Balanced hierarchical modern-Hopfield routing, contiguous exact leaf scans.
use crate::hnsw::Hnsw;
use crate::math::{normalize, select_dot, Dot};
use rayon::prelude::*;
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashSet};
use std::ops::Range;

#[derive(Clone, Copy)]
pub struct Config {
    pub leaf_capacity: usize,
    pub branching_factor: usize,
    pub beta: f32,
    pub rebuild_threshold: usize,
    pub max_tree_depth: usize,
    pub hnsw_threshold: usize,
    pub hnsw_degree: usize,
    pub ef_construction: usize,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            leaf_capacity: 128,
            branching_factor: 8,
            beta: 16.0,
            rebuild_threshold: 4096,
            max_tree_depth: usize::MAX,
            hnsw_threshold: 512,
            hnsw_degree: 16,
            ef_construction: 128,
        }
    }
}

#[derive(Clone, Copy)]
pub struct SearchOptions {
    pub top_k: usize,
    pub candidate_budget: usize,
    pub beam_width: usize,
    pub exact: bool,
    pub leaf_ef: usize, // zero forces exact scans within selected leaves
}
impl SearchOptions {
    pub fn validate(self) -> Result<(), String> {
        if self.beam_width == 0 {
            return Err("beam_width must be positive".into());
        }
        Ok(())
    }
}

struct Node {
    children: Vec<usize>,
    rows: Range<usize>,
    graph: Option<Hnsw>,
}

struct LeafTask<'a> {
    rows: Range<usize>,
    graph: Option<&'a Hnsw>,
}

#[derive(Clone, Copy, Debug)]
pub struct Hit {
    pub row: usize,
    pub score: f32,
}
impl PartialEq for Hit {
    fn eq(&self, rhs: &Self) -> bool {
        self.cmp(rhs) == Ordering::Equal
    }
}
impl Eq for Hit {}
impl PartialOrd for Hit {
    fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
        Some(self.cmp(rhs))
    }
}
impl Ord for Hit {
    fn cmp(&self, rhs: &Self) -> Ordering {
        self.score
            .total_cmp(&rhs.score)
            .then_with(|| rhs.row.cmp(&self.row))
    }
}
pub struct SearchResult {
    pub hits: Vec<Hit>,
    pub vectors_scored: usize,
    pub nodes_expanded: usize,
}
#[derive(Clone, Copy)]
struct Route {
    node: usize,
    log_mass: f64,
}
fn route_order(a: &Route, b: &Route) -> Ordering {
    b.log_mass
        .total_cmp(&a.log_mass)
        .then_with(|| a.node.cmp(&b.node))
}

pub struct Index {
    pub dim: usize,
    config: Config,
    ids: Vec<String>,
    known_ids: HashSet<String>,
    vectors: Vec<f32>,
    nodes: Vec<Node>,
    centers: Vec<f32>,
    indexed_len: usize,
    min_leaf_size: usize,
    dot: Dot,
}
impl Index {
    pub fn new(dim: usize, config: Config) -> Result<Self, String> {
        if dim == 0 || config.leaf_capacity == 0 || config.rebuild_threshold == 0 {
            return Err("dimension, leaf_capacity and rebuild_threshold must be positive".into());
        }
        if !(2..=32).contains(&config.branching_factor) {
            return Err("branching_factor must be between 2 and 32".into());
        }
        if !(2..=64).contains(&config.hnsw_degree)
            || config.ef_construction < config.hnsw_degree * 2
            || config.hnsw_threshold < 2
        {
            return Err(
                "HNSW requires degree in [2, 64], ef_construction >= 2*degree, threshold >= 2"
                    .into(),
            );
        }
        if !config.beta.is_finite() || config.beta <= 0.0 || config.beta > 1000.0 {
            return Err("beta must be finite and in (0, 1000]".into());
        }
        Ok(Self {
            dim,
            config,
            ids: Vec::new(),
            known_ids: HashSet::new(),
            vectors: Vec::new(),
            nodes: Vec::new(),
            centers: Vec::new(),
            indexed_len: 0,
            min_leaf_size: 1,
            dot: select_dot(),
        })
    }
    pub fn len(&self) -> usize {
        self.ids.len()
    }
    pub fn pending_count(&self) -> usize {
        self.len() - self.indexed_len
    }
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
    pub fn graph_count(&self) -> usize {
        self.nodes
            .iter()
            .filter(|node| node.graph.is_some())
            .count()
    }
    fn vector(&self, row: usize) -> &[f32] {
        &self.vectors[row * self.dim..(row + 1) * self.dim]
    }
    fn center(&self, node: usize) -> &[f32] {
        &self.centers[node * self.dim..(node + 1) * self.dim]
    }

    pub fn insert(
        &mut self,
        ids: Vec<String>,
        mut vectors: Vec<f32>,
        auto_rebuild: bool,
    ) -> Result<(), String> {
        if ids.len().checked_mul(self.dim) != Some(vectors.len()) {
            return Err("vector shape does not match IDs and dimension".into());
        }
        let mut incoming = HashSet::with_capacity(ids.len());
        for id in &ids {
            if id.is_empty() || self.known_ids.contains(id) || !incoming.insert(id.as_str()) {
                return Err("IDs must be nonempty and unique; insertion is not an upsert".into());
            }
        }
        // Validate the entire batch before mutation.
        for row in vectors.chunks_mut(self.dim) {
            normalize(row)?;
        }
        self.known_ids.extend(ids.iter().cloned());
        self.ids.extend(ids);
        self.vectors.extend(vectors);
        if auto_rebuild
            && (self.nodes.is_empty() || self.pending_count() >= self.config.rebuild_threshold)
        {
            self.rebuild();
        }
        Ok(())
    }

    pub fn rebuild(&mut self) {
        if self.pending_count() == 0 {
            return;
        }
        let mut order: Vec<usize> = (0..self.len()).collect();
        self.nodes.clear();
        self.centers.clear();
        self.min_leaf_size = usize::MAX;
        self.build_node(&mut order, 0, 0);
        // Pack leaves sequentially; no query-time vector gathers.
        let mut packed = Vec::with_capacity(self.vectors.len());
        let mut old_ids = std::mem::take(&mut self.ids);
        self.ids.reserve(order.len());
        for row in order {
            packed.extend_from_slice(self.vector(row));
            self.ids.push(std::mem::take(&mut old_ids[row]));
        }
        self.vectors = packed;
        self.indexed_len = self.len();
        // Each graph borrows the packed vectors at search time; no vector copy.
        let dim = self.dim;
        let dot = self.dot;
        let vectors = &self.vectors;
        let config = self.config;
        self.nodes.par_iter_mut().for_each(|node| {
            if node.children.is_empty() && node.rows.len() >= config.hnsw_threshold {
                node.graph = Some(Hnsw::build(
                    &vectors[node.rows.start * dim..node.rows.end * dim],
                    dim,
                    dot,
                    config.hnsw_degree,
                    config.ef_construction,
                ));
            }
        });
    }

    fn mean(&self, rows: &[usize]) -> Vec<f32> {
        let mut sum = vec![0.0f64; self.dim];
        for &row in rows {
            for (s, &x) in sum.iter_mut().zip(self.vector(row)) {
                *s += x as f64;
            }
        }
        let norm = sum.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-12 {
            return self.vector(rows[0]).to_vec();
        }
        sum.into_iter().map(|x| (x / norm) as f32).collect()
    }

    /// Balanced spherical two-means. Median assignment guarantees progress
    /// for duplicates, antipodal inputs, and collapsed distributions.
    fn split(&self, rows: &mut [usize]) {
        let farthest = |anchor| {
            *rows
                .iter()
                .min_by(|&&a, &&b| {
                    (self.dot)(self.vector(anchor), self.vector(a))
                        .total_cmp(&(self.dot)(self.vector(anchor), self.vector(b)))
                        .then_with(|| a.cmp(&b))
                })
                .unwrap()
        };
        let a = farthest(rows[0]);
        let b = farthest(a);
        let mut left = self.vector(a).to_vec();
        let mut right = self.vector(b).to_vec();
        let mid = rows.len() / 2;
        let mut scored = Vec::with_capacity(rows.len());
        for iteration in 0..4 {
            let direction: Vec<f32> = left.iter().zip(&right).map(|(a, b)| a - b).collect();
            scored.clear();
            scored.extend(
                rows.iter()
                    .map(|&row| ((self.dot)(self.vector(row), &direction), row)),
            );
            scored.select_nth_unstable_by(mid, |a, b| {
                b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1))
            });
            for (row, &(_, id)) in rows.iter_mut().zip(&scored) {
                *row = id;
            }
            if iteration < 3 {
                left = self.mean(&rows[..mid]);
                right = self.mean(&rows[mid..]);
            }
        }
    }

    fn build_node(&mut self, rows: &mut [usize], offset: usize, depth: usize) -> usize {
        let node = self.nodes.len();
        let center = self.mean(rows);
        self.centers.extend(center);
        self.nodes.push(Node {
            children: Vec::new(),
            rows: offset..offset + rows.len(),
            graph: None,
        });
        if rows.len() <= self.config.leaf_capacity || depth >= self.config.max_tree_depth {
            self.min_leaf_size = self.min_leaf_size.min(rows.len());
            return node;
        }
        let mut groups = vec![0..rows.len()];
        while groups.len() < self.config.branching_factor {
            let (which, largest) = groups
                .iter()
                .enumerate()
                .max_by_key(|(_, r)| r.len())
                .unwrap();
            if largest.len() <= self.config.leaf_capacity {
                break;
            }
            let range = largest.clone();
            self.split(&mut rows[range.clone()]);
            let mid = range.start + range.len() / 2;
            groups[which] = range.start..mid;
            groups.push(mid..range.end);
        }
        groups.sort_unstable_by_key(|r| r.start);
        let children: Vec<_> = groups
            .into_iter()
            .map(|r| {
                let start = r.start;
                self.build_node(&mut rows[r], offset + start, depth + 1)
            })
            .collect();
        self.nodes[node].children = children;
        node
    }

    pub fn search(&self, query: &[f32], options: SearchOptions) -> Result<SearchResult, String> {
        options.validate()?;
        if query.len() != self.dim {
            return Err("query has incorrect embedding dimension".into());
        }
        let mut q = query.to_vec();
        normalize(&mut q)?;
        let k = options.top_k.min(self.len());
        if k == 0 {
            return Ok(SearchResult {
                hits: Vec::new(),
                vectors_scored: 0,
                nodes_expanded: 0,
            });
        }
        if options.exact {
            return Ok(SearchResult {
                hits: self.scan(&q, 0..self.len(), k),
                vectors_scored: self.len(),
                nodes_expanded: 0,
            });
        }
        let budget = options.candidate_budget.max(k).min(self.indexed_len);
        // Widen enough to fill requested k even with tiny leaves.
        let required_width = k / self.min_leaf_size + usize::from(k % self.min_leaf_size != 0);
        let width = options
            .beam_width
            .max(required_width)
            .min(self.indexed_len.max(1));
        let mut beam = if self.nodes.is_empty() {
            Vec::new()
        } else {
            vec![Route {
                node: 0,
                log_mass: 0.0,
            }]
        };
        let mut expanded = 0;
        let mut next = Vec::new();
        loop {
            next.clear();
            let mut branched = false;
            for route in &beam {
                let children = &self.nodes[route.node].children;
                if children.is_empty() {
                    next.push(*route);
                    continue;
                }
                expanded += 1;
                branched = true;
                // Modern Hopfield association weights p = softmax(beta M q).
                // Preserve q; recurrent attraction can erase NN information.
                let mut logits = [0.0f64; 32];
                let mut max = f64::NEG_INFINITY;
                for (j, &child) in children.iter().enumerate() {
                    logits[j] = self.config.beta as f64 * (self.dot)(&q, self.center(child)) as f64;
                    max = max.max(logits[j]);
                }
                let log_z = max
                    + logits[..children.len()]
                        .iter()
                        .map(|s| (s - max).exp())
                        .sum::<f64>()
                        .ln();
                for (j, &child) in children.iter().enumerate() {
                    next.push(Route {
                        node: child,
                        log_mass: route.log_mass + logits[j] - log_z,
                    });
                }
            }
            if !branched {
                break;
            }
            if next.len() > width {
                next.select_nth_unstable_by(width, route_order);
                next.truncate(width);
            }
            std::mem::swap(&mut beam, &mut next);
        }
        beam.sort_unstable_by(route_order);
        let mut tasks = Vec::with_capacity(beam.len() + 1);
        let mut candidates = 0;
        for route in beam {
            if candidates >= budget {
                break;
            }
            let range = self.nodes[route.node].rows.clone();
            candidates += range.len();
            tasks.push(LeafTask {
                rows: range,
                graph: self.nodes[route.node].graph.as_ref(),
            });
        }
        // Pending inserts are always scanned, regardless of beam pruning.
        if self.pending_count() != 0 {
            tasks.push(LeafTask {
                rows: self.indexed_len..self.len(),
                graph: None,
            });
        }
        let covered_rows = candidates + self.pending_count();
        let mut vectors_scored = 0;
        let mut heap = BinaryHeap::with_capacity(k);
        // Parallelize only substantial single-query work. Batch queries already
        // occupy Rayon workers, and tiny per-leaf tasks cost more than scoring.
        let workers = if covered_rows >= 4096
            && covered_rows.saturating_mul(self.dim) >= 1_048_576
            && tasks.len() > 1
            && rayon::current_thread_index().is_none()
        {
            rayon::current_num_threads().min(covered_rows / 2048)
        } else {
            1
        };
        if workers > 1 {
            let group_size = (tasks.len() + workers - 1) / workers;
            let local: Vec<_> = tasks
                .par_chunks(group_size)
                .map(|group| {
                    let mut local_heap = BinaryHeap::with_capacity(k);
                    let mut evaluations = 0;
                    for task in group {
                        evaluations +=
                            self.score_leaf(&q, task, k, options.leaf_ef, &mut local_heap);
                    }
                    (finish(local_heap), evaluations)
                })
                .collect();
            for (hits, evaluations) in local {
                vectors_scored += evaluations;
                for hit in hits {
                    offer(&mut heap, hit, k);
                }
            }
        } else {
            for task in &tasks {
                vectors_scored += self.score_leaf(&q, task, k, options.leaf_ef, &mut heap);
            }
        }
        Ok(SearchResult {
            hits: finish(heap),
            vectors_scored,
            nodes_expanded: expanded,
        })
    }

    fn score_leaf(
        &self,
        query: &[f32],
        task: &LeafTask<'_>,
        k: usize,
        ef: usize,
        heap: &mut BinaryHeap<Reverse<Hit>>,
    ) -> usize {
        let mut evaluations = 0;
        if ef > 0 && ef.max(k) < task.rows.len() {
            if let Some(graph) = task.graph {
                let (hits, work) = graph.search(
                    &self.vectors[task.rows.start * self.dim..task.rows.end * self.dim],
                    self.dim,
                    self.dot,
                    query,
                    k,
                    ef,
                );
                evaluations += work;
                if hits.len() == k.min(task.rows.len()) {
                    for hit in hits {
                        offer(
                            heap,
                            Hit {
                                row: hit.row + task.rows.start,
                                score: hit.score,
                            },
                            k,
                        );
                    }
                    return evaluations;
                }
                // A disconnected graph must never silently underfill top-k.
            }
        }
        for row in task.rows.clone() {
            offer(heap, self.score(query, row), k);
        }
        evaluations + task.rows.len()
    }

    fn score(&self, query: &[f32], row: usize) -> Hit {
        Hit {
            row,
            score: (self.dot)(query, self.vector(row)).clamp(-1.0, 1.0),
        }
    }
    fn scan(&self, query: &[f32], rows: Range<usize>, k: usize) -> Vec<Hit> {
        let mut heap = BinaryHeap::with_capacity(k.min(rows.len()));
        for row in rows {
            offer(&mut heap, self.score(query, row), k);
        }
        finish(heap)
    }
    pub fn named(&self, hits: Vec<Hit>) -> Vec<(String, f32)> {
        hits.into_iter()
            .map(|h| (self.ids[h.row].clone(), h.score))
            .collect()
    }
}

fn offer(heap: &mut BinaryHeap<Reverse<Hit>>, hit: Hit, k: usize) {
    if heap.len() < k {
        heap.push(Reverse(hit));
    } else if hit > heap.peek().unwrap().0 {
        *heap.peek_mut().unwrap() = Reverse(hit);
    }
}
fn finish(heap: BinaryHeap<Reverse<Hit>>) -> Vec<Hit> {
    heap.into_sorted_vec().into_iter().map(|h| h.0).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture(n: usize, d: usize) -> Index {
        let mut index = Index::new(
            d,
            Config {
                leaf_capacity: 7,
                branching_factor: 4,
                ..Config::default()
            },
        )
        .unwrap();
        let data = (0..n * d)
            .map(|i| ((i * 79 + i * i * 13) as f32 * 0.013).sin())
            .collect();
        index
            .insert((0..n).map(|i| format!("id{i}")).collect(), data, true)
            .unwrap();
        index
    }
    fn options(k: usize) -> SearchOptions {
        SearchOptions {
            top_k: k,
            candidate_budget: usize::MAX,
            beam_width: usize::MAX,
            exact: false,
            leaf_ef: 64,
        }
    }
    #[test]
    fn exhaustive_beam_matches_independent_cosine_oracle() {
        let index = fixture(213, 19);
        let q: Vec<_> = (0..19).map(|i| (i as f32 * 1.7).cos()).collect();
        let norm = q.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let mut oracle: Vec<_> = (0..index.len())
            .map(|row| {
                let v = index.vector(row);
                let vn = v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
                let score = q
                    .iter()
                    .zip(v)
                    .map(|(a, b)| *a as f64 * *b as f64)
                    .sum::<f64>()
                    / (norm * vn);
                (row, score)
            })
            .collect();
        oracle.sort_by(|a, b| b.1.total_cmp(&a.1));
        for exact in [false, true] {
            let result = index
                .search(
                    &q,
                    SearchOptions {
                        exact,
                        ..options(17)
                    },
                )
                .unwrap();
            assert_eq!(result.vectors_scored, index.len());
            for (hit, expected) in result.hits.iter().zip(&oracle) {
                assert_eq!(hit.row, expected.0);
                assert!((hit.score as f64 - expected.1).abs() < 1e-5);
            }
        }
    }
    #[test]
    fn pending_insert_visible_and_rebuild_preserves_results() {
        let mut index = fixture(100, 9);
        let q = vec![1.0; 9];
        index.insert(vec!["new".into()], q.clone(), false).unwrap();
        assert_eq!(index.pending_count(), 1);
        let before = index.named(index.search(&q, options(10)).unwrap().hits);
        assert_eq!(before[0].0, "new");
        index.rebuild();
        let after = index.named(index.search(&q, options(10)).unwrap().hits);
        assert_eq!(before, after);
    }
    #[test]
    fn invalid_batch_is_atomic() {
        let mut index = fixture(10, 3);
        for (ids, data) in [
            (vec!["id0".into()], vec![1.0; 3]),
            (vec!["x".into(), "x".into()], vec![1.0; 6]),
            (
                vec!["x".into(), "y".into()],
                vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0],
            ),
            (vec!["x".into()], vec![f32::NAN; 3]),
        ] {
            assert!(index.insert(ids, data, true).is_err());
            assert_eq!(index.len(), 10);
        }
    }
    #[test]
    fn degenerate_tree_terminates_and_fills_k() {
        let mut index = Index::new(
            1,
            Config {
                leaf_capacity: 1,
                ..Config::default()
            },
        )
        .unwrap();
        index
            .insert(
                (0..137).map(|i| i.to_string()).collect(),
                vec![1.0; 137],
                true,
            )
            .unwrap();
        for k in [0, 1, 25, 137, 200] {
            let result = index
                .search(
                    &[1.0],
                    SearchOptions {
                        beam_width: 1,
                        candidate_budget: 0,
                        ..options(k)
                    },
                )
                .unwrap();
            assert_eq!(result.hits.len(), k.min(137));
            assert!(result.hits.iter().all(|h| h.score == 1.0));
        }
        assert!(index
            .nodes
            .iter()
            .all(|n| !n.children.is_empty() || n.rows.len() == 1));
    }
    #[test]
    fn bounded_beam_avoids_full_scan() {
        let index = fixture(4000, 11);
        let result = index
            .search(
                index.vector(73),
                SearchOptions {
                    top_k: 5,
                    candidate_budget: 50,
                    beam_width: 8,
                    exact: false,
                    leaf_ef: 64,
                },
            )
            .unwrap();
        assert_eq!(result.hits.len(), 5);
        assert!(result.vectors_scored < 100);
        assert!(result.nodes_expanded < index.node_count());
    }
    #[test]
    fn empty_and_invalid_queries() {
        let index = fixture(0, 3);
        assert!(index
            .search(&[1.0, 0.0, 0.0], options(5))
            .unwrap()
            .hits
            .is_empty());
        assert!(index.search(&[1.0], options(5)).is_err());
        assert!(index.search(&[0.0; 3], options(5)).is_err());
        assert!(index
            .search(
                &[1.0; 3],
                SearchOptions {
                    beam_width: 0,
                    ..options(5)
                }
            )
            .is_err());
    }
}
