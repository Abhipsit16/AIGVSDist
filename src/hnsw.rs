//! Immutable per-leaf HNSW. Vectors are borrowed from the packed index arena.
//! Upper layers use greedy descent; layer zero uses bounded best-first search.
use crate::index::Hit;
use crate::math::Dot;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

pub struct Hnsw {
    links: Vec<Vec<Vec<usize>>>, // local row -> layer -> neighbors
    entry: usize,
    top_level: usize,
    degree: usize,
}

struct Space<'a> {
    data: &'a [f32],
    dim: usize,
    dot: Dot,
}
impl Space<'_> {
    fn vector(&self, row: usize) -> &[f32] {
        &self.data[row * self.dim..(row + 1) * self.dim]
    }
    fn hit(&self, query: &[f32], row: usize) -> Hit {
        Hit {
            row,
            score: (self.dot)(query, self.vector(row)).clamp(-1.0, 1.0),
        }
    }
}

impl Hnsw {
    pub fn build(data: &[f32], dim: usize, dot: Dot, degree: usize, ef: usize) -> Self {
        let space = Space { data, dim, dot };
        let n = data.len() / dim;
        let mut graph = Self {
            links: Vec::with_capacity(n),
            entry: 0,
            top_level: 0,
            degree,
        };
        let mut state = 0x243f6a8885a308d3u64;
        for row in 0..n {
            // Geometric level distribution, P(level >= l) = degree^(-l).
            let mut level = 0;
            while level < 16 && random(&mut state) % degree as u64 == 0 {
                level += 1;
            }
            graph.links.push(vec![Vec::new(); level + 1]);
            if row == 0 {
                graph.top_level = level;
                continue;
            }
            let query = space.vector(row);
            let mut entry = space.hit(query, graph.entry);
            let mut evaluations = 0;
            for layer in ((level + 1)..=graph.top_level).rev() {
                entry = graph.greedy(&space, query, entry, layer, &mut evaluations);
            }
            for layer in (0..=level.min(graph.top_level)).rev() {
                let candidates = graph.layer_search(
                    &space,
                    query,
                    entry,
                    ef.max(degree),
                    layer,
                    &mut evaluations,
                );
                entry = candidates[0];
                let limit = graph.limit(layer);
                let selected = graph.select_neighbors(&space, candidates, limit);
                graph.links[row][layer] = selected.clone();
                for neighbor in selected {
                    graph.links[neighbor][layer].push(row);
                    if graph.links[neighbor][layer].len() > limit {
                        let candidates = graph.links[neighbor][layer]
                            .iter()
                            .map(|&other| space.hit(space.vector(neighbor), other))
                            .collect();
                        graph.links[neighbor][layer] =
                            graph.select_neighbors(&space, candidates, limit);
                    }
                }
            }
            if level > graph.top_level {
                graph.entry = row;
                graph.top_level = level;
            }
        }
        graph
    }

    fn limit(&self, layer: usize) -> usize {
        if layer == 0 {
            self.degree * 2
        } else {
            self.degree
        }
    }

    /// Diversify outgoing edges rather than keeping only a tight local clique.
    fn select_neighbors(
        &self,
        space: &Space<'_>,
        mut candidates: Vec<Hit>,
        limit: usize,
    ) -> Vec<usize> {
        candidates.sort_unstable_by(|a, b| b.cmp(a));
        let mut selected = Vec::with_capacity(limit);
        let mut rejected = Vec::new();
        for candidate in candidates {
            if selected.iter().all(|&chosen| {
                (space.dot)(space.vector(candidate.row), space.vector(chosen)) < candidate.score
            }) {
                selected.push(candidate.row);
                if selected.len() == limit {
                    return selected;
                }
            } else {
                rejected.push(candidate.row);
            }
        }
        // Keep available connectivity for tiny or degenerate partitions.
        selected.extend(rejected.into_iter().take(limit - selected.len()));
        selected
    }

    fn greedy(
        &self,
        space: &Space<'_>,
        query: &[f32],
        mut best: Hit,
        layer: usize,
        evaluations: &mut usize,
    ) -> Hit {
        loop {
            let previous = best.row;
            for &neighbor in &self.links[previous][layer] {
                let candidate = space.hit(query, neighbor);
                *evaluations += 1;
                if candidate > best {
                    best = candidate;
                }
            }
            if best.row == previous {
                return best;
            }
        }
    }

    fn layer_search(
        &self,
        space: &Space<'_>,
        query: &[f32],
        entry: Hit,
        ef: usize,
        layer: usize,
        evaluations: &mut usize,
    ) -> Vec<Hit> {
        let ef = ef.min(self.links.len()).max(1);
        let mut frontier = BinaryHeap::new();
        let mut retained = BinaryHeap::with_capacity(ef);
        // A compact query-local bitset avoids locks and hashing. Scratch
        // initialization is O(leaf_size / 64), separate from distance work.
        let mut visited = vec![0u64; (self.links.len() + 63) / 64];
        visited[entry.row / 64] |= 1u64 << (entry.row % 64);
        frontier.push(entry);
        retained.push(Reverse(entry));
        while let Some(best) = frontier.pop() {
            if retained.len() == ef && best < retained.peek().unwrap().0 {
                break;
            }
            for &neighbor in &self.links[best.row][layer] {
                let mask = 1u64 << (neighbor % 64);
                let word = &mut visited[neighbor / 64];
                if *word & mask != 0 {
                    continue;
                }
                *word |= mask;
                let candidate = space.hit(query, neighbor);
                *evaluations += 1;
                if retained.len() < ef || candidate > retained.peek().unwrap().0 {
                    frontier.push(candidate);
                    retained.push(Reverse(candidate));
                    if retained.len() > ef {
                        retained.pop();
                    }
                }
            }
        }
        retained
            .into_sorted_vec()
            .into_iter()
            .map(|h| h.0)
            .collect()
    }

    pub fn search(
        &self,
        data: &[f32],
        dim: usize,
        dot: Dot,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> (Vec<Hit>, usize) {
        let space = Space { data, dim, dot };
        let mut evaluations = 1;
        let mut entry = space.hit(query, self.entry);
        for layer in (1..=self.top_level).rev() {
            entry = self.greedy(&space, query, entry, layer, &mut evaluations);
        }
        let mut hits = self.layer_search(&space, query, entry, ef.max(k), 0, &mut evaluations);
        hits.truncate(k);
        (hits, evaluations)
    }
}

fn random(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{normalize, select_dot};
    #[test]
    fn layers_respect_degree_and_retrieve_self() {
        let dim = 13;
        let mut state = 12345;
        let mut vectors: Vec<_> = (0..600 * dim)
            .map(|_| (random(&mut state) >> 40) as f32 / 8_388_608.0 - 1.0)
            .collect();
        for row in vectors.chunks_mut(dim) {
            normalize(row).unwrap();
        }
        let graph = Hnsw::build(&vectors, dim, select_dot(), 12, 100);
        assert!(graph.top_level > 0);
        for (row, layers) in graph.links.iter().enumerate() {
            for (layer, neighbors) in layers.iter().enumerate() {
                assert!(neighbors.len() <= graph.limit(layer));
                assert!(neighbors
                    .iter()
                    .all(|&n| n != row && graph.links[n].len() > layer));
            }
        }
        for row in (0..600).step_by(17) {
            let (hits, _) = graph.search(
                &vectors,
                dim,
                select_dot(),
                &vectors[row * dim..(row + 1) * dim],
                1,
                64,
            );
            assert_eq!(hits[0].row, row);
        }
    }
}
