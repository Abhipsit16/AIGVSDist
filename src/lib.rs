//! Python boundary; all indexing and retrieval live in the Rust core.
mod hnsw;
mod index;
mod math;

use index::{Config, Index, SearchOptions};
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

fn value_error(message: String) -> PyErr {
    PyValueError::new_err(message)
}

#[pyclass]
struct AIGVSEngine {
    index: Index,
}

#[pymethods]
impl AIGVSEngine {
    #[new]
    #[pyo3(signature = (embedding_dim, projection_matrix=None, *, leaf_capacity=128, branching_factor=8, beta=16.0, rebuild_threshold=4096, max_tree_depth=None, leaf_backend="auto", hnsw_threshold=512, hnsw_degree=16, ef_construction=128))]
    fn new(
        embedding_dim: usize,
        projection_matrix: Option<PyReadonlyArray2<f32>>,
        leaf_capacity: usize,
        branching_factor: usize,
        beta: f32,
        rebuild_threshold: usize,
        max_tree_depth: Option<usize>,
        leaf_backend: &str,
        hnsw_threshold: usize,
        hnsw_degree: usize,
        ef_construction: usize,
    ) -> PyResult<Self> {
        // Legacy projection argument is accepted but unused by the tree.
        if let Some(projection) = projection_matrix {
            if projection.as_array().nrows() != embedding_dim {
                return Err(PyValueError::new_err(
                    "projection rows must equal embedding_dim",
                ));
            }
        }
        let config = Config {
            leaf_capacity,
            branching_factor,
            beta,
            rebuild_threshold,
            max_tree_depth: max_tree_depth.unwrap_or(usize::MAX),
            hnsw_threshold: match leaf_backend {
                "auto" => hnsw_threshold,
                "flat" => usize::MAX,
                "hnsw" => 2,
                _ => {
                    return Err(PyValueError::new_err(
                        "leaf_backend must be auto, flat or hnsw",
                    ))
                }
            },
            hnsw_degree,
            ef_construction,
        };
        Ok(Self {
            index: Index::new(embedding_dim, config).map_err(value_error)?,
        })
    }

    #[pyo3(signature = (chunk_ids, vectors, *, rebuild=true))]
    fn native_batch_insert(
        &mut self,
        py: Python<'_>,
        chunk_ids: Vec<String>,
        vectors: PyReadonlyArray2<f32>,
        rebuild: bool,
    ) -> PyResult<()> {
        let view = vectors.as_array();
        if view.nrows() != chunk_ids.len() || view.ncols() != self.index.dim {
            return Err(PyValueError::new_err(
                "vectors must have shape (len(chunk_ids), embedding_dim)",
            ));
        }
        // Copy before releasing the GIL: NumPy aliases may be writable.
        // Iteration accepts strided / Fortran arrays without panicking.
        let owned: Vec<f32> = view.iter().copied().collect();
        py.allow_threads(|| self.index.insert(chunk_ids, owned, rebuild))
            .map_err(value_error)
    }

    fn rebuild(&mut self, py: Python<'_>) {
        py.allow_threads(|| self.index.rebuild());
    }

    #[pyo3(signature = (query_vector, top_k=10, rerank_k=4096, beam_width=64, exact=false, leaf_ef=64))]
    fn search(
        &self,
        py: Python<'_>,
        query_vector: PyReadonlyArray1<f32>,
        top_k: usize,
        rerank_k: usize,
        beam_width: usize,
        exact: bool,
        leaf_ef: usize,
    ) -> PyResult<Vec<(String, f32)>> {
        let q: Vec<f32> = query_vector.as_array().iter().copied().collect();
        let options = SearchOptions {
            top_k,
            candidate_budget: rerank_k,
            beam_width,
            exact,
            leaf_ef,
        };
        py.allow_threads(|| {
            self.index
                .search(&q, options)
                .map(|r| self.index.named(r.hits))
        })
        .map_err(value_error)
    }

    #[pyo3(signature = (query_vector, top_k, rerank_k=4096, beam_width=64, exact=false, leaf_ef=64))]
    fn unified_search(
        &self,
        py: Python<'_>,
        query_vector: PyReadonlyArray1<f32>,
        top_k: usize,
        rerank_k: usize,
        beam_width: usize,
        exact: bool,
        leaf_ef: usize,
    ) -> PyResult<Vec<String>> {
        Ok(self
            .search(
                py,
                query_vector,
                top_k,
                rerank_k,
                beam_width,
                exact,
                leaf_ef,
            )?
            .into_iter()
            .map(|(id, _)| id)
            .collect())
    }

    #[pyo3(signature = (queries, top_k=10, rerank_k=4096, beam_width=64, exact=false, leaf_ef=64))]
    fn batch_search(
        &self,
        py: Python<'_>,
        queries: PyReadonlyArray2<f32>,
        top_k: usize,
        rerank_k: usize,
        beam_width: usize,
        exact: bool,
        leaf_ef: usize,
    ) -> PyResult<Vec<Vec<(String, f32)>>> {
        let view = queries.as_array();
        if view.ncols() != self.index.dim {
            return Err(PyValueError::new_err(
                "queries have incorrect embedding dimension",
            ));
        }
        let owned: Vec<f32> = view.iter().copied().collect();
        let options = SearchOptions {
            top_k,
            candidate_budget: rerank_k,
            beam_width,
            exact,
            leaf_ef,
        };
        options.validate().map_err(value_error)?;
        py.allow_threads(|| {
            owned
                .par_chunks(self.index.dim)
                .map(|q| {
                    self.index
                        .search(q, options)
                        .map(|r| self.index.named(r.hits))
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .map_err(value_error)
    }

    #[pyo3(signature = (query_vector, top_k=10, rerank_k=4096, beam_width=64, exact=false, leaf_ef=64))]
    fn search_diagnostics(
        &self,
        py: Python<'_>,
        query_vector: PyReadonlyArray1<f32>,
        top_k: usize,
        rerank_k: usize,
        beam_width: usize,
        exact: bool,
        leaf_ef: usize,
    ) -> PyResult<(Vec<(String, f32)>, usize, usize)> {
        let q: Vec<f32> = query_vector.as_array().iter().copied().collect();
        let options = SearchOptions {
            top_k,
            candidate_budget: rerank_k,
            beam_width,
            exact,
            leaf_ef,
        };
        py.allow_threads(|| {
            self.index
                .search(&q, options)
                .map(|r| (self.index.named(r.hits), r.vectors_scored, r.nodes_expanded))
        })
        .map_err(value_error)
    }

    #[getter]
    fn size(&self) -> usize {
        self.index.len()
    }
    #[getter]
    fn pending_count(&self) -> usize {
        self.index.pending_count()
    }
    #[getter]
    fn node_count(&self) -> usize {
        self.index.node_count()
    }

    #[getter]
    fn graph_count(&self) -> usize {
        self.index.graph_count()
    }
}

#[pymodule]
fn aigvs_rust(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<AIGVSEngine>()?;
    Ok(())
}
