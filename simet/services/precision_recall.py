import faiss
import numpy as np


class PrecisionRecallService:
    """Utilities for feature normalization and batched k-NN search (FAISS)."""

    @staticmethod
    def safe_norm(x: np.ndarray) -> None:
        """In-place L2 normalization of row vectors with zero-safe clipping.

        Normalizes each row of `x` to unit L2 norm: `x[i] /= max(||x[i]||_2, 1e-12)`.
        Operates **in place** and returns `None`.

        Args:
            x (np.ndarray): 2D array of shape `(n_samples, n_dims)`.

        Notes:
            - Uses an epsilon `1e-12` to avoid division by zero for all-zero rows.
            - The dtype is preserved; if you need `float32`, cast before calling.
        """
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        x /= norms

    @staticmethod
    def batched_search(
        index: faiss.Index,
        queries: np.ndarray,
        k: int,
        batch_size: int | None,
        return_indices: bool = False,
    ) -> np.ndarray:
        """Run a k-NN search against a FAISS index with optional batching.

        Wraps `index.search(queries, k)` and allows chunking the query set to
        control memory usage. Returns **distances** by default, or **indices**
        if `return_indices=True`.

        Args:
            index: A FAISS index (e.g., `faiss.IndexFlatL2`, `IndexIVFFlat`, etc.)
                already trained (if required) and populated with database vectors.
            queries (np.ndarray): 2D array of query vectors, shape `(n_queries, d)`.
                The dimension `d` must match the index’s dimension.
            k (int): Number of nearest neighbors to retrieve per query.
            batch_size (int | None): If `None` or `<= 0`, performs a single pass.
                Otherwise, processes queries in chunks of this size.
            return_indices (bool): If `True`, return neighbor indices (`int64`);
                otherwise return distances (`float32`).

        Returns:
            np.ndarray:
                - If `return_indices=False`: distances of shape `(n_queries, k)`,
                  dtype `float32`. For L2 indexes, **smaller is nearer**.
                - If `return_indices=True`: indices of shape `(n_queries, k)`,
                  dtype `int64`.

        Notes:
            - No normalization is applied here. If needed, call `safe_norm` on your
              database and query vectors beforehand (and use a cosine-compatible index).
            - For FAISS IVF/HNSW indexes, performance/accuracy also depends on probe
              settings (`nprobe`, etc.), which you should configure on `index` prior
              to calling this function.
        """
        n = len(queries)
        if batch_size is None or batch_size <= 0:
            # Let faiss handle in one go
            distances, indices = index.search(queries, k)
            return indices if return_indices else distances

        out_D = np.empty((n, k), dtype=np.float32)
        out_I = np.empty((n, k), dtype=np.int64)
        start = 0
        while start < n:
            end = min(start + batch_size, n)
            distances, indices = index.search(queries[start:end], k)
            out_D[start:end] = distances
            out_I[start:end] = indices
            start = end
        return out_I if return_indices else out_D
