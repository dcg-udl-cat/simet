import logging
from typing import Literal, Optional, Tuple, override

import faiss
import numpy as np

from simet.dataset_loaders import DatasetLoader
from simet.metrics import Metric
from simet.services import PrecisionRecallService

logger = logging.getLogger(__name__)

MetricType = Literal["l2", "cosine"]
IndexType = Literal["flat", "ivf"]


class PrecisionRecall(Metric[tuple[float, float]]):
    """Precision/Recall between real and synthetic feature sets via FAISS k-NN.

    Computes:
        - **Precision**: fraction of synthetic samples whose 1-NN in the **real**
          set lies within the synthetic sample’s **real-set** k-th neighbor radius.
        - **Recall**: fraction of real samples whose 1-NN in the **synthetic**
          set lies within the real sample’s **synthetic-set** k-th neighbor radius.

    Distance & index backends:
        - `metric="l2"`: uses squared L2 distances.
        - `metric="cosine"`: uses inner-product search on **L2-normalized** vectors
          (we L2-normalize both sets in-place) and converts sims to cosine distances
          as `1 - sim`.
        - `index_type="flat"`: exact search (`IndexFlatL2` / `IndexFlatIP`).
        - `index_type="ivf"`: coarse-quantized IVF (`IndexIVFFlat`), requires training.

    GPU:
        - If `use_gpu=True` and GPUs are available, the FAISS index is cloned to GPU.
        - With `num_gpus > 1`, uses a **sharded** multi-GPU index.
        - `use_fp16=True` stores/searches in fp16 on GPU (memory/speed trade-offs).

    Args:
        metric (Literal["l2", "cosine"], optional): Distance/similarity type. Defaults to "l2".
        index_type (Literal["flat", "ivf"], optional): FAISS index type. Defaults to "flat".
        nlist (int, optional): Number of IVF lists if `index_type="ivf"`. Defaults to 1024.
        use_gpu (bool, optional): Enable GPU indices when available. Defaults to True.
        num_gpus (int | None, optional): Number of GPUs to use (`None`→all). Defaults to None.
        batch_size (int | None, optional): Query batch size for FAISS search (`None`→auto). Defaults to None.
        use_fp16 (bool, optional): Use fp16 storage/search on GPU. Defaults to False.
        random_state (int | None, optional): Seed for FAISS training (IVF). Defaults to 1234.

    Notes:
        - If either feature set is empty, returns `(0.0, 0.0)` with a warning.
        - If `k` ≥ min(|real|, |synth|), `k` is **clamped** down to avoid degeneracy.
        - For `metric="cosine"`, features are L2-normalized **in place**.
    """

    def __init__(
        self,
        *,
        metric: MetricType = "l2",
        index_type: IndexType = "flat",
        nlist: int = 1024,  # IVF lists if index_type="ivf"
        use_gpu: bool = True,
        num_gpus: Optional[int] = None,  # None => all available
        batch_size: Optional[int] = None,  # None => faiss decides
        use_fp16: bool = False,  # GPU only: store/search in fp16
        random_state: Optional[int] = 1234,
    ) -> None:
        logger.info("Initializing Precision/Recall metric")
        super().__init__()
        self.metric = metric
        self.index_type = index_type
        self.nlist = int(nlist)
        self.use_gpu = bool(use_gpu)
        self.num_gpus = num_gpus
        self.batch_size = batch_size
        self.use_fp16 = use_fp16
        self.random_state = random_state

    @property
    @override
    def name(self) -> str:
        """Human-readable metric name."""
        return "Precision/Recall"

    @override
    def compute(self, loader: DatasetLoader, k: int = 5) -> Tuple[float, float]:
        """Compute `(precision, recall)` between real/synth feature sets.

        Steps:
            1) Validate shapes: both 2D and same `D`.
            2) (Cosine only) L2-normalize `real` and `synth` **in place**.
            3) Build **k-th neighbor radii** per-domain (real→real, synth→synth).
            4) Precision: 1-NN of synth in real vs real radii.
            5) Recall: 1-NN of real in synth vs synth radii.

        Args:
            loader (DatasetLoader): Provides `real_features` and `synth_features` arrays.
            k (int, optional): Neighborhood size for radii (k-th neighbor). Defaults to 5.

        Returns:
            Tuple[float, float]: `(precision, recall)` each in `[0.0, 1.0]`.

        Raises:
            ValueError: On invalid shapes, NaN/Inf, or invalid `k`.

        Notes:
            - If `k >= min(len(real), len(synth))`, `k` will be clamped to `min(n)-1`.
            - For IVF indices, training is performed on the same domain used to build radii/DB.
        """
        real = np.ascontiguousarray(loader.real_features, dtype=np.float32)
        synth = np.ascontiguousarray(loader.synth_features, dtype=np.float32)
        logger.debug("Loaded features from loader")

        logger.debug("Checking feature shapes and types")
        if real.ndim != 2 or synth.ndim != 2:
            logger.error("Features must be 2D: (n_samples, n_dims).")
            raise ValueError("Features must be 2D: (n_samples, n_dims).")
        if real.shape[1] != synth.shape[1]:
            logger.error(
                "Real and synthetic features must have the same dimensionality."
            )
            raise ValueError(
                f"Dim mismatch: real {real.shape[1]} vs synth {synth.shape[1]}"
            )
        if len(real) == 0 or len(synth) == 0:
            logger.warning("One of the feature sets is empty; returning 0.0, 0.0")
            return 0.0, 0.0
        if not np.isfinite(real).all() or not np.isfinite(synth).all():
            logger.error("Features contain NaN/Inf.")
            raise ValueError("Features contain NaN/Inf.")
        if k < 1:
            logger.error("k must be >= 1.")
            raise ValueError("k must be >= 1.")
        if k >= len(real) or k >= len(synth):
            logger.warning(
                "k is greater than or equal to the number of samples; clamping."
            )
            # With k >= n, radius degenerates; clamp safely
            k = max(1, min(k, len(real) - 1, len(synth) - 1))

        if self.metric == "cosine":
            logger.debug("Normalizing features for cosine metric")
            PrecisionRecallService.safe_norm(real)
            PrecisionRecallService.safe_norm(synth)

        # --- build radii for each domain
        logger.debug("Computing k-th neighbor radii")
        real_rad = self._kth_neighbor_radius(real, k=k)
        synth_rad = self._kth_neighbor_radius(synth, k=k)

        # --- precision: synth -> real (1-NN)
        logger.debug("Computing precision and recall")
        d_sr, idx_sr = self._nn_search(real, synth, k=1)
        # Compare squared L2 (or cosine distance converted from IP below)
        precision = float((d_sr[:, 0] <= real_rad[idx_sr[:, 0]]).mean())
        logger.info(f"Computed precision: {precision}")

        # --- recall: real -> synth (1-NN)
        d_rs, idx_rs = self._nn_search(synth, real, k=1)
        recall = float((d_rs[:, 0] <= synth_rad[idx_rs[:, 0]]).mean())
        logger.info(f"Computed recall: {recall}")

        return precision, recall

    def _faiss_metric(self) -> int:
        """Map `self.metric` to the corresponding FAISS metric enum."""
        if self.metric == "l2":
            return faiss.METRIC_L2
        elif self.metric == "cosine":
            return faiss.METRIC_INNER_PRODUCT
        else:
            logger.error(f"Unknown metric: {self.metric}")
            raise ValueError(f"Unknown metric: {self.metric}")

    def _make_index(self, dim: int, train_on: Optional[np.ndarray] = None):
        """Create (and optionally train) a FAISS index, then move to GPU if requested.

        Args:
            dim (int): Feature dimensionality `D`.
            train_on (np.ndarray | None): Training data required for IVF; ignored for flat.

        Returns:
            faiss.Index: CPU or GPU index, potentially sharded across GPUs.

        Raises:
            ValueError: If `index_type` is unknown or IVF training data is missing.
        """
        metric = self._faiss_metric()

        if self.index_type == "flat":
            if metric == faiss.METRIC_L2:
                cpu_index = faiss.IndexFlatL2(dim)
            else:
                cpu_index = faiss.IndexFlatIP(dim)
        elif self.index_type == "ivf":
            # IVF requires training; use coarse quantizer + IVF
            nlist = max(1, self.nlist)
            if metric == faiss.METRIC_L2:
                quantizer = faiss.IndexFlatL2(dim)
            else:
                quantizer = faiss.IndexFlatIP(dim)
            cpu_index = faiss.IndexIVFFlat(quantizer, dim, nlist, metric)
            if train_on is None:
                raise ValueError("IVF index requires training data.")
            if self.random_state is not None:
                faiss.random_seed(self.random_state)  # type: ignore
            cpu_index.train(train_on)  # type: ignore
        else:
            logger.error(f"Unknown index_type: {self.index_type}")
            raise ValueError(f"Unknown index_type: {self.index_type}")

        if self.use_gpu and faiss.get_num_gpus() > 0:
            # Decide how many GPUs
            ng = self.num_gpus if self.num_gpus is not None else faiss.get_num_gpus()

            if ng > 1:
                logger.debug(f"Using {ng} GPUs for FAISS index")
                co = faiss.GpuMultipleClonerOptions()
                co.shard = True  # sharded across GPUs
                co.useFloat16 = bool(self.use_fp16)
                gpu_index = faiss.index_cpu_to_all_gpus(cpu_index, co=co)
            else:
                logger.debug("Using single GPU for FAISS index")
                res = faiss.StandardGpuResources()
                co = faiss.GpuClonerOptions()
                co.useFloat16 = bool(self.use_fp16)
                gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index, co)

            return gpu_index

        return cpu_index

    def _kth_neighbor_radius(self, feats: np.ndarray, k: int) -> np.ndarray:
        """Return each vector’s distance to its k-th neighbor within the same set.

        Implementation details:
            - Builds a FAISS index on `feats` (training if IVF).
            - Searches for `kk = min(k+1, max(2, n))` neighbors, dropping the
              self-match at column 0.
            - For cosine, converts FAISS inner products to cosine distances via `1 - sim`.

        Args:
            feats (np.ndarray): Feature array `(n, d)`.
            k (int): Neighborhood order.

        Returns:
            np.ndarray: Vector of shape `(n,)` with the k-th neighbor distance.
        """
        n, d = feats.shape

        # Build index and add feats
        index = self._make_index(
            d, train_on=feats if self.index_type == "ivf" else None
        )
        index.add(feats)  # type: ignore

        # Search (k+1) neighbors so the first is self (distance 0 or sim=1)
        kk = min(k + 1, max(2, n))  # ensure at least 2
        distances = PrecisionRecallService.batched_search(
            index, feats, kk, self.batch_size
        )

        # Drop the first column (self); take k-th neighbor distance
        # distances are squared L2 for L2 metric; for IP on normalized vectors
        # we convert to cosine distance threshold: d = 1 - sim
        dists_excl_self = distances[:, 1:]  # shape (n, k)
        kth = dists_excl_self[:, -1]
        if self.metric == "cosine":
            kth = 1.0 - kth  # convert sim -> cosine distance
        return kth.astype(np.float32, copy=False)

    def _nn_search(
        self, db: np.ndarray, queries: np.ndarray, k: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run k-NN search from `queries` into `db`; return distances and indices.

        Args:
            db (np.ndarray): Database vectors `(n_db, d)`.
            queries (np.ndarray): Query vectors `(n_q, d)`.
            k (int, optional): Number of neighbors to retrieve. Defaults to 1.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - distances: `(n_q, k)` squared L2 or cosine distance (if metric="cosine").
                - indices: `(n_q, k)` int64 indices of neighbors in `db`.

        Notes:
            - For cosine, FAISS returns inner products; we convert to cosine distance `1 - sim`.
            - Uses `_make_index` (GPU-enabled if configured) and batched searches for memory control.
        """
        d = db.shape[1]
        index = self._make_index(d, train_on=db if self.index_type == "ivf" else None)
        index.add(db)  # type: ignore

        distances = PrecisionRecallService.batched_search(
            index, queries, k, self.batch_size
        )
        idx = PrecisionRecallService.batched_search(
            index, queries, k, self.batch_size, return_indices=True
        )

        if self.metric == "cosine":
            # distances currently are inner products; convert to cosine distance
            distances = 1.0 - distances

        logger.debug(f"Performed k-NN search with k={k}")
        return distances, idx
