# Simet

**Simet** is a lightweight, extensible framework for **evaluating synthetic image datasets** against real datasets. It gives you:

- A unified **pipeline** to load images (from local folders or built-in datasets), apply **transforms**, **extract features** (Inception-v3 out of the box), and run **metrics** such as **FID**, **Precision/Recall (FAISS)**, and **ROC-AUC**.
- **Restraints** (threshold checks) to enforce quality bars (e.g., FID ≤ 40, ROC-AUC ≥ 0.85) and return a single pass/fail.
- A **CLI** for simple or fully-configurable runs, and a clean **Python API** for scripting.
- Practical scalability features: **feature caching**, **subsampling**, **FAISS IVF** indexes, optional **AMP**, and deterministic **seeding**.

---

## Installation

```bash
pip install simet
```

Python 3.12 is needed. CUDA is optional (for faster feature extraction and FAISS GPU support).

---

## Quick start

### 1) Fast check with the CLI (simple two-paths mode)

```bash
simet simple /path/to/real_images /path/to/synth_images
```

Runs a minimal pipeline (no restraints) using default transforms and Inception-v3 feature extraction, printing metric (FID, Precision/Recall, ROC AUC) results.

### 2) Full pipeline from YAML

```bash
simet pipeline pipeline.yaml
```

**`pipeline.yaml` example:**

```yaml
pipeline:
  loader:
    real_provider:
      type: LocalProviderWithoutClass
      path: data/real
    synth_provider:
      type: LocalProviderWithoutClass
      path: data/synth
    provider_transform:
      type: InceptionTransform
    feature_extractor:
      type: InceptionFeatureExtractor

  restraints:
    - type: FIDRestraint
      upper_bound: 40.0
    - type: PrecisionRecallRestraint
      lower_bound: [0.70, 0.60]
    - type: RocAucRestraint
      lower_bound: 0.85
```

CLI returns exit code 0 if all restraints pass.

---

## Python API

```python
from pathlib import Path
from simet.dataset_loaders import DatasetLoader
from simet.feature_extractor import InceptionFeatureExtractor
from simet.pipeline import Pipeline
from simet.providers import LocalProviderWithoutClass
from simet.restraints import FIDRestraint, PrecisionRecallRestraint, RocAucRestraint
from simet.services import LoggingService, SeedingService
from simet.transforms import InceptionTransform

LoggingService.setup_logging()
SeedingService.set_global_seed(42)

loader = DatasetLoader(
    real_provider=LocalProviderWithoutClass(Path("data/real")),
    synth_provider=LocalProviderWithoutClass(Path("data/synth")),
    provider_transform=InceptionTransform(),
    feature_extractor=InceptionFeatureExtractor(),
)

pipeline = Pipeline(
    loader=loader,
    restraints=[
        FIDRestraint(upper_bound=40.0),
        PrecisionRecallRestraint(lower_bound=[0.70, 0.60]),
        RocAucRestraint(lower_bound=0.85),
    ],
)

ok = pipeline.run()
print("PASS" if ok else "FAIL")
```

---

## CLI reference

### `simet simple REAL_PATH SYNTH_PATH [--log-path LOG_DIR]`
Run a simple pipeline with default settings.

### `simet simple-pipeline CONFIG_PATH [--log-path LOG_DIR]`
Run YAML mode without restraints.

**Example `simple.yaml`:**

```yaml
pipeline:
  real_path: data/real
  synth_path: data/synth
  metrics: [FID, PrecisionRecall, RocAuc]
```
Specifying no metrics will run all available (FID, Precision/Recall, ROC AUC).

### `simet pipeline CONFIG_PATH [--log-path LOG_DIR]`
Run the full configurable pipeline (providers, transforms, feature extractor, restraints).

---

## Concepts & Extensibility

- **Providers** → `simet.providers`: local or built-in datasets.
- **Transforms** → `simet.transforms`: image preprocessing (e.g. `InceptionTransform`).
- **Feature Extractors** → `simet.feature_extractor`: cached Inception-v3 features.
- **Metrics** → `simet.metrics`: FID, Precision/Recall (FAISS), ROC-AUC.
- **Restraints** → `simet.restraints`: wrap metrics with thresholds.
- **Services** → logging, seeding, caching, subsampling.

Add custom components by subclassing:
```
Provider → providers.base.Provider
Transform → transforms.base.Transform
Feature Extractor → feature_extractor.FeatureExtractor
Metric → metrics.base.Metric
Restraint → restraints.base.Restraint[T]
```

and register them via the corresponding `*Parser`.

---

## Scalability notes

- **FAISS** IVF & batched search for large sets  
- **GPU / AMP** support  
- **Feature caching** per dataset hash  
- **Subsampling** to balance dataset sizes  
- **Deterministic seeding** via `SeedingService`

---

## Examples

See [main](examples/main.py) for a complete demo.

---

## License

MIT (or your chosen license)
