# Getting Started

A quick guide to run your first Simet evaluation.

---

## Install

```bash
pip install simet
```

---

## CLI Usage

### Minimal two-paths run

```bash
simet simple /path/to/real /path/to/synth
```

Computes FID, Precision/Recall, and ROC-AUC with default Inception settings.

Add logging:
```bash
simet simple /path/to/real /path/to/synth --log-path ~/.simet/logs
```

---

### Simple YAML run

```bash
simet simple-pipeline simple.yaml
```

**`simple.yaml`**
```yaml
pipeline:
  real_path: data/real
  synth_path: data/synth
  metrics: [FID, PrecisionRecall, RocAuc]
```
No metrics specified will run all available (FID, Precision/Recall, ROC AUC).

---

### Full pipeline with restraints

```bash
simet pipeline pipeline.yaml
```

**`pipeline.yaml`**
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

Returns exit code 0 if all restraints pass.

---

## Python API Example

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

## Building Blocks Overview

| Component | Description | Module |
|------------|--------------|---------|
| **Providers** | Data sources (local folders, CIFAR, binary splits) | `simet.providers` |
| **Transforms** | Preprocessing pipelines | `simet.transforms` |
| **Feature Extractors** | Converts images to feature vectors + cache | `simet.feature_extractor` |
| **Metrics** | FID, Precision/Recall, ROC-AUC | `simet.metrics` |
| **Restraints** | Threshold logic for metrics pass/fail | `simet.restraints` |
| **Services** | Logging, seeding, subsampling, feature cache | `simet.services` |

---

## Performance & Reproducibility

- **Feature caching** on disk  
- **FAISS IVF** + batched queries  
- **GPU/AMP** support in Inception extractor  
- **Subsampling** to balance datasets  
- **Global seeding** for deterministic runs  

---

## Extending Simet

Subclass a base and register it:

| Type | Base class | Parser/Registry |
|------|-------------|----------------|
| Provider | `providers.base.Provider` | `ProviderParser` |
| Transform | `transforms.base.Transform` | `TransformParser` |
| Feature Extractor | `feature_extractor.FeatureExtractor` | `FeatureExtractorParser` |
| Metric | `metrics.base.Metric` | —  |
| Restraint | `restraints.base.Restraint[T]` | `RestraintParser` |

Then reference it in YAML via its `type`.

---

## Troubleshooting

- 📂 Check dataset paths and extensions  
- ⚖️ Ensure transform + extractor match (e.g., Inception norm range)

---

## Next Steps

- Add custom metrics or providers  
