# Simet

**Simet** is a modular toolkit for evaluating **synthetic** images against **real** ones with reproducible metrics and threshold-based **restraints**.

```bash
pip install simet
```

---

## Key features

- **Load** images from folders or built-in datasets  
- **Transform** for feature extraction (Inception-v3 provided)  
- **Extract** and **cache** features automatically  
- **Measure** quality (FID, Precision/Recall, ROC-AUC)  
- **Gate** results with simple threshold rules  

---

## Why Simet?

- 🔁 **Fast iteration** — cache features and tweak thresholds freely  
- ⚙️ **Extensible** — plug custom providers, transforms, metrics, restraints  
- 🚀 **Scalable** — FAISS IVF, batching, GPU/AMP, subsampling  
- 🧩 **Deterministic** — consistent seeds & reproducible logs  

---

## Typical workflow

1. Point to **real** and **synthetic** image folders  
2. Pick **metrics**  
3. Optionally add **restraints**  
4. Run via CLI or Python API  
5. Inspect reports and JSON logs  

→ Continue to **[Getting Started](getting-started.md)** to run your first evaluation.
