# Multimodal AI for predicting comorbid RBD in MDD

Code accompanying the study:

**Multimodal AI for predicting comorbid REM sleep behavior disorder in major depressive disorder**

This repository contains model code and analysis/plotting utilities for a **smartphone video + audio** pipeline that predicts the probability of **comorbid REM sleep behavior disorder (RBD)** among patients with **major depressive disorder (MDD)**, and surfaces **interpretable facial and vocal digital biomarkers** (e.g., via SHAP).

> ⚠️ Research code only. The model does **not** diagnose disease and is **not** intended for direct clinical decision-making.

---

## What the method does (high level)

- **Data**: short smartphone recordings (reading + spontaneous speech), segmented into ~10s clips; low-quality clips excluded (tracking failure, occlusion, severe noise).
- **Visual features** (per frame): HOG + facial Action Units (AUs) (e.g., OpenFace).
- **Audio features** (per clip/utterance): acoustic descriptors (e.g., OpenSMILE).
- **Model**:
  - Visual sequence → **sLSTM** → video-level visual embedding
  - Audio features → **MLP encoder** → audio embedding
  - Late fusion (concat) → **MLP classifier** → clip-level probability of MDD-RBD
- **Patient-level score**: average clip-level probabilities per participant.
- **Evaluation**: **5-fold cross-validation at the participant level** (no leakage across folds).
- **Interpretability**: baseline feature-aggregation ML models + **SHAP** to highlight candidate biomarkers.

---

## Repository structure

```text
├── Analysis/
    ├── 1.1plot_feature_heatmap.py
    ├── 1.2plot_shap_analysis.py
    ├── 2.plot_summary_metrics.py
    ├── 3.plot_pred_clinical_corr.py
    ├── 4.plot_predicted_class.py
    ├── SI_1.plot_audio_visualization.py
    ├── SI_2.plot_ml_roc_curves.py
    └── SI_3.plot_confusion_matrics.py
└── Model/
    ├── dl.py
    └── slstm.py
```

### `Model/`
- **`slstm.py`**: sLSTM sequence model used by dl.py.
- **`dl.py`**: deep learning multimodal model implementation (TriStreamModel: visual sLSTM streams + audio MLP + fusion classifier).

### `Analysis/`
Plotting and post-hoc analysis scripts used to generate paper-style figures:
- **Confusion matrices**: `SI_3.plot_confusion_matrics.py`
- **ROC curves for baseline ML models**: `SI_2.plot_ml_roc_curves.py`
- **SHAP visualizations**: `1.2plot_shap_analysis.py`, `1.1plot_feature_heatmap.py`
- **Patient-level prediction summaries**: `2.plot_summary_metrics.py`, `4.plot_predicted_class.py`
- **Prediction–clinical scale correlations**: `3.plot_pred_clinical_corr.py`
- **Audio visualization utilities**: `SI_1.plot_audio_visualization.py`

---

## Requirements

Typical environment:
- Python 3.9+
- `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`, `seaborn` (optional), `shap`
- Deep learning framework used by `Model/` (commonly `torch` or `tensorflow`) — see imports in `slstm.py`
- External feature extractors (if you follow the paper):
  - **OpenFace** for AUs
  - **OpenSMILE** for acoustic features

> This repo (based on the visible structure) focuses on **modeling + plotting**. If feature extraction scripts are not included here, prepare features externally and save them in the format expected by `Model/` and `Analysis/`.

---

## Data expectations (conceptual)

You will generally need:
- **Clip-level visual features**: per-frame HOG + AU arrays (with a clip/patient ID)
- **Clip-level audio features**: OpenSMILE feature vectors (with a clip/patient ID)
- **Labels**: patient-level binary label (`MDD-RBD` vs `MDD-only`)
- **Metadata**: mapping from clips → patient; task type (reading vs speech) if used
- **Clinical scales**: UPDRS / BDI / HADS totals for correlation analysis

If scripts use file paths hard-coded at the top (common in plotting scripts), edit those constants to point to your processed CSV/NPY outputs.

---

## Typical workflow

1) **Extract features** (outside Python training loop)
- Run OpenFace on video → AU time series
- Compute HOG per frame
- Run OpenSMILE on audio → acoustic feature vector per clip

2) **Train/evaluate models**
- Use `Model/dl.py` and `Model/slstm.py` for the multimodal deep model (visual sLSTM + audio encoder + fusion classifier).
- Ensure **participant-level 5-fold CV** (all clips from the same participant stay in the same fold).

3) **Aggregate to patient level**
- Average clip probabilities per patient to obtain a single “probability of comorbid RBD” score.

4) **Generate figures**
- Run scripts under `Analysis/` to create:
  - ROC curves / confusion matrices
  - SHAP feature importance and heatmaps
  - Patient-level prediction distribution plots
  - Correlation plots with UPDRS/BDI/HADS

---

## Reproducibility notes

- Use **fixed random seeds** for fold splits and model initialization where possible.
- Perform CV splits **by patient**, not by clip.
- Report mean ± SD over folds for: accuracy, balanced accuracy, precision, recall, specificity, F1, AUROC.

---
