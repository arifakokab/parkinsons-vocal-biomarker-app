# Parkinsons-Vocal-Biomarker-App

**Parkinsons-Vocal-Biomarker-App** is a two-part system that turns short sustained “aaah” recordings into actionable signals for Parkinson’s care. The goal is twofold: (1) provide a fast, feature-based **screening** path that’s simple enough to deploy on the web, and (2) explore **progression monitoring** by forecasting visit-to-visit changes in clinician-rated motor\_UPDRS for already-diagnosed patients.

* **Part 1 — Classifier (screening):** A Random Forest on robust dysphonia features (jitter, shimmer, HNR, pitch stats) that proves the voice → features → inference pipeline and serves as the deployable demo.
* **Part 2 — Severity progression:** A compact BiGRU+1D-CNN that predicts next-visit Δmotor\_UPDRS to help flag sudden or unusual shifts during ongoing treatment (research-grade; not a diagnostic tool).

This project was completed as the **M.Sc(Eng) Applied AI capstone at the University of San Diego**, **built solo by Arifa Kokab (Group 11)**. The screening classifier is publicly deployed under the **CarePath AI Foundation**; see the links in the sections below.


## Live demo & external deployment repo (Part 1)

* GitHub (deployment code):

  ```
  https://github.com/arifakokab/classification-for-parkinsons
  ```
* Public web app:

  ```
  www.parkinsonsaiscreening-carepathai-foundation.care
  ```

The deployed app records a short sustained “aaah,” extracts features on the backend, and returns a screening result using the trained Random Forest at a fixed threshold (0.63). Audio is converted to mono 16 kHz WAV; features are computed via Parselmouth (Praat); missing values are handled conservatively.

> This screening app is a **research/education prototype** under the CarePath AI Foundation and **not a diagnostic device**. No voice data or personal information is stored.

## Repository structure (this repo)

```
Classifier Model (part 1 of 2)/
    # Notebook/code for Random Forest training & evaluation
Severity Progression Predictor (Part 2 of 2)/
    # Notebook/code for BiGRU+1D-CNN Δmotor_UPDRS prediction
README.md
```

## Datasets

**Screening (Part 1).** UCI “Parkinson’s Disease Detection” (Oxford): tabular acoustic features with `status` (0=Healthy, 1=PD). Used to train and validate the Random Forest and to demonstrate deployable, explainable feature-based inference.

**Progression (Part 2).** Oxford Parkinson’s Telemonitoring: longitudinal visits with clinician-rated UPDRS; we use **motor\_UPDRS** as the target. Multiple phonations per visit are aggregated to visit-level rows; sequences are constructed per subject over consecutive visits.

> All dataset usage follows their respective licenses/terms. This project is for research/education only and is not a medical device.

## Methods Brief Overview:

### Part 1 — Screening classifier

**Features.** Standard dysphonia measures suitable for short phonations: pitch statistics (`MDVP:Fo/Fhi/Flo`), perturbation metrics (`Jitter(%)`, `RAP`, `PPQ`, `DDP`), amplitude perturbation (`Shimmer`, `Shimmer(dB)`, `APQ3/5`), and noise measures (`NHR`, `HNR`). The deployed model uses a robust subset of 16 features that compute reliably from browser audio.

**Modeling.** `RandomForestClassifier` with GridSearchCV (5-fold; 405 configs; 2,025 fits).
**Best parameters:**
`{'n_estimators': 100, 'max_depth': None, 'max_features': 'sqrt', 'min_samples_split': 2, 'min_samples_leaf': 1}`

**Operating points.**

* Threshold **0.50** (research test set): Acc 0.90; PD F1 0.93; ROC AUC 0.97.
* Threshold **0.63** (deployed): Healthy **Recall=1.00**, PD **Precision=1.00** in our test—chosen to reduce false alarms while preserving high PD precision.

**Feature importance (top, decreasing).** `MDVP:APQ`, `Shimmer:APQ5`, `Jitter:DDP`, `MDVP:Fo(Hz)`, `MDVP:Fhi(Hz)`, `RAP`, `Shimmer(dB)`, `Shimmer`, `NHR`, `APQ3`, `Flo`, `PPQ`, `Jitter(%)`, `DDA`, `HNR`, `Jitter(Abs)`.

### Part 2 — Severity progression predictor

**Task.** Predict **Δmotor\_UPDRS** (next visit’s change) from short sequences of per-visit features; reconstruct the next absolute severity as `last_score + Δ`.

**Preprocessing.** Visit-level aggregation, per-subject z-scoring to remove personal baselines, fixed-length sequences over consecutive visits, guardrails for short/constant series.

**Architecture.** Lightweight **BiGRU + 1D-CNN** hybrid: recurrent branch captures temporal trends; convolutional branch captures local patterns. L2 regularization; mixed precision when available; ReduceLROnPlateau.

**Evaluation.** Grouped CV by subject (GroupKFold=5); final **subject-held-out** test. Hyperparameter grid: `seq_len ∈ {5,7,9}`, `L2 ∈ {1e-3, 3e-4}` → **best** `seq_len=5`, `L2=1e-3`.

**Performance (held-out subjects).**

* Absolute motor\_UPDRS MAE ≈ **0.433** (naïve last-value ≈ 0.427; Ridge ≈ 0.481).
* Δ direction accuracy ≈ **46.5%** (many small “stable” changes make sign prediction difficult).
* Calibration: slope ≈ **1.007**, intercept ≈ **−0.353**, **R² ≈ 0.996**.
* Bland–Altman: bias ≈ −0.186; LoA ≈ (−1.147, 0.774).
* Clinical framing: `THRESH_VISIT = 0.5` (single-visit change) and `THRESH_CLINIC = 3.0` (3-visit rolling sum; UPDRS-III MID).

**Interpretation.** The Δ-task is challenging but the model is well-calibrated and can flag **sudden/drastic** severity changes for triage and remote check-ins (not diagnosis).

## Results summary

**Screening (Part 1):** ROC AUC **0.97**; test Acc **0.90** at 0.50; at 0.63, Healthy **Recall=1.00** and PD **Precision=1.00** in our test set. Fast and stable feature pipeline supports web deployment.

**Progression (Part 2):** Test **MAE ≈ 0.433** on absolute motor\_UPDRS; strong calibration; modest directional skill on Δ; clinically-meaningful change thresholds documented.

## Deployment & ops (Part 1)

The production demo lives in the separate repo noted above. Key details:

* **Backend:** Flask API on Render.com; Python 3.12 (pinned). Gunicorn as WSGI. Audio normalized to mono 16 kHz WAV; features via Parselmouth; NaNs handled; serialized RF loaded from `rf_model.pkl`.
* **Endpoint:** `/predict` (expects audio; returns probability + label using fixed threshold 0.63).
* **Frontend:** React + Vite; “Start Recording → Stop → Run Analysis”; update `API_URL` in `App.jsx` for the deployed backend.
* **Privacy:** No storage of voice or PII; in-memory processing only.
* **License:** MIT (deployment repo).

## Ethics, privacy, and clinical use

* Voice data can be identifiable; handle per HIPAA/IRB when collecting from humans.
* Models are **assistive**; clinical decisions must remain with qualified professionals.
* Longitudinal deployment requires drift monitoring (hardware/channel shifts), recalibration, and ongoing validation.

## License

MIT License

## Status & roadmap

* Deployed screening demo (separate repo + live site).
* Research-grade progression modeling with subject-held-out evaluation.
* Next: larger longitudinal cohorts; richer acoustic embeddings; uncertainty quantification; clinician-defined alerting; prospective validation.
