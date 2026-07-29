# Classifying Fermi-LAT Blazar Candidates of Uncertain Type

Capstone project for the AI for Science Summer School.

## Background

Blazars are a class of Active Galactic Nuclei characterised by relativistic jets pointing
almost directly toward Earth. They divide into two physical subclasses based on their
environments. **Flat Spectrum Radio Quasars (FSRQs)** sit in gas-rich environments, so their
jets up-scatter external photons from the accretion disc and broad-line region — producing
highly variable, *softer* (steeper) gamma-ray spectra. Conversely, **BL Lacertae objects
(BL Lacs)** sit in gas-poor environments where the jet up-scatters its own synchrotron
photons, producing more stable, *harder* (flatter) gamma-ray spectra.

Roughly a third of the blazars Fermi-LAT detects are **BCUs** — *blazar candidates of
uncertain type*. These have a confident counterpart association, but lack the optical
spectroscopy needed to sort them into BL Lac or FSRQ.

## Research question

> To what extent do the decision boundaries of tree-based ML models align with theoretical
> blazar physics when distinguishing between BL Lacs and FSRQs, and can we confidently use
> these learned boundaries to classify Fermi-LAT blazar candidates of uncertain type (BCUs)?

## Dataset

[Fermi-LAT 4LAC-DR3, high-latitude sample](https://fermi.gsfc.nasa.gov/ssc/data/access/lat/4LACDR3/)
(`table-4LAC-DR3-h.fits`), read with `astropy` and converted to a `pandas` DataFrame.

| Population | Count | Role |
|---|---:|---|
| BL Lac (`bll`) | 1,379 | training |
| FSRQ (`fsrq`) | 755 | training |
| BCU (`bcu`) | 1,208 | held out — the prediction target |
| Other AGN (`rdg`, `nlsy1`, …) | 65 | dropped, not blazars |
| **Total** | **3,407** | |

Four gamma-ray features, each measurable for a BCU: `PL_Index` (photon index Γ),
`Variability_Index`, `Flux1000`, `Pivot_Energy`.

## Approach

Feature engineering to isolate leakage-free physical measurements, a Random Forest classifier
with `class_weight="balanced"`, then SHAP and permutation importance to extract which physical
features drive the decision boundaries — categorising the BCU population while testing whether
the model's underlying logic aligns with known high-energy astrophysics.

### Pre-registered prediction

Recorded before any model is trained, so the physics-alignment claim is falsifiable rather
than post-hoc:

> `PL_Index` will be the dominant feature. FSRQs will concentrate at Γ ≳ 2.3 and BL Lacs at
> Γ ≲ 2.1, with the primary decision boundary near Γ ≈ 2.2. Variability is secondary, with
> FSRQs the more variable population.

Day 1 data-level check: BL Lac ⟨Γ⟩ = 2.030 ± 0.211, FSRQ ⟨Γ⟩ = 2.469 ± 0.202.

## Failure modes and mitigations

| Risk | Mitigation |
|---|---|
| **Accuracy illusion** — 1.83:1 imbalance means "always BL Lac" scores ~65% | Report ROC-AUC and per-class precision/recall; accuracy is not used |
| **Class imbalance** | `class_weight="balanced"`. *Not* SMOTE — interpolated points in (Γ, variability) space are physically meaningless, and the imbalance is mild |
| **Leakage** | Drop all identifiers, sky coordinates, and `Unc_*` columns |
| **Optical-derived features** | Drop `Redshift`, `nu_syn`, `SED_class` — a BCU by definition lacks these, so training on them yields a model that cannot be applied |
| **Overfitting** | Limit `max_depth` / `min_samples_leaf`; validate with stratified 5-fold CV |
| **Overconfidence on BCUs** | Random Forest probabilities are poorly calibrated by default — check a reliability curve before applying any confidence threshold |
| **Distribution shift** | BCUs are fainter and messier than the training set; sources below the confidence threshold stay labelled "unknown". Measured with a domain classifier (AUC 0.739) and priced by re-running the evaluation on a flux-matched labelled sample before predicting anything |
| **Misreading feature importance** | `Pivot_Energy` is ρ = −0.86 with Γ, so SHAP and permutation importance both over-credit it. Confirm every importance claim with a retrain-without-it ablation |

## 5-day plan

| Day | Work | Deliverable |
|---|---|---|
| 1 | Load, audit schema and labels, split, feature selection | ✅ Clean CSVs in `data/processed/` |
| 2 | EDA scatter + **baseline Random Forest, end to end** | ✅ ROC-AUC 0.958, confusion matrix + ROC in `figures/` |
| 3 | Repeated stratified CV, tuning, calibration check | ✅ AUC 0.946 ± 0.009, isotonic calibration, reliability curve |
| 4 | SHAP, ablation, 2D boundary, noise-ceiling analysis | ✅ All 6 pre-registered claims hold; LBL mechanism identified; problem shown to be photon-limited |
| 5 | Distribution-shift gate, transfer test, predict BCUs | ✅ 603 of 1,208 classified; forest's margin over Γ shown not to transfer |


**Day 2 is the milestone that matters.** Once an end-to-end model exists, every later day is
optional improvement.

**Out of scope:** SMOTE, XGBoost comparison, grid search, redshift/`nu_syn` features,
low-latitude sources, neural networks.

## Repository layout

```
blazar_aiss/
├── data/
│   ├── raw/          # 4LAC FITS (gitignored, auto-downloaded by notebook 01)
│   └── processed/    # cleaned CSVs (gitignored, regenerated by notebook 01)
├── docs/             # planning notes (local only, not tracked)
├── figures/          # exported plots
├── notebooks/
│   ├── 01_initial_data_exploration.ipynb
│   ├── 02_baseline_model.ipynb
│   ├── 03_cross_validation_and_calibration.ipynb
│   ├── 04_interpretability_and_physics.ipynb
│   └── 05_bcu_classification.ipynb
└── src/
```

## Results so far

### Held-out test set (n = 427)

| Model | Accuracy | FSRQ precision | FSRQ recall | ROC-AUC |
|---|---:|---:|---:|---:|
| Always predict BL Lac | 0.646 | 0.000 | 0.000 | 0.500 |
| Cut on Γ alone | 0.862 | 0.745 | 0.927 | 0.945 |
| Day 2 RF (guessed params) | 0.888 | 0.832 | 0.854 | 0.958 |
| **Day 3 RF (tuned + isotonic)** | **0.888** | **0.860** | **0.815** | **0.960** |

### Cross-validated, 50 folds (the number to quote)

| Ranker | ROC-AUC |
|---|---:|
| Random Forest, 4 features | **0.946 ± 0.009** |
| `PL_Index` (Γ) alone | 0.930 ± 0.010 |
| **paired difference** | **+0.016 ± 0.007**, positive in **50/50 folds** |

Day 2's single-split 0.958 was optimistic by ~1.3σ. The forest's margin over spectral index
alone is small but its sign never flipped across 50 folds — the three non-spectral features carry
a real, modest signal, worth **+0.053 average precision**. Most of the gamma-ray discriminating
power really is concentrated in spectral slope, as blazar emission theory predicts.

Note on the baseline: ROC-AUC is threshold-independent, so "a cut at Γ > 2.2" and "Γ used alone
as a ranking score" have the *same* AUC. The threshold only affects precision and recall.

### Tuning and calibration

Grid search over `max_depth` × `min_samples_leaf` moved CV AUC by **+0.003** — inside the
fold-to-fold scatter of 0.013, so Day 2's guessed hyperparameters were already fine. The search
did prefer shallower trees (depth 4, not 8), consistent with a genuinely simple decision surface.

Calibration mattered more. As `class_weight="balanced"` implies, the raw forest over-stated
`P(FSRQ)` (mean 0.388 vs a true rate of 0.354). Isotonic regression corrected it —
Brier **0.078 → 0.074**; Platt scaling made it slightly worse. Thresholding the calibrated
probabilities at `P ≥ 0.90` (or `≤ 0.10`) retains **72%** of test sources at **97.7%** accuracy,
which sets the confidence/coverage trade-off Day 5 inherits.

### Physics alignment (Day 4 — the answer to the research question)

All six pre-registered claims from Day 1 hold. The learned decision boundary sits at
median **Γ = 2.251** against a prediction of 2.2, made before any model was trained.

**The forest's advantage over a spectral cut has a name.** Of the 24 test sources it rescues from
a Γ > 2.2 rule, **24 are BL Lacs and none are FSRQs**. They sit at high Γ (2.25) with anomalously
low variability — *low-peaked BL Lacs*, whose soft spectra mimic FSRQs but which stay quiet
because, lacking a broad-line region, they have no external-Compton cooling to drive fast
variability. The learned boundary tilts by **+0.078 in Γ** in exactly the direction this predicts:
a quiet source must look measurably softer before the model will call it an FSRQ.

| Feature set | CV ROC-AUC |
|---|---:|
| Γ alone | 0.9293 |
| Γ + variability | 0.9480 |
| all four features | 0.9504 |

Variability supplies ~90% of everything gained over spectral index alone.

**The problem is photon-limited, not algorithm-limited.** Using `Unc_PL_Index` as a diagnostic
(never as a feature — the leakage rule stands), ~75% of each class's spread in Γ is intrinsic
astrophysical diversity, so the class overlap is real. But the remaining ~25% is measurement
noise, and removing it is worth more than any modelling we did:

| Source of improvement | AUC gain |
|---|---:|
| Random Forest over Γ alone | +0.016 |
| all three extra features | +0.021 |
| **perfect measurement of Γ** | **+0.023** |

The residual ambiguity splits roughly half and half: **51%** of low-confidence sources sit within
1σ of the boundary (measurement-limited), while **31%** are firmly measured yet still ambiguous —
and most of those sit where the training data's own class mixture is genuinely intermediate. Not
a defect; the blazar sequence is a continuum. Matched on flux, ambiguous and confident sources
have the same Γ uncertainty (ratio 1.06), so they are not badly observed — just awkwardly placed.

**Methodological caution worth reporting.** `Pivot_Energy` receives the second-largest SHAP
attribution *and* is the cheapest feature to delete (ablation cost 0.0029, the lowest of the
four). It is ρ = −0.86 correlated with Γ because the pivot energy is a byproduct of the same
spectral fit — so attribution methods hand it credit that belongs to Γ. Reporting the SHAP
ranking alone would have produced a confident and wrong physical claim. **Attribution measures
credit; only ablation measures unique information.**

### The BCUs (Day 5 — the other half of the research question)

Three predictions were registered before any BCU was scored. All three held.

**The model's advantage does not transfer to the population it was built for.** A domain
classifier separates BCUs from labelled blazars at AUC 0.739, driven by flux (0.699 alone) — BCUs
are unclassified *because* they are faint. Resampling the labelled set to the BCU flux
distribution (rejection sampling, no duplicates) and re-measuring:

| Population | RF | Γ alone | margin |
|---|---:|---:|---:|
| full labelled sample (n = 2,134) | 0.9460 | 0.9298 | **+0.0163** |
| random subsample (n ≈ 991) — control | 0.9465 | 0.9337 | +0.0128 |
| **flux-matched to the BCUs** (n ≈ 991) | 0.9311 | 0.9295 | **+0.0016 ± 0.0022** |

The margin does not shrink, it vanishes. The control rules out sample size. Note *which* number
moved: Γ alone is untouched (0.9298 → 0.9295) because spectral shape degrades gracefully, while
the forest falls, because all three features it adds are flux-sensitive — Variability_Index most
of all, since a χ²-like statistic loses power with photon count. The Day 4 mechanism is real
*and* fragile: on a faint blazar everything looks quiet, and non-detection of variability is not
evidence of steadiness.

**Three estimates of the BCU FSRQ fraction, and they disagree informatively:**

| Estimate | FSRQ fraction | Uses |
|---|---:|---|
| training prior | 0.354 | nothing about the BCUs |
| model's mean P(FSRQ) | 0.374 | all four features |
| EM on the score distribution | 0.394 | features + score spread |
| moment matching on Γ, flux-corrected | **0.444** | Γ only |
| moment matching on Γ, naive | 0.520 | Γ only, biased reference means |

Correcting the class reference means for flux drops the moment estimate 0.520 → 0.444: within the
FSRQ class ρ(log Flux, Γ) = −0.41, so faint FSRQs are softer and comparing BCUs against
bright-FSRQ means inflates the answer. The model's 0.374 sits *below* the Γ-only estimate, in
exactly the direction suppressed variability predicts — it under-calls FSRQs.

### Result

| | Labelled set | Flux-matched rehearsal | BCUs |
|---|---:|---:|---:|
| coverage at `P ≥ 0.90 / ≤ 0.10` | 70.6% | 67.8% | **49.9%** |
| accuracy within coverage | 96.6% | 96.6% | — |

**603 of 1,208 BCUs classified — 418 BL Lacs, 185 FSRQs — at an expected 96.6% accuracy within
the confident set, measured on a rehearsal rather than assumed. The remaining 605 are declared
uncertain.** Accuracy holds while coverage falls: when the evidence degrades the model does not
become confidently wrong, it abstains. That is the calibration from Day 3 earning its place, and
it is the part of this model worth defending — not its margin over a spectral cut.

Predictions in `data/processed/bcu_predictions.csv`.

### The claim, stated honestly

> A model can be right about the physics and still not transfer. Ours learned the correct
> boundary, for the correct reason, and then met a population where the evidence it relies on
> stops being measurable — so it says so, and abstains on half of them.

The Day 4 LBL mechanism *replicates in direction* on the BCUs: the forest disagrees with a Γ > 2.2
cut 349 times, always pulling soft sources back to BL Lac. That asymmetry is not an artifact — on
the labelled set the same comparison gives 283:3, and 67.5% of those 283 really are BL Lacs. But
it cannot be **confirmed**: as flux falls, mean Γ falls, variability falls, and P(FSRQ) falls
together, and two explanations — genuine selection toward hard-spectrum sources at the faint end,
versus the variability artifact — predict identical trends in every column. Separating them needs
a flux-independent variability measure, or optical spectra.

## Setup

```bash
pip install -r requirements.txt
jupyter lab notebooks/01_initial_data_exploration.ipynb
```

Notebook 01 downloads the catalog automatically on first run.
