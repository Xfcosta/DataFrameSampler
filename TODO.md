# TODO: JMLR DataFrameSampler Paper

This checklist tracks the current JMLR-oriented paper and experiment plan. The
paper's central claim is deliberately narrow: DataFrameSampler is an
inspectable, low-setup generator for mixed-type example data, with mechanism
and manifold claims treated as empirical diagnostics rather than guarantees.

## 1. JMLR Formatting And Structure

- [x] Migrate `publication/main.tex` to the official JMLR-style structure.
- [x] Vendor `jmlr2e.sty` under `publication/`.
- [x] Use JMLR metadata macros with explicit TODO placeholders.
- [x] Use author-year citations through the JMLR/natbib path.
- [x] Move references after appendices.
- [x] Start the appendix on a new page.
- [ ] Replace TODO metadata before submission:
  - [ ] paper id, volume, date range, and page range;
  - [ ] editor;
  - [ ] author email and full address;
  - [ ] funding and competing-interest disclosure.
- [ ] Decide whether generated build files should remain ignored or be removed
  from the working tree before submission.

## 2. JMLR Paper Content

- [x] Introduction framed around WHAT, WHY, and HOW.
- [x] Add formal problem setting for mixed-type dataframe generation.
- [x] Expand related work into a taxonomy:
  - [x] tabular generators and SDV-style systems;
  - [x] CTGAN/deep tabular generation;
  - [x] SMOTE and interpolation-based oversampling;
  - [x] manifold augmentation and manifold conformance;
  - [x] NCA and supervised metric-learning components.
- [x] Rewrite method as a formal composite latent-space construction.
- [x] Add pseudocode for fitting and generation.
- [x] Add stochastic transport notation:
  - [x] `Z = [X_num | f_1(X_-1) | ... | f_q(X_-q)]`;
  - [x] `T(A; B, C) = A + lambda (C - B)`.
- [x] Add complexity analysis for preprocessing, NCA, decoding, neighbour
  search, generation, and diagnostics.
- [x] Add an interpretation section:
  - [x] composite metric-space view;
  - [x] metric-learning view;
  - [x] local tangent-transport analogy;
  - [x] lightweight sanity assumptions without theorem claims.
- [x] Rewrite the claim-specific evidence section as an evaluation framework.
- [x] Add a structured failure-mode and limitations section.
- [x] Expand implementation and usage around API, CLI, persistence, and KNN
  backends.
- [ ] Read the generated PDF end to end for narrative continuity after the
  JMLR reorganization.

## 3. Evidence Artifacts

- [x] Primary comparison CSVs and tables:
  - [x] nearest-neighbour distance diagnostics;
  - [x] real-versus-synthetic discrimination;
  - [x] utility lift;
  - [x] distribution similarity.
- [x] Mechanism validation:
  - [x] NCA block vs majority baseline;
  - [x] NCA block vs same-width PCA baseline;
  - [x] raw-context classifier reference.
- [x] Decoder diagnostics:
  - [x] negative log loss;
  - [x] Brier score;
  - [x] top-class confidence;
  - [x] expected calibration error.
- [x] Manifold diagnostics:
  - [x] convex-hull membership in fitted latent space;
  - [x] frozen-Isomap insertion stress;
  - [x] latent interpolation baseline;
  - [x] latent bootstrap baseline.
- [x] Sensitivity validation code:
  - [x] fast setup: `n_iterations=0`, no retries, no calibration;
  - [x] default setup: `n_iterations=1`, five retries, no calibration;
  - [x] accurate setup: `n_iterations=2`, twenty retries, calibrated decoders.
- [x] Sensitivity table and figure generation hooks.
- [ ] Refresh `adult_sensitivity_validation.csv` for the final paper cap.
  - Current policy: show the three named setups on Adult Census Income only,
    so setup choice remains an illustrative speed--accuracy tradeoff rather
    than a full cross-dataset benchmark.
  - Recommended final cap: `max_train_rows=80`, `n_samples=80` if the accurate
    setup remains slow in local builds; increase only if runtime is acceptable.
- [ ] Optional CTGAN reference artifact:
  - [x] SDV CTGAN adapter and Adult-only reference workflow;
  - [x] publication table hook for `deep_reference_comparison.tex`;
  - [ ] install `dataframe-sampler[deep-baselines]` and run
    `python experiments/run_deep_reference.py` if the CTGAN table should appear
    in the final PDF snapshot.

## 4. Tables And Figures

- [x] Dataset table.
- [x] Method/baseline metadata table.
- [x] Primary measures table.
- [x] Distributional similarity table.
- [x] Downstream utility table.
- [x] Usability/runtime table.
- [x] Controlled synthetic dataset and result tables.
- [x] Manifold validation table.
- [x] Mechanism validation table.
- [x] Decoder calibration table.
- [x] Sensitivity validation table when sensitivity CSVs exist.
- [x] Limitations/scope table.
- [x] Distribution dashboard figure.
- [x] Baseline similarity figure.
- [x] Utility-cost frontier figure.
- [x] Controlled synthetic similarity figure.
- [x] Categorical stress figure.
- [x] Manifold stress figure.
- [x] Mechanism validation figure.
- [x] Decoder calibration figure.
- [x] Sensitivity validation figure when sensitivity CSVs exist.
- [ ] Review all figure sizes under the one-column JMLR layout.
- [ ] Fix remaining LaTeX float-size warnings where they hurt readability.

## 5. Experiments And Notebooks

- [x] `run_configured_dataset_experiment` returns:
  - [x] baseline comparison;
  - [x] manifold validation;
  - [x] mechanism validation;
  - [x] decoder calibration;
  - [x] sensitivity validation.
- [x] Dataset notebooks import and display sensitivity summaries.
- [ ] Re-run notebooks after the JMLR changes if refreshed outputs are needed.
- [ ] Clear notebook outputs before commit unless executed notebooks are
  intentionally part of the snapshot.
- [ ] Record final runtime caps used for each paper artifact in the appendix or
  table captions.
- [ ] Consider one larger modern dataset before final submission:
  - recommended: UCI Credit Card Default for a larger mixed-type real dataset;
  - optional: Covertype only if a mostly numeric scaling reference is useful.

## 6. Validation

- [x] `python -m pytest tests/test_sensitivity_validation.py`
- [x] `python -m pytest tests/test_manifold_validation.py tests/test_mechanism_validation.py tests/test_experiment_workflow.py tests/test_experiment_metrics.py`
- [x] `python -m pytest tests/test_dataframe_sampler.py`
- [x] `python experiments/make_tables.py`
- [x] `python experiments/plot_results.py`
- [x] `jupyter nbconvert --clear-output --inplace experiments/notebooks/*.ipynb`
- [x] `cd publication && latexmk -pdf main.tex`

## 7. Submission Boundaries

- [x] No formal privacy claim.
- [x] No general superiority claim over deep tabular generators.
- [x] No theorem-level manifold preservation claim.
- [x] No claim that decoder probabilities are reliable without calibration
  diagnostics.
- [x] State NCA mechanism evidence only where held-out lift over PCA/majority is
  observed.
- [x] State setup-comparison results as a representative speed--accuracy
  illustration, not hyperparameter optimization.
