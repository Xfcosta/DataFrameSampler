# TODO: Claim-Specific Evidence Plan For The DataFrameSampler Paper

This checklist translates the claim-specific evidence framework from
`from_benchmark_wins_to_claim_specific_evidence.pdf` into concrete work for the
DataFrameSampler paper. The guiding rule is: every major claim needs primary
evidence, supporting evidence, boundary evidence, and an allowable conclusion.
If evidence is missing, the claim must be weakened.

## 1. Lock The Paper Claim

- [x] State the main claim as a narrow practical-use and mechanism-supported
  claim, not as general synthetic-data superiority.
- [x] Proposed claim:
  "DataFrameSampler provides a simple, inspectable way to create mixed-type
  tabular examples for business and medical software workflows; its generated
  rows can be explained through local-neighbour bin-space operations, while
  utility and privacy-safeguard behaviour remain conditional on dataset and
  configuration."
- [x] Explicitly avoid unsupported claims:
  - [x] "privacy preserving"
  - [x] "formally anonymized"
  - [x] "better than deep synthetic data generators"
  - [x] "clinically valid simulator"
  - [x] "generally superior synthetic data method"

## 2. Evidence Roles

### Primary Evidence

- [ ] Usability/ease-of-use evidence:
  - [ ] number of lines/commands needed for common workflows;
  - [ ] time-to-first-sample from CSV or dataframe;
  - [ ] success rate of default and LLM-assisted configuration;
  - [ ] comparison against setup complexity of competitor packages.
- [ ] Inspectability evidence:
  - [x] trace generated rows back to anchor row, neighbour chain, latent
  difference, bins, and decoded values;
  - [ ] show examples of generated-row explanations;
  - [ ] measure how often generated rows have complete trace information.
- [ ] Practical utility evidence:
  - [x] nearest-neighbour distance test against natural real-real distances;
  - [x] discrimination test for real-versus-synthetic distinguishability;
  - [x] utility lift test from adding generated rows to real training data;
  - [x] distribution similarity test using histogram overlap and divergence;
  - [ ] schema/interface preservation for software testing.
- [x] Safeguard evidence:
  - [x] exact source-value overlap before and after anonymization;
  - [x] consistency of repeated sensitive-value replacement;
  - [x] failure when LLM replacement collides with source values.

### Supporting Evidence

- [ ] Mechanism ablations:
  - [ ] frequency categorical encoding only;
  - [ ] helper-column embedding;
  - [ ] PCA embedding;
  - [ ] MDS embedding;
  - [ ] nonlinear embedding option, for example Isomap or Kernel PCA;
  - [ ] mutual neighbours only;
  - [ ] mutual neighbours plus one-nearest-neighbour fallback;
  - [ ] exact KNN versus approximate KNN backend.
- [ ] Configuration evidence:
  - [ ] manual vectorizing-column configuration;
  - [x] LLM-assisted configuration with user overrides;
  - [ ] invalid or weak LLM recommendation rate;
  - [ ] configuration reproducibility across prompts/seeds/models if feasible.
- [ ] Error analysis:
  - [ ] examples where generated rows are implausible;
  - [ ] columns with poor marginal preservation;
  - [ ] rare-category loss or over-reuse;
  - [ ] datasets where helper embeddings hurt.

### Boundary Evidence

- [x] Runtime and memory:
  - [x] fit time and sample time versus row count;
  - [ ] exact KNN versus approximate KNN runtime;
  - [x] memory footprint for fitted sampler.
- [ ] Dataset regimes:
  - [ ] small tables;
  - [ ] medium tables;
  - [ ] high-cardinality categorical columns;
  - [ ] many numeric columns;
  - [ ] missing values;
  - [ ] imbalanced categorical values.
- [x] Privacy limits:
  - [x] source-value reuse for non-anonymized columns;
  - [x] nearest-neighbour memorization risk;
  - [x] statement that exact-overlap checks do not prevent linkage attacks.
- [x] Medical/business governance limits:
  - [x] no claim of HIPAA/GDPR compliance;
  - [x] no claim of clinical validity;
  - [x] require local disclosure and governance review.

## 3. Required Figures

- [x] Figure 1, WHAT: intended use context.
  - [x] Show real source table, governance constraint, synthetic example table,
  and downstream uses: tests, dashboards, demos, notebooks.
  - [x] Include business and medical examples visually without implying formal
  de-identification.
- [ ] Figure 2, WHY: why black-box benchmark wins are not the point.
  - [ ] Contrast opaque generator output with an inspectable generated row.
  - [ ] Show the failure mode: teams need realistic examples they can explain,
  not only high benchmark scores.
- [x] Figure 3, HOW: DataFrameSampler pipeline.
  - [x] dataframe input;
  - [x] optional anonymization of selected columns;
  - [x] numeric conversion and categorical vectorization;
  - [x] bin encoding;
  - [x] neighbour-chain sampling;
  - [x] bin decoding;
  - [x] output dataframe;
  - [x] explanation trace.
- [ ] Figure 4: generated-row trace example.
  - [ ] One source anchor row;
  - [ ] neighbour rows;
  - [ ] latent difference;
  - [ ] generated latent bins;
  - [ ] decoded output row.
- [x] Figure 5: distributional similarity dashboard.
  - [x] numeric histogram overlays;
  - [x] categorical frequency bars;
  - [x] correlation heatmap difference.
- [x] Figure 6: utility versus simplicity/cost frontier.
  - [x] x-axis: runtime, configuration effort, or code complexity;
  - [x] y-axis: utility/similarity score;
  - [x] compare DataFrameSampler with baselines and competitor generators.
- [ ] Figure 7: anonymization safeguard check.
  - [ ] source sensitive values;
  - [ ] surrogate values;
  - [ ] zero-overlap check;
  - [ ] collision failure example.
- [x] Figure 8: LLM-assisted configuration flow.
  - [x] dataframe profile;
  - [x] LLM recommendation dictionary;
  - [x] user overrides;
  - [x] sampler configuration.

## 4. Required Tables

- [x] Table 1: claim-to-evidence matrix for this paper.
  - [x] Rows: simplicity, inspectability, practical utility, anonymization
  safeguard, LLM configuration.
  - [x] Columns: claim type, primary evidence, supporting evidence, boundary
  evidence, allowable conclusion.
- [x] Table 2: datasets.
  - [x] dataset name;
  - [x] domain;
  - [x] rows;
  - [x] numeric columns;
  - [x] categorical columns;
  - [x] missingness;
  - [x] sensitive columns if any;
  - [x] inclusion rationale.
- [x] Table 3: competitor and baseline methods.
  - [x] method;
  - [x] package;
  - [x] model family;
  - [x] handles mixed types;
  - [x] setup effort;
  - [x] inspectability;
  - [x] optional dependencies.
- [x] Table 4: distributional similarity results.
  - [x] per dataset and method;
  - [x] numeric distance;
  - [x] categorical divergence;
  - [x] correlation difference;
  - [x] privacy/similarity caveat.
- [x] Table 5: downstream utility results.
  - [x] train-on-synthetic, test-on-real;
  - [x] real-train baseline;
  - [x] bootstrap baseline;
  - [x] method comparisons;
  - [x] uncertainty intervals.
- [x] Table 6: usability and runtime.
  - [x] commands/LOC;
  - [x] fit time;
  - [x] sample time;
  - [x] memory;
  - [x] tuning/configuration steps.
- [x] Table 7: ablations.
  - [x] component removed or changed;
  - [x] expected effect;
  - [x] observed effect;
  - [x] claim supported, weakened, or bounded.
- [x] Table 8: limitations and allowable conclusions.
  - [x] what is known;
  - [x] what is not known;
  - [x] scope of claim permitted.

## 5. Datasets To Use

- [ ] Business-like datasets:
  - [ ] customer churn dataset;
  - [ ] credit scoring or loan approval dataset;
  - [ ] sales, transactions, or retail dataset;
  - [ ] employee/HR-style dataset if available.
- [ ] Medical or health-informatics datasets:
  - [ ] UCI Heart Disease;
  - [ ] diabetes readmission or diabetes classification;
  - [ ] breast cancer diagnostic dataset;
  - [ ] MIMIC-derived public benchmark only if governance and preprocessing are
  practical.
- [ ] Generic mixed-type benchmarks:
  - [x] Adult Census Income;
  - [x] Titanic;
  - [ ] Bank Marketing;
  - [ ] OpenML mixed-type classification datasets.
- [x] Synthetic controlled datasets:
  - [x] correlated numeric plus categorical helper columns;
  - [x] high-cardinality categorical column;
  - [x] known rare-category structure;
  - [x] known sensitive identifier column.

## 6. Baselines

### Simple Baselines

- [x] Row bootstrap with replacement.
- [x] Independent column-wise sampler.
- [x] Stratified column-wise sampler if target labels exist.
- [x] Gaussian/copula baseline for numeric columns plus categorical empirical sampling.
- [x] SMOTE/SMOTENC for supervised tabular augmentation where applicable.

### Competitor Synthetic-Data Methods

- [ ] SDV GaussianCopula.
- [ ] SDV CTGAN.
- [ ] SDV TVAE.
- [ ] SDV CopulaGAN if stable in the environment.
- [ ] Synthcity tabular plugins if installation is practical.
- [ ] ydata-synthetic only if dependency burden is acceptable.

### Configuration Competitors

- [x] DataFrameSampler manual configuration.
- [x] DataFrameSampler default configuration.
- [x] DataFrameSampler LLM-assisted configuration.
- [ ] Competitor defaults with minimal tuning.
- [ ] Competitor tuned under a fixed budget.

## 7. Primary Metrics

The paper now treats four tests as the main measures. Lower-level
distributional and downstream metrics remain supporting diagnostics.

### Nearest-Neighbor Distance Test

- [x] Minimal distance from each synthetic row to real rows.
- [x] Natural nearest-neighbour distance inside the original real table.
- [x] Ratio of synthetic-to-real distance against natural real-real distance.
- [x] Suspiciously-close rate based on the lower tail of natural real-real
  distances.

### Discrimination Test

- [x] Train a classifier to distinguish real from synthetic rows.
- [x] Report discrimination accuracy.
- [x] Report ROC AUC where possible.
- [x] Report a chance-proximity privacy score derived from accuracy.

### Utility Lift Test

- [x] Train a model on real training data and evaluate on held-out real data.
- [x] Train a model on real plus generated rows and evaluate on the same
  held-out real data.
- [x] Report the score difference as utility lift.

### Distribution Similarity Test

- [x] Numeric histogram overlap.
- [x] Numeric histogram KL divergence.
- [x] Categorical Jensen-Shannon divergence.
- [x] Aggregate distribution similarity score.

### Supporting Distributional Similarity

- [x] Numeric:
  - [x] mean/standard deviation error;
  - [x] Kolmogorov-Smirnov statistic;
  - [x] Wasserstein distance;
  - [x] histogram overlap.
- [x] Categorical:
  - [x] total variation distance;
  - [x] Jensen-Shannon divergence;
  - [x] category coverage;
  - [x] rare-category preservation.
- [x] Dependence:
  - [x] Pearson/Spearman correlation difference;
  - [x] Cramer's V or Theil's U for categorical associations;
  - [x] mixed-type association matrix difference.

### Supporting Downstream Utility

- [x] Train-on-synthetic, test-on-real score.
- [x] Train-on-real, test-on-real upper reference.
- [x] Train-on-bootstrap, test-on-real baseline.
- [x] Classification metrics:
  - [x] accuracy;
  - [x] ROC AUC;
  - [x] F1;
  - [x] Brier score if probabilities are used.
- [x] Regression metrics if regression datasets are added:
  - [x] MAE;
  - [x] RMSE;
  - [x] R2.

### Inspectability And Practicality

- [x] Explanation trace completeness rate.
- [x] Average trace size per generated row.
- [x] Number of user configuration choices required.
- [x] Lines of Python for common workflow.
- [x] CLI command length for common workflow.
- [x] Fit/sample wall-clock time.
- [x] Peak memory where practical.

### Anonymization Safeguard

- [x] Exact source-value overlap count.
- [x] Normalized source-value overlap count.
- [x] Collision rate in LLM replacements.
- [x] Repeated-value consistency rate.
- [x] Manual review of surrogate plausibility.

## 8. Statistical And Reporting Protocol

- [ ] Use fixed train/test splits or repeated cross-validation per dataset.
- [ ] Use identical splits for all methods.
- [ ] Separate tuning data from final evaluation data.
- [ ] Report random seeds.
- [ ] Report package versions.
- [ ] Report hardware.
- [ ] Report tuning budgets for competitors.
- [ ] Report per-dataset raw scores, not only averages.
- [ ] Report paired differences versus relevant baselines.
- [ ] Use uncertainty intervals over seeds/splits.
- [ ] For multi-dataset comparisons, summarize wins/losses/ties and paired
  differences rather than relying only on ranks.
- [ ] State the unit of uncertainty:
  - [ ] seeds;
  - [ ] splits;
  - [ ] datasets;
  - [ ] generated samples.

## 9. Ablation Plan

- [ ] Remove anonymization layer.
- [ ] Replace LLM-assisted config with defaults.
- [ ] Replace helper-column embeddings with frequency encoding.
- [ ] Compare embedding methods:
  - [ ] PCA;
  - [ ] MDS;
  - [ ] Isomap;
  - [ ] Kernel PCA if practical;
  - [ ] custom transformer object.
- [ ] Compare KNN backends:
  - [ ] exact;
  - [ ] sklearn;
  - [ ] pynndescent;
  - [ ] hnswlib;
  - [ ] annoy.
- [ ] Disable one-nearest-neighbour fallback.
- [ ] Vary number of bins.
- [ ] Vary interpolation range.
- [ ] Vary number of neighbours.
- [ ] Use different sampled column subsets.

## 10. Experiments To Implement

- [ ] E1: Quick-start usability.
  - [ ] Compare setup steps and LOC for DataFrameSampler, SDV GaussianCopula,
  CTGAN, and TVAE.
  - [ ] Primary claim tested: simplicity of use.
- [ ] E2: Traceable generation case study.
  - [ ] Generate a small business-like and medical-like table.
  - [ ] Show row-level traces for 5 generated examples.
  - [ ] Primary claim tested: human-auditable generation.
- [ ] E3: Four-measure benchmark.
  - [ ] Run all simple baselines, DataFrameSampler variants, and feasible
  competitor methods on all selected datasets.
  - [ ] Report nearest-neighbour distance, discrimination, utility lift, and
  distribution similarity tests.
  - [ ] Primary claim tested: generated data is plausible, not suspiciously
  close, hard to distinguish when successful, and useful only when utility lift
  is observed.
- [ ] E4: Utility lift.
  - [ ] Train simple models on real data with and without generated rows and
  test on real holdout data.
  - [ ] Use random forest by default and add other models if feasible.
  - [ ] Supporting claim tested: generated data improves modelling utility in
  some regimes without relying on overfitted copies.
- [ ] E5: Anonymization safeguard.
  - [ ] Add sensitive columns to selected datasets or use existing identifier
  columns.
  - [ ] Compare generation before and after anonymization.
  - [ ] Test zero exact/normalized overlap for selected columns.
  - [ ] Boundary claim tested: safeguard prevents exact re-emission only for
  selected values.
- [ ] E6: LLM-assisted configuration.
  - [ ] Compare defaults, manual configuration, and LLM-assisted configuration.
  - [ ] Measure syntactic validity, compatible column choices, user override
  preservation, and downstream similarity/utility.
  - [ ] Primary claim tested: LLM improves ease of configuration, not modelling
  superiority.
- [ ] E7: Scalability and KNN backend comparison.
  - [ ] Vary row count and number of columns.
  - [ ] Compare exact KNN with approximate KNN backends.
  - [ ] Boundary claim tested: approximate backends improve runtime while
  possibly changing local graph behaviour.
- [ ] E8: Failure-mode study.
  - [ ] Identify datasets where DataFrameSampler performs poorly.
  - [ ] Explain whether the issue comes from high cardinality, weak helper
  columns, sparse bins, rare categories, or neighbour structure.
  - [ ] Synthesis purpose: sharpen the allowable conclusion.

## 11. Paper Sections To Revise

- [x] Abstract:
  - [x] state claim as practical-use plus inspectability;
  - [x] avoid broad superiority wording.
- [x] Introduction:
  - [x] add explicit WHAT-WHY-HOW paragraphs;
  - [x] add a claim-to-evidence preview.
- [x] Related Work:
  - [x] separate deep synthetic-data generators from simple transparent
  baselines;
  - [x] discuss SDV, CTGAN, TVAE, copulas, bootstrap, SMOTE/SMOTENC;
  - [x] discuss privacy-preserving synthetic data separately and state that this
  paper does not make that formal claim.
- [x] Method:
  - [x] add row-level trace definition;
  - [x] add pseudocode for fit/sample/anonymize;
  - [x] identify which mechanisms predict which empirical effects.
- [ ] Experiments:
  - [ ] replace stub with claim-specific experiment design;
  - [ ] define dataset inclusion criteria;
  - [ ] define metrics and uncertainty procedure;
  - [ ] define tuning budgets.
- [ ] Results:
  - [ ] add distributional, utility, usability, safeguard, runtime, and ablation
  results.
- [x] Discussion:
  - [x] write the integrated summary;
  - [x] state where the method should and should not be preferred.
- [x] Limitations:
  - [x] add a dedicated limitations section if space allows.
- [ ] Reproducibility:
  - [ ] include package versions, scripts, seeds, and data-preparation details.

## 12. Implementation Work Needed

- [ ] Add an experiment runner under `experiments/`.
- [x] Add dataset download/preparation scripts.
- [ ] Add a common method interface for baselines and competitors.
- [x] Add metric implementations for mixed-type similarity.
- [ ] Add row-trace export from `DataFrameSampler.sample`.
- [ ] Add benchmark configuration files.
- [ ] Add result caching.
- [ ] Add plotting scripts for all required figures.
- [ ] Add table-generation scripts for LaTeX.
- [ ] Add a reproducibility README for experiments.

## 13. Allowable Conclusion Template

Use this form when writing the final paper:

> DataFrameSampler supports a practical-use and mechanism-supported claim: under
> the studied business, medical, and mixed-type tabular datasets, it provides a
> simple and inspectable mechanism for generating example data, with utility
> comparable to simple baselines in specified regimes and with clearer row-level
> explanations than opaque generators. The anonymization layer reduces exact
> re-emission of selected sensitive values but does not provide formal privacy.
> The evidence does not support a claim of general superiority over all
> synthetic-data methods or clinical-grade de-identification.
