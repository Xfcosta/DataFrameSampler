# Pima Indians Diabetes

This dataset contains clinical measurements and a binary diabetes outcome. In these experiments it is used as a compact medical classification benchmark with missing clinical measurements. During preprocessing, zero values in measurement fields where zero is not clinically meaningful are converted to missing values.

## Columns

### pregnancies

`pregnancies` represents the number of times the patient has been pregnant. It is a numeric reproductive-history feature.

### glucose

`glucose` represents plasma glucose concentration measured during an oral glucose tolerance test. It is a numeric metabolic measurement and one of the strongest predictors of the diabetes label.

### blood_pressure

`blood_pressure` represents diastolic blood pressure. It is a numeric cardiovascular measurement.

### skin_thickness

`skin_thickness` represents triceps skinfold thickness. It is a numeric body-composition measurement and has missing values after preprocessing impossible zero values.

### insulin

`insulin` represents two-hour serum insulin. It is a numeric metabolic measurement and has substantial missingness after preprocessing impossible zero values.

### bmi

`bmi` represents body mass index. It is a numeric body-composition measurement and is treated as missing when the source value is an impossible zero.

### diabetes_pedigree

`diabetes_pedigree` represents a pedigree-function score related to diabetes family history. It is a numeric risk-context feature.

### age

`age` represents the patient's age in years. It is a numeric demographic feature.

### diabetes

`diabetes` is the binary outcome label indicating whether the patient is recorded as positive or negative for diabetes in the source data. It is the primary classification target in this dataset.

## POTENTIAL TARGET COLUMNS

- `diabetes`: classification. This is the primary target for diabetes risk classification.
- `glucose`: regression. This can be used as a clinical-measurement prediction or imputation target.
- `blood_pressure`: regression. This can be used as a cardiovascular measurement prediction target.
- `skin_thickness`: regression. This can be used as a body-composition measurement prediction or imputation target.
- `insulin`: regression. This can be used as a metabolic measurement prediction or imputation target, with care because missingness is substantial.
- `bmi`: regression. This can be used as a body-composition prediction target.
- `age`: regression. This can be used as a demographic reconstruction target, though it is rarely the primary clinical task.
- `pregnancies`: regression or ordinal classification. This can be used as a reproductive-history prediction target when appropriate.
