# UCI Heart Disease

This dataset contains the processed Cleveland subset of the UCI Heart Disease data. It combines demographic, clinical, electrocardiographic, exercise-test, and angiographic variables. In these experiments it is used as a small medical classification benchmark. The original multi-level disease score is converted into a binary `heart_disease` label.

## Columns

### age

`age` represents the patient age in years. It is a numeric demographic feature.

### sex

`sex` represents the patient sex category as provided by the source data. It is a categorical demographic feature.

### chest_pain

`chest_pain` represents the reported chest-pain type, mapped to readable categories such as typical angina, atypical angina, non-anginal pain, and asymptomatic. It is a categorical symptom feature.

### resting_blood_pressure

`resting_blood_pressure` represents resting blood pressure in mm Hg. It is a numeric cardiovascular measurement.

### cholesterol

`cholesterol` represents serum cholesterol in mg/dl. It is a numeric laboratory measurement.

### fasting_blood_sugar

`fasting_blood_sugar` indicates whether fasting blood sugar is above 120 mg/dl. It is a categorical clinical-threshold feature.

### resting_ecg

`resting_ecg` represents the resting electrocardiographic result category. It is a categorical cardiac-test feature.

### max_heart_rate

`max_heart_rate` represents the maximum heart rate achieved during exercise testing. It is a numeric exercise-test measurement.

### exercise_angina

`exercise_angina` indicates whether exercise-induced angina was observed. It is a categorical exercise-test feature.

### oldpeak

`oldpeak` represents ST depression induced by exercise relative to rest. It is a numeric exercise-test measurement.

### slope

`slope` represents the slope of the peak exercise ST segment. It is a categorical exercise-test feature.

### major_vessels

`major_vessels` represents the number of major vessels colored by fluoroscopy. It is a numeric angiographic feature and may contain missing values.

### thal

`thal` represents the thalassemia-related test category, mapped to readable categories such as normal, fixed defect, and reversible defect. It is a categorical clinical-test feature and may contain missing values.

### heart_disease

`heart_disease` is the binary label derived from the original diagnosis score. It indicates absence or presence of heart disease and is the primary predictive target for this dataset.

## POTENTIAL TARGET COLUMNS

- `heart_disease`: classification. This is the primary target for heart-disease presence prediction.
- `chest_pain`: classification. This can be used to predict symptom category from demographic and clinical measurements.
- `exercise_angina`: classification. This can be used to predict exercise-induced angina from test and clinical context.
- `resting_ecg`: classification. This can be used to predict electrocardiographic category from other measurements.
- `thal`: classification. This can be used to predict thalassemia-related test category, with care because missingness is present.
- `age`: regression. This can be used as a demographic reconstruction target, though it is rarely the primary clinical task.
- `resting_blood_pressure`: regression. This can be used as a cardiovascular measurement prediction target.
- `cholesterol`: regression. This can be used as a laboratory measurement prediction target.
- `max_heart_rate`: regression. This can be used as an exercise-test measurement prediction target.
- `oldpeak`: regression. This can be used as an exercise-test measurement prediction target.
- `major_vessels`: regression or ordinal classification. This can be used as an angiographic count prediction target.
