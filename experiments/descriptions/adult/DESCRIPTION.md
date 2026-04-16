# Adult Census Income

This dataset contains census-style demographic, employment, education, and income records. In these experiments it is used as a generic mixed-type benchmark with both numeric and categorical columns.

## Columns

### age

`age` represents the person's age in years. It is a numeric demographic attribute and is useful both as a contextual feature for income prediction and as a regression target in age-imputation style tasks.

### workclass

`workclass` represents the person's employment sector or work arrangement, such as private employment, government employment, self-employment, or no pay. It is a categorical labour-market feature.

### fnlwgt

`fnlwgt` is the final census weight assigned to the record. It estimates how many people in the population the row represents and should usually be treated as a sampling or survey-design variable rather than a behavioural attribute.

### education

`education` represents the highest education category reported for the person, such as high school, bachelor's degree, or master's degree. It is a categorical version of educational attainment.

### education_num

`education_num` is the ordinal numeric encoding of educational attainment. Higher values indicate higher educational levels, making it convenient for numeric models while still reflecting an ordered category.

### marital_status

`marital_status` describes the person's marital status, such as never married, married, divorced, separated, or widowed. It is a categorical demographic attribute.

### occupation

`occupation` describes the person's broad occupation category. It is a categorical employment feature and is often strongly related to income, hours worked, and education.

### relationship

`relationship` describes the person's relationship role within the household, such as husband, wife, own child, or not in family. It is a categorical household-composition feature.

### race

`race` records the race category present in the source data. It is a sensitive demographic attribute and should be handled carefully in modelling, reporting, and synthetic-data demonstrations.

### sex

`sex` records the binary sex category present in the source data. It is a sensitive demographic attribute and should be handled carefully when used as a feature, target, or grouping variable.

### capital_gain

`capital_gain` records reported capital gains. It is a numeric financial attribute with a highly skewed distribution and many zero values.

### capital_loss

`capital_loss` records reported capital losses. It is a numeric financial attribute with a highly skewed distribution and many zero values.

### hours_per_week

`hours_per_week` represents the number of hours the person usually works per week. It is a numeric labour-market variable and can support both income prediction and workload modelling.

### native_country

`native_country` records the person's country of origin as provided in the source data. It is a categorical demographic attribute and may be sensitive depending on the application.

### income

`income` is the binary income band label, indicating whether income is at most 50K or above 50K. It is the primary predictive target used in the Adult experiments.

## POTENTIAL TARGET COLUMNS

- `income`: classification. This is the primary target for predicting whether a record falls in the `<=50K` or `>50K` income band.
- `workclass`: classification. This can be used for employment-sector prediction from demographic, education, and income-related features.
- `education`: classification. This can be used to predict education category from employment, age, and household features.
- `education_num`: ordinal classification or regression. It can be treated as an ordered education-level target.
- `marital_status`: classification. This can be predicted from demographic and household context, though it should not be over-interpreted.
- `occupation`: classification. This can be used for occupation-category prediction from education, workclass, hours, and demographic features.
- `hours_per_week`: regression. This can be used as a workload prediction target.
- `capital_gain`: regression. This can be used as a skewed financial-value target, but zero inflation should be considered.
- `capital_loss`: regression. This can be used as a skewed financial-value target, but zero inflation should be considered.
