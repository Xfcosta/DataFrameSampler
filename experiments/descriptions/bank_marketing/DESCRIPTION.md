# Bank Marketing

This dataset contains records from direct marketing campaigns of a Portuguese banking institution. The rows combine client attributes, contact history, previous campaign outcomes, and macroeconomic indicators. In these experiments it is used as a mixed-type business classification benchmark.

## Columns

### age

`age` represents the client age in years. It is a numeric demographic feature.

### job

`job` represents the client's occupation category. It is a categorical socioeconomic feature and may contain missing values where the source value was unknown.

### marital

`marital` represents the client's marital status. It is a categorical demographic feature.

### education

`education` represents the client's education level. It is a categorical socioeconomic feature and may contain missing values where the source value was unknown.

### default

`default` records whether the client has credit in default according to the source data. It is a categorical banking-risk feature and may contain missing values.

### housing

`housing` records whether the client has a housing loan. It is a categorical financial-product feature and may contain missing values.

### loan

`loan` records whether the client has a personal loan. It is a categorical financial-product feature and may contain missing values.

### contact

`contact` represents the communication channel used for the campaign contact, such as cellular or telephone. It is a categorical campaign-process feature.

### month

`month` represents the month of the last contact in the campaign. It is a categorical calendar feature.

### day_of_week

`day_of_week` represents the day of week of the last campaign contact. It is a categorical calendar feature.

### duration

`duration` records the duration of the last contact in seconds. It is a numeric campaign-interaction feature and is highly predictive of subscription, although it may not be known before a call is completed.

### campaign

`campaign` records the number of contacts performed during the current campaign for the client. It is a numeric contact-intensity feature.

### pdays

`pdays` records the number of days since the client was last contacted in a previous campaign. A large sentinel value indicates no previous contact in the source data.

### previous

`previous` records the number of contacts performed before the current campaign for the client. It is a numeric prior-contact feature.

### poutcome

`poutcome` records the outcome of the previous marketing campaign for the client. It is a categorical historical-outcome feature.

### emp_var_rate

`emp_var_rate` represents the employment variation rate at the time of the campaign. It is a numeric macroeconomic context feature.

### cons_price_idx

`cons_price_idx` represents the consumer price index at the time of the campaign. It is a numeric macroeconomic context feature.

### cons_conf_idx

`cons_conf_idx` represents the consumer confidence index at the time of the campaign. It is a numeric macroeconomic context feature.

### euribor3m

`euribor3m` represents the three-month Euribor rate. It is a numeric macroeconomic context feature.

### nr_employed

`nr_employed` represents the number of employees in the economy indicator used by the source dataset. It is a numeric macroeconomic context feature.

### subscribed

`subscribed` records whether the client subscribed to the term deposit product. It is the primary predictive target for this dataset.

## POTENTIAL TARGET COLUMNS

- `subscribed`: classification. This is the primary target for predicting term-deposit subscription.
- `poutcome`: classification. This can be used to predict previous campaign outcome from client and contact features.
- `job`: classification. This can be used as an occupation-category prediction target, though it is usually more natural as a feature.
- `education`: classification. This can be used as an education-level prediction target.
- `housing`: classification. This can be used to predict whether a client has a housing loan.
- `loan`: classification. This can be used to predict whether a client has a personal loan.
- `duration`: regression. This can be used to predict call duration from campaign and client context, but it should be treated carefully because it is post-contact information.
- `campaign`: regression or ordinal classification. This can be used to predict contact intensity during the campaign.
- `previous`: regression or ordinal classification. This can be used to predict prior contact count.
