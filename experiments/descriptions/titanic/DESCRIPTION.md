# Titanic

This dataset contains passenger-level records from the Titanic example dataset. In these experiments it is used as a small mixed-type benchmark with demographic, ticket, cabin, and survival columns.

## Columns

### survived

`survived` indicates whether the passenger survived. It is encoded as a binary numeric label and is the primary predictive target in the Titanic experiments.

### pclass

`pclass` represents the passenger ticket class as an integer category. Lower values indicate higher ticket class, so it can be treated as an ordinal socioeconomic proxy.

### sex

`sex` records the passenger sex category present in the source data. It is a categorical demographic feature and should be treated carefully in modelling and interpretation.

### age

`age` represents the passenger age in years. It is numeric and can contain missing values in the source data.

### sibsp

`sibsp` records the number of siblings or spouses aboard with the passenger. It is a numeric family-context variable.

### parch

`parch` records the number of parents or children aboard with the passenger. It is a numeric family-context variable.

### fare

`fare` represents the passenger fare paid. It is a numeric ticket-price variable and is strongly related to class and cabin information.

### embarked

`embarked` records the port-of-embarkation code. It is a categorical travel-origin feature.

### class

`class` is a human-readable version of ticket class, such as First, Second, or Third. It duplicates the information in `pclass` in categorical form.

### who

`who` is a derived demographic grouping that distinguishes men, women, and children. It combines age and sex information into a compact categorical feature.

### adult_male

`adult_male` is a boolean indicator for adult male passengers. It is derived from age and sex and is useful as an interpretable survival-related feature.

### deck

`deck` records the cabin deck when known. It is a categorical location feature and has substantial missingness in the source data.

### embark_town

`embark_town` is the human-readable embarkation town corresponding to `embarked`. It is a categorical travel-origin feature.

### alive

`alive` is a human-readable duplicate of `survived`, using labels such as yes and no. It should not be used as a feature when `survived` is the target because it leaks the answer.

### alone

`alone` is a boolean indicator for whether the passenger travelled without family members recorded in `sibsp` or `parch`. It is a derived categorical or boolean feature.

## POTENTIAL TARGET COLUMNS

- `survived`: classification. This is the primary target for survival prediction.
- `alive`: classification. This is equivalent to `survived` and should only be used as an alternative label, not together with `survived`.
- `pclass`: ordinal classification. This can be used to predict ticket class from fare, demographics, and embarkation information.
- `class`: classification. This is the categorical equivalent of `pclass` and should not be used as a feature when predicting `pclass`.
- `age`: regression. This can be used for age imputation or demographic reconstruction tasks.
- `fare`: regression. This can be used to predict ticket fare from class, cabin, and passenger context.
- `embarked`: classification. This can be used to predict embarkation port from ticket and passenger attributes.
- `deck`: classification. This can be used as a cabin-deck prediction target, but missingness is high.
- `alone`: classification. This can be used as a derived family-travel target.
