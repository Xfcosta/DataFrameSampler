# Forest Covertype

This dataset contains cartographic measurements for forest cells and a cover-type label. In these experiments it is used as a larger public benchmark that complements the mid-sized Adult and Bank Marketing datasets. The prepared version keeps the continuous terrain measurements and collapses the source one-hot wilderness and soil indicators back into categorical columns.

## Columns

### elevation

`elevation` is the elevation of the cell. It is a numeric terrain feature.

### aspect

`aspect` is the compass aspect of the terrain. It is a numeric terrain-orientation feature.

### slope

`slope` is the terrain slope. It is a numeric terrain-steepness feature.

### horizontal_distance_to_hydrology

`horizontal_distance_to_hydrology` is the horizontal distance from the cell to the nearest surface-water feature.

### vertical_distance_to_hydrology

`vertical_distance_to_hydrology` is the vertical distance from the cell to the nearest surface-water feature.

### horizontal_distance_to_roadways

`horizontal_distance_to_roadways` is the horizontal distance from the cell to the nearest roadway.

### hillshade_9am

`hillshade_9am` is a numeric hillshade index measured for morning illumination.

### hillshade_noon

`hillshade_noon` is a numeric hillshade index measured for noon illumination.

### hillshade_3pm

`hillshade_3pm` is a numeric hillshade index measured for afternoon illumination.

### horizontal_distance_to_fire_points

`horizontal_distance_to_fire_points` is the horizontal distance from the cell to fire ignition points.

### wilderness_area

`wilderness_area` is the categorical wilderness-area code reconstructed from the source one-hot indicators.

### soil_type

`soil_type` is the categorical soil-type code reconstructed from the source one-hot indicators.

### cover_type

`cover_type` is the forest cover-type class label. It is the primary predictive target for this dataset.

## POTENTIAL TARGET COLUMNS

- `cover_type`: classification. This is the primary target for predicting forest cover type from terrain and categorical context.
- `soil_type`: classification. This can be used to predict soil type from terrain and wilderness context.
- `wilderness_area`: classification. This can be used to predict wilderness area from terrain and soil context.
- `elevation`: regression. This can be used as a continuous terrain target, though it is usually more natural as a feature.
