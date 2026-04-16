# Wisconsin Diagnostic Breast Cancer

This dataset contains diagnostic features computed from digitized images of fine needle aspirate samples of breast masses. In these experiments it is used as a small medical binary-classification benchmark. The original identifier is dropped during preprocessing.

## Columns

### diagnosis

`diagnosis` represents the clinical diagnostic class in the source dataset: benign or malignant. It is the primary classification target for this dataset.

### mean_radius

`mean_radius` represents the average radius of cell nuclei in the sample image. Radius summarizes the average distance from the nucleus center to its perimeter.

### mean_texture

`mean_texture` represents the average texture of cell nuclei. Texture summarizes variation in gray-scale intensity in the nucleus image region.

### mean_perimeter

`mean_perimeter` represents the average perimeter of cell nuclei. It captures typical boundary length and is related to nucleus size.

### mean_area

`mean_area` represents the average area of cell nuclei. It captures typical nucleus size in the sample.

### mean_smoothness

`mean_smoothness` represents the average local variation in nucleus radius lengths. It measures boundary smoothness or irregularity.

### mean_compactness

`mean_compactness` represents the average compactness of cell nuclei. It is derived from perimeter and area and reflects how compact or spread out the nucleus shape is.

### mean_concavity

`mean_concavity` represents the average severity of concave portions of the nucleus contour. Higher values indicate stronger inward boundary irregularities.

### mean_concave_points

`mean_concave_points` represents the average number or extent of concave points on the nucleus contour. It captures localized boundary indentations.

### mean_symmetry

`mean_symmetry` represents the average symmetry of cell nuclei. It summarizes how balanced the nucleus shapes are.

### mean_fractal_dimension

`mean_fractal_dimension` represents the average fractal-dimension estimate of nucleus contours. It is a measure of boundary complexity.

### se_radius

`se_radius` represents the standard error of nucleus radius measurements across the sample. It captures variability in radius estimates.

### se_texture

`se_texture` represents the standard error of texture measurements. It captures variability in gray-scale intensity variation.

### se_perimeter

`se_perimeter` represents the standard error of perimeter measurements. It captures variability in nucleus boundary length.

### se_area

`se_area` represents the standard error of area measurements. It captures variability in nucleus size.

### se_smoothness

`se_smoothness` represents the standard error of smoothness measurements. It captures variability in boundary smoothness.

### se_compactness

`se_compactness` represents the standard error of compactness measurements. It captures variability in compactness across nuclei.

### se_concavity

`se_concavity` represents the standard error of concavity measurements. It captures variability in boundary concavity.

### se_concave_points

`se_concave_points` represents the standard error of concave-point measurements. It captures variability in localized contour indentations.

### se_symmetry

`se_symmetry` represents the standard error of symmetry measurements. It captures variability in nucleus symmetry.

### se_fractal_dimension

`se_fractal_dimension` represents the standard error of fractal-dimension measurements. It captures variability in contour complexity.

### worst_radius

`worst_radius` represents the largest or most extreme radius value summarized from the sample. It captures the high-end radius behaviour among measured nuclei.

### worst_texture

`worst_texture` represents the largest or most extreme texture value summarized from the sample. It captures high-end image texture variation.

### worst_perimeter

`worst_perimeter` represents the largest or most extreme perimeter value summarized from the sample. It captures high-end nucleus boundary length.

### worst_area

`worst_area` represents the largest or most extreme area value summarized from the sample. It captures high-end nucleus size.

### worst_smoothness

`worst_smoothness` represents the largest or most extreme smoothness value summarized from the sample. It captures high-end boundary irregularity.

### worst_compactness

`worst_compactness` represents the largest or most extreme compactness value summarized from the sample. It captures high-end compactness behaviour.

### worst_concavity

`worst_concavity` represents the largest or most extreme concavity value summarized from the sample. It captures strong contour concavity.

### worst_concave_points

`worst_concave_points` represents the largest or most extreme concave-point value summarized from the sample. It captures strong localized contour indentation.

### worst_symmetry

`worst_symmetry` represents the largest or most extreme symmetry value summarized from the sample. It captures high-end symmetry behaviour.

### worst_fractal_dimension

`worst_fractal_dimension` represents the largest or most extreme fractal-dimension value summarized from the sample. It captures high-end boundary complexity.

## POTENTIAL TARGET COLUMNS

- `diagnosis`: classification. This is the primary target for benign-versus-malignant diagnostic prediction.
- `mean_radius`, `mean_texture`, `mean_perimeter`, and `mean_area`: regression. These can be used as measurement-reconstruction targets from the other image-derived features.
- `worst_radius`, `worst_texture`, `worst_perimeter`, and `worst_area`: regression. These can be used to predict extreme measurement summaries from the mean and standard-error features.
- `mean_smoothness`, `mean_compactness`, `mean_concavity`, `mean_concave_points`, `mean_symmetry`, and `mean_fractal_dimension`: regression. These can be used as shape-property prediction targets.
- `se_radius`, `se_texture`, `se_perimeter`, `se_area`, `se_smoothness`, `se_compactness`, `se_concavity`, `se_concave_points`, `se_symmetry`, and `se_fractal_dimension`: regression. These can be used as variability-estimation targets, although they are less natural as primary application targets.
