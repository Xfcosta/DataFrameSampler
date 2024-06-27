from collections import defaultdict
from scipy.sparse.csr import csr_matrix
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
from scipy.stats import rankdata
from scipy.stats import ttest_ind
from sklearn.manifold import MDS
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import KBinsDiscretizer
import click
import copy
import numpy as np
import pandas as pd
import pickle
import scipy as sp
import seaborn as sns
import yaml
import sys

def yaml_load(fname='data.txt'):
    with open(fname, "r") as fh:
        python_object = yaml.load(fh, Loader=yaml.SafeLoader)
    return python_object

class NearestMutualNeighboursEstimator(object):
    """
    A class to estimate the nearest mutual neighbours of data points using a specified metric.

    Attributes:
    n_neighbours (int): The number of nearest neighbours to find.
    metric (str): The distance metric to use for finding nearest neighbours.
    """

    def __init__(self, n_neighbours=10, metric='euclidean'):
        """
        Initializes the NearestMutualNeighboursEstimator with specified parameters.

        Parameters:
        n_neighbours (int, optional): The number of nearest neighbours to find. Default is 10.
        metric (str, optional): The distance metric to use for finding nearest neighbours. Default is 'euclidean'.
        """
        self.n_neighbours = n_neighbours
        self.metric = metric

    def dist(self, objects, metric='euclidean', diagval=np.inf):
        """
        Computes the pairwise distances between objects using the specified metric.

        Parameters:
        objects (array-like): The data points to compute distances between.
        metric (str, optional): The distance metric to use. Default is 'euclidean'.
        diagval (float, optional): The value to fill in the diagonal of the distance matrix. Default is infinity.

        Returns:
        array-like: The distance matrix with diagonal values set to diagval.
        """
        # Compute pairwise distances using the specified metric
        distvec = pdist(objects, metric=metric)

        # Convert the condensed distance matrix to a square matrix
        out = squareform(distvec)

        # Set the diagonal values to diagval
        np.fill_diagonal(out, diagval)

        return out

    def fit(self, X, y=None):
        """
        Fits the estimator to the data. This method is currently a no-op.

        Parameters:
        X (array-like): The feature matrix.
        y (array-like, optional): The target vector. Default is None.

        Returns:
        self: Returns the instance itself.
        """
        return self

    def fit_predict_single(self, X):
        """
        Fits the estimator to the data and predicts the nearest mutual neighbours for each data point.

        Parameters:
        X (array-like): The feature matrix.

        Returns:
        list: A list of arrays containing the indices of the nearest mutual neighbours for each data point.
        """
        # Compute the pairwise distance matrix for the data points
        pdists = self.dist(X, self.metric)

        # Sort the distances to find the nearest neighbours for each data point
        nearest_neighbours = np.argsort(pdists)

        # Select the indices of the k-nearest neighbours
        k_nearest_neighbours = nearest_neighbours[:, :self.n_neighbours]

        # Initialize a mask to denote the k-nearest neighbours
        k_nearest_mutual_neighbours_mask = np.zeros(pdists.shape, bool)
        for _mask_row, _neighbours_row in zip(k_nearest_mutual_neighbours_mask, k_nearest_neighbours):
            _mask_row[_neighbours_row] = True

        # Perform element-wise AND with the transposed mask to remove non-mutual nearest neighbours
        k_nearest_mutual_neighbours_mask &= k_nearest_mutual_neighbours_mask.T

        # Extract the indices of the mutual nearest neighbours
        k_nearest_mutual_neighbours = [np.where(row == True)[0] for row in k_nearest_mutual_neighbours_mask]

        return k_nearest_mutual_neighbours

    def fit_predict(self, X, y=None):
        """
        Fits the estimator to the data and predicts the nearest mutual neighbours for each data point.
        If target labels are provided, mutual neighbours are found within the same class.

        Parameters:
        X (array-like): The feature matrix.
        y (array-like, optional): The target vector. Default is None.

        Returns:
        list: A list of arrays containing the indices of the nearest mutual neighbours for each data point.
        """
        # If no target labels are provided, use the single fit_predict method
        if y is None:
            return self.fit_predict_single(X)

        # Convert target labels to a numpy array
        targets = np.asarray(y)

        # Create a mask for each unique target value
        targets_masks_list = [targets == t for t in sorted(set(y))]

        # Initialize a list to store the nearest mutual neighbours for each data point
        k_nearest_mutual_neighbours = [[] for _ in range(len(targets))]

        # Iterate over each target mask
        for targets_mask in targets_masks_list:
            idxs = [i for i, t in enumerate(targets_mask) if t]

            # Find mutual neighbours for data points within the same class
            k_nearest_mutual_neighbours_single = self.fit_predict_single(X[targets_mask])

            # Map the mutual neighbours back to the original indices
            for idx, ngbs in zip(idxs, k_nearest_mutual_neighbours_single):
                k_nearest_mutual_neighbours[idx] = [idxs[ngb] for ngb in ngbs]

        return k_nearest_mutual_neighbours



class NearestMutualNeighboursProbabilityEstimator(object):
    """
    A class to estimate the sampling probability of data points based on their nearest mutual neighbours.

    Attributes:
    nearest_mutual_neighbours_estimator (NearestMutualNeighboursEstimator): An instance of NearestMutualNeighboursEstimator.
    """

    def __init__(self, n_neighbours=10, metric='euclidean'):
        """
        Initializes the NearestMutualNeighboursProbabilityEstimator with specified parameters.

        Parameters:
        n_neighbours (int, optional): The number of nearest neighbours to find. Default is 10.
        metric (str, optional): The distance metric to use for finding nearest neighbours. Default is 'euclidean'.
        """
        self.nearest_mutual_neighbours_estimator = NearestMutualNeighboursEstimator(n_neighbours, metric)

    def fit(self, X, y=None):
        """
        Fits the estimator to the data. This method is currently a no-op.

        Parameters:
        X (array-like): The feature matrix.
        y (array-like, optional): The target vector. Default is None.

        Returns:
        self: Returns the instance itself.
        """
        return self

    def fit_predict_proba_single(self, X):
        """
        Fits the estimator to the data and predicts the sampling probability for each data point.

        Parameters:
        X (array-like): The feature matrix.

        Returns:
        array-like: The sampling probability for each data point.
        """
        # Find the nearest mutual neighbours for each data point
        k_nearest_mutual_neighbours = self.nearest_mutual_neighbours_estimator.fit_predict(X)

        # Calculate the sampling probability based on the number of mutual neighbours
        p = np.array([len(neighbours) / self.nearest_mutual_neighbours_estimator.n_neighbours for neighbours in k_nearest_mutual_neighbours])

        # Normalize the sampling probabilities to sum to 1
        sampling_probability = p / np.sum(p)

        return sampling_probability

    def fit_predict_proba(self, X, y=None):
        """
        Fits the estimator to the data and predicts the sampling probability for each data point.
        If target labels are provided, sampling probabilities are calculated within each class.

        Parameters:
        X (array-like): The feature matrix.
        y (array-like, optional): The target vector. Default is None.

        Returns:
        array-like: The sampling probability for each data point.
        """
        # If no target labels are provided, use the single fit_predict_proba method
        if y is None:
            return self.fit_predict_proba_single(X)

        # Convert target labels to a numpy array
        targets = np.asarray(y)

        # Create a mask for each unique target value
        targets_masks_list = [targets == t for t in sorted(set(y))]

        # Initialize an array to store the sampling probabilities
        sampling_probability = np.zeros(len(y))

        # Iterate over each target mask
        for targets_mask in targets_masks_list:
            # Calculate the sampling probabilities for data points within the same class
            sampling_probability[targets_mask] = self.fit_predict_proba_single(X[targets_mask])

        # Normalize the sampling probabilities to sum to 1
        sampling_probability = sampling_probability / np.sum(sampling_probability)

        return sampling_probability



class ProbabilityEstimator(object):
    """
    A class to estimate sampling probabilities using a combination of multiple probability estimators.

    Attributes:
    probability_estimators (list): A list of probability estimator instances.
    """

    def __init__(self, probability_estimators=[]):
        """
        Initializes the ProbabilityEstimator with a list of probability estimators.

        Parameters:
        probability_estimators (list, optional): A list of probability estimator instances. Default is an empty list.
        """
        self.probability_estimators = probability_estimators

    def fit(self, X, y):
        """
        Fits all probability estimators to the data.

        Parameters:
        X (array-like): The feature matrix.
        y (array-like): The target vector.

        Returns:
        self: Returns the instance itself.
        """
        # Fit each probability estimator to the data
        self.probability_estimators = [probability_estimator.fit(X, y) for probability_estimator in self.probability_estimators]
        return self

    def fit_predict_proba(self, X, y):
        """
        Fits all probability estimators to the data and predicts the combined sampling probability for each data point.

        Parameters:
        X (array-like): The feature matrix.
        y (array-like): The target vector.

        Returns:
        array-like: The combined sampling probability for each data point.
        """
        # Compute the sampling probabilities for each estimator
        probs_mtx = np.array([probability_estimator.fit_predict_proba(X, y) for probability_estimator in self.probability_estimators]).T

        # Compute the product of probabilities across all estimators
        p = np.product(probs_mtx, axis=1)

        # Normalize the combined probabilities to sum to 1
        sampling_probability = p / np.sum(p)

        return sampling_probability



class NearestMutualNeighboursSampler(object):
    """
    A class to sample new data points based on nearest mutual neighbours and specified interpolation factors.

    Attributes:
    nearest_mutual_neighbours_estimator (NearestMutualNeighboursEstimator): An instance of NearestMutualNeighboursEstimator.
    probability_estimator (ProbabilityEstimator): An instance of ProbabilityEstimator.
    interpolation_factor (float): The maximum interpolation factor for generating new samples.
    min_interpolation_factor (float): The minimum interpolation factor for generating new samples.
    use_min_max_constraints (bool): Whether to apply min-max constraints on the generated samples.
    """

    def __init__(self, nearest_mutual_neighbours_estimator=None, probability_estimator=None, interpolation_factor=1, min_interpolation_factor=1, use_min_max_constraints=False):
        """
        Initializes the NearestMutualNeighboursSampler with specified parameters.

        Parameters:
        nearest_mutual_neighbours_estimator (NearestMutualNeighboursEstimator, optional): An instance of NearestMutualNeighboursEstimator.
        probability_estimator (ProbabilityEstimator, optional): An instance of ProbabilityEstimator.
        interpolation_factor (float, optional): The maximum interpolation factor for generating new samples. Default is 1.
        min_interpolation_factor (float, optional): The minimum interpolation factor for generating new samples. Default is 1.
        use_min_max_constraints (bool, optional): Whether to apply min-max constraints on the generated samples. Default is False.
        """
        self.nearest_mutual_neighbours_estimator = nearest_mutual_neighbours_estimator
        self.probability_estimator = probability_estimator
        self.interpolation_factor = interpolation_factor
        self.min_interpolation_factor = min_interpolation_factor
        self.use_min_max_constraints = use_min_max_constraints

    def fit(self, X, y=None):
        """
        Fits the sampler to the data and computes the necessary attributes for sampling.

        Parameters:
        X (array-like): The feature matrix.
        y (array-like, optional): The target vector. Default is None.

        Returns:
        self: Returns the instance itself.
        """
        # Deep copy the data matrix and targets to avoid modifying the original data
        self.data_mtx = copy.deepcopy(X)
        self.targets = copy.deepcopy(np.asarray(y))

        # Compute the sampling probability and nearest mutual neighbours
        self.sampling_probability = self.probability_estimator.fit_predict_proba(X, y)
        self.k_nearest_mutual_neighbours = self.nearest_mutual_neighbours_estimator.fit_predict(X)

        return self

    def generate(self, data_mtx, k_nearest_mutual_neighbours, sampling_probability, interpolation_factor, min_interpolation_factor):
        """
        Generates a new data point based on nearest mutual neighbours and interpolation.

        Parameters:
        data_mtx (array-like): The data matrix.
        k_nearest_mutual_neighbours (list): A list of nearest mutual neighbours for each data point.
        sampling_probability (array-like): The sampling probability for each data point.
        interpolation_factor (float): The maximum interpolation factor.
        min_interpolation_factor (float): The minimum interpolation factor.

        Returns:
        array-like: A newly generated data point.
        """
        # Select an instance at random based on the sampling probability
        idx1 = np.random.choice(len(k_nearest_mutual_neighbours), size=1, p=sampling_probability)[0]

        # Select one of its neighbours at random
        idx2 = np.random.choice(k_nearest_mutual_neighbours[idx1])

        # Select one of the neighbours of the second instance at random
        idx3 = np.random.choice(k_nearest_mutual_neighbours[idx2])

        # Compute the scaled offset for interpolation
        alpha = np.random.rand() * (interpolation_factor - min_interpolation_factor) + min_interpolation_factor
        xn = data_mtx[idx1] + alpha * (data_mtx[idx3] - data_mtx[idx2])

        return xn

    def min_max_constraints(self, X, Xp):
        """
        Applies min-max constraints to the generated data points to ensure they are within the original data range.

        Parameters:
        X (array-like): The original data matrix.
        Xp (array-like): The generated data matrix.

        Returns:
        array-like: The constrained data matrix.
        """
        # Compute the min and max values for each feature in the original data
        mn = np.min(X, axis=0)
        mx = np.max(X, axis=0)

        # Apply min-max constraints to each feature in the generated data
        Xn = []
        for i in range(X.shape[1]):
            Xpi = Xp[:, i]
            Xpi[Xpi < mn[i]] = mn[i]
            Xpi[Xpi > mx[i]] = mx[i]
            Xn.append(Xpi.reshape(-1, 1))
        Xn = np.hstack(Xn)

        return Xn

    def sample(self, n_samples, target=None):
        """
        Generates new samples based on the fitted data and specified parameters.

        Parameters:
        n_samples (int): The number of samples to generate.
        target (int, optional): The target class for which to generate samples. Default is None.

        Returns:
        array-like: The generated samples.
        """
        # Adjust the sampling probability if a target class is specified
        if target is not None:
            sampling_probability = copy.deepcopy(self.sampling_probability)
            sampling_probability[self.targets != target] = 0
            sampling_probability = sampling_probability / np.sum(sampling_probability)
        else:
            sampling_probability = self.sampling_probability

        # Generate new samples
        sampled_data_mtx = []
        for _ in range(n_samples):
            try:
                x = self.generate(self.data_mtx, self.k_nearest_mutual_neighbours, sampling_probability, self.interpolation_factor, self.min_interpolation_factor)
                sampled_data_mtx.append(x)
            except:
                pass

        sampled_data_mtx = np.array(sampled_data_mtx)

        # Apply min-max constraints if specified
        if self.use_min_max_constraints:
            sampled_data_mtx = self.min_max_constraints(self.data_mtx, sampled_data_mtx)

        return sampled_data_mtx


def ConcreteNearestMutualNeighboursSampler(n_neighbours=10, interpolation_factor=1, min_interpolation_factor=1, metric='euclidean', use_min_max_constraints=False):
    """
    Creates an instance of NearestMutualNeighboursSampler with specified parameters and components.

    Parameters:
    n_neighbours (int, optional): The number of nearest neighbours to find. Default is 10.
    interpolation_factor (float, optional): The maximum interpolation factor for generating new samples. Default is 1.
    min_interpolation_factor (float, optional): The minimum interpolation factor for generating new samples. Default is 1.
    metric (str, optional): The distance metric to use for finding nearest neighbours. Default is 'euclidean'.
    use_min_max_constraints (bool, optional): Whether to apply min-max constraints on the generated samples. Default is False.

    Returns:
    NearestMutualNeighboursSampler: An instance of NearestMutualNeighboursSampler configured with the specified parameters.
    """
    # Create an instance of NearestMutualNeighboursEstimator with the specified number of neighbours and metric
    nearest_mutual_neighbours_estimator = NearestMutualNeighboursEstimator(n_neighbours, metric)

    # Create an instance of NearestMutualNeighboursProbabilityEstimator with the specified number of neighbours and metric
    probability_estimator = NearestMutualNeighboursProbabilityEstimator(n_neighbours, metric)

    # Create an instance of NearestMutualNeighboursSampler with the specified parameters and components
    sampler = NearestMutualNeighboursSampler(
        nearest_mutual_neighbours_estimator,
        probability_estimator,
        interpolation_factor=interpolation_factor,
        min_interpolation_factor=min_interpolation_factor,
        use_min_max_constraints=use_min_max_constraints
    )

    return sampler



def make_2d_grid(D, nsteps=20):
    """
    Creates a 2D grid of points based on the minimum and maximum values of a given dataset.

    Parameters:
    D (array-like): Input data array with at least two columns representing 2D points.
    nsteps (int, optional): Number of steps to divide the range between min and max values. Default is 20.

    Returns:
    np.ndarray: A 2D array of grid points.
    """
    # Determine the minimum and maximum values along each axis (column) of the input data
    mns, mxs = np.min(D, axis=0), np.max(D, axis=0)
    
    # Initialize an empty list to store grid points
    mtx = []
    
    # Create grid points by iterating over the range of values for each axis
    for i in np.linspace(mns[0], mxs[0], nsteps):  # Iterate over the range for the first axis
        for j in np.linspace(mns[1], mxs[1], nsteps):  # Iterate over the range for the second axis
            mtx.append((i, j))  # Append the grid point as a tuple to the list
    
    # Convert the list of grid points to a NumPy array
    mtx = np.array(mtx)
    
    return mtx


def compute_symmetrized_kullback_leibler_divergence_single(latent_data_mtx, generated_latent_data_mtx, idxs, grid_nsteps=20, n_components=2):
    """
    Computes the symmetrized Kullback-Leibler divergence between two sets of latent data using Gaussian Mixture Models.

    Parameters:
    latent_data_mtx (array-like): The original latent data matrix.
    generated_latent_data_mtx (array-like): The generated latent data matrix.
    idxs (array-like): Indices to select specific columns from the latent data matrices.
    grid_nsteps (int, optional): Number of steps for grid creation. Default is 20.
    n_components (int, optional): Number of components for Gaussian Mixture Model. Default is 2.

    Returns:
    float: Symmetrized Kullback-Leibler divergence.
    """
    # Select specific columns from the latent data matrices based on the provided indices
    X = latent_data_mtx[:, idxs]
    Z = generated_latent_data_mtx[:, idxs]

    # Combine both datasets into a single matrix
    D = np.vstack([X, Z])

    # Create a grid of points based on the combined data matrix
    G = make_2d_grid(D, nsteps=grid_nsteps)

    # Fit a Gaussian Mixture Model to the original latent data
    est = GaussianMixture(n_components=n_components, covariance_type='full').fit(X)

    # Calculate the log probabilities of the grid points under the fitted model
    probs_r = est.score_samples(G)

    # Convert log probabilities to probabilities
    probs_r = np.exp(probs_r)

    # Cap the probabilities at 1
    probs_r[probs_r > 1] = 1

    # Fit a Gaussian Mixture Model to the generated latent data
    est = GaussianMixture(n_components=n_components, covariance_type='full').fit(Z)

    # Calculate the log probabilities of the grid points under the fitted model
    probs_g = est.score_samples(G)

    # Convert log probabilities to probabilities
    probs_g = np.exp(probs_g)

    # Cap the probabilities at 1
    probs_g[probs_g > 1] = 1

    # Compute the symmetrized Kullback-Leibler divergence
    symmetrized_kullback_leibler_divergence = np.mean([
        entropy(probs_r, probs_g),  # KL divergence from probs_r to probs_g
        entropy(probs_g, probs_r)   # KL divergence from probs_g to probs_r
    ])

    return symmetrized_kullback_leibler_divergence


def compute_symmetrized_kullback_leibler_divergence(latent_data_mtx, generated_latent_data_mtx, grid_nsteps=20, n_components=2):
    """
    Computes the average symmetrized Kullback-Leibler divergence between all pairs of features in the latent data matrices.

    Parameters:
    latent_data_mtx (array-like): The original latent data matrix.
    generated_latent_data_mtx (array-like): The generated latent data matrix.
    grid_nsteps (int, optional): Number of steps for grid creation. Default is 20.
    n_components (int, optional): Number of components for Gaussian Mixture Model. Default is 2.

    Returns:
    float: Average symmetrized Kullback-Leibler divergence over all feature pairs.
    """
    # Determine the number of features in the latent data matrix
    n_features = latent_data_mtx.shape[1]

    # Initialize a list to store the symmetrized Kullback-Leibler divergences for each feature pair
    symmetrized_kullback_leibler_divergences = []

    # Iterate over all pairs of features
    for i in range(n_features - 1):
        for j in range(i + 1, n_features):
            # Select the current pair of feature indices
            idxs = [i, j]

            # Compute the symmetrized Kullback-Leibler divergence for the current feature pair
            symmetrized_kullback_leibler_divergence = compute_symmetrized_kullback_leibler_divergence_single(
                latent_data_mtx, generated_latent_data_mtx, idxs, grid_nsteps=grid_nsteps, n_components=n_components)

            # If the computed divergence is finite, append it to the list
            if np.isfinite(symmetrized_kullback_leibler_divergence):
                symmetrized_kullback_leibler_divergences.append(symmetrized_kullback_leibler_divergence)

    # Compute the average symmetrized Kullback-Leibler divergence over all feature pairs
    avg_symmetrized_kullback_leibler_divergence = np.mean(symmetrized_kullback_leibler_divergences)

    return avg_symmetrized_kullback_leibler_divergence


def is_numerical_column_type(series):
    """
    Checks if the given pandas Series has a numerical data type.

    Parameters:
    series (pandas.Series): The series whose data type is to be checked.

    Returns:
    bool: True if the series is of numerical data type (int or float), False otherwise.
    """
    # Get the data type of the series as a string
    column_type = '%s' % series.dtype
    
    # Check if 'int' or 'float' is in the string representation of the data type
    return 'int' in column_type or 'float' in column_type



class DataFrameVectorizer(object):
    """
    A class to vectorize a pandas DataFrame by handling both numerical and categorical columns.

    Attributes:
    vectorizing_columns_dict (dict): A dictionary specifying which columns to use for vectorizing categorical columns.
    """

    def __init__(self, vectorizing_columns_dict=None):
        """
        Initializes the DataFrameVectorizer with an optional dictionary for vectorizing columns.

        Parameters:
        vectorizing_columns_dict (dict, optional): Dictionary specifying columns for vectorizing categorical columns. Default is None.
        """
        self.vectorizing_columns_dict = vectorizing_columns_dict

    def fit(self, dataframe):
        """
        Fits the vectorizer to the dataframe. This method is currently a no-op.

        Parameters:
        dataframe (pandas.DataFrame): The dataframe to fit.

        Returns:
        self: Returns the instance itself.
        """
        return self
    
    def transform(self, dataframe):
        """
        Transforms the dataframe by vectorizing its columns.

        Parameters:
        dataframe (pandas.DataFrame): The dataframe to transform.

        Returns:
        pandas.DataFrame: A new dataframe with vectorized columns.
        """
        # Initialize an empty DataFrame to store the vectorized columns
        vectorizing_dataframe = pd.DataFrame()

        # Get the list of columns in the dataframe
        columns = dataframe.columns

        # Iterate over each column in the dataframe
        for column in columns:
            # Get the values of the current column
            column_values = dataframe[column].values

            # Check if the column is numerical
            if is_numerical_column_type(dataframe[column]):
                # If numerical, directly add it to the vectorizing dataframe
                vectorizing_dataframe[column] = dataframe[column]
            else:
                # If categorical, check if vectorizing_columns_dict is provided and contains the current column
                if self.vectorizing_columns_dict is not None and column in self.vectorizing_columns_dict:
                    # Select the specified columns for vectorizing
                    selected_cols = self.vectorizing_columns_dict[column]
                    X = dataframe.loc[:, selected_cols].values

                    # Apply Multi-Dimensional Scaling (MDS) to reduce to 1 dimension and flatten the result
                    mds_values = MDS(n_components=1).fit_transform(X).flatten()
                    
                    # Add the MDS-transformed values to the vectorizing dataframe
                    vectorizing_dataframe[column] = mds_values
                else:
                    # If no specific columns are provided, vectorize based on frequency of unique values
                    unique, counts = np.unique(column_values, return_counts=True)
                    column_values_to_freq_map = {uniq: count for uniq, count in zip(unique, counts)}
                    freq_column_values = [column_values_to_freq_map[column_value] for column_value in column_values]

                    # Add the frequency-based values to the vectorizing dataframe
                    vectorizing_dataframe[column] = freq_column_values
        
        return vectorizing_dataframe 
    
    def fit_transform(self, dataframe):
        """
        Fits the vectorizer to the dataframe and then transforms it.

        Parameters:
        dataframe (pandas.DataFrame): The dataframe to fit and transform.

        Returns:
        pandas.DataFrame: A new dataframe with vectorized columns.
        """
        return self.fit(dataframe).transform(dataframe)



class ColumnDataFrameEncoderDecoder(object):
    """
    A class to encode and decode column data using a discretizer and binning strategy.

    Attributes:
    n_bins (int): Number of bins for discretization.
    strategy (str): Strategy for binning ('uniform', 'quantile', or 'kmeans').
    default (int): Default value for encoding.
    max_n_attempts (int): Maximum number of attempts for decoding.
    discretizer (KBinsDiscretizer): The discretizer object.
    histogram (defaultdict): Dictionary to map discretized values to original column values.
    """

    def __init__(self, n_bins=5, strategy='uniform'):
        """
        Initializes the encoder-decoder with specified binning parameters.

        Parameters:
        n_bins (int, optional): Number of bins for discretization. Default is 5.
        strategy (str, optional): Strategy for binning. Default is 'uniform'.
        """
        self.n_bins = n_bins
        self.strategy = strategy
        self.default = -1
        self.max_n_attempts = 100
        self.discretizer = None

    def fit(self, column_values, vectorizing_column_values):
        """
        Fits the discretizer to the vectorizing column values and associates original column values to bins.

        Parameters:
        column_values (array-like): Original column values.
        vectorizing_column_values (array-like): Values used for vectorizing and binning.

        Returns:
        self: Returns the instance itself.
        """
        # Determine the number of unique values and their counts
        unique, counts = np.unique(vectorizing_column_values, return_counts=True)

        # Set the number of bins to the minimum of unique values or specified bins
        n_bins = min(len(unique), self.n_bins)

        # Initialize the discretizer with the specified strategy
        self.discretizer = KBinsDiscretizer(n_bins=n_bins, strategy=self.strategy, encode='ordinal')

        # Fit the discretizer to the vectorizing column values
        self.discretizer.fit(vectorizing_column_values.reshape(-1, 1))

        # Encode the vectorizing column values to discretized values
        discretized_column_values = self.encode(vectorizing_column_values)

        # Associate each bin with the corresponding original column entries
        self.histogram = defaultdict(list)
        for discretized_column_value, column_value in zip(discretized_column_values, column_values):
            self.histogram[discretized_column_value].append(column_value)

        return self

    def encode(self, vectorizing_column_values):
        """
        Encodes the vectorizing column values into discretized bin values.

        Parameters:
        vectorizing_column_values (array-like): Values to be encoded.

        Returns:
        array: Discretized bin values.
        """
        # Transform the vectorizing column values using the discretizer
        Xt = self.discretizer.transform(vectorizing_column_values.reshape(-1, 1))

        # Flatten the transformed values and convert to integers
        discretized_column_values = Xt[:, 0].flatten().astype(int)
        return discretized_column_values

    def decode(self, discretized_column_values):
        """
        Decodes discretized bin values back to original column values by sampling from the bins.

        Parameters:
        discretized_column_values (array-like): Discretized bin values to be decoded.

        Returns:
        list: Decoded original column values.
        """
        results = []

        # Iterate over each discretized bin value
        for discretized_column_value in discretized_column_values:
            selected = None

            # Attempt to sample a value from the bin up to the maximum number of attempts
            for it in range(self.max_n_attempts):
                bin_data = self.histogram[discretized_column_value]
                if len(bin_data) > 0:
                    # Randomly select a value from the bin data
                    selected = np.random.choice(bin_data)
                    break
                else:
                    # If bin is empty, increment the bin value
                    discretized_column_value += 1
            
            # If no value is selected, use the default value
            if selected is None:
                selected = self.default
            
            results.append(selected)
        
        return results



class DataFrameEncoderDecoder(object):
    """
    A class to encode and decode a pandas DataFrame using binning and a vectorizing strategy.

    Attributes:
    n_bins (int): Number of bins for discretization.
    strategy (str): Strategy for binning ('uniform', 'quantile', or 'kmeans').
    """

    def __init__(self, n_bins=5, strategy='uniform'):
        """
        Initializes the encoder-decoder with specified binning parameters.

        Parameters:
        n_bins (int, optional): Number of bins for discretization. Default is 5.
        strategy (str, optional): Strategy for binning. Default is 'uniform'.
        """
        self.n_bins = n_bins
        self.strategy = strategy

    def fit(self, dataframe, vectorizing_dataframe):
        """
        Fits the encoder-decoder to the dataframe using a vectorizing dataframe.

        Parameters:
        dataframe (pandas.DataFrame): The original dataframe to be encoded and decoded.
        vectorizing_dataframe (pandas.DataFrame): The dataframe used for vectorizing.

        Returns:
        self: Returns the instance itself.
        """
        # Store the column names of the dataframe
        self.columns = dataframe.columns

        # Create and fit a ColumnDataFrameEncoderDecoder for each column
        self.column_dataframe_encoder_decoders = [
            copy.deepcopy(ColumnDataFrameEncoderDecoder(n_bins=self.n_bins, strategy=self.strategy)).fit(
                dataframe[column].values, vectorizing_dataframe[column].values
            ) for column in dataframe.columns
        ]

        return self

    def encode(self, vectorizing_dataframe):
        """
        Encodes the vectorizing dataframe into a matrix of discretized bin values.

        Parameters:
        vectorizing_dataframe (pandas.DataFrame): The dataframe to be encoded.

        Returns:
        np.ndarray: A matrix of discretized bin values.
        """
        # Encode each column of the vectorizing dataframe
        data_mtx = [
            column_dataframe_encoder_decoder.encode(vectorizing_dataframe[self.columns[i]].values)
            for i, column_dataframe_encoder_decoder in enumerate(self.column_dataframe_encoder_decoders)
        ]

        # Transpose the matrix to match the original dataframe's structure
        data_mtx = np.array(data_mtx).T
        return data_mtx

    def decode(self, data_mtx):
        """
        Decodes a matrix of discretized bin values back into the original dataframe.

        Parameters:
        data_mtx (np.ndarray): A matrix of discretized bin values to be decoded.

        Returns:
        pandas.DataFrame: The decoded original dataframe.
        """
        # Initialize an empty DataFrame to store the decoded columns
        res = pd.DataFrame()

        # Decode each column of the matrix
        for i, col in enumerate(data_mtx.T):
            res[self.columns[i]] = self.column_dataframe_encoder_decoders[i].decode(col)

        return res

    def sample(self, n_samples):
        """
        Generates a sample of discretized bin values for each column.

        Parameters:
        n_samples (int): Number of samples to generate.

        Returns:
        np.ndarray: A matrix of sampled discretized bin values.
        """
        all_values = []

        # Generate samples for each column based on the number of bins
        for column_dataframe_encoder_decoder in self.column_dataframe_encoder_decoders:
            max_val = column_dataframe_encoder_decoder.discretizer.n_bins_
            values = np.random.randint(0, max_val + 1, size=n_samples)
            all_values.append(values)

        # Transpose the matrix to match the original dataframe's structure
        all_values = np.array(all_values).T
        return all_values

    def info(self, k=1):
        """
        Prints information about the histogram bins for each column.

        Parameters:
        k (int, optional): Number of unique elements to display per bin. Default is 1.
        """
        for i in range(len(self.columns)):
            print()
            print('%2d   %s' % (i, self.columns[i]))

            # Iterate over each bin in the histogram and display information
            for key in sorted(self.column_dataframe_encoder_decoders[i].histogram.keys()):
                elements_in_bin = self.column_dataframe_encoder_decoders[i].histogram[key]
                n_elements_in_bin = len(elements_in_bin)
                unique_elements_in_bin = sorted(list(np.unique(elements_in_bin)))
                size = min(len(unique_elements_in_bin), k)

                if size > 0:
                    if size > 2:
                        selected_elements = np.random.choice(unique_elements_in_bin, size=size)
                    else:
                        selected_elements = '|'
                    if size == 1:
                        last_element = '|'
                    else:
                        last_element = unique_elements_in_bin[-1]

                    print('%d  #%3d   <%s  %s  %s>' % (key, n_elements_in_bin, unique_elements_in_bin[0], selected_elements, last_element))




class DataFrameSampler(object):
    """
    A class to handle sampling from a pandas DataFrame using vectorization, encoding, and custom sampling techniques.

    Attributes:
    dataframe_vectorizer (DataFrameVectorizer): An instance of DataFrameVectorizer for vectorizing the DataFrame.
    dataframe_encoder_decoder (DataFrameEncoderDecoder): An instance of DataFrameEncoderDecoder for encoding and decoding the DataFrame.
    sampler (Sampler): A custom sampler instance to generate samples from the encoded data.
    sampled_columns (list): A list of columns to be sampled.
    latent_data_mtx (np.ndarray): The latent matrix representation of the encoded data.
    generated_latent_data_mtx (np.ndarray): The latent matrix representation of the generated data.
    """

    def __init__(self, dataframe_vectorizer=None, dataframe_encoder_decoder=None, sampler=None, sampled_columns=None):
        """
        Initializes the DataFrameSampler with specified components for vectorization, encoding, and sampling.

        Parameters:
        dataframe_vectorizer (DataFrameVectorizer, optional): An instance of DataFrameVectorizer.
        dataframe_encoder_decoder (DataFrameEncoderDecoder, optional): An instance of DataFrameEncoderDecoder.
        sampler (Sampler, optional): A custom sampler instance.
        sampled_columns (list, optional): A list of columns to be sampled.
        """
        self.dataframe_vectorizer = dataframe_vectorizer
        self.dataframe_encoder_decoder = dataframe_encoder_decoder
        self.sampler = sampler
        self.sampled_columns = sampled_columns
        self.latent_data_mtx = None

    def fit(self, orig_dataframe):
        """
        Fits the DataFrameSampler to the original dataframe by vectorizing, encoding, and preparing the sampler.

        Parameters:
        orig_dataframe (pandas.DataFrame): The original dataframe to fit.

        Returns:
        self: Returns the instance itself.
        """
        # Make a copy of the original dataframe to avoid modifying it
        dataframe = copy.copy(orig_dataframe)

        # Vectorize the dataframe
        vectorizing_dataframe = self.dataframe_vectorizer.fit_transform(dataframe)

        # If sampled_columns is specified, use only those columns for vectorizing and encoding
        if self.sampled_columns is not None:
            vectorizing_dataframe = vectorizing_dataframe[self.sampled_columns]
            dataframe = dataframe[self.sampled_columns]

        # Fit the encoder-decoder with the dataframe and vectorized dataframe
        self.dataframe_encoder_decoder.fit(dataframe, vectorizing_dataframe)

        # Encode the vectorized dataframe to obtain the latent matrix
        self.latent_data_mtx = self.dataframe_encoder_decoder.encode(vectorizing_dataframe)

        # Fit the sampler with the latent matrix
        self.sampler.fit(self.latent_data_mtx)

        return self

    def sample(self, n_samples):
        """
        Generates samples from the fitted data.

        Parameters:
        n_samples (int): Number of samples to generate.

        Returns:
        pandas.DataFrame: A dataframe of generated samples.
        """
        # Generate latent data matrix using the sampler
        generated_latent_data_mtx = self.sampler.sample(n_samples=n_samples)

        # Convert the generated latent data matrix to integers
        self.generated_latent_data_mtx = generated_latent_data_mtx.astype(int)

        # Decode the generated latent data matrix to obtain the generated dataframe
        generated_df = self.dataframe_encoder_decoder.decode(self.generated_latent_data_mtx)

        return generated_df

    def sample_to_file(self, n_samples, filename='data.csv'):
        """
        Generates samples and saves them to a CSV file.

        Parameters:
        n_samples (int): Number of samples to generate.
        filename (str, optional): Filename for the CSV file. Default is 'data.csv'.

        Returns:
        pandas.DataFrame: A dataframe of generated samples.
        """
        # Generate samples
        generated_df = self.sample(n_samples)

        # Save the generated dataframe to a CSV file
        generated_df.to_csv(filename, index=False)

        return generated_df

    def quality_score_(self, grid_nsteps=20, n_components=2):
        """
        Computes the quality score of the generated data by comparing it to the original data.

        Parameters:
        grid_nsteps (int, optional): Number of steps for grid creation. Default is 20.
        n_components (int, optional): Number of components for Gaussian Mixture Model. Default is 2.

        Returns:
        float: The Kullback-Leibler divergence score.
        """
        # Generate a latent data matrix with the same number of samples as the original latent data matrix
        generated_latent_data_mtx = self.sampler.sample(n_samples=self.latent_data_mtx.shape[0])

        # Compute the symmetrized Kullback-Leibler divergence between the original and generated latent data matrices
        symmetrized_kullback_leibler_divergence = compute_symmetrized_kullback_leibler_divergence(
            self.latent_data_mtx, generated_latent_data_mtx, grid_nsteps=grid_nsteps, n_components=n_components
        )

        return symmetrized_kullback_leibler_divergence

    def quality_score(self, grid_nsteps=20, n_components=2, n_iter=10):
        """
        Computes the quality score of the generated data over multiple iterations to assess stability.

        Parameters:
        grid_nsteps (int, optional): Number of steps for grid creation. Default is 20.
        n_components (int, optional): Number of components for Gaussian Mixture Model. Default is 2.
        n_iter (int, optional): Number of iterations for quality assessment. Default is 10.

        Returns:
        float: The p-value from a t-test comparing generated and real data KL divergence scores.
        """
        assert self.latent_data_mtx is not None, 'ERROR: estimator is not fit.'
        n_instances = self.latent_data_mtx.shape[0]
        real_symmetrized_kullback_leibler_divergences = []

        # Calculate KL divergence for real data by sampling subsets
        for it in range(n_iter):
            idxs1 = np.random.randint(n_instances, size=n_instances // 2)
            idxs2 = np.random.randint(n_instances, size=n_instances // 2)
            mtx1 = self.latent_data_mtx[idxs1]
            mtx2 = self.latent_data_mtx[idxs2]
            symmetrized_kullback_leibler_divergence = compute_symmetrized_kullback_leibler_divergence(
                mtx1, mtx2, grid_nsteps=grid_nsteps, n_components=n_components
            )
            real_symmetrized_kullback_leibler_divergences.append(symmetrized_kullback_leibler_divergence)

        # Calculate KL divergence for generated data
        generated_symmetrized_kullback_leibler_divergences = [
            self.quality_score_(grid_nsteps=grid_nsteps, n_components=n_components) for _ in range(n_iter)
        ]

        # Perform t-test to compare the KL divergence scores
        pvalue = ttest_ind(generated_symmetrized_kullback_leibler_divergences, real_symmetrized_kullback_leibler_divergences).pvalue

        return pvalue

    def plot(self, dataframe, filename='df'):
        """
        Plots the dataframe using seaborn pairplot and saves the plot as PNG and SVG files.

        Parameters:
        dataframe (pandas.DataFrame): The dataframe to plot.
        filename (str, optional): Base filename for saving the plots. Default is 'df'.
        """
        # Create a pairplot of the dataframe with histograms on the diagonal and KDE plots in the lower triangle
        sns_figure = sns.pairplot(dataframe, diag_kind="hist", corner=True)
        sns_figure.map_lower(sns.kdeplot, levels=5, color=".8")

        # Save the plot as PNG and SVG files
        sns_figure.figure.savefig(filename + '.png', format='png')
        sns_figure.figure.savefig(filename + '.svg', format='svg', dpi=1200)

    def save(self, filename='model.obj'):
        """
        Saves the current state of the DataFrameSampler to a file.

        Parameters:
        filename (str, optional): Filename for saving the model. Default is 'model.obj'.
        """
        # Save the current instance to a file using pickle
        with open(filename, 'wb') as filehandler:
            pickle.dump(self, filehandler)

        return self

    def load(self, filename='model.obj'):
        """
        Loads the state of the DataFrameSampler from a file.

        Parameters:
        filename (str, optional): Filename for loading the model. Default is 'model.obj'.

        Returns:
        self: Returns the loaded instance.
        """
        # Load the instance from a file using pickle
        with open(filename, 'rb') as filehandler:
            self = pickle.load(filehandler)

        return self



def ConcreteDataFrameSampler(n_bins=20, n_neighbours=10, vectorizing_columns_dict=None, sampled_columns=None):
    """
    Creates an instance of DataFrameSampler with specified parameters and components for vectorizing, encoding, and sampling.

    Parameters:
    n_bins (int, optional): Number of bins for discretization. Default is 20.
    n_neighbours (int, optional): Number of neighbors for the sampling algorithm. Default is 10.
    vectorizing_columns_dict (dict, optional): Dictionary specifying columns for vectorizing categorical columns.
    sampled_columns (list, optional): List of columns to be sampled.

    Returns:
    DataFrameSampler: An instance of DataFrameSampler configured with the specified parameters.
    """
    # Create an instance of DataFrameVectorizer with the provided vectorizing columns dictionary
    dataframe_vectorizer = DataFrameVectorizer(vectorizing_columns_dict=vectorizing_columns_dict)

    # Create an instance of DataFrameEncoderDecoder with the specified number of bins and strategy
    dataframe_encoder_decoder = DataFrameEncoderDecoder(n_bins=n_bins, strategy='uniform')

    # Create an instance of ConcreteNearestMutualNeighboursSampler with the specified parameters
    sampler = ConcreteNearestMutualNeighboursSampler(
        n_neighbours=n_neighbours, 
        interpolation_factor=1, 
        min_interpolation_factor=1, 
        metric='euclidean', 
        use_min_max_constraints=True
    )

    # Return an instance of DataFrameSampler with the created components and sampled columns
    return DataFrameSampler(
        dataframe_vectorizer=dataframe_vectorizer, 
        dataframe_encoder_decoder=dataframe_encoder_decoder, 
        sampler=sampler, 
        sampled_columns=sampled_columns
    )


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--input_filename', '-i', type=click.Path(exists=True), help='Path to input CSV file.')
@click.option('--output_filename', '-o', type=click.Path(), default='data.csv', help='Path to CSV file to generate. By default a file called "data.csv" will be generated.')
@click.option('--input_model_filename', '-m', type=click.Path(exists=True), help='Path to fit model.')
@click.option('--output_model_filename', '-d', type=click.Path(), default='dataframe_sampler_model.obj', help='Path to model to save.')
@click.option('--vectorizing_columns_dict_filename', '-f', type=click.Path(exists=True), help='Path to vectorizing_columns_dict.')
@click.option('--n_samples', '-n', type=click.IntRange(min=0, max_open=True, clamp=True), default=100, help='Number of samples to generate. If 0 then generate the same number of samples as there are in input. Default is 100.')
@click.option('--n_bins', type=click.IntRange(min=2, max_open=True, clamp=True), default=9, help='Number of bins.')
@click.option('--n_neighbours', type=click.IntRange(min=1, max_open=True, clamp=True), default=5, help='Number of neighbours.')
@click.option('--sampled_columns', '-c', multiple=True, help='Selected columns to generate.')
@click.version_option("0.1.2", '--version', '-v')
def dataframe_sampler_main(input_filename, output_filename, input_model_filename, output_model_filename, vectorizing_columns_dict_filename, n_samples, n_bins, n_neighbours, sampled_columns):
    """
    The program dataframe_sampler can generate a CSV file similar to the one given in input.
    """
    # If no sampled columns are provided, set to None
    if len(sampled_columns) == 0:
        sampled_columns = None
    else:
        sampled_columns = list(sampled_columns)

    if vectorizing_columns_dict_filename: vectorizing_columns_dict = yaml_load(fname=vectorizing_columns_dict_filename)
    else: vectorizing_columns_dict = None
    
    # If an input model filename is provided, load the model
    if input_model_filename:
        sampler = ConcreteDataFrameSampler().load(input_model_filename)
    else:
        # Otherwise, create a new ConcreteDataFrameSampler with specified parameters
        sampler = ConcreteDataFrameSampler(n_bins=n_bins, n_neighbours=n_neighbours, vectorizing_columns_dict=vectorizing_columns_dict, sampled_columns=sampled_columns)
    
    # If an input filename is provided, read the CSV file, fit the sampler, and save the model
    if input_filename:
        df = pd.read_csv(input_filename)
        sampler.fit(df)
        sampler.save(output_model_filename)
        # If n_samples is 0, set it to the number of rows in the input dataframe
        if n_samples == 0:
            n_samples = len(df)

    # Generate samples and save them to the specified output filename
    generated_df = sampler.sample_to_file(n_samples=n_samples, filename=output_filename)

# Entry point for the command-line interface
if __name__ == '__main__':
    if len(sys.argv) == 1: dataframe_sampler_main.main(['--help'])
    else: dataframe_sampler_main()
