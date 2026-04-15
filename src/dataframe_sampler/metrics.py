import numpy as np
from scipy.stats import entropy, ttest_ind
from sklearn.mixture import GaussianMixture


def make_2d_grid(D, nsteps=20):
    mns, mxs = np.min(D, axis=0), np.max(D, axis=0)
    mtx = []
    for i in np.linspace(mns[0], mxs[0], nsteps):
        for j in np.linspace(mns[1], mxs[1], nsteps):
            mtx.append((i, j))
    return np.array(mtx)


def compute_symmetrized_kullback_leibler_divergence_single(
    latent_data_mtx,
    generated_latent_data_mtx,
    idxs,
    grid_nsteps=20,
    n_components=2,
):
    X = latent_data_mtx[:, idxs]
    Z = generated_latent_data_mtx[:, idxs]
    D = np.vstack([X, Z])
    G = make_2d_grid(D, nsteps=grid_nsteps)

    real_components = min(n_components, len(X))
    generated_components = min(n_components, len(Z))
    probs_r = np.exp(GaussianMixture(n_components=real_components, covariance_type="full").fit(X).score_samples(G))
    probs_g = np.exp(
        GaussianMixture(n_components=generated_components, covariance_type="full").fit(Z).score_samples(G)
    )
    probs_r[probs_r > 1] = 1
    probs_g[probs_g > 1] = 1
    return np.mean([entropy(probs_r, probs_g), entropy(probs_g, probs_r)])


def compute_symmetrized_kullback_leibler_divergence(
    latent_data_mtx,
    generated_latent_data_mtx,
    grid_nsteps=20,
    n_components=2,
):
    n_features = latent_data_mtx.shape[1]
    symmetrized_kullback_leibler_divergences = []
    for i in range(n_features - 1):
        for j in range(i + 1, n_features):
            divergence = compute_symmetrized_kullback_leibler_divergence_single(
                latent_data_mtx,
                generated_latent_data_mtx,
                [i, j],
                grid_nsteps=grid_nsteps,
                n_components=n_components,
            )
            if np.isfinite(divergence):
                symmetrized_kullback_leibler_divergences.append(divergence)

    if not symmetrized_kullback_leibler_divergences:
        return np.nan
    return np.mean(symmetrized_kullback_leibler_divergences)


def divergence_ttest(generated_divergences, real_divergences):
    return ttest_ind(generated_divergences, real_divergences).pvalue
