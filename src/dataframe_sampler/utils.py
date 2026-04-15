import numpy as np
import yaml


def yaml_save(obj, fname="data.txt"):
    with open(fname, "w") as fh:
        yaml.dump(obj, fh)


def yaml_load(fname="data.txt"):
    with open(fname, "r") as fh:
        return yaml.load(fh, Loader=yaml.SafeLoader)


def make_random_state(random_state=None):
    if random_state is None:
        return np.random.RandomState()
    if random_state is np.random:
        return np.random.RandomState()
    if isinstance(random_state, (np.random.RandomState, np.random.Generator)):
        return random_state
    return np.random.RandomState(random_state)


def random_choice(rng, values, size=None, replace=True, p=None):
    return rng.choice(values, size=size, replace=replace, p=p)


def random_uniform(rng):
    if hasattr(rng, "random"):
        return rng.random()
    return rng.rand()


def random_integers(rng, low, high=None, size=None):
    if hasattr(rng, "integers"):
        return rng.integers(low, high=high, size=size)
    return rng.randint(low, high=high, size=size)
