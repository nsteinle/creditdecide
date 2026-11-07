import numpy as np
import pandas as pd
import pytest

from creditdecide import Config, FeatureSpec


def _toy():
    """600 rows, two overlapping classes, one categorical, one heavy-tailed, one protected,
    an ID column, and five exact feature-duplicates of row 0."""
    rng = np.random.default_rng(0)
    n = 600
    y = rng.integers(0, 2, n)
    df = pd.DataFrame({
        "id": np.arange(n),
        "x1": rng.normal(y * 1.0, 1.0),
        "x2": rng.normal(-y * 0.5, 1.0),
        "amount": rng.lognormal(3 + y, 1.0) * rng.choice([-1, 1], n),
        "region": rng.choice(["a", "b", "c"], n),
        "age": rng.integers(20, 70, n),
        "label": y,
    })
    dup = df.iloc[[0] * 5].copy()
    dup["id"] = np.arange(n, n + 5)
    return pd.concat([df, dup], ignore_index=True)


def _toy_spec():
    return FeatureSpec(target="label", categorical=("region",), signed_log=("amount",),
                       protected=("age",), drop=("id",))


@pytest.fixture
def fast():
    return Config(rf_n_estimators=20)


@pytest.fixture
def toy():
    return _toy()


@pytest.fixture
def toy_spec():
    return _toy_spec()


@pytest.fixture(scope="module")
def toy_module():
    return _toy()


@pytest.fixture(scope="module")
def toy_spec_module():
    return _toy_spec()
