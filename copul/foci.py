"""
Conditional Dependence Coefficient (CODEC) Implementation

This module provides functions to calculate the conditional dependence coefficient (CODEC),
a measure of conditional dependence between random variables based on an i.i.d. sample.

The implementation is based on the paper "An Empirical Study on New Model-Free Multi-output
Variable Selection Methods" by Ansari et al.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import rankdata

from copul._lazy import is_pandas_instance

if TYPE_CHECKING:
    import pandas as pd


def codec(
    Y: np.ndarray | pd.Series | pd.DataFrame | list,
    Z: np.ndarray | pd.Series | pd.DataFrame | list,
    X: np.ndarray | pd.Series | pd.DataFrame | list | None = None,
    na_rm: bool = True,
) -> float | dict[str, float]:
    """
    Calculate the conditional dependence coefficient (CODEC).

    CODEC measures the amount of conditional dependence between a random variable Y
    and a random vector Z given a random vector X, based on an i.i.d. sample of (Y, Z, X).
    The coefficient is asymptotically guaranteed to be between 0 and 1.

    If X is None, the unconditional CODEC is calculated, corresponding to xi(Y|Z)
    from the Ansari et al. paper.

    Parameters
    ----------
    Y : array-like
        The response variable.
    Z : array-like
        The conditioning variable.
    X : array-like, optional
        The conditioning variable. If None, the unconditional CODEC is calculated.
    na_rm : bool, optional
        Whether to remove NAs. Default is True.

    Returns
    -------
    float or dict
        The conditional dependence coefficient or a dictionary of coefficients
        when Y is a DataFrame.

    Raises
    ------
    ValueError
        If the number of rows of Y, X, and Z are not equal.
        If the number of rows with no NAs is less than 2.

    Examples
    --------
    >>> import numpy as np
    >>> n = 1000
    >>> x = np.random.rand(n, 2)
    >>> y = (x[:, 0] + x[:, 1]) % 1
    >>> # Calculate unconditional CODEC
    >>> codec_y_x = codec(y, x)
    >>> # Calculate conditional CODEC
    >>> z = np.random.randn(n, 1)
    >>> codec_y_z_x = codec(y, z, x)
    """
    # Handle DataFrame case for Y (multiple response variables)
    if is_pandas_instance(Y, "DataFrame"):
        results = {}
        for i in range(Y.shape[1]):
            results[Y.columns[i]] = codec(Y.iloc[:, i], Z, X, na_rm)
        return results

    # Convert inputs to numpy arrays
    Y = _ensure_numpy_array(Y)
    Z = _ensure_numpy_array(Z)

    # Handle unconditional case
    if X is None:
        if len(Y) != Z.shape[0]:
            raise ValueError("Number of rows of Y and Z should be equal.")

        if na_rm:
            # Create mask for finite values, ensuring compatible shapes
            y_mask = np.isfinite(Y).ravel()  # Flatten to 1D
            z_mask = np.all(np.isfinite(Z), axis=1)
            mask = y_mask & z_mask

            # Apply mask to select valid rows
            Z = Z[mask, :]  # Keep the second dimension
            Y = Y[mask].reshape(-1, 1)  # Reshape to maintain column vector

        if len(Y) < 2:
            raise ValueError("Number of rows with no NAs should be at least 2.")

        return estimate_t(Y, Z)

    # Convert X to numpy array for conditional case
    X = _ensure_numpy_array(X)

    # Check dimensions
    if len(Y) != X.shape[0] or len(Y) != Z.shape[0]:
        raise ValueError("Number of rows of Y, X, and Z should be equal.")

    # Remove NAs if requested
    if na_rm:
        # Create mask for finite values, ensuring compatible shapes
        y_mask = np.isfinite(Y).ravel()  # Flatten to 1D
        z_mask = np.all(np.isfinite(Z), axis=1)
        x_mask = np.all(np.isfinite(X), axis=1)
        mask = y_mask & z_mask & x_mask

        # Apply mask to select valid rows
        Z = Z[mask, :]  # Keep the second dimension
        Y = Y[mask].reshape(-1, 1)  # Reshape to maintain column vector
        X = X[mask, :]  # Keep the second dimension

    if len(Y) < 2:
        raise ValueError("Number of rows with no NAs should be at least 2.")

    return estimate_conditional_t(Y, Z, X)


def estimate_conditional_q(Y: np.ndarray, X: np.ndarray, Z: np.ndarray) -> float:
    """
    Estimate the conditional Q statistic for CODEC calculation.

    Parameters
    ----------
    Y : np.ndarray
        The response variable.
    X : np.ndarray
        First conditioning variable.
    Z : np.ndarray
        Second conditioning variable.

    Returns
    -------
    float
        The estimated conditional Q statistic.
    """
    n = len(Y)
    W = np.hstack((X, Z))

    # Find nearest neighbors for X and W
    nn_index_X = find_nearest_neighbors(X)
    nn_index_W = find_nearest_neighbors(W)

    # Calculate rank statistics
    R_Y = rankdata(Y.ravel(), method="max")

    # Calculate minimums
    minimum_1 = np.minimum(R_Y, R_Y[nn_index_W])
    minimum_2 = np.minimum(R_Y, R_Y[nn_index_X])

    # Calculate Q statistic
    Q_n = np.sum(minimum_1 - minimum_2) / (n**2)

    return Q_n


def estimate_conditional_s(Y: np.ndarray, X: np.ndarray) -> float:
    """
    Estimate the conditional S statistic for CODEC calculation.

    Parameters
    ----------
    Y : np.ndarray
        The response variable.
    X : np.ndarray
        The conditioning variable.

    Returns
    -------
    float
        The estimated conditional S statistic.
    """
    n = len(Y)

    # Find nearest neighbors for X
    nn_index_X = find_nearest_neighbors(X)

    # Calculate rank statistics
    R_Y = rankdata(Y.ravel(), method="max")

    # Calculate S statistic
    S_n = np.sum(R_Y - np.minimum(R_Y, R_Y[nn_index_X])) / (n**2)

    return S_n


def estimate_conditional_t(Y: np.ndarray, Z: np.ndarray, X: np.ndarray) -> float:
    """
    Estimate the conditional T statistic (the conditional CODEC).

    Parameters
    ----------
    Y : np.ndarray
        The response variable.
    Z : np.ndarray
        The primary conditioning variable.
    X : np.ndarray
        The secondary conditioning variable.

    Returns
    -------
    float
        The estimated conditional T statistic (CODEC value).
    """
    S = estimate_conditional_s(Y, X)

    if np.isclose(S, 0):
        return 1.0
    else:
        q = estimate_conditional_q(Y, X, Z)
        return q / S


def estimate_q(Y: np.ndarray, X: np.ndarray) -> float:
    """
    Estimate the Q statistic for unconditional CODEC calculation.

    Parameters
    ----------
    Y : np.ndarray
        The response variable.
    X : np.ndarray
        The conditioning variable.

    Returns
    -------
    float
        The estimated Q statistic.
    """
    n = len(Y)

    # Find nearest neighbors for X
    nn_index_X = find_nearest_neighbors(X)

    # Calculate rank statistics (integers; exact in float64 for n < 2**26)
    R_Y = rankdata(Y.ravel(), method="max").astype(float)
    L_Y = rankdata(-Y.ravel(), method="max").astype(float)

    # Calculate Q statistic
    min_values = np.minimum(R_Y, R_Y[nn_index_X])
    Q_n = np.mean(min_values - L_Y**2 / n) / n

    return float(Q_n)


def estimate_s(Y: np.ndarray) -> float:
    """
    Estimate the S statistic for unconditional CODEC calculation.

    Parameters
    ----------
    Y : np.ndarray
        The response variable.

    Returns
    -------
    float
        The estimated S statistic.
    """
    n = len(Y)

    # Calculate rank statistics
    L_Y = rankdata(-Y.ravel(), method="max").astype(float)

    # Calculate S statistic
    S_n = np.sum(L_Y * (n - L_Y)) / float(n) ** 3

    return float(S_n)


def estimate_t(Y: np.ndarray, X: np.ndarray) -> float:
    """
    Estimate the T statistic (the unconditional CODEC).

    Parameters
    ----------
    Y : np.ndarray
        The response variable.
    X : np.ndarray
        The conditioning variable.

    Returns
    -------
    float
        The estimated T statistic (CODEC value).
    """
    S = estimate_s(Y)

    if np.isclose(S, 0):
        return 1.0
    else:
        q = estimate_q(Y, X)
        return q / S


def find_nearest_neighbors(X: np.ndarray, random_state=None) -> np.ndarray:
    """
    Find the nearest neighbour of each point in X, handling repeats and ties.

    A point is never its own neighbour.  Repeated points (zero distance to
    the nearest other point) get a uniformly random *other* member of their
    group of identical points; ties between several other points at the same
    minimal distance are broken uniformly at random (as in the FOCI R
    package).

    Parameters
    ----------
    X : np.ndarray
        The data points, shape ``(n,)`` or ``(n, d)``.
    random_state : int, numpy Generator or None
        Randomness for breaking repeats/ties.  ``None`` uses NumPy's global
        generator (never reseeded).

    Returns
    -------
    np.ndarray
        Indices of nearest neighbors.
    """
    from copul.checkerboard._biv_engine import resolve_rng

    rng = resolve_rng(random_state)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = X.shape[0]
    if n < 2:
        return np.zeros(n, dtype=int)

    tree = cKDTree(X)
    k = min(3, n)
    distances, nn_indices = tree.query(X, k=k)
    own = np.arange(n)

    # the returned neighbour list may start with a duplicate instead of the
    # point itself -> take the first index that differs from the point
    nn_index_X = np.where(nn_indices[:, 0] != own, nn_indices[:, 0], nn_indices[:, 1])

    # Repeated data points: pick a random *other* point of the same group
    repeat_data = np.where(distances[:, 1] == 0)[0]
    if repeat_data.size > 0:
        _, groups = np.unique(X[repeat_data], axis=0, return_inverse=True)
        groups = np.asarray(groups).ravel()
        for g in np.unique(groups):
            members = repeat_data[groups == g]
            m = members.size
            # draw from the other m - 1 members uniformly
            draw = (rng.random(m) * (m - 1)).astype(int)
            draw = np.minimum(draw, m - 2)
            draw = draw + (draw >= np.arange(m))
            nn_index_X[members] = members[draw]

    # Ties: equal distances to the 2nd and 3rd nearest points
    if k > 2:
        ties = np.where(distances[:, 1] == distances[:, 2])[0]
        ties = np.setdiff1d(ties, repeat_data)
        for a in ties:
            d = np.linalg.norm(X - X[a], axis=1)
            d[a] = np.inf
            candidates = np.flatnonzero(d == d.min())
            nn_index_X[a] = candidates[int(rng.random() * candidates.size) % candidates.size]

    return nn_index_X


def _ensure_numpy_array(data: Any) -> np.ndarray:
    """
    Convert input data to a properly formatted numpy array.

    Parameters
    ----------
    data : array-like
        Input data to convert.

    Returns
    -------
    np.ndarray
        Converted numpy array.
    """
    if isinstance(data, list):
        data = np.array(data)
    elif is_pandas_instance(data, "Series", "DataFrame"):
        data = data.to_numpy()

    # Ensure 2D array for matrix data
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    # Reshape 1D array to column vector
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)

    return data


if __name__ == "__main__":
    import pandas as pd

    # Example usage and tests
    rng = np.random.default_rng(42)  # For reproducibility

    # Generate example data
    n_samples = 1000
    X = rng.random((n_samples, 2))
    Y_dependent = (X[:, 0] + X[:, 1]) % 1  # Y depends on X
    Y_independent = rng.random(n_samples)  # Y independent of X
    Z = rng.standard_normal((n_samples, 1))

    # Calculate various CODEC values
    print("CODEC between Y_dependent and X:", codec(Y_dependent, X))
    print("CODEC between Y_independent and X:", codec(Y_independent, X))
    print("Conditional CODEC (Y_dependent | Z, X):", codec(Y_dependent, Z, X))

    # Test with pandas DataFrame
    df_Y = pd.DataFrame({"dependent": Y_dependent, "independent": Y_independent})
    print("\nCODEC with multiple response variables:")
    print(codec(df_Y, X))

    # Verify edge cases
    try:
        print("\nTesting mismatched dimensions:")
        codec(Y_dependent, X[:500])
    except ValueError as e:
        print(f"Caught expected error: {e}")
