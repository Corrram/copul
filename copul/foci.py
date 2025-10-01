"""
Conditional Dependence Coefficient (CODEC) Implementation

This module provides functions to calculate the conditional dependence coefficient (CODEC),
a measure of conditional dependence between random variables based on an i.i.d. sample.

The implementation is based on the paper "An Empirical Study on New Model-Free Multi-output
Variable Selection Methods" by Ansari et al.
"""

import decimal
from typing import Union, Optional, Dict, List, Any
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import rankdata


def codec(
    Y: Union[np.ndarray, pd.Series, pd.DataFrame, List],
    Z: Union[np.ndarray, pd.Series, pd.DataFrame, List],
    X: Optional[Union[np.ndarray, pd.Series, pd.DataFrame, List]] = None,
    na_rm: bool = True,
) -> Union[float, Dict[str, float]]:
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
    if isinstance(Y, pd.DataFrame):
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

    # Calculate rank statistics
    R_Y = rankdata(Y.ravel(), method="max")
    L_Y = rankdata(-Y.ravel(), method="max")

    # Convert to decimal for numerical stability
    L_Y_dec = np.array([decimal.Decimal(str(float(val))) for val in L_Y])
    R_Y_dec = np.array([decimal.Decimal(str(float(val))) for val in R_Y])

    # Get ranks at nearest neighbor indices
    R_Y_nn = R_Y_dec[nn_index_X]

    # Calculate minimums and L_Y squared
    min_values = np.minimum(R_Y_dec, R_Y_nn)
    L_Y_squared = L_Y_dec**2
    n_dec = decimal.Decimal(str(n))

    # Calculate Q statistic
    Q_n = np.mean(min_values - L_Y_squared / n_dec) / n

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
    L_Y = rankdata(-Y.ravel(), method="max")

    # Convert to decimal for numerical stability
    L_Y_dec = np.array([decimal.Decimal(str(float(val))) for val in L_Y])
    n_dec = decimal.Decimal(str(n))

    # Calculate S statistic
    S_n = np.sum(L_Y_dec * (n_dec - L_Y_dec)) / (n_dec**3)

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


def find_nearest_neighbors(X: np.ndarray) -> np.ndarray:
    """
    Find the nearest neighbors for each point in X, handling repeats and ties.

    Parameters
    ----------
    X : np.ndarray
        The data points.

    Returns
    -------
    np.ndarray
        Indices of nearest neighbors.
    """
    # Ensure X is a numpy array
    X = np.asarray(X)

    # Use cKDTree for nearest neighbor search
    tree = cKDTree(X)
    distances, nn_indices = tree.query(X, k=min(3, X.shape[0]))

    # Get the second nearest neighbor (first is the point itself)
    nn_index_X = (
        nn_indices[:, 1].copy()
        if nn_indices.shape[1] > 1
        else np.zeros(X.shape[0], dtype=int)
    )

    # Find data points that are not unique (zero distance to nearest neighbor)
    repeat_data = np.where(distances[:, 1] == 0)[0] if distances.shape[1] > 1 else []

    # Handle repeated data points using DataFrame approach (like original code)
    if len(repeat_data) > 0:
        # Create a DataFrame to manage repeated data
        df_X = pd.DataFrame({"id": repeat_data, "group": nn_indices[repeat_data, 0]})

        # Function to select a random nearest neighbor
        def random_nn(group_ids):
            if len(group_ids) > 0:
                return np.random.choice(group_ids)
            return None

        # Apply to each group
        df_X["rnn"] = df_X.groupby("group")["id"].transform(random_nn)

        # Update nearest neighbor indices
        for idx, rnn in zip(repeat_data, df_X["rnn"]):
            if rnn is not None:
                nn_index_X[idx] = rnn

    # Handle ties (equal distances to second and third nearest neighbors)
    if nn_indices.shape[1] > 2:
        ties = np.where(distances[:, 1] == distances[:, 2])[0]
        ties = np.setdiff1d(ties, repeat_data)

        if len(ties) > 0:
            for a in ties:
                # Take current point
                a_point = X[a].reshape(1, -1)

                # Get all other points
                rest_points = np.delete(X, a, axis=0)
                rest_indices = np.delete(np.arange(X.shape[0]), a)

                # Find distances to all other points
                distances_to_others = np.linalg.norm(rest_points - a_point, axis=1)

                # Find points at minimum distance
                min_indices = np.where(
                    distances_to_others == distances_to_others.min()
                )[0]

                # Adjust indices and randomly select one
                adjusted_indices = rest_indices[min_indices]
                nn_index_X[a] = np.random.choice(adjusted_indices)

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
    elif isinstance(data, pd.Series):
        data = data.to_numpy()
    elif isinstance(data, pd.DataFrame):
        data = data.to_numpy()

    # Ensure 2D array for matrix data
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    # Reshape 1D array to column vector
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)

    return data


if __name__ == "__main__":
    # Example usage and tests
    np.random.seed(42)  # For reproducibility

    # Generate example data
    n_samples = 1000
    X = np.random.rand(n_samples, 2)
    Y_dependent = (X[:, 0] + X[:, 1]) % 1  # Y depends on X
    Y_independent = np.random.rand(n_samples)  # Y independent of X
    Z = np.random.randn(n_samples, 1)

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
