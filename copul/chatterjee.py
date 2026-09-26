"""
Chatterjee's Xi coefficient for measuring nonlinear dependence.

This module implements Chatterjee's Xi coefficient, a rank-based measure of dependence
that can detect both linear and non-linear associations between variables.

References:
    Chatterjee, S. (2021). A new coefficient of correlation.
    Journal of the American Statistical Association, 116(536), 2009-2022.
"""

import numpy as np
from scipy import stats


def _validate_pair(xvec, yvec) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(xvec, dtype=float).ravel()
    y = np.asarray(yvec, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError(f"xvec and yvec must have the same length, got {x.size} and {y.size}.")
    if np.isnan(x).any() or np.isnan(y).any():
        raise ValueError("Input contains NaN values.")
    return x, y


def xi_ncalculate(xvec: np.ndarray, yvec: np.ndarray, random_state=None) -> float:
    r"""
    Chatterjee's rank correlation :math:`\xi_n(X, Y)` (Chatterjee, 2021).

    The data are ordered by ``xvec`` (ties in ``xvec`` are broken uniformly at
    random) and, with :math:`r_i = \#\{j : Y_j \le Y_{(i)}\}` and
    :math:`l_i = \#\{j : Y_j \ge Y_{(i)}\}`,

    .. math::

       \xi_n = 1 - \frac{n \sum_{i=1}^{n-1} |r_{i+1} - r_i|}
                        {2 \sum_{i=1}^n l_i (n - l_i)} .

    Without ties in ``yvec`` this reduces to
    :math:`1 - 3\sum_i |r_{i+1} - r_i| / (n^2 - 1)`.

    Parameters
    ----------
    xvec, yvec : array-like
        Samples of equal length.
    random_state : int, numpy Generator or None, optional
        Randomness for breaking ties in ``xvec`` (irrelevant without ties).

    Returns
    -------
    float
        The Xi_n dependence measure.

    Raises
    ------
    ValueError
        If the inputs have different lengths or contain NaN values.

    Notes
    -----
    - The measure is not symmetric: xi_n(x, y) may not equal xi_n(y, x).
    - For ``Y`` a strictly monotone function of ``X`` (no ties),
      :math:`\xi_n = 1 - 3/(n+1)`, which tends to 1 (it is *not* 0.5 in
      general; for ``n = 5`` it happens to equal 0.5).
    - For constant ``yvec`` the coefficient is undefined and ``nan`` is returned.
    - Empty or single-element vectors return NaN.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(1, 11)
    >>> round(xi_ncalculate(x, x), 4)  # 1 - 3/11
    0.7273
    """
    x, y = _validate_pair(xvec, yvec)
    n = x.size
    if n < 2:
        return np.nan

    if np.unique(x).size < n:
        rng = np.random.default_rng(random_state)
        order = np.lexsort((rng.random(n), x))
    else:
        order = np.argsort(x, kind="stable")
    ys = y[order]
    sorted_y = np.sort(y)
    r = np.searchsorted(sorted_y, ys, side="right")
    l_ = n - np.searchsorted(sorted_y, ys, side="left")
    denom = 2.0 * np.sum(l_ * (n - l_), dtype=float)
    if denom == 0:
        return np.nan
    return float(1.0 - n * np.sum(np.abs(np.diff(r)), dtype=float) / denom)


def xi_nvarcalculate(xvec: np.ndarray, yvec: np.ndarray) -> float:
    r"""
    Estimate the asymptotic variance of :math:`\sqrt{n}\,\xi_n`.

    The returned value estimates :math:`\sigma^2` in
    :math:`\sqrt{n}(\xi_n - \xi) \to N(0, \sigma^2)` (general, possibly
    dependent case); the standard error of :math:`\xi_n` itself is therefore
    ``sqrt(xi_nvarcalculate(x, y) / n)``.  All sums are evaluated in
    :math:`O(n \log n)` via sorting.  The estimator is consistent but tends
    to underestimate the variance in small samples (e.g. by ~30% at
    ``n = 500`` for moderately dependent data).

    Parameters
    ----------
    xvec, yvec : array-like
        Samples of equal length.

    Returns
    -------
    float
        The estimated asymptotic variance (non-negative), ``nan`` for fewer
        than two observations.

    Raises
    ------
    ValueError
        If the inputs have different lengths or contain NaN values.
    """
    x, y = _validate_pair(xvec, yvec)
    n = x.size
    if n < 2:
        return np.nan

    # ordinal ranks, y ranks ordered by x
    yrank_temp = np.argsort(np.argsort(y, kind="stable"), kind="stable") + 1
    yrank = yrank_temp[np.argsort(x, kind="stable")].astype(float)

    # Create shifted versions of the y ranks
    yrank1 = np.concatenate((yrank[1:n], [yrank[n - 1]]))
    yrank2 = np.concatenate((yrank[2:n], [yrank[n - 1]] * min(2, n)))[:n]
    yrank3 = np.concatenate((yrank[3:n], [yrank[n - 1]] * min(3, n)))[:n]

    # Compute the terms needed for variance calculation
    term1 = np.minimum(yrank, yrank1)
    term2 = np.minimum(yrank, yrank2)
    term3 = np.minimum(yrank2, yrank3)
    term5 = np.minimum(yrank1, yrank2)

    # term4[i] = #{k != i : yrank[i] <= term1[k]}  (sorting instead of O(n^2))
    t1_sorted = np.sort(term1)
    term4 = n - np.searchsorted(t1_sorted, yrank, side="left")
    term4 = term4 - (term1 >= yrank)

    # sum6_terms[i] = sum_{k != i} min(term1[i], term1[k])
    csum = np.concatenate(([0.0], np.cumsum(t1_sorted)))
    idx = np.searchsorted(t1_sorted, term1, side="left")
    sum6_terms = csum[idx] + term1 * (n - idx) - term1

    # Compute the sums needed for variance calculation
    sum1 = np.mean((term1 / n) ** 2)
    sum2 = np.mean(term1 * term2 / n**2)
    sum3 = np.mean(term1 * term3 / n**2)
    sum4 = np.mean(term4 * term1 / (n * (n - 1)))
    sum5 = np.mean(term4 * term5 / (n * (n - 1)))
    sum6 = np.mean(sum6_terms / (n * (n - 1)))
    sum7 = (np.mean(term1 / n)) ** 2

    variance = 36 * (sum1 + 2 * sum2 - 2 * sum3 + 4 * sum4 - 2 * sum5 + sum6 - 4 * sum7)
    return float(max(0.0, variance))


def xi_null_variance(yvec: np.ndarray) -> float:
    r"""
    Asymptotic variance of :math:`\sqrt{n}\,\xi_n` under independence.

    Equals :math:`2/5` for continuous ``Y`` (Chatterjee 2021, Thm. 2.1); with
    ties in ``yvec`` the consistent estimator of Chatterjee (2021, Thm. 2.2)
    is used.
    """
    y = np.asarray(yvec, dtype=float).ravel()
    n = y.size
    if n < 2:
        return np.nan
    if np.unique(y).size == n:
        return 0.4
    sorted_y = np.sort(y)
    fr = np.searchsorted(sorted_y, y, side="right") / n  # rank (ties: max) / n
    gr = (n - np.searchsorted(sorted_y, y, side="left")) / n  # rank of -y / n
    cu = np.mean(gr * (1.0 - gr))
    if cu == 0:
        return np.nan
    qfr = np.sort(fr)
    ind = np.arange(1, n + 1)
    ind2 = 2 * n - 2 * ind + 1
    ai = np.mean(ind2 * qfr * qfr) / n
    ci = np.mean(ind2 * qfr) / n
    cq = np.cumsum(qfr)
    m = (cq + (n - ind) * qfr) / n
    b = np.mean(m**2)
    return float((ai - 2.0 * b + ci**2) / cu**2)


def xi_n_with_ci(
    xvec: np.ndarray, yvec: np.ndarray, alpha: float = 0.05
) -> tuple[float, tuple[float, float]]:
    """
    Calculate Xi_n dependence measure with an asymptotic confidence interval.

    Uses ``xi_n +- z_{1-alpha/2} * sqrt(sigma^2 / n)`` with the variance
    estimate ``sigma^2`` of :func:`xi_nvarcalculate` (which refers to
    ``sqrt(n) * xi_n``), clipped to ``[0, 1]``.

    Parameters
    ----------
    xvec : np.ndarray
        First vector of data.
    yvec : np.ndarray
        Second vector of data.
    alpha : float, optional
        Significance level, default is 0.05 for 95% confidence interval.

    Returns
    -------
    Tuple[float, Tuple[float, float]]
        The Xi_n dependence measure and its confidence interval as (xi_n, (lower, upper)).
    """
    x, y = _validate_pair(xvec, yvec)
    n = x.size
    xi = xi_ncalculate(x, y)
    var = xi_nvarcalculate(x, y)
    se = np.sqrt(var / n)
    z = stats.norm.ppf(1 - alpha / 2)
    lower = max(0.0, xi - z * se)
    upper = min(1.0, xi + z * se)
    return xi, (lower, upper)


def test_independence(
    xvec: np.ndarray, yvec: np.ndarray, alpha: float = 0.05
) -> tuple[float, float, bool]:
    r"""
    One-sided asymptotic test of the null hypothesis of independence.

    Under :math:`H_0`, :math:`\sqrt{n}\,\xi_n \to N(0, \tau^2)` with
    :math:`\tau^2 = 2/5` for continuous ``Y`` (see :func:`xi_null_variance`
    for ties); the p-value is :math:`1 - \Phi(\sqrt{n}\,\xi_n / \tau)`.

    Returns
    -------
    Tuple[float, float, bool]
        The Xi_n value, p-value, and boolean indicating if the null hypothesis
        of independence should be rejected.
    """
    x, y = _validate_pair(xvec, yvec)
    n = x.size
    xi = xi_ncalculate(x, y)
    var0 = xi_null_variance(y)
    if not np.isfinite(xi) or not np.isfinite(var0) or var0 <= 0:
        return xi, np.nan, False
    z_score = np.sqrt(n) * xi / np.sqrt(var0)
    p_value = float(stats.norm.sf(z_score))
    return xi, p_value, bool(p_value < alpha)


# prevent pytest from collecting the statistical test function above
test_independence.__test__ = False
