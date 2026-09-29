r"""
The empirical copula of a sample.

:class:`EmpiricalCopula` wraps the pseudo-observations
:math:`\hat U_i = (\hat U_{i1},\dots,\hat U_{id})` of a sample and provides

* the empirical copula :math:`C_n(u) = \frac1n\sum_i 1\{\hat U_i\le u\}`
  (Deheuvels, 1979) as a vectorized ``cdf``;
* its smooth/continuous versions: the empirical checkerboard copula
  (:meth:`~EmpiricalCopula.to_checkerboard`, Genest, Nešlehová & Rémillard,
  2017) and the empirical Bernstein copula
  (:meth:`~EmpiricalCopula.to_bernstein`, Sancetta & Satchell, 2004;
  Segers, Sibuya & Tsukahara, 2017);
* sample versions of all dependence measures of :mod:`copul.measures`
  (bivariate case), with the same method names as copula objects
  (``spearmans_rho()``, ``kendalls_tau()``, ``chatterjees_xi()``, ...) and
  keyed access through :meth:`~EmpiricalCopula.measure` /
  :meth:`~EmpiricalCopula.measures`;
* scatter and contour plots.

References
----------
* Deheuvels, P. (1979). La fonction de dépendance empirique et ses
  propriétés. *Acad. Roy. Belg. Bull. Cl. Sci.* 65, 274--292.
* Genest, C., Nešlehová, J. G. and Rémillard, B. (2017). Asymptotic behavior
  of the empirical multilinear copula process under broad conditions.
  *J. Multivariate Anal.* 159, 82--110.
* Sancetta, A. and Satchell, S. (2004). The Bernstein copula and its
  applications to modeling and approximations of multivariate
  distributions. *Econometric Theory* 20, 535--562.
* Segers, J., Sibuya, M. and Tsukahara, H. (2017). The empirical beta
  copula. *J. Multivariate Anal.* 155, 35--51.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from copul._lazy import plt
from copul.measures.registry import _iter_keys
from copul.stats._utils import RandomLike, as_data, chunk_size
from copul.stats.estimators import SAMPLE_ESTIMATORS, sample_measure
from copul.stats.pseudo_obs import pseudo_obs

__all__ = ["EmpiricalCopula", "checkerboard_mass"]


def _overlap_matrix(r: np.ndarray, m: int, n: int) -> np.ndarray:
    r"""``A[i, a] = n * |[(r_i-1)/n, r_i/n] \cap [a/m, (a+1)/m]|`` (rows sum to 1)."""
    lo = (r - 1.0)[:, None] / n
    hi = r[:, None] / n
    a = np.arange(m, dtype=float)[None, :]
    ov = np.clip(np.minimum(hi, (a + 1.0) / m) - np.maximum(lo, a / m), 0.0, None)
    return ov * n


def checkerboard_mass(data: Any, m: int | tuple[int, int] = 10) -> np.ndarray:
    r"""Mass matrix of the empirical checkerboard copula on an ``m x m`` grid.

    Each observation :math:`i` with ordinal ranks :math:`(R_i, S_i)` carries
    mass :math:`1/n` spread uniformly over the rank cell
    :math:`[\frac{R_i-1}n,\frac{R_i}n]\times[\frac{S_i-1}n,\frac{S_i}n]`
    (the multilinear/checkerboard extension of the empirical copula,
    Genest, Nešlehová & Rémillard, 2017); the matrix collects the mass of each
    cell of the ``m``-grid.  Its margins are *exactly* uniform for every
    ``m`` (also when ``m`` does not divide ``n``).  Ties are broken by order
    of appearance.

    Returns
    -------
    numpy.ndarray of shape (m1, m2), summing to one.
    """
    arr = as_data(data, min_dim=2, max_dim=2)
    m1, m2 = (int(m), int(m)) if np.ndim(m) == 0 else (int(m[0]), int(m[1]))
    if m1 < 1 or m2 < 1:
        raise ValueError("grid size must be positive")
    n = arr.shape[0]
    r = pseudo_obs(arr, ties="first", scale="n") * n
    A = _overlap_matrix(r[:, 0], m1, n)
    B = _overlap_matrix(r[:, 1], m2, n)
    return (A.T @ B) / n


class EmpiricalCopula:
    r"""Empirical copula of a sample.

    Parameters
    ----------
    data : array_like of shape (n, d)
        Raw data or pseudo-observations (one column per variable); the
        pseudo-observations are recomputed from the ranks either way.
    ties : str
        Tie handling of the ranks (see :func:`copul.stats.pseudo_obs`).
    scale : {"n+1", "n"}
        Scaling of the ranks for :attr:`U` (default :math:`R/(n+1)`).
    random_state : int, Generator or None
        Seed for random tie breaking.

    Attributes
    ----------
    U : numpy.ndarray of shape (n, d)
        The pseudo-observations.
    data : numpy.ndarray of shape (n, d)
        The data as given.
    n, dim : int
        Sample size and dimension.

    Examples
    --------
    >>> import copul as cp
    >>> from copul.stats import EmpiricalCopula
    >>> X = cp.Clayton(theta=2).rvs(500, random_state=0)
    >>> ec = EmpiricalCopula(X)
    >>> round(ec.kendalls_tau(), 2)  # doctest: +SKIP
    0.5
    >>> ec.measures(["tau", "rho", "xi"])  # doctest: +SKIP
    {'tau': 0.49..., 'rho': 0.67..., 'xi': 0.31...}
    """

    def __init__(
        self,
        data: Any,
        ties: str = "average",
        scale: str = "n+1",
        random_state: RandomLike = None,
    ) -> None:
        self.data = as_data(data, min_dim=2)
        self.ties = ties
        self.U = pseudo_obs(self.data, ties=ties, scale=scale, random_state=random_state)
        self.n, self.dim = self.U.shape

    # ------------------------------------------------------------------
    # basics
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"EmpiricalCopula(n={self.n}, dim={self.dim})"

    def __len__(self) -> int:
        return self.n

    @property
    def u(self) -> np.ndarray:
        """First coordinate of the pseudo-observations."""
        return self.U[:, 0]

    @property
    def v(self) -> np.ndarray:
        """Second coordinate of the pseudo-observations."""
        return self.U[:, 1]

    def _require_bivariate(self) -> None:
        if self.dim != 2:
            raise ValueError(f"this method needs bivariate data, got dimension {self.dim}.")

    def cdf(self, u: Any, v: Any = None) -> np.ndarray | float:
        r"""Empirical copula :math:`C_n(u) = \frac1n\#\{i : \hat U_i \le u\}`.

        Call as ``cdf(u, v)`` with broadcastable arrays (bivariate) or
        ``cdf(points)`` with ``points`` of shape ``(m, d)``.  Evaluated by
        chunked vectorized comparisons (:math:`O(nm)` time).

        Returns a float for scalar input, else an array of the broadcast shape
        (``(m,)`` for points).
        """
        if v is None:
            pts = np.asarray(u, dtype=float)
            scalar = pts.ndim == 1
            pts = np.atleast_2d(pts)
            if pts.shape[-1] != self.dim:
                raise ValueError(f"points must have {self.dim} columns.")
            out_shape = pts.shape[:-1]
            pts = pts.reshape(-1, self.dim)
        else:
            self._require_bivariate()
            uu, vv = np.broadcast_arrays(np.asarray(u, dtype=float), np.asarray(v, dtype=float))
            scalar = uu.ndim == 0
            out_shape = uu.shape
            pts = np.column_stack([uu.ravel(), vv.ravel()])
        m = pts.shape[0]
        out = np.empty(m, dtype=float)
        step = chunk_size(self.n * self.dim)
        for s in range(0, m, step):
            p = pts[s : s + step]
            inside = np.ones((p.shape[0], self.n), dtype=bool)
            for j in range(self.dim):
                inside &= self.U[None, :, j] <= p[:, j, None]
            out[s : s + step] = np.count_nonzero(inside, axis=1) / self.n
        if scalar:
            return float(out[0])
        return out.reshape(out_shape)

    __call__ = cdf

    # ------------------------------------------------------------------
    # smooth versions
    # ------------------------------------------------------------------
    def checkerboard_mass(self, m: int | tuple[int, int] = 10) -> np.ndarray:
        """Mass matrix of the empirical checkerboard copula (see
        :func:`checkerboard_mass`)."""
        self._require_bivariate()
        return checkerboard_mass(self.data, m)

    def to_checkerboard(self, m: int | tuple[int, int] = 10, checkerboard_type: str = "CheckPi"):
        r"""Empirical checkerboard copula with an ``m x m`` grid.

        Parameters
        ----------
        m : int or (int, int)
            Grid size.
        checkerboard_type : {"CheckPi", "CheckMin", "CheckW"}
            How mass is spread within a cell (uniformly, on the diagonal, on
            the anti-diagonal).

        Returns
        -------
        BivCheckPi, BivCheckMin or BivCheckW
            A genuine copula (margins exactly uniform, see
            :func:`checkerboard_mass`).
        """
        from copul.checkerboard.biv_check_min import BivCheckMin
        from copul.checkerboard.biv_check_pi import BivCheckPi
        from copul.checkerboard.biv_check_w import BivCheckW

        cls = {
            "checkpi": BivCheckPi,
            "bivcheckpi": BivCheckPi,
            "pi": BivCheckPi,
            "checkmin": BivCheckMin,
            "bivcheckmin": BivCheckMin,
            "min": BivCheckMin,
            "checkw": BivCheckW,
            "bivcheckw": BivCheckW,
            "w": BivCheckW,
        }.get(str(checkerboard_type).lower())
        if cls is None:
            raise ValueError(f"unknown checkerboard_type {checkerboard_type!r}")
        return cls(self.checkerboard_mass(m))

    def to_bernstein(self, m: int | tuple[int, int] = 10):
        r"""Empirical Bernstein copula of degree ``m``.

        .. math::

           C_{n,m}(u,v) = \sum_{k=1}^m\sum_{l=1}^m
           C^\#_n\bigl(\tfrac km, \tfrac lm\bigr)\,B_{m,k}(u)\,B_{m,l}(v),

        the Bernstein smoothing (Sancetta & Satchell, 2004; Segers, Sibuya &
        Tsukahara, 2017) of the empirical checkerboard copula
        :math:`C^\#_n`, which is a genuine copula for every ``m``.

        Returns
        -------
        BivBernsteinCopula
        """
        from copul.checkerboard.bernstein import BernsteinCopula

        return BernsteinCopula(self.checkerboard_mass(m))

    # ------------------------------------------------------------------
    # dependence measures
    # ------------------------------------------------------------------
    def measure(self, key: str, **options: Any) -> float:
        """Sample estimate of the dependence measure ``key`` (see
        :mod:`copul.stats.estimators`); options such as ``k`` (tail
        estimators), ``p`` (``"lp"``) or ``random_state`` are forwarded."""
        self._require_bivariate()
        return sample_measure(self.data[:, 0], self.data[:, 1], key, **options)

    def measures(self, keys: str | Iterable[str] | None = None, **options: Any) -> dict:
        """``{key: estimate}`` for several measures (default: the registry's
        default set ``xi, rho, tau, footrule, gamma, beta, nu``)."""
        return {k: self.measure(k, **options) for k in _iter_keys(keys)}

    def estimate(self, measures: str | Iterable[str] | None = None, **kwargs: Any):
        """Tidy DataFrame of estimates with optional confidence intervals, see
        :func:`copul.stats.estimate`."""
        from copul.stats.inference import estimate

        return estimate(self.data, measures=measures, **kwargs)

    def chatterjees_xi(self, condition_on_y: bool = False, random_state: RandomLike = None):
        r"""Chatterjee's :math:`\xi_n` (``condition_on_y=True``: :math:`\xi_n(Y, X)`)."""
        return self.measure("xi_2" if condition_on_y else "xi", random_state=random_state)

    def spearmans_rho(self) -> float:
        r"""Spearman's :math:`\rho_n`."""
        return self.measure("rho")

    def kendalls_tau(self, variant: str = "b") -> float:
        r"""Kendall's :math:`\tau_n` (:math:`\tau_b` by default)."""
        return self.measure("tau", variant=variant)

    def spearmans_footrule(self) -> float:
        r"""Spearman's footrule :math:`\psi_n`."""
        return self.measure("footrule")

    def ginis_gamma(self) -> float:
        r"""Gini's :math:`\gamma_n`."""
        return self.measure("gamma")

    def blomqvists_beta(self) -> float:
        r"""Blomqvist's :math:`\beta_n`."""
        return self.measure("beta")

    def blests_nu(self) -> float:
        r"""Blest's :math:`\nu_n`."""
        return self.measure("nu")

    def hoeffdings_d(self) -> float:
        r"""Hoeffding's :math:`\Phi^2_n` (registry key ``"hoeffdings_d"``)."""
        return self.measure("hoeffdings_d")

    def schweizer_wolff_sigma(self) -> float:
        r"""Schweizer--Wolff :math:`\sigma_n`."""
        return self.measure("sigma")

    def uniform_distance(self) -> float:
        r"""Uniform distance :math:`\kappa_n = 4\sup|C_n - \Pi|`."""
        return self.measure("kappa")

    def lp_distance(self, p: float = 2.0) -> float:
        r""":math:`L^p` distance :math:`\delta_{p,n}`."""
        return self.measure("lp", p=p)

    def blum_kiefer_rosenblatt(self) -> float:
        r"""Hoeffding's :math:`D_n`, estimating the BKR coefficient :math:`B`."""
        return self.measure("bkr")

    def mutual_information(self, k: int = 3, random_state: RandomLike = None) -> float:
        """KSG estimator of the mutual information."""
        return self.measure("mutual_information", k=k, random_state=random_state)

    def lambda_L(self, method: str = "ss", k: int | None = None) -> float:
        r"""Lower tail dependence estimate (Schmidt--Stadtmüller or CFG)."""
        return self.measure("lambda_l", method=method, k=k)

    def lambda_U(self, method: str = "ss", k: int | None = None) -> float:
        r"""Upper tail dependence estimate (Schmidt--Stadtmüller or CFG)."""
        return self.measure("lambda_u", method=method, k=k)

    @staticmethod
    def available_measures() -> list[str]:
        """Canonical keys with a sample estimator."""
        return list(SAMPLE_ESTIMATORS)

    # ------------------------------------------------------------------
    # plotting
    # ------------------------------------------------------------------
    def scatter_plot(self, ax=None, **kwargs: Any):
        """Scatter plot of the pseudo-observations; returns the axes."""
        self._require_bivariate()
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5))
        kw = {"s": 6, "alpha": 0.6}
        kw.update(kwargs)
        ax.scatter(self.u, self.v, **kw)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("u")
        ax.set_ylabel("v")
        ax.set_title(f"Pseudo-observations (n={self.n})")
        return ax

    def plot_contour(
        self,
        ax=None,
        grid: int = 101,
        levels: int | Iterable[float] = 10,
        compare_with: Any = None,
        **kwargs: Any,
    ):
        """Contour plot of :math:`C_n` (optionally together with the cdf of a
        fitted copula ``compare_with``, dashed); returns the axes."""
        self._require_bivariate()
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5))
        t = np.linspace(0.0, 1.0, int(grid))
        uu, vv = np.meshgrid(t, t, indexing="xy")
        z = self.cdf(uu, vv)
        cs = ax.contour(uu, vv, z, levels=levels, **kwargs)
        ax.clabel(cs, inline=True, fontsize=7)
        if compare_with is not None:
            from copul.stats._adapters import cdf as model_cdf

            # a FitResult carries the fitted copula in ``.copula``
            cop = getattr(compare_with, "copula", compare_with)
            zc = model_cdf(cop, uu.ravel(), vv.ravel()).reshape(uu.shape)
            ax.contour(uu, vv, zc, levels=cs.levels, linestyles="dashed", colors="gray")
        ax.set_aspect("equal")
        ax.set_xlabel("u")
        ax.set_ylabel("v")
        ax.set_title("Empirical copula")
        return ax

    def plot(self, **kwargs: Any):
        """Alias of :meth:`scatter_plot`."""
        return self.scatter_plot(**kwargs)


def _as_empirical(data: Any) -> EmpiricalCopula:
    return data if isinstance(data, EmpiricalCopula) else EmpiricalCopula(data)
