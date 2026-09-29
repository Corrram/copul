r"""BB9 (Crowder) copula, Joe (2014) Sec. 4.17.7."""

from __future__ import annotations

import numpy as np
import sympy as sp
from scipy.special import xlogy

from copul.family.bb._frailty import log_tilted_stable_rv
from copul.family.bb.lt_archimedean import LTArchimedeanCopula


class BB9(LTArchimedeanCopula):
    r"""BB9 (Crowder) copula, an exponentially tilted positive-stable frailty family.

    .. math::

       C(u,v) = \exp\Bigl\{-\bigl[(\delta^{-1}-\log u)^{\theta}
                + (\delta^{-1}-\log v)^{\theta} - \delta^{-\theta}\bigr]^{1/\theta}
                + \delta^{-1}\Bigr\},
       \qquad \theta\ge 1,\ \delta>0 .

    :math:`\psi(s) = \exp\{-(\delta^{-\theta}+s)^{1/\theta} + \delta^{-1}\}`,
    :math:`\varphi(t) = (\delta^{-1}-\log t)^{\theta} - \delta^{-\theta}`.  The
    frailty is positive stable with index :math:`1/\theta`, exponentially
    tilted by :math:`\delta^{-\theta}`.

    Closed forms: :math:`\lambda_L = \lambda_U = 0`.
    Special cases: :math:`\theta = 1` is the independence copula;
    :math:`\delta\to\infty` gives Gumbel–Hougaard(:math:`\theta`).

    References
    ----------
    Crowder, M. (1989). A multivariate distribution with Weibull connections.
    *J. R. Stat. Soc. B* 51, 93–107.

    Joe, H. (2014). *Dependence Modeling with Copulas*, CRC Press, Sec. 4.17.7.
    """

    theta, delta = sp.symbols("theta delta", positive=True)
    params = [theta, delta]
    intervals = {
        "theta": sp.Interval(1, sp.oo, left_open=False, right_open=True),
        "delta": sp.Interval(0, sp.oo, left_open=True, right_open=True),
    }

    def _phi_sym(self, t, th, de):
        return (1 / de - sp.log(t)) ** th - de ** (-th)

    def _psi_sym(self, s, th, de):
        return sp.exp(-((de ** (-th) + s) ** (1 / th)) + 1 / de)

    def _log_phi(self, t, th, de):
        x = -np.log(t)
        return -th * np.log(de) + np.log(np.expm1(th * np.log1p(de * x)))

    def _log_mdphi(self, t, th, de):
        x = -np.log(t)
        return np.log(th) + xlogy(th - 1.0, 1.0 / de + x) + x

    def _parts(self, ls, th, de):
        las = np.logaddexp(-th * np.log(de), ls)  # log(delta^-theta + s)
        z = np.exp(las / th)
        return las, z

    def _psi_ls(self, ls, th, de):
        las, z = self._parts(ls, th, de)
        return np.exp(1.0 / de - z)

    def _log_mdpsi_ls(self, ls, th, de):
        las, z = self._parts(ls, th, de)
        return 1.0 / de - z + np.log(z) - np.log(th) - las

    def _log_d2psi_ls(self, ls, th, de):
        las, z = self._parts(ls, th, de)
        return 1.0 / de - z + np.log(z) - np.log(th) - 2.0 * las + np.log(z / th + 1.0 - 1.0 / th)

    def _log_frailty(self, n, rng, th, de):
        return log_tilted_stable_rv(1.0 / th, de ** (-th), n, rng)

    def lambda_L(self, *args, **kwargs):
        r""":math:`\lambda_L = 0`."""
        return 0

    def lambda_U(self, *args, **kwargs):
        r""":math:`\lambda_U = 0`."""
        return 0
