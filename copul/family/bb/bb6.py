r"""BB6 (Joe–Gumbel) copula, Joe (2014) Sec. 4.17.4."""

from __future__ import annotations

import numpy as np
import sympy as sp
from scipy.special import xlogy

from copul.family.bb._frailty import log_positive_stable_rv, sibuya_rv
from copul.family.bb.lt_archimedean import LTArchimedeanCopula


def _log_m(t, th):
    r""":math:`\log(1-(1-t)^\theta)` computed accurately."""
    return np.log(-np.expm1(th * np.log1p(-t)))


class BB6(LTArchimedeanCopula):
    r"""BB6 copula (Joe & Hu 1996), a Joe–Gumbel two-parameter family.

    .. math::

       C(u,v) = 1 - \Bigl(1 - \exp\Bigl\{-\bigl[(-\log(1-\bar u^{\theta}))^{\delta}
                + (-\log(1-\bar v^{\theta}))^{\delta}\bigr]^{1/\delta}\Bigr\}\Bigr)^{1/\theta},

    :math:`\bar u = 1-u`, :math:`\theta\ge 1`, :math:`\delta\ge 1`.

    :math:`\psi(s) = 1 - [1 - \exp(-s^{1/\delta})]^{1/\theta}`,
    :math:`\varphi(t) = [-\log(1-(1-t)^{\theta})]^{\delta}`.  Frailty
    :math:`V = N^{\delta}S` with :math:`N\sim\mathrm{Sibuya}(1/\theta)` and
    :math:`S` positive stable with Laplace transform :math:`e^{-s^{1/\delta}}`.

    Closed forms: :math:`\lambda_L = 0`,
    :math:`\lambda_U = 2 - 2^{1/(\theta\delta)}`.
    Special cases: :math:`\theta = 1` is Gumbel–Hougaard(:math:`\delta`),
    :math:`\delta = 1` is Joe(:math:`\theta`).

    References
    ----------
    Joe, H. (2014). *Dependence Modeling with Copulas*, CRC Press, Sec. 4.17.4.
    """

    theta, delta = sp.symbols("theta delta", positive=True)
    params = [theta, delta]
    intervals = {
        "theta": sp.Interval(1, sp.oo, left_open=False, right_open=True),
        "delta": sp.Interval(1, sp.oo, left_open=False, right_open=True),
    }

    def _phi_sym(self, t, th, de):
        return (-sp.log(1 - (1 - t) ** th)) ** de

    def _psi_sym(self, s, th, de):
        return 1 - (1 - sp.exp(-(s ** (1 / de)))) ** (1 / th)

    def _log_phi(self, t, th, de):
        return de * np.log(-_log_m(t, th))

    def _log_mdphi(self, t, th, de):
        lm = _log_m(t, th)
        return np.log(de * th) + xlogy(th - 1.0, 1.0 - t) + xlogy(de - 1.0, -lm) - lm

    def _psi_ls(self, ls, th, de):
        w = np.exp(ls / de)
        return -np.expm1(np.log(-np.expm1(-w)) / th)

    def _log_mdpsi_ls(self, ls, th, de):
        w = np.exp(ls / de)
        return -np.log(th * de) + (1.0 / th - 1.0) * np.log(-np.expm1(-w)) - w + ls / de - ls

    def _log_d2psi_ls(self, ls, th, de):
        w = np.exp(ls / de)
        bracket = (1.0 - 1.0 / th) * w / (de * np.expm1(w)) + w / de + 1.0 - 1.0 / de
        return (
            -np.log(th * de)
            + (1.0 / th - 1.0) * np.log(-np.expm1(-w))
            - w
            + ls / de
            - 2.0 * ls
            + np.log(bracket)
        )

    def _log_frailty(self, n, rng, th, de):
        return de * np.log(sibuya_rv(1.0 / th, n, rng)) + log_positive_stable_rv(1.0 / de, n, rng)

    def lambda_L(self, *args, **kwargs):
        r""":math:`\lambda_L = 0`."""
        return 0

    def lambda_U(self, *args, **kwargs):
        r""":math:`\lambda_U = 2 - 2^{1/(\theta\delta)}`."""
        return 2 - 2 ** (1 / (self.theta * self.delta))
