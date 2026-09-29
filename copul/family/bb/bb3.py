r"""BB3 copula, Joe (2014) Sec. 4.17.3."""

from __future__ import annotations

import numpy as np
import sympy as sp
from scipy.special import xlogy

from copul.family.bb._frailty import log_gamma_rv, log_positive_stable_rv
from copul.family.bb.lt_archimedean import LTArchimedeanCopula, log_expm1, softplus


class BB3(LTArchimedeanCopula):
    r"""BB3 copula (Joe & Hu 1996), a positive-stable–gamma frailty family.

    .. math::

       C(u,v) = \exp\Bigl\{-\Bigl[\delta^{-1}\log\bigl(e^{\delta\tilde u^{\theta}}
                + e^{\delta\tilde v^{\theta}} - 1\bigr)\Bigr]^{1/\theta}\Bigr\},
       \quad \tilde u = -\log u,\qquad \theta\ge 1,\ \delta>0 .

    :math:`\psi(s) = \exp\{-[\delta^{-1}\log(1+s)]^{1/\theta}\}`,
    :math:`\varphi(t) = e^{\delta(-\log t)^{\theta}} - 1`.  Frailty:
    :math:`V\mid S\sim\mathrm{Gamma}(S/\delta)` with :math:`S` positive stable,
    Laplace transform :math:`e^{-s^{1/\theta}}`.

    Closed forms: :math:`\lambda_U = 2 - 2^{1/\theta}`;
    :math:`\lambda_L = 1` for :math:`\theta>1` and :math:`2^{-1/\delta}` for
    :math:`\theta = 1`.  Special case :math:`\theta = 1`: Clayton copula with
    parameter :math:`\delta`.

    References
    ----------
    Joe, H. (2014). *Dependence Modeling with Copulas*, CRC Press, Sec. 4.17.3.
    """

    theta, delta = sp.symbols("theta delta", positive=True)
    params = [theta, delta]
    intervals = {
        "theta": sp.Interval(1, sp.oo, left_open=False, right_open=True),
        "delta": sp.Interval(0, sp.oo, left_open=True, right_open=True),
    }

    def _phi_sym(self, t, th, de):
        return sp.exp(de * (-sp.log(t)) ** th) - 1

    def _psi_sym(self, s, th, de):
        return sp.exp(-((sp.log(1 + s) / de) ** (1 / th)))

    def _log_phi(self, t, th, de):
        return log_expm1(de * (-np.log(t)) ** th)

    def _log_mdphi(self, t, th, de):
        x = -np.log(t)
        return de * x**th + np.log(de * th) + xlogy(th - 1.0, x) + x

    def _psi_ls(self, ls, th, de):
        return np.exp(-((softplus(ls) / de) ** (1.0 / th)))

    def _parts(self, ls, th, de):
        L = softplus(ls)
        r = (L / de) ** (1.0 / th)
        log_g = np.log(r) - np.log(th) - np.log(L) - L
        return L, r, log_g

    def _log_mdpsi_ls(self, ls, th, de):
        L, r, log_g = self._parts(ls, th, de)
        return -r + log_g

    def _log_d2psi_ls(self, ls, th, de):
        L, r, log_g = self._parts(ls, th, de)
        return -r + log_g - L + np.log(r / (th * L) + 1.0 + (1.0 - 1.0 / th) / L)

    def _log_frailty(self, n, rng, th, de):
        ls = log_positive_stable_rv(1.0 / th, n, rng)
        return log_gamma_rv(np.exp(ls) / de, rng)

    def lambda_L(self, *args, **kwargs):
        r""":math:`\lambda_L = 1` (:math:`\theta>1`), :math:`2^{-1/\delta}` (:math:`\theta=1`)."""
        th, de = self.theta, self.delta
        if isinstance(th, sp.Basic) and not th.is_number:
            return sp.Piecewise((2 ** (-1 / de), sp.Eq(th, 1)), (1, True))
        return 2 ** (-1 / de) if float(th) == 1.0 else 1

    def lambda_U(self, *args, **kwargs):
        r""":math:`\lambda_U = 2 - 2^{1/\theta}`."""
        return 2 - 2 ** (1 / self.theta)
