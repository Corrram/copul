r"""BB2 copula, Joe (2014) Sec. 4.17.2."""

from __future__ import annotations

import numpy as np
import sympy as sp

from copul.family.bb._frailty import log_gamma_rv
from copul.family.bb.lt_archimedean import LTArchimedeanCopula, log_expm1, softplus


class BB2(LTArchimedeanCopula):
    r"""BB2 copula (Joe & Hu 1996), a gamma–gamma frailty family.

    .. math::

       C(u,v) = \Bigl[1 + \delta^{-1}\log\bigl(e^{\delta(u^{-\theta}-1)}
                + e^{\delta(v^{-\theta}-1)} - 1\bigr)\Bigr]^{-1/\theta},
       \qquad \theta>0,\ \delta>0 .

    :math:`\psi(s) = [1 + \delta^{-1}\log(1+s)]^{-1/\theta}`,
    :math:`\varphi(t) = e^{\delta(t^{-\theta}-1)} - 1`.  Frailty:
    :math:`V\mid M \sim \mathrm{Gamma}(M/\delta)`,
    :math:`M\sim\mathrm{Gamma}(1/\theta)`.

    Closed forms: :math:`\lambda_L = 1`, :math:`\lambda_U = 0`.
    As :math:`\delta\to 0^+` the Clayton copula with parameter :math:`\theta`
    is obtained.

    References
    ----------
    Joe, H. (2014). *Dependence Modeling with Copulas*, CRC Press, Sec. 4.17.2.
    """

    theta, delta = sp.symbols("theta delta", positive=True)
    params = [theta, delta]
    intervals = {
        "theta": sp.Interval(0, sp.oo, left_open=True, right_open=True),
        "delta": sp.Interval(0, sp.oo, left_open=True, right_open=True),
    }

    def _phi_sym(self, t, th, de):
        return sp.exp(de * (t ** (-th) - 1)) - 1

    def _psi_sym(self, s, th, de):
        return (1 + sp.log(1 + s) / de) ** (-1 / th)

    def _log_phi(self, t, th, de):
        return log_expm1(de * np.expm1(-th * np.log(t)))

    def _log_mdphi(self, t, th, de):
        lt = np.log(t)
        return de * np.expm1(-th * lt) + np.log(de * th) - (th + 1.0) * lt

    def _psi_ls(self, ls, th, de):
        return np.exp(-np.log1p(softplus(ls) / de) / th)

    def _log_mdpsi_ls(self, ls, th, de):
        L = softplus(ls)
        return -np.log(th * de) - (1.0 / th + 1.0) * np.log1p(L / de) - L

    def _log_d2psi_ls(self, ls, th, de):
        L = softplus(ls)
        q = 1.0 + L / de
        return (
            -np.log(th * de)
            - (1.0 / th + 2.0) * np.log(q)
            - 2.0 * L
            + np.log((1.0 + th) / (th * de) + q)
        )

    def _log_frailty(self, n, rng, th, de):
        lm = log_gamma_rv(np.full(n, 1.0 / th), rng)
        return log_gamma_rv(np.exp(lm) / de, rng)

    def lambda_L(self, *args, **kwargs):
        r""":math:`\lambda_L = 1`."""
        return 1

    def lambda_U(self, *args, **kwargs):
        r""":math:`\lambda_U = 0`."""
        return 0
