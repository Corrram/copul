r"""BB10 copula, Joe (2014) Sec. 4.17.8."""

from __future__ import annotations

import numpy as np
import sympy as sp

from copul.family.bb.lt_archimedean import LTArchimedeanCopula


class BB10(LTArchimedeanCopula):
    r"""BB10 copula, a negative-binomial frailty family extending Ali–Mikhail–Haq.

    .. math::

       C(u,v) = uv\,\bigl[1 - \pi(1-u^{\theta})(1-v^{\theta})\bigr]^{-1/\theta},
       \qquad \theta>0,\ 0\le\pi<1 .

    :math:`\psi(s) = [(1-\pi)/(e^{s}-\pi)]^{1/\theta}`,
    :math:`\varphi(t) = \log[(1-\pi)t^{-\theta} + \pi]`.  Frailty:
    :math:`V = 1/\theta + K` with :math:`K` negative binomial
    (:math:`P(K=k)\propto\Gamma(1/\theta+k)\pi^k/k!`).

    Closed forms: :math:`\lambda_L = \lambda_U = 0`.
    Special cases: :math:`\pi = 0` is the independence copula,
    :math:`\theta = 1` is Ali–Mikhail–Haq(:math:`\pi`); :math:`\pi\to 1` gives
    Clayton(:math:`\theta`).

    References
    ----------
    Joe, H. (2014). *Dependence Modeling with Copulas*, CRC Press, Sec. 4.17.8.
    """

    theta, pi = sp.symbols("theta pi", positive=True)
    params = [theta, pi]
    intervals = {
        "theta": sp.Interval(0, sp.oo, left_open=True, right_open=True),
        "pi": sp.Interval(0, 1, left_open=False, right_open=True),
    }

    @property
    def _cdf_expr(self):
        th, p = self._sym_params()
        u, v = self.u, self.v
        return u * v * (1 - p * (1 - u**th) * (1 - v**th)) ** (-1 / th)

    def _phi_sym(self, t, th, p):
        return sp.log((1 - p) * t ** (-th) + p)

    def _psi_sym(self, s, th, p):
        return ((1 - p) / (sp.exp(s) - p)) ** (1 / th)

    def _log_phi(self, t, th, p):
        return np.log(np.log1p((1.0 - p) * np.expm1(-th * np.log(t))))

    def _log_mdphi(self, t, th, p):
        lt = np.log(t)
        phi = np.log1p((1.0 - p) * np.expm1(-th * lt))
        return np.log((1.0 - p) * th) - (th + 1.0) * lt - phi

    def _log_psi(self, s, th, p):
        return (np.log1p(-p) - s - np.log1p(-p * np.exp(-s))) / th

    def _psi_ls(self, ls, th, p):
        return np.exp(self._log_psi(np.exp(ls), th, p))

    def _log_mdpsi_ls(self, ls, th, p):
        s = np.exp(ls)
        return self._log_psi(s, th, p) - np.log(th) - np.log1p(-p * np.exp(-s))

    def _log_d2psi_ls(self, ls, th, p):
        s = np.exp(ls)
        q = p * np.exp(-s)
        return self._log_psi(s, th, p) - np.log(th) - 2.0 * np.log1p(-q) + np.log(1.0 / th + q)

    def _log_frailty(self, n, rng, th, p):
        k = rng.negative_binomial(1.0 / th, 1.0 - p, size=n) if p > 0 else np.zeros(n)
        return np.log(1.0 / th + k)

    def lambda_L(self, *args, **kwargs):
        r""":math:`\lambda_L = 0`."""
        return 0

    def lambda_U(self, *args, **kwargs):
        r""":math:`\lambda_U = 0`."""
        return 0
