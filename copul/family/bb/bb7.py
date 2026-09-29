r"""BB7 (Joe–Clayton) copula, Joe (2014) Sec. 4.17.5."""

from __future__ import annotations

import numpy as np
import sympy as sp
from scipy.special import xlogy

from copul.family.bb._frailty import log_gamma_rv, sibuya_rv
from copul.family.bb.lt_archimedean import LTArchimedeanCopula, log_expm1, softplus


def _log_m(t, th):
    return np.log(-np.expm1(th * np.log1p(-t)))


class BB7(LTArchimedeanCopula):
    r"""BB7 (Joe–Clayton) copula (Joe & Hu 1996).

    .. math::

       C(u,v) = 1 - \Bigl(1 - \bigl[(1-\bar u^{\theta})^{-\delta}
                + (1-\bar v^{\theta})^{-\delta} - 1\bigr]^{-1/\delta}\Bigr)^{1/\theta},

    :math:`\bar u = 1-u`, :math:`\theta\ge 1`, :math:`\delta>0`.

    :math:`\psi(s) = 1 - [1 - (1+s)^{-1/\delta}]^{1/\theta}`,
    :math:`\varphi(t) = (1-(1-t)^{\theta})^{-\delta} - 1`.  Frailty:
    :math:`V\mid N\sim\mathrm{Gamma}(N/\delta)` with
    :math:`N\sim\mathrm{Sibuya}(1/\theta)`.

    Closed forms
    ------------
    * :math:`\lambda_L = 2^{-1/\delta}`, :math:`\lambda_U = 2 - 2^{1/\theta}`
      (the tail coefficients can be chosen independently);
    * for :math:`1\le\theta<2`,
      :math:`\tau = 1 - \dfrac{2}{\delta(2-\theta)}
      + \dfrac{4}{\theta^2\delta}\,B\bigl(\delta+2, \tfrac{2}{\theta}-1\bigr)`.

    Special cases: :math:`\theta = 1` is Clayton(:math:`\delta`);
    :math:`\delta\to 0^+` gives Joe(:math:`\theta`).

    References
    ----------
    Joe, H. (1997). *Multivariate Models and Dependence Concepts*, Chapman &
    Hall, family BB7 (p. 153).

    Joe, H. (2014). *Dependence Modeling with Copulas*, CRC Press, Sec. 4.17.5.
    """

    theta, delta = sp.symbols("theta delta", positive=True)
    params = [theta, delta]
    intervals = {
        "theta": sp.Interval(1, sp.oo, left_open=False, right_open=True),
        "delta": sp.Interval(0, sp.oo, left_open=True, right_open=True),
    }

    def _phi_sym(self, t, th, de):
        return (1 - (1 - t) ** th) ** (-de) - 1

    def _psi_sym(self, s, th, de):
        return 1 - (1 - (1 + s) ** (-1 / de)) ** (1 / th)

    def _log_phi(self, t, th, de):
        return log_expm1(-de * _log_m(t, th))

    def _log_mdphi(self, t, th, de):
        return np.log(de * th) - (de + 1.0) * _log_m(t, th) + xlogy(th - 1.0, 1.0 - t)

    def _psi_ls(self, ls, th, de):
        L = softplus(ls)
        return -np.expm1(np.log(-np.expm1(-L / de)) / th)

    def _log_mdpsi_ls(self, ls, th, de):
        L = softplus(ls)
        return -np.log(th * de) + (1.0 / th - 1.0) * np.log(-np.expm1(-L / de)) - L / de - L

    def _log_d2psi_ls(self, ls, th, de):
        L = softplus(ls)
        bracket = (1.0 - 1.0 / th) / (de * np.expm1(L / de)) + 1.0 / de + 1.0
        return (
            -np.log(th * de)
            + (1.0 / th - 1.0) * np.log(-np.expm1(-L / de))
            - L / de
            - 2.0 * L
            + np.log(bracket)
        )

    def _log_frailty(self, n, rng, th, de):
        return log_gamma_rv(sibuya_rv(1.0 / th, n, rng) / de, rng)

    def kendalls_tau(self, *args, **kwargs):
        r""":math:`1 - \frac{2}{\delta(2-\theta)} + \frac{4}{\theta^2\delta}B(\delta+2, \frac2\theta-1)`, :math:`1\le\theta<2`."""
        th, de = self.theta, self.delta
        if isinstance(th, sp.Basic) and not th.is_number:
            return 1 - 2 / (de * (2 - th)) + 4 / (th**2 * de) * sp.beta(de + 2, 2 / th - 1)
        th, de = float(th), float(de)
        if not th < 2.0:
            raise NotImplementedError("closed form of tau only for 1 <= theta < 2")
        from scipy.special import beta

        return 1.0 - 2.0 / (de * (2.0 - th)) + 4.0 / (th**2 * de) * beta(de + 2.0, 2.0 / th - 1.0)

    def lambda_L(self, *args, **kwargs):
        r""":math:`\lambda_L = 2^{-1/\delta}`."""
        return 2 ** (-1 / self.delta)

    def lambda_U(self, *args, **kwargs):
        r""":math:`\lambda_U = 2 - 2^{1/\theta}`."""
        return 2 - 2 ** (1 / self.theta)
