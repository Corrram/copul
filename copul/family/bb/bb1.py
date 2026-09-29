r"""BB1 (Clayton–Gumbel) copula, Joe (2014) Sec. 4.17.1."""

from __future__ import annotations

import numpy as np
import sympy as sp

from copul.family.bb._frailty import log_gamma_rv, log_positive_stable_rv
from copul.family.bb.lt_archimedean import LTArchimedeanCopula, log_expm1, softplus


class BB1(LTArchimedeanCopula):
    r"""BB1 copula (Joe & Hu 1996), a Clayton–Gumbel two-parameter family.

    .. math::

       C(u,v) = \Bigl\{1 + \bigl[(u^{-\theta}-1)^{\delta}
               + (v^{-\theta}-1)^{\delta}\bigr]^{1/\delta}\Bigr\}^{-1/\theta},
       \qquad \theta>0,\ \delta\ge 1 .

    Generator inverse (Laplace transform)
    :math:`\psi(s) = (1+s^{1/\delta})^{-1/\theta}`, generator
    :math:`\varphi(t) = (t^{-\theta}-1)^{\delta}`.  The frailty is
    :math:`V = G^{\delta} S` with :math:`G\sim\mathrm{Gamma}(1/\theta)` and
    :math:`S` positive stable with Laplace transform :math:`e^{-s^{1/\delta}}`.

    Closed forms
    ------------
    * Kendall's :math:`\tau = 1 - \dfrac{2}{\delta(\theta+2)}`,
    * :math:`\lambda_L = 2^{-1/(\theta\delta)}`,
      :math:`\lambda_U = 2 - 2^{1/\delta}`.

    Special cases: :math:`\delta = 1` is the Clayton copula with parameter
    :math:`\theta`; :math:`\theta\to 0^+` gives the Gumbel–Hougaard copula
    with parameter :math:`\delta`.

    References
    ----------
    Joe, H. & Hu, T. (1996). Multivariate distributions from mixtures of
    max-infinitely divisible distributions. *J. Multivariate Anal.* 57, 240–265.

    Joe, H. (2014). *Dependence Modeling with Copulas*, CRC Press, Sec. 4.17.1.
    """

    theta, delta = sp.symbols("theta delta", positive=True)
    params = [theta, delta]
    intervals = {
        "theta": sp.Interval(0, sp.oo, left_open=True, right_open=True),
        "delta": sp.Interval(1, sp.oo, left_open=False, right_open=True),
    }

    # -- symbolic ------------------------------------------------------------------
    def _phi_sym(self, t, th, de):
        return (t ** (-th) - 1) ** de

    def _psi_sym(self, s, th, de):
        return (1 + s ** (1 / de)) ** (-1 / th)

    # -- numeric -------------------------------------------------------------------
    def _log_phi(self, t, th, de):
        return de * log_expm1(-th * np.log(t))

    def _log_mdphi(self, t, th, de):
        lt = np.log(t)
        return np.log(th * de) + (de - 1.0) * log_expm1(-th * lt) - (th + 1.0) * lt

    def _psi_ls(self, ls, th, de):
        return np.exp(-softplus(ls / de) / th)

    def _log_mdpsi_ls(self, ls, th, de):
        return -np.log(th * de) - (1.0 / th + 1.0) * softplus(ls / de) + (1.0 / de - 1.0) * ls

    def _log_d2psi_ls(self, ls, th, de):
        a = (1.0 + th) / (th * de)
        b = 1.0 - 1.0 / de
        with np.errstate(divide="ignore"):
            bracket = np.logaddexp(np.log(a + b) + ls / de, np.log(b))
        return (
            -np.log(th * de)
            - (1.0 / th + 2.0) * softplus(ls / de)
            + (1.0 / de - 2.0) * ls
            + bracket
        )

    def _log_frailty(self, n, rng, th, de):
        return de * log_gamma_rv(np.full(n, 1.0 / th), rng) + log_positive_stable_rv(
            1.0 / de, n, rng
        )

    # -- closed forms --------------------------------------------------------------
    def kendalls_tau(self, *args, **kwargs):
        r""":math:`\tau = 1 - 2/(\delta(\theta+2))` (Joe 2014, Sec. 4.17.1)."""
        return 1 - 2 / (self.delta * (self.theta + 2))

    def lambda_L(self, *args, **kwargs):
        r""":math:`\lambda_L = 2^{-1/(\theta\delta)}`."""
        return 2 ** (-1 / (self.theta * self.delta))

    def lambda_U(self, *args, **kwargs):
        r""":math:`\lambda_U = 2 - 2^{1/\delta}`."""
        return 2 - 2 ** (1 / self.delta)
