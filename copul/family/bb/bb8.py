r"""BB8 (Joe–Frank) copula, Joe (2014) Sec. 4.17.6."""

from __future__ import annotations

import numpy as np
import sympy as sp
from scipy.special import xlogy

from copul.family.bb._frailty import sibuya_rv
from copul.family.bb.lt_archimedean import LTArchimedeanCopula


class BB8(LTArchimedeanCopula):
    r"""BB8 (Joe–Frank) copula.

    .. math::

       C(u,v) = \delta^{-1}\Bigl(1 - \bigl\{1 - \eta^{-1}[1-(1-\delta u)^{\theta}]
                [1-(1-\delta v)^{\theta}]\bigr\}^{1/\theta}\Bigr),
       \qquad \eta = 1-(1-\delta)^{\theta},

    :math:`\theta\ge 1`, :math:`0<\delta\le 1`.

    :math:`\psi(s) = \delta^{-1}[1 - (1-\eta e^{-s})^{1/\theta}]`,
    :math:`\varphi(t) = -\log\bigl([1-(1-\delta t)^{\theta}]/\eta\bigr)`.
    The frailty is discrete, :math:`P(V=k)\propto p_k\eta^k` with the
    Sibuya(:math:`1/\theta`) probabilities :math:`p_k` (sampled by rejection
    from the Sibuya law).

    Closed forms: :math:`\lambda_L = 0`; :math:`\lambda_U = 0` for
    :math:`\delta<1` and :math:`2-2^{1/\theta}` for :math:`\delta = 1`.
    Special cases: :math:`\delta = 1` is Joe(:math:`\theta`),
    :math:`\theta = 1` the independence copula.

    References
    ----------
    Joe, H. (2014). *Dependence Modeling with Copulas*, CRC Press, Sec. 4.17.6.
    """

    theta, delta = sp.symbols("theta delta", positive=True)
    params = [theta, delta]
    intervals = {
        "theta": sp.Interval(1, sp.oo, left_open=False, right_open=True),
        "delta": sp.Interval(0, 1, left_open=True, right_open=False),
    }

    @staticmethod
    def _eta(th, de):
        return -np.expm1(th * np.log1p(-de)) if de < 1 else 1.0

    @staticmethod
    def _log1m(s, th, de):
        r""":math:`\log(1-\eta e^{-s}) = \log((1-\delta)^\theta - \eta\,\mathrm{expm1}(-s))`."""
        eta = BB8._eta(th, de)
        return np.log((1.0 - de) ** th - eta * np.expm1(-s))

    def _phi_sym(self, t, th, de):
        eta = 1 - (1 - de) ** th
        return -sp.log((1 - (1 - de * t) ** th) / eta)

    def _psi_sym(self, s, th, de):
        eta = 1 - (1 - de) ** th
        return (1 - (1 - eta * sp.exp(-s)) ** (1 / th)) / de

    def _log_phi(self, t, th, de):
        eta = self._eta(th, de)
        # (1-delta)^theta - (1-delta t)^theta, accurate near t = 1
        with np.errstate(divide="ignore"):
            r = np.log1p(-de * (1.0 - t) / (1.0 - de * t)) if de < 1 else np.log(1.0 - t)
        if de < 1:
            d = np.exp(th * np.log1p(-de * t)) * np.expm1(th * r)
        else:
            d = -np.exp(th * r)
        return np.log(-np.log1p(d / eta))

    def _log_mdphi(self, t, th, de):
        lm = np.log(-np.expm1(th * np.log1p(-de * t)))
        return np.log(th * de) + xlogy(th - 1.0, 1.0 - de * t) - lm

    def _psi_ls(self, ls, th, de):
        return -np.expm1(self._log1m(np.exp(ls), th, de) / th) / de

    def _log_mdpsi_ls(self, ls, th, de):
        eta = self._eta(th, de)
        s = np.exp(ls)
        return np.log(eta) - np.log(de * th) + (1.0 / th - 1.0) * self._log1m(s, th, de) - s

    def _log_d2psi_ls(self, ls, th, de):
        eta = self._eta(th, de)
        s = np.exp(ls)
        x = eta * np.exp(-s)
        return (
            np.log(eta)
            - np.log(de * th)
            + (1.0 / th - 2.0) * self._log1m(s, th, de)
            - s
            + np.log1p(-x / th)
        )

    def _log_frailty(self, n, rng, th, de):
        eta = self._eta(th, de)
        out = np.empty(n)
        todo = np.arange(n)
        while todo.size:
            k = sibuya_rv(1.0 / th, todo.size, rng)
            acc = np.log(rng.random(todo.size)) <= (k - 1.0) * np.log(eta)
            out[todo[acc]] = k[acc]
            todo = todo[~acc]
        return np.log(out)

    def lambda_L(self, *args, **kwargs):
        r""":math:`\lambda_L = 0`."""
        return 0

    def lambda_U(self, *args, **kwargs):
        r""":math:`\lambda_U = 0` (:math:`\delta<1`), :math:`2-2^{1/\theta}` (:math:`\delta=1`)."""
        th, de = self.theta, self.delta
        if isinstance(de, sp.Basic) and not de.is_number:
            return sp.Piecewise((2 - 2 ** (1 / th), sp.Eq(de, 1)), (0, True))
        return 2 - 2 ** (1 / th) if float(de) == 1.0 else 0
