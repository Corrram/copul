import sympy as sp
import numpy as np

from copul.exceptions import PropertyUnavailableException
from copul.family.core.biv_copula import BivCopula
from copul.family.other.upper_frechet import UpperFrechet
from copul.family.other.lower_frechet import LowerFrechet
from copul.wrapper.cd2_wrapper import CD2Wrapper
from copul.wrapper.cdf_wrapper import CDFWrapper
from scipy.optimize import brentq


class VThresholdCopula(BivCopula):
    r"""
    V–threshold copula family tracing the **upper boundary** of the
    :math:`(\rho,\nu)` region.

    Definition (conditional distribution in :math:`u`):
    .. math::

        h_\mu(t,v) \;=\; \mathbf{1}\{|t - t_\star|\ge (1-v)/2\},\qquad
        t_\star \;=\; 1 - \mu/2,\;\; \mu\in[0,2].

    Closed-form rank measures:
    - For :math:`0\le \mu\le 1`:
      :math:`\rho(\mu)=1-\mu^3,\;\; \nu(\mu)=1-\tfrac34 \mu^4`.
    - For :math:`1\le \mu\le 2`:
      :math:`\rho(\mu)=-\mu^3+6\mu^2-12\mu+7, \;\;
      \nu(\mu)=-\tfrac34 \mu^4+4\mu^3-6\mu^2+3`.
    """

    # SymPy symbols & meta
    mu = sp.symbols("mu", real=True)
    params = [mu]
    intervals = {"mu": sp.Interval(0, 2, left_open=False, right_open=False)}
    special_cases = {
        0: UpperFrechet,  # comonotone
        2: LowerFrechet,  # countermonotone
    }

    u, v = sp.symbols("u v", real=True)

    def __init__(self, *args, **kwargs):
        if args and len(args) == 1:
            kwargs["mu"] = args[0]
        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # Rank measures (closed form)
    # ------------------------------------------------------------------

    def spearmans_rho(self):
        mu = float(self.mu)
        if mu <= 1.0:
            return 1.0 - mu**3
        else:
            return -(mu**3) + 6 * mu**2 - 12 * mu + 7

    def blests_nu(self):
        mu = float(self.mu)
        if mu <= 1.0:
            return 1.0 - 0.75 * mu**4
        else:
            return -0.75 * mu**4 + 4.0 * mu**3 - 6.0 * mu**2 + 3.0

    @classmethod
    def from_rho(cls, rho_target: float):
        """
        Construct boundary copula with target Spearman's :math:`\\rho\\in[-1,1]`.

        Inversion:
        - If :math:`\\rho\\in[0,1]` then :math:`\\mu=(1-\\rho)^{1/3}`.
        - If :math:`\\rho\\in[-1,0]` solve the monotone cubic
          :math:`\\rho(\\mu)=-\\mu^3+6\\mu^2-12\\mu+7` on :math:`[1,2]`.
        """
        if not (-1.0 <= rho_target <= 1.0):
            raise ValueError("rho_target must be in [-1, 1].")

        if rho_target >= 0:
            mu = (1.0 - rho_target) ** (1.0 / 3.0)
            return cls(mu=float(mu))

        # rho_target in [-1,0]: solve rho(mu) = -mu^3 + 6 mu^2 - 12 mu + 7
        def rho_branch(mu):
            return -(mu**3) + 6 * mu**2 - 12 * mu + 7

        mu = brentq(
            lambda m: rho_branch(m) - rho_target, 1.0, 2.0, xtol=1e-14, rtol=1e-12
        )
        return cls(mu=float(mu))

    # ------------------------------------------------------------------
    # Symbolic CDF & conditional distributions (like in Frechet exemplar)
    # ------------------------------------------------------------------

    # --- inside class VThresholdCopula ---

    @property
    def cdf(self):
        u, v, mu = self.u, self.v, self.mu
        ts = 1 - mu / 2

        # regime thresholds
        left_reg = sp.And(sp.Le(mu, 1), sp.Le(v, 1 - mu))  # μ ≤ 1 and v ≤ 1−μ
        right_reg = sp.And(sp.Ge(mu, 1), sp.Le(v, mu - 1))  # μ ≥ 1 and v ≤ μ−1

        # one-tail expressions
        expr_left = sp.Min(u, v)  # a(v)=v, s(v)=1
        expr_right = sp.Max(u - (1 - v), 0)  # a(v)=0, s(v)=1−v

        # two-tail expressions
        a_in = ts - (1 - v) / 2
        s_in = ts + (1 - v) / 2
        expr_interior = sp.Min(u, a_in) + sp.Max(u - s_in, 0)

        return CDFWrapper(
            sp.Piecewise(
                (expr_left, left_reg), (expr_right, right_reg), (expr_interior, True)
            )
        )

    def cond_distr_1(self, u=None, v=None):
        # h(u,v) = 1{u ≤ a(v)} + 1{u ≥ s(v)} with regimes
        ts = 1 - self.mu / 2
        left_reg = sp.And(sp.Le(self.mu, 1), sp.Le(self.v, 1 - self.mu))
        right_reg = sp.And(sp.Ge(self.mu, 1), sp.Le(self.v, self.mu - 1))

        # one-tail regimes
        h_left = sp.Heaviside(self.v - self.u, 0)  # u ≤ v
        h_right = sp.Heaviside(self.u - (1 - self.v), 0)  # u ≥ 1−v

        # two-tail regime
        a_in = ts - (1 - self.v) / 2
        s_in = ts + (1 - self.v) / 2
        h_in = sp.Heaviside(a_in - self.u, 0) + sp.Heaviside(self.u - s_in, 0)

        expr = sp.Piecewise((h_left, left_reg), (h_right, right_reg), (h_in, True))
        return CD2Wrapper(expr)(u, v)

    def cond_distr_2(self, u=None, v=None):
        ts = 1 - self.mu / 2
        left_reg = sp.And(sp.Le(self.mu, 1), sp.Le(self.v, 1 - self.mu))
        right_reg = sp.And(sp.Ge(self.mu, 1), sp.Le(self.v, self.mu - 1))

        dC_left = sp.Heaviside(self.u - self.v, 0)
        dC_right = sp.Heaviside(self.u - (1 - self.v), 0)

        a_in = ts - (1 - self.v) / 2
        s_in = ts + (1 - self.v) / 2
        dC_in = sp.Rational(1, 2) * (
            sp.Heaviside(self.u - a_in, 0) + sp.Heaviside(self.u - s_in, 0)
        )

        expr = sp.Piecewise((dC_left, left_reg), (dC_right, right_reg), (dC_in, True))
        return CD2Wrapper(expr)(u, v)

    # ------------------------------------------------------------------
    # Vectorized numerics (same as before)
    # ------------------------------------------------------------------

    def cdf_vectorized(self, u, v):
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        mu = float(self.mu)
        t_star = 1.0 - mu / 2.0

        u_flat = u.ravel()
        v_flat = v.ravel()
        out = np.empty_like(
            np.broadcast_to(u_flat, np.broadcast(u_flat, v_flat).shape), dtype=float
        )

        def C_scalar(ui, vi):
            if mu <= 1.0:
                if vi <= 1.0 - mu:  # left-only
                    a, s = vi, 1.0
                else:
                    a = t_star - (1.0 - vi) / 2.0
                    s = t_star + (1.0 - vi) / 2.0
            else:
                if vi <= mu - 1.0:  # right-only
                    a, s = 0.0, 1.0 - vi
                else:
                    a = t_star - (1.0 - vi) / 2.0
                    s = t_star + (1.0 - vi) / 2.0
            return min(ui, a) + max(ui - s, 0.0)

        # vectorize via np.frompyfunc or loop (simple loop is fine here)
        out = np.array(
            [C_scalar(ui, vi) for ui, vi in np.broadcast(u_flat, v_flat)], dtype=float
        )
        return out.reshape(np.broadcast(u, v).shape)

    def pdf_vectorized(self, u, v):
        """
        Vectorized density c(u,v) = ∂^2 C/∂u∂v = ∂ h(u,v)/∂v
        computed by a finite-difference in v.
        """
        u, v = np.atleast_1d(u), np.atleast_1d(v)
        pdf_vals = np.zeros_like(u, dtype=float)
        eps = 1e-7
        for i in np.ndindex(u.shape):
            u_i, v_i = u[i], v[i]
            C_plus = self.cdf_vectorized(u_i, min(1.0, v_i + eps))
            C = self.cdf_vectorized(u_i, v_i)
            pdf_vals[i] = (C_plus - C) / eps
        return pdf_vals.reshape(np.asarray(u).shape)

    # (Optional) expose a “no symbolic PDF” like in Frechet
    @property
    def pdf(self):
        raise PropertyUnavailableException(
            "This copula has singular components; no purely absolutely-continuous PDF."
        )


if __name__ == "__main__":
    # Quick demo
    for mu in [0.5, 1.0, 1.5]:
        cop = VThresholdCopula(mu=mu)
        print(f"mu={mu:.2f} -> rho={cop.spearmans_rho():.6f}, nu={cop.blests_nu():.6f}")
        # cop.plot_cdf()
        # cop.plot_pdf()
        cop.plot_cond_distr_1(plot_type="contour", grid_size=1000, cmap="viridis")
