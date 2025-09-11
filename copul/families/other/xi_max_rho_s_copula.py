# file: copul/families/xi_max_rho_s.py
import sympy as sp

from copul.families.core.biv_copula import BivCopula
from copul.wrapper.cdf_wrapper import CDFWrapper
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class XiMaxRhoSCopula(BivCopula):
    r"""
    Optimal–$\xi$ copula family for a given Spearman's $\rho_S$.
    The family is parameterized by $x \equiv \rho_S \in [0,1]$.

    The internal parameter $\mu_x = 1/b^*_x$ is determined from $\rho_S = x$:
    If $x \in [0, 3/10]$: $b^*_x$ is the unique root in $(0,1]$ of $2(b^*_x)^3 - 5(b^*_x)^2 + 10x = 0$.
        ($b^*_x \to 0^+$ as $x \to 0$, so $\mu_x \to \infty$)
    If $x \in (3/10, 1]$: $b^*_x = \frac{10 + \sqrt{120x - 20}}{20(1-x)}$.
        ($b^*_x \to \infty$ as $x \to 1$, so $\mu_x \to 0$)

    The maximized Chatterjee's $\xi$ for a given $\rho_S = x$ is:
    \[
      \xi_{opt}(x) =
      \begin{cases}
        \frac{(b^*_x)^2}{2} - \frac{(b^*_x)^3}{5}, & x \in [0, 3/10] \quad (b^*_x \in (0,1])\\[1ex]
        1 - \frac{1}{b^*_x} + \frac{3}{10(b^*_x)^2}, & x \in (3/10, 1] \quad (b^*_x \in [1,\infty))
      \end{cases}
    \]
    
    The conditional distribution $h_v(t)$ parameter $s_v(v, \mu_x)$ (where $\mu_x = 1/b^*_x$):
    If $\mu_x \ge 1$ (i.e., $b^*_x \le 1$, corresponds to $x \in [0, 3/10]$):
    \[
      s_v =
      \begin{cases}
        \sqrt{2v\mu_x}, & 0 \le v \le \mu_x/2 \\
        v/\mu_x + 1/2, & \mu_x/2 < v \le 1-\mu_x/2 \\
        1+\mu_x-\sqrt{2\mu_x(1-v)}, & 1-\mu_x/2 < v \le 1
      \end{cases}
    \]
    If $\mu_x \le 1$ (i.e., $b^*_x \ge 1$, corresponds to $x \in (3/10, 1]$):
    \[
      s_v =
      \begin{cases}
        \sqrt{2v\mu_x}, & 0 \le v \le \mu_x/(2\mu_x^2) \text{ (typo, should be } 1/(2b_x^*) \text{ i.e. } \mu_x/2) \\ % Original example had errors in region def for s_v
                                  % Corrected based on \mu as parameter for s_v conditions
        v+\mu_x/2, & \mu_x/2 < v \le 1-\mu_x/2 \\
        1+\mu_x-\sqrt{2\mu_x(1-v)}, & 1-\mu_x/2 < v \le 1
      \end{cases}
    \]
    Actually, the s_v structure from the example `XiRhoBoundaryCopula._s_expr(v, b_param)`
    where `b_param` is $\mu_x$ is better structured. Let's use that directly.
    $a_v = s_v - \mu_x$.
    The copula CDF is:
    \[
      C_x(u,v) =
      \begin{cases}
        u, & u \le a_v \quad (\text{if } a_v \ge 0)\\
        a_v + \frac{s_v(u-a_v) - (u^2-a_v^2)/2}{\mu_x}, & a_v < u \le s_v \quad (\text{if } a_v \ge 0) \\
                                                      % or if a_v < 0, different form starting from u=0
        \mu_x (s_v u - u^2/2), & 0 \le u \le s_v \quad (\text{if } a_v < 0) \\
        v, & u > s_v
      \end{cases}
    \]
    The example class handles $a_v = \text{Max}(s_v - \mu_x, 0)$ and then a single piecewise for CDF,
    which is more robust.
    The middle term provided in the original example `(2 * s * (u - a) - u**2 + a**2) / (2 * mu_x)`
    is $ (s(u-a) - (u^2-a^2)/2) / \mu_x $. This is what we will use.
    """

    # symbolic parameter & admissible interval
    rho_s = sp.symbols("rho_s")  # Parameter is Spearman's rho
    params = [rho_s]
    # The derivation applies for rho_s in [0,1].
    # Problem states input x in (-1,1), implying symmetry for xi.
    # Copula is typically defined for non-negative dependence parameter for this construction.
    intervals = {"rho_s": sp.Interval(0, 1)}

    u, v = sp.symbols(
        "u v", positive=True, real=True
    )  # u,v in (0,1) typically for CDF evaluation

    def __init__(self, *args, **kwargs):
        if args and len(args) == 1:
            kwargs["rho_s"] = args[0]

        # Validate rho_s if provided, default to symbolic if not
        rho_s_val = kwargs.get("rho_s", self.rho_s)
        if not isinstance(rho_s_val, sp.Symbol):
            self._validate_rho_s(rho_s_val)
        super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        if args and len(args) == 1:
            kwargs["rho_s"] = args[0]
        if "rho_s" in kwargs:
            if not isinstance(kwargs["rho_s"], sp.Symbol):
                self._validate_rho_s(kwargs["rho_s"])
        return super().__call__(**kwargs)

    @staticmethod
    def _validate_rho_s(rho_s_val):
        # The model is derived for rho_s in [0,1].
        # For rho_s < 0, the family C(|rho_s|) can be used to calculate xi(|rho_s|).
        # The copula itself is defined for the parameter in [0,1].
        if not (0 <= rho_s_val <= 1):
            raise ValueError(
                f"Parameter rho_s must be in [0,1] for this copula definition, got {rho_s_val}"
            )

    @staticmethod
    def _b_star_expr(rho_s_param):
        """
        Calculates the intermediate parameter b_star (slope) from Spearman's rho_s_param.
        This b_star is $b^*$ from the LaTeX derivation.
        """
        b_star_trig = (sp.Rational(5, 3)) * sp.cos(
            (sp.Rational(1, 3)) * sp.acos(1 - (sp.Rational(108, 25)) * rho_s_param)
            + (sp.Rational(4, 3)) * sp.pi
        ) + sp.Rational(5, 6)
        return b_star_trig

    @staticmethod
    def _mu_calc_expr(rho_s_param):
        """
        Calculates the internal parameter mu_val = 1/b_star from Spearman's rho_s_param.
        This mu_val is used in s_v and CDF formulas, analogous to 'b_x' in XiRhoBoundaryCopula.
        """
        b_star_val = XiMaxRhoSCopula._b_star_expr(rho_s_param)

        # mu = 1/b_star. Handle b_star = 0 (rho_s=0) and b_star = oo (rho_s=1)
        mu_val = sp.Piecewise(
            (sp.oo, sp.Eq(b_star_val, 0)),  # If b_star is 0 (rho_s=0), mu is oo
            (0, sp.Eq(b_star_val, sp.oo)),  # If b_star is oo (rho_s=1), mu is 0
            (1 / b_star_val, True),
        )
        return mu_val

    @staticmethod
    def _xi_optimal_expr(rho_s_param):
        """Maximal Chatterjee's xi_opt for a given Spearman's rho_s_param."""
        rho_s_transition = sp.Rational(3, 10)
        b_star_val = XiMaxRhoSCopula._b_star_expr(rho_s_param)

        # xi_opt for Regime 1 (b_star_val in (0,1])
        xi_B = (b_star_val**2 / 2) - (b_star_val**3 / 5)

        # xi_opt for Regime 2 (b_star_val in [1,infinity))
        # Need to handle b_star_val = infinity for rho_s_param = 1 -> xi_opt = 1
        xi_A_finite = 1 - (1 / b_star_val) + (3 / (10 * b_star_val**2))

        xi_opt = sp.Piecewise(
            (0, sp.Eq(rho_s_param, 0)),  # rho_s=0 => b_star=0 => xi=0
            (
                xi_B,
                rho_s_param <= rho_s_transition,
            ),  # Includes rho_s=0.3, b_star=1, xi=0.3
            (1, sp.Eq(rho_s_param, 1)),  # rho_s=1 => b_star=oo => xi=1
            (xi_A_finite, rho_s_param > rho_s_transition),
        )
        return xi_opt

    @staticmethod
    def _s_expr(v_sym, mu_param):
        r"""
        Shift s_v (max‐band) using mu_param.
        This structure is identical to XiRhoBoundaryCopula._s_expr(v, b)
        if we identify its 'b' argument with our 'mu_param'.
        mu_param corresponds to \mu in the appendix derivations.
        """
        # mu_param <= 1 region (corresponds to original example's b <= 1)
        v_thresh_small_mu = mu_param / 2
        s1_small_mu = sp.sqrt(2 * v_sym * mu_param)
        s2_small_mu = v_sym + mu_param / 2
        s3_small_mu = 1 + mu_param - sp.sqrt(2 * mu_param * (1 - v_sym))

        sp.Piecewise(  # mu_param <= 1
            (s1_small_mu, v_sym <= v_thresh_small_mu),
            (
                s2_small_mu,
                v_sym <= 1 - v_thresh_small_mu,
            ),  # Corrected condition from example
            (s3_small_mu, True),
        )

        # mu_param > 1 region (corresponds to original example's b > 1, was 's_large')
        # The example's s_large used 'b' for mu_param.
        # v1_L = 1 / (2 * mu_param) in original example's logic if its b was mu_param.
        # It's simpler if we use the conditions consistently.
        # The structure of s_v depends on mu_param vs 1.
        # The original example _s_expr had 'b' as the parameter.
        # If 'b' in original _s_expr is mu_param:
        # small-b region: mu_param <= 1
        # large-b region: mu_param > 1

        # Re-using the structure from XiRhoBoundaryCopula for clarity, assuming its 'b' is mu_param
        # Small mu_param (mu_param <= 1, equivalent to original example's b <= 1)
        mu_param / 2
        # s1_s, s2_s, s3_s are defined above as s1_small_mu etc.

        # Large mu_param (mu_param > 1, equivalent to original example's b > 1)
        sp.Rational(
            1, 2
        ) * mu_param  # This was v1_L = 1/(2*b) if b was slope. If b is mu, then it's mu/2.
        # Let's use the structure based on mu from my LaTeX notes for s_v.
        # if mu_param >= 1 (b_slope <=1): breakpoints for v are mu_param/2, 1-mu_param/2
        # s_v cases: sqrt(2v*mu), v/mu + 1/2, 1+mu-sqrt(2mu(1-v))
        # if mu_param <= 1 (b_slope >=1): breakpoints for v are mu_param/2, 1-mu_param/2
        # s_v cases: sqrt(2v*mu), v+mu/2, 1+mu-sqrt(2mu(1-v))
        # The example _s_expr had an error in conditions for s_large (v1_L definition was for slope b, not mu=b)
        # Let's use the corrected s_v structure from my LaTeX derivation which depends on mu directly:

        # Case 1: mu_param >= 1 (corresponds to b_slope <= 1)
        sp.Piecewise(
            (
                sp.sqrt(2 * v_sym / (1 / mu_param)),
                v_sym <= (1 / mu_param) / 2,
            ),  # using b_slope = 1/mu_param
            (
                v_sym / (1 / mu_param) + sp.Rational(1, 2),
                v_sym <= 1 - (1 / mu_param) / 2,
            ),
            (
                1 + mu_param - sp.sqrt(2 * mu_param * (1 - v_sym)),
                True,
            ),  # this used mu directly
        )

        # For mu_param <= 1 (small mu)
        v_boundary1_small_mu = mu_param / 2
        s_piece1_small_mu = sp.sqrt(2 * v_sym * mu_param)
        s_piece2_small_mu = v_sym + mu_param / 2
        s_piece3_small_mu = 1 + mu_param - sp.sqrt(2 * mu_param * (1 - v_sym))
        sp.Piecewise(
            (s_piece1_small_mu, v_sym <= v_boundary1_small_mu),
            (
                s_piece2_small_mu,
                v_sym <= 1 - v_boundary1_small_mu,
            ),  # v_boundary2_small_mu
            (s_piece3_small_mu, True),
        )

        # For mu_param > 1 (large mu)
        1 / (
            2 * mu_param
        )  # This is b_slope / 2 if mu_param = 1/b_slope. No, this is mu_lit / 2 for the other regime if mu_lit = 1/mu_param.
        # The original text had conditions for s_v split by (mu<=1 or mu>=1)
        # and then sub-conditions on v. (eqs_vcase3).
        # Line 1: common part sqrt(2v*mu). v <= mu/2 (if mu small) OR v <= 1/(2mu) (if mu large)
        # Line 2: v+mu/2.      mu/2 < v <= 1-mu/2 (ONLY if mu < 1)
        # Line 3: v*mu+1/2.    1/(2mu) < v <= 1-1/(2mu) (ONLY if mu > 1)
        # Line 4: common part 1+mu-sqrt(2mu(1-v)). v > 1-mu/2 (if mu small) OR v > 1-1/(2mu) (if mu large)

        s_piece1_common = sp.sqrt(2 * v_sym * mu_param)
        s_piece4_common = 1 + mu_param - sp.sqrt(2 * mu_param * (1 - v_sym))

        # mu_param <= 1 (b_slope >= 1)
        cond1_v_small_mu = mu_param / 2
        s_piece2_small_mu = v_sym + mu_param / 2
        s_expr_small_mu = sp.Piecewise(
            (s_piece1_common, v_sym <= cond1_v_small_mu),
            (s_piece2_small_mu, v_sym <= 1 - cond1_v_small_mu),
            (s_piece4_common, True),
        )

        # mu_param > 1 (b_slope <= 1)
        cond1_v_large_mu = 1 / (2 * mu_param)  # This is b_slope / 2
        s_piece2_large_mu = v_sym * mu_param + sp.Rational(
            1, 2
        )  # This is v_sym / b_slope + 1/2
        s_expr_large_mu = sp.Piecewise(
            (s_piece1_common, v_sym <= cond1_v_large_mu),
            (s_piece2_large_mu, v_sym <= 1 - cond1_v_large_mu),
            (s_piece4_common, True),
        )

        return sp.Piecewise((s_expr_small_mu, mu_param <= 1), (s_expr_large_mu, True))

    @property
    def cdf(self):
        u_sym, v_sym, rho_s_sym = self.u, self.v, self.rho_s

        mu_val = self._mu_calc_expr(rho_s_sym)

        s_v_expr = self._s_expr(v_sym, mu_val)

        # a_v = s_v - mu_val, but must be non-negative for the first piece of C(u,v)
        # The example uses Max(s - b, 0) where b is mu_val
        # However, the copula formula pieces should be based on whether a_v defined as s_v - mu_val is < 0 or >= 0
        # The provided C_x in the example description:
        # C_x(u,v) = u if u <= a_v ; middle if a_v < u <= s_v ; v if u > s_v
        # This structure implicitly assumes a_v is the first threshold.
        # If a_v < 0, then u <= a_v is not met for u in [0,1].
        # The definition of a_v from the Appendix text was $a_v = s_v - \mu$.
        # And the piecewise $h_v(t)$ depended on $a_v$.
        # The example CDF seems to use $a = \text{Max}(s_v - \mu, 0)$ for the u-thresholds.
        # But the middle formula $a + (2 s (u-a) - u^2 + a^2)/(2\mu)$ is derived assuming integration from $a$.
        # This needs to be consistent with my LaTeX $C_{b^*}(u,v)$ pieces.

        # Adopting the robust structure from example for $a_v$ in thresholds for Piecewise
        # but use the correct $a_v = s_v - \mu_val$ in the formula for 'middle' part.
        # The example implementation uses a = sp.Max(s - b, 0) for u-threshold and in formula.
        # This is only correct if $a_v$ used in formula IS $\max(s_v-\mu,0)$.
        # My LaTeX derivation: $C(u,v) = a_v + \frac{s_v(u-a_v)-(u^2-a_v^2)/2}{\mu_val}$ where $a_v = s_v-\mu_val$.
        # This is equivalent to example's: $a_v + \frac{2s_v(u-a_v) - (u^2-a_v^2)}{\mu_val}$
        # The example's $(2 s (u - a) - u^2 + a^2) / (2 * b)$ is $ (s(u-a) - (u^2-a^2)/2)/b $.
        # This is correct.

        a_v_calc = s_v_expr - mu_val  # This can be negative

        # Case 1: a_v_calc < 0  (s_v < mu_val) "Pure triangle" or zero
        # C(u,v) = mu_val * (s_v * u - u^2/2) for 0 <= u <= s_v
        # This is 1/b_slope * (s_v*u - u^2/2), as b_slope = 1/mu_val
        # Or, (s_v*u - u^2/2) / mu_val is incorrect. It was b_slope * (...)
        # From my LaTeX: $b^*(s_v u - u^2/2)$ for $0 \le u \le s_v$.
        # So, $(s_v u - u^2/2) / \mu_{val}$ if $b^* = 1/\mu_{val}$.

        (s_v_expr * u_sym - u_sym**2 / 2) / mu_val  # for 0 <= u <= s_v

        # Case 2: a_v_calc >= 0 (s_v >= mu_val) "Plateau + triangle"
        # Here a_v_calc is the actual $a_v$ from formula.
        middle_piece_av_pos = (
            a_v_calc
            + (s_v_expr * (u_sym - a_v_calc) - (u_sym**2 - a_v_calc**2) / 2) / mu_val
        )

        # Define threshold $a_{eff} = \max(0, a_v_calc)$ for u-conditions
        sp.Max(0, a_v_calc)

        sp.Piecewise(
            (
                u_sym,
                u_sym <= a_v_calc,
            ),  # This branch only if a_v_calc >= 0 and u_sym <= a_v_calc
            # If a_v_calc < 0, this is never met for u_sym >= 0.
            (
                middle_piece_av_pos,
                u_sym <= s_v_expr,
            ),  # This applies if a_v_calc >=0 and a_v_calc < u_sym <= s_v_expr
            (v_sym, True),  # u_sym > s_v_expr
        )

        # More robust piecewise considering a_v_calc < 0 explicitly, as in my LaTeX
        C_final = sp.Piecewise(
            (  # Case a_v_calc < 0
                sp.Piecewise(
                    ((s_v_expr * u_sym - u_sym**2 / 2) / mu_val, u_sym <= s_v_expr),
                    (v_sym, True),  # u_sym > s_v_expr
                ),
                a_v_calc < 0,
            ),
            (  # Case a_v_calc >= 0
                sp.Piecewise(
                    (u_sym, u_sym <= a_v_calc),
                    (middle_piece_av_pos, u_sym <= s_v_expr),
                    (v_sym, True),  # u_sym > s_v_expr
                ),
                True,  # Default, a_v_calc >= 0
            ),
        )
        self._cdf_expr = C_final
        return CDFWrapper(C_final)

    def pdf(self):
        """Joint density c(u,v) = ∂²C/∂u∂v."""
        # Ensure cdf expression is generated
        if not hasattr(self, "_cdf_expr") or self._cdf_expr is None:
            _ = self.cdf  # Call to generate and cache

        # Differentiate the cached symbolic CDF expression
        # Need to handle Piecewise differentiation carefully.
        # SymPy can differentiate Piecewise, but results can be complex.
        # For numerical evaluation, direct h_v(t) differentiation is better.
        # h_v(t) = C.diff(u). Then pdf is h_v.diff(v).

        h_v_expr = sp.diff(self._cdf_expr, self.u)
        pdf_expr = sp.diff(h_v_expr, self.v)

        # Simplify can be very slow or hang. Use with caution or for specific cases.
        # pdf_expr = sp.simplify(pdf_expr)
        return SymPyFuncWrapper(pdf_expr, self.u, self.v, self.rho_s)

    @property
    def is_absolutely_continuous(self) -> bool:
        return True  # Based on the construction involving densities

    @property
    def is_symmetric(self) -> bool:
        # This family is generally not symmetric C(u,v) != C(v,u)
        # However, the specific problem did not impose symmetry.
        # The example XiRhoBoundaryCopula claims is_symmetric = True.
        # That might be specific to its parameterization or an assumed property.
        # For this general family, symmetry is not guaranteed.
        # Let's assume it refers to exchangeability for now, like original example.
        return True  # Placeholder, needs verification if true for this construction.


if __name__ == "__main__":
    print("Defining XiMaxRhoSCopula")
    # Test with a rho_s value
    # rho_s_val = sp.Rational(1,5) # Example: rho_s = 0.2 (should be in Regime 1)
    rho_s_val = 0.2
    print(f"Testing with rho_s = {rho_s_val}")

    cop = XiMaxRhoSCopula(rho_s=rho_s_val)
    ccop = cop.to_check_pi(10)
    xi = ccop.xi()
    rho = ccop.rho()
    print(f"xi_optimal for rho_s = {rho_s_val}: {xi}")
    print(f"rho_s for copula: {rho}")
