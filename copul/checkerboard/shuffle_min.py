from typing import Sequence, Union
import numpy as np

from copul.families.core.biv_core_copula import BivCoreCopula
from copul.families.core.copula_plotting_mixin import CopulaPlottingMixin


# --- Start of ShuffleOfMin Class Definition ---
class ShuffleOfMin(BivCoreCopula, CopulaPlottingMixin):
    r"""
    Straight shuffle-of-Min copula C_\pi of order n.

    *Parameters*
    ----------
    pi : Sequence[int]
        A permutation of {1,…,n} in 1-based notation. `pi[i]` is \pi(i+1).

    Notes
    -----
    The support consists of n diagonal line segments
        S_i = {((i-1+t)/n , (pi[i]-1+t)/n) : 0 ≤ t ≤ 1 }.
    The copula is singular: `pdf` returns 0 everywhere.
    The CDF is given by C(u,v) = (1/n) * sum_{i=1}^n min(max(0, min(nu - (i-1), nv - (pi[i]-1))), 1).
    The conditional CDF C_1(v|u) for u in ((i-1)/n, i/n) is a step function:
    0 if v < v_0, 1 if v >= v_0, where v_0 = (pi[i]-1+t)/n and t = n*u - (i-1).
    At boundaries u=0 or u=1, C_1(v|u)=v.
    Similarly for C_2(u|v). At boundaries v=0 or v=1, C_2(u|v)=u.
    """

    def __init__(self, pi: Sequence[int]) -> None:
        self.pi = np.asarray(pi, dtype=int)
        if self.pi.ndim != 1:
            raise ValueError("pi must be a 1-D permutation array.")
        self.n = len(self.pi)
        if self.n == 0:
            raise ValueError("Permutation cannot be empty.")
        # Check if it's a valid permutation of 1..n
        if sorted(self.pi.tolist()) != list(range(1, self.n + 1)):
            raise ValueError("pi must be a permutation of 1..n")

        # Pre-compute 0-based permutation and its inverse for efficiency
        self.pi0 = self.pi - 1  # 0-based permutation: pi0[i] = pi(i+1)-1
        if self.n > 0:
            # 0-based inverse: pi0_inv[j] = k means pi0[k] = j
            self.pi0_inv = np.argsort(self.pi0)
        else:
            self.pi0_inv = np.array([], dtype=int)

        # Check if this is identity or reverse permutation (for optimized calculations)
        self.is_identity = np.array_equal(self.pi, np.arange(1, self.n + 1))
        self.is_reverse = np.array_equal(self.pi, np.arange(self.n, 0, -1))
        BivCoreCopula.__init__(self)  # Sets self.dim = 2

    # ---------- helper -------------------------------------------------------

    def _process_args(
        self, args
    ) -> tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Process arguments to extract u and v coordinates.

        Supports:
        - _process_args((u, v))
        - _process_args(([u, v],))
        - _process_args(([[u1, v1], [u2, v2], ...],))
        """
        if not args:
            raise ValueError("No arguments provided.")

        if len(args) == 2:
            # Direct u, v arguments
            return args[0], args[1]

        if len(args) == 1:
            arr = np.asarray(args[0], dtype=float)

            if arr.ndim == 1:
                if arr.size == 2:
                    # Single point as 1D array [u, v]
                    return arr[0], arr[1]
                else:
                    raise ValueError("1D input must have length 2.")

            elif arr.ndim == 2:
                if arr.shape[1] == 2:
                    # Multiple points as 2D array [[u1, v1], [u2, v2], ...]
                    return arr[:, 0], arr[:, 1]
                else:
                    raise ValueError("2D input must have 2 columns.")
            else:
                raise ValueError("Input must be 1D or 2D array.")

        raise ValueError(f"Expected 1 or 2 arguments, got {len(args)}.")

    # ---------- CDF ----------------------------------------------------------

    def cdf(self, *args) -> Union[float, np.ndarray]:
        """
        Copula C_\\pi(u,v). Vectorized implementation.

        Multiple call signatures are supported:
        - cdf(u, v): u and v are scalars or arrays
        - cdf([u, v]): Single point as a 1D array
        - cdf([[u1, v1], [u2, v2], ...]): Multiple points as a 2D array

        Returns
        -------
        float or numpy.ndarray
            The CDF values at the specified points.
        """
        u, v = self._process_args(args)
        # Ensure inputs are arrays and broadcastable
        u_arr, v_arr = np.broadcast_arrays(u, v)

        # Check bounds
        if np.any((u_arr < 0) | (u_arr > 1) | (v_arr < 0) | (v_arr > 1)):
            raise ValueError("u, v must lie in [0,1].")

        # Store if the original input was scalar to return scalar at the end
        input_is_scalar = np.isscalar(u) and np.isscalar(v)

        # --- Optimization for identity permutation ---
        if self.is_identity:
            # For identity, CDF is simply min(u, v)
            result = np.minimum(u_arr, v_arr)
            return result.item() if input_is_scalar else result

        # Initialize output array with the same shape as broadcast inputs
        out = np.zeros_like(u_arr, dtype=float)

        # --- Handle boundary conditions using masks ---
        tol = 1e-9  # Tolerance for floating point comparisons
        mask_u0 = np.abs(u_arr) < tol
        mask_v0 = np.abs(v_arr) < tol
        mask_u1 = np.abs(u_arr - 1.0) < tol
        mask_v1 = np.abs(v_arr - 1.0) < tol

        # Condition: C(u,v) = 0 if u=0 or v=0
        out[mask_u0 | mask_v0] = 0.0

        # Condition: C(1,v) = v (apply where u=1 but v!=0)
        # Ensure C(1,0)=0 is preserved
        out[mask_u1 & ~mask_v0] = v_arr[mask_u1 & ~mask_v0]

        # Condition: C(u,1) = u (apply where v=1 but u!=0 and u!=1)
        # Ensure C(0,1)=0 and C(1,1)=1 are preserved
        out[mask_v1 & ~mask_u0 & ~mask_u1] = u_arr[mask_v1 & ~mask_u0 & ~mask_u1]

        # Identify points strictly inside (0,1)x(0,1) for the main calculation
        mask_in = ~(mask_u0 | mask_v0 | mask_u1 | mask_v1)

        # Perform calculation only if there are interior points
        if np.any(mask_in):
            # Select only interior points
            u_in = u_arr[mask_in]
            v_in = v_arr[mask_in]

            # --- Vectorized calculation for interior points ---
            nu = self.n * u_in[:, None]
            nv = self.n * v_in[:, None]
            i_idx = np.arange(self.n)[None, :]  # Segment indices (0 to n-1)
            pi_i_0based = self.pi0[None, :]  # 0-based permutation values

            # Calculate contribution from each segment:
            # min(max(0, min(nu-(i-1), nv-(pi(i)-1))), 1)
            t_u_comp = nu - i_idx
            t_v_comp = nv - pi_i_0based
            min_t = np.minimum(t_u_comp, t_v_comp)
            # Capped contribution: min( max(0, min_t), 1.0 )
            capped_contribution = np.minimum(np.maximum(0.0, min_t), 1.0)

            # Sum contributions over segments and divide by n
            cdf_values_in = np.sum(capped_contribution, axis=1) / self.n

            # Assign results back to the output array
            out[mask_in] = cdf_values_in

        # Return scalar if input was scalar, otherwise return the array
        if input_is_scalar:
            return out.item()
        else:
            return out

    # ---------- PDF ----------------------------------------------------------
    def pdf(self, *args) -> Union[float, np.ndarray]:
        """
        The straight shuffle-of-Min copula is singular ⇒ density is 0 a.e.
        """
        raise NotImplementedError(
            "The straight shuffle-of-Min copula is singular. PDF does not exist."
        )

    # ---------- Conditional Distribution -------------------------------------
    def cond_distr_1(self, *args) -> Union[float, np.ndarray]:
        """
        Conditional distribution of V given U=u: C_1(v|u) = P(V <= v | U = u).
        Same calling conventions as cdf() and pdf().
        """
        return self.cond_distr(1, *args)

    def cond_distr_2(self, *args) -> Union[float, np.ndarray]:
        """
        Conditional distribution of U given V=v: C_2(u|v) = P(U <= u | V = v).
        Same calling conventions as cdf() and pdf().
        """
        return self.cond_distr(2, *args)

    def cond_distr(self, i: int, *args) -> Union[float, np.ndarray]:
        """
        Conditional distribution function (vectorized).

        Calculates:
        - C_1(v|u) if i=1: conditional of V given U=u
        - C_2(u|v) if i=2: conditional of U given V=v

        For a fixed conditioning value, the conditional CDF is a step
        function that jumps from 0 to 1 at the corresponding value on the
        copula's support segment. At boundaries (u/v = 0 or 1), it's uniform.

        Parameters
        ----------
        i : int
            Index of the conditioning variable (1 for V|U, 2 for U|V)
        *args :
            Same calling conventions as cdf() and pdf().

        Returns
        -------
        float or numpy.ndarray
            The conditional distribution values.
        """
        if not (1 <= i <= self.dim):
            raise ValueError(f"i must be between 1 and {self.dim}")

        u, v = self._process_args(args)
        # Ensure inputs are arrays and broadcastable
        u_arr, v_arr = np.broadcast_arrays(u, v)

        # Check bounds
        if np.any((u_arr < 0) | (u_arr > 1) | (v_arr < 0) | (v_arr > 1)):
            raise ValueError("u, v must lie in [0,1].")

        # Store if the original input was scalar
        input_is_scalar = np.isscalar(u) and np.isscalar(v)

        # Initialize output array
        out = np.zeros_like(u_arr, dtype=float)

        # Use a small tolerance for floating point comparisons
        tol = 1e-9

        # --- C_1(v|u): Conditional of V given U=u ---
        if i == 1:
            # Handle boundaries: C_1(v|0)=v, C_1(v|1)=v
            mask_u0 = np.abs(u_arr) < tol
            mask_u1 = np.abs(u_arr - 1.0) < tol
            mask_boundary = mask_u0 | mask_u1
            mask_in = ~mask_boundary  # Interior u values

            # Apply boundary condition
            out[mask_boundary] = v_arr[mask_boundary]

            # Process interior u values
            if np.any(mask_in):
                u_in = u_arr[mask_in]
                v_in = v_arr[mask_in]

                # Find segment index i (0-based) based on u: i/n <= u < (i+1)/n
                # Use floor(n*u - tol) to handle edge cases near boundaries like u=1/n
                i_idx = np.floor(self.n * u_in - tol).astype(int)
                # Clip to handle potential edge case u slightly less than 0 or exactly 1
                i_idx = np.clip(i_idx, 0, self.n - 1)

                # Calculate parameter t = n*u - i
                t = self.n * u_in - i_idx

                # Find the corresponding v0 on the segment's diagonal
                # v0 = (pi(i+1)-1 + t) / n = (pi0[i_idx] + t) / n
                pi_i_0based = self.pi0[i_idx]  # Get pi(i+1)-1 using 0-based index
                v0 = (pi_i_0based + t) / self.n

                # Conditional CDF is 1 if v >= v0, else 0
                out[mask_in] = (v_in >= v0 - tol).astype(
                    float
                )  # Add tol for comparison robustness

        # --- C_2(u|v): Conditional of U given V=v ---
        elif i == 2:
            # Handle boundaries: C_2(u|0)=u, C_2(u|1)=u
            mask_v0 = np.abs(v_arr) < tol
            mask_v1 = np.abs(v_arr - 1.0) < tol
            mask_boundary = mask_v0 | mask_v1
            mask_in = ~mask_boundary  # Interior v values

            # Apply boundary condition
            out[mask_boundary] = u_arr[mask_boundary]

            # Process interior v values
            if np.any(mask_in):
                u_in = u_arr[mask_in]
                v_in = v_arr[mask_in]

                # Find the segment index k (0-based) and parameter t corresponding to v
                # Find the rank j (0-based) of v: j/n <= v < (j+1)/n
                j_idx = np.floor(self.n * v_in - tol).astype(int)
                j_idx = np.clip(j_idx, 0, self.n - 1)

                # Find the segment index k (0-based) such that pi0[k] = j_idx
                # This is k = pi0_inv[j_idx]
                k_idx = self.pi0_inv[j_idx]

                # Calculate parameter t = n*v - j
                t = self.n * v_in - j_idx

                # Find the corresponding u0 on the segment's diagonal
                # u0 = (k + t) / n
                u0 = (k_idx + t) / self.n

                # Conditional CDF is 1 if u >= u0, else 0
                out[mask_in] = (u_in >= u0 - tol).astype(
                    float
                )  # Add tol for comparison robustness

        # Return scalar if input was scalar, otherwise return the array
        if input_is_scalar:
            return out.item()
        else:
            return out

    # ---------- utilities ----------------------------------------------------
    def __str__(self):
        return f"ShuffleOfMin(order={self.n}, pi={self.pi.tolist()})"

    # simple generators for simulation / plotting -----------------------------
    def rvs(self, size: int, **kwargs) -> np.ndarray:
        """Generate `size` iid samples from C_{\\pi}."""
        # Choose a random segment index (0 to n-1) for each sample
        seg_idx = np.random.randint(0, self.n, size=size)
        # Choose a random parameter t (0 to 1) along the segment diagonal
        t = np.random.random(size=size)

        # Calculate u and v based on the chosen segment and t
        # u = (i + t) / n where i = seg_idx
        u = (seg_idx + t) / self.n
        # v = (pi(i+1)-1 + t) / n = (pi0[i] + t) / n
        v = (self.pi0[seg_idx] + t) / self.n

        return np.column_stack([u, v])

    # --- Association measures ------------------------------------------------
    def kendall_tau(self) -> float:
        """Population Kendall's tau via inversion count."""
        # Handle n=1 case first
        if self.n <= 1:
            return np.nan

        if self.is_identity:
            return 1.0

        # Correct calculation using 0-based indexing internally for pi0
        pi0_temp = self.pi0  # Use precomputed 0-based perm
        N_inv = sum(
            1
            for i in range(self.n)
            for j in range(i + 1, self.n)
            if pi0_temp[i] > pi0_temp[j]
        )
        # Denominator n*(n-1)/2 is the total number of pairs
        # Tau = 1 - 2 * (N_inv / (n*(n-1)/2)) = 1 - 4*N_inv/(n*(n-1))
        tau = 1.0 - 4.0 * N_inv / (self.n**2)
        return tau

    def spearman_rho(self) -> float:
        """Population Spearman's rho via squared rank differences."""
        # Handle n=1 case first
        if self.n <= 1:
            return np.nan

        if self.is_identity:
            return 1.0

        # Ranks for u are essentially 1, 2, ..., n based on segment index
        # Ranks for v are pi(1), pi(2), ..., pi(n)
        i_ranks = np.arange(1, self.n + 1)
        pi_ranks = self.pi  # Use 1-based perm for rank difference calculation
        d_sq = np.sum((i_ranks - pi_ranks) ** 2)
        # Rho = 1 - 6 * sum(d^2) / (n * (n^2 - 1))
        return 1.0 - 6.0 * d_sq / self.n**3

    def chatterjee_xi(self) -> float:
        """Chatterjee's xi = 1 for any straight shuffle-of-Min (functional dependence)."""
        # V is a piecewise linear function of U, so xi should be 1.
        # For n=1, dependence is perfect, so 1 seems reasonable, though ranks are trivial.
        if self.n == 0:
            return np.nan  # Or raise error?
        return 1.0

    def tail_lower(self) -> float:
        """Lower tail dependence: positive only if first segment is on diagonal."""
        if self.n == 0:
            return np.nan
        return 1.0 if self.pi[0] == 1 else 0.0

    def tail_upper(self) -> float:
        """Upper tail dependence: positive only if last segment is on diagonal."""
        if self.n == 0:
            return np.nan
        return 1.0 if self.pi[-1] == self.n else 0.0


if __name__ == "__main__":
    # Example usage
    copula = ShuffleOfMin([2, 1, 3])
    copula.plot_cdf()
    copula.plot_cond_distr_1()
    copula.plot_cond_distr_2()
