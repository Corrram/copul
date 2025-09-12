from typing import Sequence, Union
import numpy as np

from copul.family.core.biv_core_copula import BivCoreCopula
from copul.family.core.copula_approximator_mixin import CopulaApproximatorMixin
from copul.family.core.copula_plotting_mixin import CopulaPlottingMixin


class ShuffleOfMin(BivCoreCopula, CopulaPlottingMixin, CopulaApproximatorMixin):
    r"""
    Straight shuffle-of-Min copula :math:`C_\pi` of order :math:`n`.

    Parameters
    ----------
    pi : Sequence[int]
        A permutation of :math:`\{1,\dots,n\}` in **1-based** notation.
        ``pi[i]`` represents :math:`\pi(i+1)`.

    Notes
    -----
    The support consists of :math:`n` diagonal line segments
    \[
    S_i \;=\; \bigl\{\,((i-1+t)/n,\; (\pi(i)-1+t)/n) : 0 \le t \le 1 \,\bigr\}.
    \]
    The copula is **singular** (the density is 0 almost everywhere).

    A convenient closed form for the CDF is
    \[
    C(u,v) \;=\; \frac{1}{n}\sum_{i=1}^n
    \min\!\Bigl( 1,\; \max\!\bigl(0,\; \min(nu-i+1,\; nv-\pi(i)+1)\bigr) \Bigr).
    \]

    For fixed :math:`u \in ((i-1)/n,\, i/n)`, the conditional CDF
    :math:`C_1(v\mid u)` is a step function: it jumps from 0 to 1 at
    \[
    v_0 \;=\; \frac{\pi(i)-1 + t}{n}, \qquad t \;=\; nu - (i-1).
    \]
    At the boundaries, :math:`C_1(v\mid 0)=v` and :math:`C_1(v\mid 1)=v`.
    Similarly for :math:`C_2(u\mid v)` with :math:`C_2(u\mid 0)=u` and
    :math:`C_2(u\mid 1)=u`.
    """

    def __init__(self, pi: Sequence[int]) -> None:
        self.pi = np.asarray(pi, dtype=int)
        if self.pi.ndim != 1:
            raise ValueError("pi must be a 1-D permutation array.")
        self.n = len(self.pi)
        if self.n == 0:
            raise ValueError("Permutation cannot be empty.")
        # Check if it's a valid permutation of 1..n
        if sorted(self.pi.tolist()) == list(range(0, self.n)):
            self.pi += 1  # Convert to 1-based permutation
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
        r"""
        Normalize positional inputs and extract ``u`` and ``v``.

        Accepted forms
        --------------
        - ``_process_args((u, v))``
        - ``_process_args(([u, v],))``  (single point as 1-D array)
        - ``_process_args(([[u1, v1], ...],))``  (multiple points as 2-D array)

        Returns
        -------
        tuple[float | ndarray, float | ndarray]
            The two coordinates ``(u, v)`` with shapes broadcastable for vectorized use.

        Raises
        ------
        ValueError
            If the input shape is invalid.
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
        r"""
        Copula :math:`C_\pi(u,v)` (vectorized).

        Multiple call signatures are supported:
        - ``cdf(u, v)`` where ``u`` and ``v`` are scalars or arrays,
        - ``cdf([u, v])`` for a single point as a 1-D array,
        - ``cdf([[u1, v1], [u2, v2], ...])`` for multiple points as a 2-D array.

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
        r"""
        The straight shuffle-of-Min copula is singular ⇒ the density is 0 a.e.
        """

        raise NotImplementedError(
            "The straight shuffle-of-Min copula is singular. PDF does not exist."
        )

    # ---------- Conditional Distribution -------------------------------------
    def cond_distr_1(self, *args) -> Union[float, np.ndarray]:
        r"""
        Conditional distribution of :math:`V` given :math:`U=u`:
        :math:`C_1(v\mid u)=\mathbb{P}(V\le v\mid U=u)`.
        Same calling conventions as :meth:`cdf`.
        """

        return self.cond_distr(1, *args)

    def cond_distr_2(self, *args) -> Union[float, np.ndarray]:
        r"""
        Conditional distribution of :math:`U` given :math:`V=v`:
        :math:`C_2(u\mid v)=\mathbb{P}(U\le u\mid V=v)`.
        Same calling conventions as :meth:`cdf`.
        """

        return self.cond_distr(2, *args)

    def cond_distr(self, i: int, *args) -> Union[float, np.ndarray]:
        r"""
        Conditional distribution (vectorized).

        Computes
        - :math:`C_1(v\mid u)` if ``i=1`` (condition on :math:`U`)
        - :math:`C_2(u\mid v)` if ``i=2`` (condition on :math:`V`)

        For interior conditioning values the conditional CDF is a step function
        jumping from :math:`0` to :math:`1` at the point where the corresponding
        support segment is reached.  At the boundaries :math:`u\in\{0,1\}` or
        :math:`v\in\{0,1\}` the conditionals are uniform (:math:`v` or :math:`u`,
        respectively).

        Parameters
        ----------
        i : int
            ``1`` for :math:`C_1(v\mid u)`, ``2`` for :math:`C_2(u\mid v)`.
        *args
            Same calling conventions as :meth:`cdf`.

        Returns
        -------
        float or numpy.ndarray
            Conditional CDF value(s).

        Raises
        ------
        ValueError
            If ``i`` is outside ``{1, 2}`` or coordinates lie outside :math:`[0,1]`.
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
        r"""
        Generate :math:`\texttt{size}` i.i.d. samples from :math:`C_{\pi}`.

        Sampling picks a segment index uniformly from :math:`\{0,\dots,n-1\}` and a
        parameter :math:`t\sim U(0,1)`, then sets
        :math:`u=(i+t)/n`, :math:`v=(\pi(i+1)-1+t)/n`.

        Parameters
        ----------
        size : int
            Number of samples.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(size, 2)`` with samples in :math:`[0,1]^2`.
        """

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
        r"""
        Population Kendall’s :math:`\tau` via inversion count.

        Let :math:`N_{\mathrm{inv}}` be the number of inversions of the
        0-based permutation ``pi0``.  Then
        :math:`\tau = 1 - \dfrac{4\,N_{\mathrm{inv}}}{n^2}`.

        Returns
        -------
        float
            Kendall’s :math:`\tau` (``nan`` if :math:`n\le 1`).
        """

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
        r"""
        Population Spearman’s :math:`\rho` via squared rank differences.

        With ranks :math:`1,\dots,n` and :math:`\pi(1),\dots,\pi(n)`,
        :math:`\rho = 1 - \dfrac{6\sum_{i=1}^n (i-\pi(i))^2}{n^3}`.

        Returns
        -------
        float
            Spearman’s :math:`\rho` (``nan`` if :math:`n\le 1`).
        """

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
        r"""
        Chatterjee’s :math:`\xi`.

        For any straight shuffle-of-Min (functional dependence along segments),
        :math:`\xi=1`.

        Returns
        -------
        float
            Always ``1.0`` (``nan`` only if :math:`n=0`).
        """

        # V is a piecewise linear function of U, so xi should be 1.
        # For n=1, dependence is perfect, so 1 seems reasonable, though ranks are trivial.
        if self.n == 0:
            return np.nan  # Or raise error?
        return 1.0

    def tail_lower(self) -> float:
        r"""
        Lower tail dependence coefficient :math:`\lambda_L`.

        It is positive (equal to 1) iff the first segment lies on the main diagonal,
        i.e. :math:`\pi(1)=1`; otherwise it is 0.

        Returns
        -------
        float
            :math:`\lambda_L \in \{0,1\}` (``nan`` if :math:`n=0`).
        """

        if self.n == 0:
            return np.nan
        return 1.0 if self.pi[0] == 1 else 0.0

    def tail_upper(self) -> float:
        r"""
        Upper tail dependence coefficient :math:`\lambda_U`.

        It is positive (equal to 1) iff the last segment lies on the main diagonal,
        i.e. :math:`\pi(n)=n`; otherwise it is 0.

        Returns
        -------
        float
            :math:`\lambda_U \in \{0,1\}` (``nan`` if :math:`n=0`).
        """

        if self.n == 0:
            return np.nan
        return 1.0 if self.pi[-1] == self.n else 0.0


if __name__ == "__main__":
    # Example usage
    copula = ShuffleOfMin([1, 3, 2])
    copula.plot_c_over_u()
    copula.plot_cdf()
    copula.plot_cond_distr_1()
    copula.plot_cond_distr_2()
