import copy
import itertools

import numpy as np
import sympy


class CheckPi:
    params = []
    intervals = {}

    def __init__(self, matr):
        if isinstance(matr, list):
            matr = np.array(matr)
        matr_sum = sum(matr) if isinstance(matr, sympy.Matrix) else matr.sum()
        self.matr = matr / matr_sum
        self.dim = matr.shape
        self.d = len(self.dim)

    def __str__(self):
        return f"CheckerboardCopula({self.dim})"

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    def cdf(self, *args):
        if len(args) != len(self.dim):
            raise ValueError(
                "Number of arguments must be equal to the dimension of the copula"
            )

        indices = []
        overlaps = []

        # Compute the indices and overlaps for each argument
        for i in range(len(args)):
            arg = args[i]
            if arg <= 0:
                return 0  # If the argument is out of bounds, return 0
            elif arg >= 1:  # If the argument exceeds 1, set it to the last index
                indices.append(self.dim[i])
                overlaps.append(0)
            else:
                shape = self.dim[i]
                index = int((arg * shape) // 1)  # Calculate the integer index
                indices.append(index)
                overlap = arg * shape - index  # Calculate the overlap for interpolation
                overlaps.append(overlap)

        # Create slices based on the computed indices
        slices = [slice(i) for i in indices]
        total_integral = self.matr[tuple(slices)].sum()

        # Now we calculate contributions from bordering hypercubes
        for i in range(self.d):
            if overlaps[i] > 0:
                border_slices = copy.deepcopy(slices)
                if indices[i] + 1 < self.dim[i]:  # Ensure we don't go out of bounds
                    border_slices[i] = indices[i]
                    border_contrib = overlaps[i] * self.matr[tuple(border_slices)].sum()
                    total_integral += border_contrib

        # Cross terms for 2D, 3D, ..., up to d-dimensional overlaps
        for r in range(2, self.d + 1):  # Start from 2D interactions up to d-dimensional
            for dims in itertools.combinations(range(self.d), r):
                border_slices = copy.deepcopy(slices)
                overlap_product = 1
                for dim in dims:
                    if overlaps[dim] > 0 and indices[dim] + 1 < self.dim[dim]:
                        border_slices[dim] = indices[dim]
                        overlap_product *= overlaps[dim]
                    else:
                        overlap_product = (
                            0  # If any dimension does not overlap, skip this term
                        )
                        break
                if overlap_product > 0:
                    total_integral += (
                        overlap_product * self.matr[tuple(border_slices)].sum()
                    )

        return total_integral

    def cond_distr(self, i: int, u):
        # Check if the provided dimension is valid
        if i > self.d:
            raise ValueError(f"Dimension {i} exceeds the number of dimensions {self.d}")

        # Adjust `i` to be zero-indexed
        i -= 1

        # Compute index values for each dimension based on `u`
        index = [min(int(u[j] * self.dim[j]), self.dim[j] - 1) for j in range(self.d)]

        # Create slice objects for multi-dimensional indexing
        slices: list = [slice(0, idx) for idx in index]

        # For the i-th dimension, we need to handle it as a regular index, not a slice
        slices[i] = index[i]  # Access the i-th dimension directly

        # Calculate the total integral over the conditioned distribution
        if isinstance(self.matr, sympy.Matrix):
            total_integral = sum(self.matr[tuple(slices)])
        else:
            total_integral = self.matr[tuple(slices)].sum()

        # Handle partial overlap for the i-th dimension
        overlap_y = u[i] * self.matr.shape[1] - index[i]
        total_integral += overlap_y * self.matr[tuple(index)]  # Adjust for the overlap

        # Adjust the result by scaling with the dimension size
        result = total_integral * self.dim[i]

        return result

    def cond_distr_1(self, u):
        return self.cond_distr(1, u)

    def cond_distr_2(self, u):
        return self.cond_distr(2, u)

    def pdf(self, *args):
        box = []
        for i in range(len(args)):
            arg = args[i]
            if arg < 0 or arg > 1:
                return 0
            box.append(int((arg * self.dim[i]) // 1))
        box = [min(i, self.dim[j] - 1) for j, i in enumerate(box)]
        return self.matr[tuple(box)]

    def rvs(self, n=1):
        sel_ele, sel_idx = self._weighted_random_selection(self.matr, n)
        samples = []
        for i in range(self.d):
            sample = np.random.rand(n) / self.dim[i]
            samples.append(sample)
        add_random = np.array(samples).T
        data_points = np.array(
            [[idx[i] / self.dim[i] for i in range(len(self.dim))] for idx in sel_idx]
        )
        data_points += add_random
        return data_points

    @staticmethod
    def _weighted_random_selection(matrix, num_samples):
        """
        Select elements from the matrix at random with likelihood proportional to their values.

        Parameters
        ----------
        matrix : numpy.ndarray
            2D array from which to select elements.
        num_samples : int
            Number of elements to select.

        Returns
        -------
        selected_elements : numpy.ndarray
            Array of selected elements.
        selected_indices : list of tuples
            List of indices of the selected elements in the original matrix.
        """
        # Flatten the matrix to a 1D array
        matrix = np.asarray(matrix, dtype=np.float64)
        flat_matrix = matrix.flatten()

        # Create the probability distribution proportional to the values
        probabilities = flat_matrix / np.sum(flat_matrix)

        # Select indices based on the probability distribution
        selected_indices_flat = np.random.choice(
            np.arange(flat_matrix.size), size=num_samples, p=probabilities
        )

        # Map the selected indices back to the original matrix
        selected_indices = [
            np.unravel_index(idx, matrix.shape) for idx in selected_indices_flat
        ]
        selected_elements = matrix[tuple(np.array(selected_indices).T)]

        return selected_elements, selected_indices

    def lambda_L(self):
        return 0

    def lambda_U(self):
        return 0
