import copy
import itertools

import numpy as np

from copul.checkerboard.check_pi import CheckPi
from copul.exceptions import PropertyUnavailableException


class CheckMin(CheckPi):
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
                overlap_min = 1
                for dim in dims:
                    if overlaps[dim] > 0 and indices[dim] + 1 < self.dim[dim]:
                        border_slices[dim] = indices[dim]
                        overlap_min = min(overlap_min, overlaps[dim])
                    else:
                        overlap_min = (
                            0  # If any dimension does not overlap, skip this term
                        )
                        break
                if overlap_min > 0:
                    total_integral += (
                        overlap_min * self.matr[tuple(border_slices)].sum()
                    )

        return total_integral

    def cond_distr(self, i, u):
        if len(u) != self.d:
            raise ValueError(
                "Number of arguments must be equal to the dimension of the copula"
            )
        if i > self.d or i < 1:
            raise ValueError("Must condition on a dimension that exists")
        i -= 1  # Adjust for zero-based indexing

        indices = []
        overlaps = []

        # Compute the indices and overlaps for each argument
        for idx in range(len(u)):
            arg = u[idx]
            if arg <= 0:
                return 0  # If the argument is out of bounds, return 0
            elif arg >= 1:
                indices.append(self.dim[idx] - 1)
                overlaps.append(1.0)
            else:
                shape = self.dim[idx]
                index = int((arg * shape) // 1)
                indices.append(index)
                overlap = arg * shape - index
                overlaps.append(overlap)

        # Denominator: Sum over all cells where U_i is in indices[i]
        slices_den = []
        for idx in range(self.d):
            if idx == i:
                slices_den.append(indices[idx])  # Fix U_i at its index
            else:
                slices_den.append(
                    slice(0, self.dim[idx])
                )  # Include all indices for U_{-i}
        denominator = self.matr[tuple(slices_den)].sum()

        # Numerator: Sum over cells where U_i is in indices[i] and U_{-i} ≥ u_i
        numerator = 0.0
        # Generate indices for U_{-i} that are greater than or equal to indices[i]
        indices_ranges = []
        for idx in range(self.d):
            if idx == i:
                indices_ranges.append([indices[idx]])
            else:
                indices_ranges.append(range(indices[i], self.dim[idx]))

        for idx_combination in itertools.product(*indices_ranges):
            cell_indices = list(idx_combination)
            cell_mass = self.matr[tuple(cell_indices)]

            # Compute within-cell contribution
            min_overlap = 1.0
            for j in range(self.d):
                if j == i:
                    continue
                if cell_indices[j] == indices[j]:
                    # Within the same cell as u[j], compute overlap
                    lower_bound = cell_indices[j] / self.dim[j]
                    upper_bound = (cell_indices[j] + 1) / self.dim[j]
                    overlap_j = (u[j] - lower_bound) / (upper_bound - lower_bound)
                    # Only consider if overlap_j >= overlaps[i]
                    if overlap_j >= overlaps[i]:
                        min_overlap = min(
                            min_overlap, (overlap_j - overlaps[i]) / (1 - overlaps[i])
                        )
                    else:
                        min_overlap = 0.0
                        break
                else:
                    # In a cell above u[j], full overlap
                    pass

            if min_overlap > 0.0:
                numerator += cell_mass * min_overlap

        if denominator == 0:
            return 0.0
        else:
            return numerator / denominator

    def rvs(self, n=1):
        sel_ele, sel_idx = self._weighted_random_selection(self.matr, n)
        sample = np.random.rand(n)
        samples = [sample / self.dim[i] for i in range(self.d)]
        add_random = np.array(samples).T
        data_points = np.array(
            [[idx[i] / self.dim[i] for i in range(len(self.dim))] for idx in sel_idx]
        )
        data_points += add_random
        return data_points

    @property
    def pdf(self):
        msg = "PDF does not exist for CheckMin"
        raise PropertyUnavailableException(msg)

    def lambda_L(self):
        return 1

    def lambda_U(self):
        return 1
