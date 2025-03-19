from copul.copula_sampler import CopulaSampler
import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt

from copul.families.copula_graphs import CopulaGraphs


class CopulaSamplingMixin:
    """
    Mixin class for copula sampling methods.
    This class provides methods for sampling from copulas using different techniques.
    """

    def rvs(self, n=1, random_state=None, approximate=False):
        """
        Generate random variates from the copula.

        Parameters
        ----------
        n : int, optional
            Number of samples to generate (default is 1).
        random_state : int or None, optional
            Seed for the random number generator.
        approximate : bool, optional
            Whether to use approximate sampling.

        Returns
        -------
        np.ndarray
            An array of shape (n, dim) containing samples from the copula.
        """
        sampler = CopulaSampler(self, random_state=random_state)
        return sampler.rvs(n, approximate)


    def scatter_plot(
        self, n=1_000, approximate=False, figsize=(10, 8), alpha=0.6, colormap="viridis"
    ):
        """
        Create a scatter plot of random variates from the copula.

        Parameters
        ----------
        n : int, optional
            The number of samples to generate (default is 1,000).
        approximate : bool, optional
            Whether to use explicit sampling from the conditional distributions or
            approximate sampling with a checkerboard copula
        figsize : tuple, optional
            Figure size as (width, height) in inches.
        alpha : float, optional
            Transparency of points (0 to 1).
        colormap : str, optional
            Colormap to use for 3D plots.

        Returns
        -------
        None
        """
        if self.dim == 2:
            data_ = self.rvs(n, approximate=approximate)
            plt.figure(figsize=figsize)
            plt.scatter(data_[:, 0], data_[:, 1], s=rcParams["lines.markersize"] ** 2)
            title = CopulaGraphs(self).get_copula_title()
            plt.title(title)
            plt.xlabel("u")
            plt.ylabel("v")
            plt.grid(True)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.show()
            plt.close()
        elif self.dim == 3:
            # Generate samples
            data = self.rvs(n, approximate=approximate)

            # Create 3D figure and axes
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")

            # Create color mapping based on the third dimension for better visualization
            colors = data[:, 2]

            # Plot the 3D scatter points
            scatter = ax.scatter(
                data[:, 0],  # x-coordinates (first margin)
                data[:, 1],  # y-coordinates (second margin)
                data[:, 2],  # z-coordinates (third margin)
                c=colors,  # color by third dimension
                cmap=colormap,
                s=rcParams["lines.markersize"] ** 2,
                alpha=alpha,
            )

            # Add a color bar to show the mapping
            cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
            cbar.set_label("w value")

            # Set title and labels
            title = CopulaGraphs(self).get_copula_title()
            ax.set_title(title)
            ax.set_xlabel("u")
            ax.set_ylabel("v")
            ax.set_zlabel("w")

            # Set axis limits to the copula domain [0,1]³
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_zlim(0, 1)

            # Add gridlines
            ax.grid(True)

            # Add a view angle that shows the 3D structure well
            ax.view_init(elev=30, azim=45)

            plt.tight_layout()
            plt.show()
            plt.close()
        else:
            # For higher dimensions, display scatter plot matrix
            data = self.rvs(n, approximate=approximate)

            # Create scatter plot matrix
            fig, axs = plt.subplots(
                self.dim,
                self.dim,
                figsize=(3 * self.dim, 3 * self.dim),
            )

            # Get copula title
            title = CopulaGraphs(self).get_copula_title()
            fig.suptitle(title, fontsize=16)

            # Variable names
            var_names = [f"u{i+1}" for i in range(self.dim)]

            # Fill the scatter plot matrix
            for i in range(self.dim):
                for j in range(self.dim):
                    if i == j:
                        # Histogram on the diagonal
                        axs[i, j].hist(data[:, i], bins=20, alpha=0.7)
                    else:
                        # Scatter plot on off-diagonal
                        axs[i, j].scatter(
                            data[:, j],
                            data[:, i],
                            s=rcParams["lines.markersize"],
                            alpha=0.5,
                        )

                    # Set labels only on the outer plots
                    if i == self.dim - 1:
                        axs[i, j].set_xlabel(var_names[j])
                    if j == 0:
                        axs[i, j].set_ylabel(var_names[i])

                    # Set limits
                    axs[i, j].set_xlim(0, 1)
                    axs[i, j].set_ylim(0, 1)

            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the title
            plt.show()
            plt.close()
