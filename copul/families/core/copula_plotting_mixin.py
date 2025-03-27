from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
from copul.families.copula_graphs import CopulaGraphs


class CopulaPlottingMixin:
    """
    Mixin class for copula sampling methods.
    This class provides methods for sampling from copulas using different techniques.
    """

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
            var_names = [f"u{i + 1}" for i in range(self.dim)]

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

    def save_plot(
        self,
        filename,
        n=1_000,
        approximate=False,
        figsize=(12, 10),
        alpha=0.7,
        colormap="viridis",
        style="default",
        point_size=None,
        dpi=300,
        grid_alpha=0.2,
        contour_levels=10,
        add_contours=True,
        add_marginals=True,
        format="png",
        transparent=False,
    ):
        """
        Create and save a scatter plot of random variates from the copula.

        Parameters
        ----------
        filename : str
            Path where the plot will be saved.
        n : int, optional
            The number of samples to generate (default is 1,000).
        approximate : bool, optional
            Whether to use explicit sampling or approximate sampling.
        figsize : tuple, optional
            Figure size as (width, height) in inches.
        alpha : float, optional
            Transparency of points (0 to 1).
        colormap : str, optional
            Colormap to use for plots.
        style : str, optional
            Matplotlib style to use.
        point_size : int, optional
            Size of scatter points.
        dpi : int, optional
            Resolution of the saved figure.
        grid_alpha : float, optional
            Transparency of grid lines (0 to 1).
        contour_levels : int, optional
            Number of contour levels to add.
        add_contours : bool, optional
            Whether to add density contours to 2D plots.
        add_marginals : bool, optional
            Whether to add marginal distributions to 2D plots.
        format : str, optional
            File format to save the plot (e.g., 'png', 'pdf', 'svg').
        transparent : bool, optional
            Whether to save with a transparent background.

        Returns
        -------
        None
        """
        # Create the figure
        fig = self.scatter_plot(
            n=n,
            approximate=approximate,
            figsize=figsize,
            alpha=alpha,
            colormap=colormap,
            style=style,
            point_size=point_size,
            dpi=dpi,
            grid_alpha=grid_alpha,
            contour_levels=contour_levels,
            add_contours=add_contours,
            add_marginals=add_marginals,
        )

        # Ensure the filename has the correct extension
        if not filename.lower().endswith(f".{format.lower()}"):
            filename = f"{filename}.{format.lower()}"

        # Save the figure
        fig.savefig(
            filename,
            dpi=dpi,
            bbox_inches="tight",
            transparent=transparent,
            format=format,
        )
        plt.close(fig)

        print(f"Plot saved to: {filename}")

    def plot_density(
        self,
        grid_size=50,
        figsize=(10, 8),
        colormap="viridis",
        style="default",
        dpi=120,
        add_contours=True,
        contour_levels=10,
        contour_colors="black",
        contour_alpha=0.6,
        add_colorbar=True,
    ):
        """
        Create a density plot of the copula.

        Parameters
        ----------
        grid_size : int, optional
            Number of grid points in each dimension.
        figsize : tuple, optional
            Figure size as (width, height) in inches.
        colormap : str, optional
            Colormap to use for the density plot.
        style : str, optional
            Matplotlib style to use.
        dpi : int, optional
            Resolution of the figure.
        add_contours : bool, optional
            Whether to add contour lines to the density plot.
        contour_levels : int, optional
            Number of contour levels to add.
        contour_colors : str or list, optional
            Color(s) for contour lines.
        contour_alpha : float, optional
            Transparency of contour lines (0 to 1).
        add_colorbar : bool, optional
            Whether to add a colorbar to the plot.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure object.
        """
        # Set the style
        if style != "default":
            plt.style.use(style)

        # Get title
        title = CopulaGraphs(self).get_copula_title()

        if self.dim == 2:
            # Create 2D grid
            x = np.linspace(0.001, 0.999, grid_size)
            y = np.linspace(0.001, 0.999, grid_size)
            X, Y = np.meshgrid(x, y)

            # Calculate the copula density
            positions = np.vstack([X.ravel(), Y.ravel()]).T
            try:
                density_values = np.array([self.pdf(pos) for pos in positions])
                Z = density_values.reshape(X.shape)

                # Create figure
                fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

                # Plot the density as a filled contour
                contourf = ax.contourf(X, Y, Z, levels=50, cmap=colormap, alpha=0.9)

                # Add contour lines if requested
                if add_contours:
                    ax.contour(
                        X,
                        Y,
                        Z,
                        levels=contour_levels,
                        colors=contour_colors,
                        alpha=contour_alpha,
                    )

                # Add colorbar if requested
                if add_colorbar:
                    cbar = fig.colorbar(contourf, ax=ax)
                    cbar.set_label("Density")

                # Set title and labels
                ax.set_title(title + " Density", fontsize=14)
                ax.set_xlabel("u", fontsize=12)
                ax.set_ylabel("v", fontsize=12)

                # Set axis limits
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)

                # Add grid
                ax.grid(True, alpha=0.2)

                plt.tight_layout()
                return fig

            except (AttributeError, NotImplementedError):
                print("Density function not available for this copula.")
                return None

        else:
            print("Density plots are currently only supported for 2D copulas.")
            return None

    def create_comparison_plot(
        self,
        other_copula,
        n=1_000,
        figsize=(15, 7),
        colormap="viridis",
        style="default",
        dpi=120,
    ):
        """
        Create a comparison plot between this copula and another one.

        Parameters
        ----------
        other_copula : Copula
            Another copula to compare with.
        n : int, optional
            Number of samples to generate from each copula.
        figsize : tuple, optional
            Figure size as (width, height) in inches.
        colormap : str, optional
            Colormap to use for the plots.
        style : str, optional
            Matplotlib style to use.
        dpi : int, optional
            Resolution of the figure.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure object.
        """
        # Set the style
        if style != "default":
            plt.style.use(style)

        # Check dimensions
        if self.dim != other_copula.dim:
            print("Copulas must have the same dimension for comparison.")
            return None

        # Generate samples
        data1 = self.rvs(n)
        data2 = other_copula.rvs(n)

        # Get titles
        title1 = CopulaGraphs(self).get_copula_title()
        title2 = CopulaGraphs(other_copula).get_copula_title()

        if self.dim == 2:
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

            # Plot first copula
            scatter1 = ax1.scatter(
                data1[:, 0],
                data1[:, 1],
                s=rcParams["lines.markersize"] * 1.5,
                alpha=0.7,
                c=data1[:, 1],
                cmap=colormap,
                edgecolor="none",
            )
            ax1.set_title(title1, fontsize=12)
            ax1.set_xlabel("u", fontsize=11)
            ax1.set_ylabel("v", fontsize=11)
            ax1.grid(True, alpha=0.2)
            ax1.set_xlim(0, 1)
            ax1.set_ylim(0, 1)

            # Plot second copula
            scatter2 = ax2.scatter(
                data2[:, 0],
                data2[:, 1],
                s=rcParams["lines.markersize"] * 1.5,
                alpha=0.7,
                c=data2[:, 1],
                cmap=colormap,
                edgecolor="none",
            )
            ax2.set_title(title2, fontsize=12)
            ax2.set_xlabel("u", fontsize=11)
            ax2.set_ylabel("v", fontsize=11)
            ax2.grid(True, alpha=0.2)
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)

            # Add colorbars
            fig.colorbar(scatter1, ax=ax1, pad=0.01)
            fig.colorbar(scatter2, ax=ax2, pad=0.01)

            # Add comparison title
            fig.suptitle(f"Comparison: {title1} vs {title2}", fontsize=14, y=0.98)

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            return fig

        else:
            print("Comparison plots are currently optimized for 2D copulas.")
            # Simply return the regular scatter plot matrix for higher dimensions
            return self.scatter_plot(
                n=n, figsize=figsize, colormap=colormap, style=style, dpi=dpi
            )
