"""
SchurVisualizer module for visualizing properties of copulas.
"""

import pathlib

import numpy as np
import sympy as sp
from matplotlib import pyplot as plt

from copul import BivCheckMin, BivCheckPi
from copul.families import archimedean, elliptical
from copul.schur_order.cis_rearranger import CISRearranger
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class SchurVisualizer:
    """
    Class for visualizing Schur-related properties of copulas.

    Attributes:
        copula: Copula object to visualize
        _v: Conditional value for visualization
        _x_vals: Array of x values for evaluation
    """

    def __init__(self, copula, v=0.5, x_vals=None):
        """
        Initialize SchurVisualizer.

        Args:
            copula: Copula object to visualize
            v: Conditional value (default: 0.5)
            x_vals: Array of x values for evaluation (default: 500 points in [0, 1])
        """
        self.copula = copula
        self._v = v
        if x_vals is not None:
            self._x_vals = x_vals
        else:
            self._x_vals = np.linspace(0, 1, 500)

    def compute(self):
        """
        Compute conditional distribution values.

        Returns:
            Array of conditional distribution values
        """
        if isinstance(self.copula, (BivCheckPi, BivCheckMin)):

            def amh1_l(u):
                return self.copula.cond_distr_1(u, self._v)
        else:
            amh1_cd1 = self.copula.cond_distr_1(v=self._v).func
            amh1_l = sp.lambdify(self.copula.u, amh1_cd1, "numpy")
        y1_vals = [amh1_l(x_val) for x_val in self._x_vals]
        return y1_vals

    def plot_for(self, thetas):
        """
        Plot conditional distributions for specified parameter values.

        Args:
            thetas: Parameter value or list of parameter values

        Returns:
            Dictionary mapping parameter values to computed y values
        """
        if isinstance(thetas, float):
            thetas = [thetas]
        param = self.copula.params[0]
        y_vals = {}
        for theta in thetas:
            amh1 = self.copula(**{str(param): theta, "v": self._v})
            amh1_cd1 = amh1.cond_distr_1()
            if isinstance(amh1_cd1, SymPyFuncWrapper):
                amh1_cd1 = amh1_cd1.func
                amh1_l = sp.lambdify(amh1.u, amh1_cd1, "numpy")
                y1_vals = [amh1_l(x) for x in self._x_vals]
            else:
                y1_vals = [amh1_cd1(x, self._v) for x in self._x_vals]
            pathlib.Path("../../../images/schur").mkdir(parents=True, exist_ok=True)
            plt.plot(self._x_vals, y1_vals, label=f"{param}={theta}", linewidth=2)
            y_vals[theta] = y1_vals
        self._finish_plot()
        self._finish_rearrangend_plot(y_vals)
        return y_vals

    def _finish_plot(self):
        """
        Helper method to finalize and save plots.
        """
        plt.legend()
        plt.grid()
        plt.xlabel(f"u (v={self._v})")
        plt.ylabel(f"Conditional CDF F(v={self._v}|u)")
        plt.title(f"Conditional CDF for {self.copula()}")
        path = self._get_schur_image_path()
        plt.savefig(f"{path}/{self.copula()}_v{self._v}.png")
        plt.show()
        plt.close()

    @staticmethod
    def _get_schur_image_path():
        """
        Get path for saving images.

        Returns:
            Path object for saving images
        """
        path = pathlib.Path(__file__).parent.parent / "docs" / "images" / "schur"
        path.mkdir(exist_ok=True)
        return path

    def _finish_rearrangend_plot(self, y_vals):
        """
        Helper method to plot rearranged data.

        Args:
            y_vals: Dictionary mapping parameter values to computed y values
        """
        param = self.copula.params[0]
        for theta, y1_vals in y_vals.items():
            # order y1_vals in decreasing order
            y1_vals = np.array(y1_vals)
            y1_vals = y1_vals[np.argsort(y1_vals)[::-1]]
            plt.plot(self._x_vals, y1_vals, label=f"{param}={theta}", linewidth=2)

        plt.legend()
        plt.grid()
        plt.xlabel(f"u (v={self._v})")
        plt.title(f"Decreasing rearrangement for {self.copula()}")
        path = self._get_schur_image_path()
        plt.savefig(f"{path}/{self.copula()}_v{self._v}_rearranged.png")
        plt.show()
        plt.close()


def visualize_rearranged(nelsen, thetas, v, grid_size=10):
    """
    Visualize rearranged copula.

    Args:
        nelsen: Copula class or factory
        thetas: Parameter values for the copula
        v: Conditional value
        grid_size: Size of grid for rearrangement (default: 10)
    """
    rearranger = CISRearranger(grid_size)
    x = np.linspace(0, 1, 100)
    ax = plt.gca()
    param = str(nelsen.params[0])
    for theta in thetas:
        schur8 = rearranger.rearrange_copula(nelsen(**{param: theta}))
        ccop = BivCheckPi(schur8)
        y1 = SchurVisualizer(ccop, v, x).compute()
        ax.plot(x, y1, label=f"{param}={theta}", linewidth=2)
    ax.legend()
    ax.grid()
    plt.xlabel(f"u (with v={v})")
    plt.title(f"Decreasing rearrangement for {nelsen()}")
    path = pathlib.Path(__file__).parent.parent / "docs" / "images" / "schur"
    path.mkdir(exist_ok=True)
    plt.savefig(f"{path}/{nelsen.__name__}_rearranged_v{v}.png")
    plt.show()
    plt.close()


if __name__ == "__main__":
    thetas = {
        # "Nelsen1": [0.1, 1, 5],
        "Nelsen2": [1.4, 2]
    }
    ell_thetas = {"Gaussian": [-0.8, 0.5]}  # ], "StudentT": [-0.3, 0, 0.9]}
    v_seq = [0.3, 0.9]
    for nelsen, nelsen_thetas in thetas.items():
        for v in v_seq:
            SchurVisualizer(archimedean.__dict__[nelsen], v=v).plot_for(nelsen_thetas)
    gaussian = elliptical.Gaussian()
    t = elliptical.StudentT(nu=1)
    if "Gaussian" in ell_thetas:
        gauss_y_vals = SchurVisualizer(gaussian, v=0.6).plot_for(ell_thetas["Gaussian"])
    if "StudentT" in ell_thetas:
        student_t_y_vals = SchurVisualizer(t, v=0.6).plot_for(ell_thetas["StudentT"])
    grid_size = 10
    visualize_rearranged(archimedean.Nelsen2, thetas["Nelsen2"], 0.1, grid_size)
    print("Done!")
