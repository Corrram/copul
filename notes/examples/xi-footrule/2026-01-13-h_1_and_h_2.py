import numpy as np
import matplotlib.pyplot as plt


def get_h1_h2(v, mu):
    """
    Computes h1 and h2 vectors for a given mu based on the piecewise definition.
    """
    v = np.asarray(v)
    h1 = np.zeros_like(v)
    h2 = np.zeros_like(v)

    v0 = mu / (2.0 + mu)
    v1 = 2.0 / (2.0 + mu)

    # Region 1: [0, v0]
    mask1 = v <= v0
    # h1 is 0
    # h2 = v / (1-v)
    # Avoid division by zero at v=1, though mask1 usually handles low v
    safe_v_m1 = v[mask1]
    h2[mask1] = safe_v_m1 / (1.0 - safe_v_m1 + 1e-12)

    # Region 2: (v0, v1]
    mask2 = (v > v0) & (v <= v1)
    factor = 1.0 + mu / 2.0
    h1[mask2] = factor * v[mask2] - mu / 2.0
    h2[mask2] = factor * v[mask2]

    # Region 3: (v1, 1]
    mask3 = v > v1
    # h1 = 2 - 1/v
    h1[mask3] = 2.0 - 1.0 / (v[mask3] + 1e-12)
    # h2 is 1
    h2[mask3] = 1.0

    return h1, h2, v0, v1


def plot_mu_grid(mu_values, save_path=None):
    """
    Plots h1 and h2 for a list of mu values in a grid.
    """
    # Plot settings
    plt.rcParams.update(
        {
            "font.size": 11,
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "lines.linewidth": 2,
        }
    )

    # Setup grid (e.g., 2x2 if 4 values provided)
    n = len(mu_values)
    cols = 2
    rows = (n + 1) // 2

    fig, axes = plt.subplots(
        rows, cols, figsize=(10, 4 * rows), sharex=True, sharey=True
    )
    axes = axes.flatten()

    v = np.linspace(0.0001, 0.9999, 1000)

    for i, mu in enumerate(mu_values):
        ax = axes[i]

        h1, h2, v0, v1 = get_h1_h2(v, mu)

        # Plot lines
        ax.plot(v, h1, label=r"$h_1(v)$", color="#1f77b4")  # Blue
        ax.plot(v, h2, label=r"$h_2(v)$", color="#d62728", linestyle="--")  # Red dashed

        # Vertical markers for v0 and v1
        ax.axvline(v0, color="gray", linestyle=":", alpha=0.6)
        ax.axvline(v1, color="gray", linestyle=":", alpha=0.6)

        # Annotate v0 and v1 on the x-axis
        # We place text slightly above the axis or use ticks
        ax.text(
            v0,
            -0.15,
            r"$v_0$",
            ha="center",
            va="top",
            color="gray",
            transform=ax.get_xaxis_transform(),
        )
        ax.text(
            v1,
            -0.15,
            r"$v_1$",
            ha="center",
            va="top",
            color="gray",
            transform=ax.get_xaxis_transform(),
        )

        # Title and Labels
        ax.set_title(rf"$\mu = {mu}$  ($v_0 \approx {v0:.2f}, v_1 \approx {v1:.2f}$)")

        # Add legend only to the first plot to avoid clutter
        if i == 0:
            ax.legend(loc="upper left", frameon=True)

    # Common labels
    for ax in axes[-cols:]:
        ax.set_xlabel(r"$v$")
    for ax in axes[::cols]:
        ax.set_ylabel(r"Value")

    # Hide unused subplots if any
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    # Recommended mu values to show progression
    mus = [0.5, 1.0, 1.5, 2.0]
    plot_mu_grid(mus)
