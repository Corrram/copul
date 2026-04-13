import numpy as np
import matplotlib.pyplot as plt


def clamp(x, lo=0.0, hi=1.0):
    return np.minimum(np.maximum(x, lo), hi)


def raw_section(t, b, q):
    return b * ((1 - t) ** 2 - q)


def clamped_section(t, b, q):
    return clamp(raw_section(t, b, q), 0.0, 1.0)


def switching_points(b, q):
    """
    Return switching points in t-coordinates.

    a: where the curve leaves the upper clamp y=1
    s: where the curve reaches the lower clamp y=0
    """
    a = None
    s = None

    # b((1-t)^2 - q) = 1
    if q + 1.0 / b >= 0:
        R = np.sqrt(q + 1.0 / b)
        if 0 <= R <= 1:
            a = 1 - R

    # b((1-t)^2 - q) = 0
    if q >= 0:
        r = np.sqrt(q)
        if 0 <= r <= 1:
            s = 1 - r

    return a, s


def classify_regime(b, q):
    if q < 0:
        max_raw = b * (1 - q)
        if max_raw > 1:
            return "upper-clamped regime"
        return "unclamped regime"
    else:
        max_raw = b * (1 - q)
        if max_raw > 1:
            return "double-clamped regime"
        return "lower-clamped regime"


def setup_style():
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10.5,
            "mathtext.fontset": "dejavuserif",
            "font.family": "DejaVu Sans",
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )


def plot_panel(ax, b, q, panel_title):
    t = np.linspace(0, 1, 2000)
    y_raw = raw_section(t, b, q)
    y_clamped = clamped_section(t, b, q)
    a, s = switching_points(b, q)
    regime = classify_regime(b, q)

    # limits
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.14, 1.22)

    # reference lines
    ax.axhline(0, color="0.55", lw=0.9, zorder=0)
    ax.axhline(1, color="0.80", lw=0.9, ls=":", zorder=0)

    # raw and clamped curves
    (raw_line,) = ax.plot(
        t,
        y_raw,
        ls="--",
        lw=2.0,
        color="#1f77b4",
        label=r"raw parabola $b((1-t)^2-q)$",
        zorder=2,
    )
    (clamp_line,) = ax.plot(
        t,
        y_clamped,
        lw=2.8,
        color="#ff7f0e",
        label=r"clamped section $h_b^{(q)}(t)$",
        zorder=3,
    )

    # switching lines and labels
    point_handles = []

    if a is not None and 0 < a < 1:
        ax.axvline(a, color="0.82", lw=0.9, ls=":", zorder=0)
        p = ax.plot(a, 1, marker="o", ms=5.5, color="#2ca02c", zorder=4)[0]
        point_handles.append(p)
        ax.annotate(
            r"$a$",
            xy=(a, 1),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=12,
        )

    if s is not None and 0 < s < 1:
        ax.axvline(s, color="0.82", lw=0.9, ls=":", zorder=0)
        p = ax.plot(s, 0, marker="o", ms=5.5, color="#d62728", zorder=4)[0]
        point_handles.append(p)
        ax.annotate(
            r"$s$",
            xy=(s, 0),
            xytext=(0, -14),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=12,
        )

    # subtle regime tag
    ax.text(
        0.03,
        0.96,
        regime,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        color="0.20",
        bbox=dict(boxstyle="round,pad=0.2", fc="0.96", ec="0.85", lw=0.8),
    )

    # titles and labels
    ax.set_title(panel_title + "\n" + rf"$b={b},\ q={q}$", pad=8)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$h_b^{(q)}(t)$")

    # cleaner ticks
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1.2, 7))

    return raw_line, clamp_line


def main():
    setup_style()

    examples = [
        (2.5, -0.2, "Upper-clamped"),
        (0.7, -0.2, "Unclamped"),
        (4.0, 0.12, "Double-clamped"),
        (2.5, 0.7, "Lower-clamped"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2))
    axes = axes.ravel()

    handles = None
    for ax, (b, q, title) in zip(axes, examples):
        handles = plot_panel(ax, b, q, title)

    # shared legend above panels
    fig.legend(
        handles,
        [r"raw parabola $b((1-t)^2-q)$", r"clamped section $h_b^{(q)}(t)$"],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )

    # fig.suptitle(
    #     r"Four clamping regimes of $h_b^{(q)}(t)=\operatorname{clamp}(b((1-t)^2-q),0,1)$",
    #     y=0.995,
    #     fontsize=18
    # )

    fig.subplots_adjust(
        top=0.89, bottom=0.09, left=0.08, right=0.98, hspace=0.35, wspace=0.16
    )

    # Uncomment for export
    # plt.savefig("clamping_regimes_clean.pdf", bbox_inches="tight")
    # plt.savefig("clamping_regimes_clean.png", dpi=300, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    main()
