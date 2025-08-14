import numpy as np
import math
import matplotlib.pyplot as plt


def d_star(q, c):
    W0 = 0.5 - q / 8 + q**2 / 30
    W1 = 0.5 - q / 2 + 4 * q**2 / 15
    target = (c + 2) / 6
    if target >= W0:
        return 0.0
    if target <= W1:
        return 1.0
    coeffs = [4 * q**2, -15 * q, 0, 20 * (1 - c)]
    roots = np.roots(coeffs)
    for r in roots:
        if abs(r.imag) < 1e-8 and 1.0 < r.real < 2.0:
            return r.real - 1.0
    raise RuntimeError("No valid root for d*")


def s_v(v, q, d):
    v1 = q * (1 + d) / 2.0
    v2 = 1 - v1
    if v <= v1:
        return math.sqrt(2 * q * (1 + d) * v)
    elif v <= v2:
        return v + q * (1 + d) / 2.0
    else:
        return 1 + q * (1 + d) - math.sqrt(2 * q * (1 + d) * (1 - v))


def h_star(t, v, q, d):
    b = 1.0 / q
    sv = s_v(v, q, d)
    core = b * (sv - t) - d * (t <= v)
    return min(1.0, max(0.0, core))


def J_closed(q, d):
    return (
        (q**3 / 20) * d**4
        + (2 * q**3 / 15) * d**3
        - (q**2 / 6) * d**3
        + (q**3 / 10) * d**2
        - (q**2 / 4) * d**2
        - (q**3 / 60)
        + (q**2 / 12)
        - (q / 2)
        + 2 / 3
    )


def compare_J(q, c, grid=301):
    d = d_star(q, c)
    vs = np.linspace(0.0, 1.0, grid)
    inner = []
    for v in vs:
        hv = [h_star(t, v, q, d) for t in vs]
        integrand = [2 * (1 - t) * h + (-q) * h * h for t, h in zip(vs, hv)]
        inner.append(np.trapz(integrand, vs))
    J_num = np.trapz(inner, vs)
    return d, J_closed(q, d), J_num, abs(J_closed(q, d) - J_num)


def plot_J_vs_c(a, c_vals, grid=201):
    q = -a
    J_vals = [J_closed(q, d_star(q, c)) for c in c_vals]
    plt.plot(c_vals, J_vals)
    plt.xlabel("c")
    plt.ylabel("J_max")
    plt.title(f"Optimal J vs c  (a={a})")
    plt.show()


if __name__ == "__main__":
    a = -0.02
    c = 0.3
    q = -a
    d, J_cl, J_num, err = compare_J(q, c, grid=301)
    print(
        f"a={a}, c={c}, d*={d:.6f}, J_cl={J_cl:.6f}, J_num={J_num:.6f}, err={err:.3e}"
    )

    # example plot
    c_grid = np.linspace(-0.5, 1.0, 200)
    plot_J_vs_c(a, c_grid, grid=301)
