import numpy as np

def s_v(v, b):
    """Compute s_v according to definition (works for b > 0)."""
    if (b >= 1 and v <= 1/(2*b)) or (b <= 1 and v <= b/2):
        return np.sqrt(2 * v / b)
    if 1/(2*b) < v <= 1 - 1/(2*b):
        return v + 1/(2*b)
    if (b <= 1 and (b/2) < v <= 1 - (b/2)) or (b >= 1 and (1/(2*b)) < v <= 1 - (b/2)):
        return v / b + 1/2
    # fourth case
    return 1 + 1/b - np.sqrt(2 * (1 - v) / b)

def a_v(v, b):
    """Compute a_v = s_v - 1/b (for b > 0)."""
    return s_v(v, b) - 1/b

def C_b_positive(u, v, b):
    """
    The 'upper' version of C_b (C_b^↑) for b > 0, as per equation (C_x).
    """
    av = a_v(v, b)
    sv = s_v(v, b)
    if u <= av:
        return u
    elif u <= sv:
        return av + b * (sv * (u - av) + (av**2 - u**2) / 2)
    else:
        return v

def C_b(u, v, b):
    """
    Full Copula C_b:
      - If b >= 0: return C_b^↑(u,v; b)
      - If b  < 0: return the decreasing rearrangement C_{-b}^↓(u,v) 
        defined by C^↓(u,v) = v - C^↑(1-u, v).
    """
    if b >= 0:
        return C_b_positive(u, v, b)
    else:
        # Define C_{|b|}^↑ at (1-u, v), then apply C^↓(u,v) = v - C^↑(1-u, v)
        return v - C_b_positive(1 - u, v, -b)

# Vectorized version for array inputs
vec_Cb = np.vectorize(C_b)

# Beispiel: Testen der Max-Stabilität erneut (optional)
if __name__ == "__main__":
    np.random.seed(0)
    u_vals = np.random.rand(1000)
    v_vals = np.random.rand(1000)
    t_vals = np.array([2.0, 3.0, 5.0])

    b_values = [-10, -2, -0.5, -0.2, -0.1, 0.1, 0.5, 1.0, 2.0, 10]
    for b_test in b_values:
        max_diff = 0.0
        for t in t_vals:
            C_uv = vec_Cb(u_vals, v_vals, b_test)
            C_u_t_v_t = vec_Cb(u_vals**t, v_vals**t, b_test)
            diff = np.abs(C_u_t_v_t - C_uv**t)
            max_diff = max(max_diff, np.nanmax(diff))
        print(f"b = {b_test}, Max Difference: {max_diff:.6f}")
