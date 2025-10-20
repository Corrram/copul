# -*- coding: utf-8 -*-
import mpmath as mp

mp.mp.dps = 60  # high precision

# ---- Closed forms for Xi(b) and Nu(b) ----

def Xi(b):
    b = mp.mpf(b)
    if b <= 1:
        return (mp.mpf(8)/15)*b**2 - (mp.mpf(8)/35)*b**3
    # b > 1
    gamma = mp.sqrt((b - 1)/b)
    A = mp.acosh(mp.sqrt(b))
    num = (mp.mpf(183)*gamma
           - mp.mpf(38)*b*gamma
           - mp.mpf(88)*b**2*gamma
           + mp.mpf(112)*b**2
           + mp.mpf(48)*b**3*gamma
           - mp.mpf(48)*b**3
           - mp.mpf(105)*A/b)
    return num/mp.mpf(210)

def Nu(b):
    b = mp.mpf(b)
    if b <= 1:
        return (mp.mpf(16)/15)*b - (mp.mpf(12)/35)*b**2
    # b > 1
    gamma = mp.sqrt((b - 1)/b)
    A = mp.acosh(mp.sqrt(b))
    num = ( (mp.mpf(87)*gamma)/b
            + mp.mpf(250)*gamma
            - mp.mpf(376)*b*gamma
            + mp.mpf(448)*b
            + mp.mpf(144)*b**2*gamma
            - mp.mpf(144)*b**2
            - mp.mpf(105)*A/b**2 )
    return num/mp.mpf(420)

def gap(b):
    return Nu(b) - Xi(b)

# ---- Golden-section argmax on (a,b) ----

def argmax_golden(f, a, b, tol=1e-12, maxit=200):
    """Maximize f on [a,b] with golden-section search."""
    invphi = (mp.sqrt(5) - 1) / 2  # 1/phi
    invphi2 = (3 - mp.sqrt(5)) / 2
    a = mp.mpf(a); b = mp.mpf(b)
    h = b - a
    if h <= tol:
        return (a + b) / 2

    # Required steps to achieve tolerance
    n = int(mp.ceil(mp.log(tol/h)/mp.log(invphi)))

    c = a + invphi2*h
    d = a + invphi*h
    yc = f(c)
    yd = f(d)

    for _ in range(n if n < maxit else maxit):
        if yc > yd:
            b, d, yd = d, c, yc
            h = invphi*h
            c = a + invphi2*h
            yc = f(c)
        else:
            a, c, yc = c, d, yd
            h = invphi*h
            d = a + invphi*h
            yd = f(d)

    xbest = c if yc > yd else d
    return xbest

# ---- Helper: coarse bracket around the max before golden search ----

def bracket_max(f, lo=1e-6, hi=100.0, ngrid=2000):
    """Coarse scan to find an interval [xL,xR] bracketing a local maximum."""
    lo = mp.mpf(lo); hi = mp.mpf(hi)
    xs = [lo + (hi - lo)*i/mp.mpf(ngrid) for i in range(ngrid+1)]
    vals = [f(x) for x in xs]
    k = max(range(1, ngrid), key=lambda i: vals[i])  # interior max index
    iL = max(0, k-1); iR = min(ngrid, k+1)
    return xs[iL], xs[iR]

# ---- Run the maximization ----

if __name__ == "__main__":
    # Step 1: coarse bracket
    L, R = bracket_max(gap, lo=1e-6, hi=100.0, ngrid=2000)

    # Step 2: golden-section refinement
    b_star = argmax_golden(gap, L, R, tol=mp.mpf('1e-14'))

    # Report
    g_star = gap(b_star)
    print("Argmax b* ≈", mp.nstr(b_star, 20))
    print("Max gap ν-ξ ≈", mp.nstr(g_star, 20))
    print("ν(b*) ≈", mp.nstr(Nu(b_star), 20))
    print("ξ(b*) ≈", mp.nstr(Xi(b_star), 20))

    # Optional: check closeness to 1
    print("\nCheck |b*-1| =", mp.nstr(abs(b_star - 1), 20))
