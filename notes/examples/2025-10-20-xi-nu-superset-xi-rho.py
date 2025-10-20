# 2025-10-20: robust Δ(b) sampler without fragile findroot
import mpmath as mp

mp.mp.dps = 80  # high precision

# ---- closed forms from your Theorem (copy exactly as in your script) ----
def Xi_of_b(b):
    b = mp.mpf(b)
    if b <= 1:
        return mp.mpf('8')/15*b**2 - mp.mpf('8')/35*b**3
    s = 1/mp.sqrt(b)
    A = mp.acosh(mp.sqrt(b))
    # Use your closed form; equivalent series shown earlier:
    # keep the “gamma/log/acosh” version if you prefer
    # Here is a numerically-stable version directly from your final formula:
    gamma = mp.sqrt((b-1)/b)
    return (183*gamma - 38*b*gamma - 88*b**2*gamma + 112*b**2
            + 48*b**3*gamma - 48*b**3 - 105*A/b) / 210

def Nu_of_b(b):
    b = mp.mpf(b)
    if b <= 1:
        return mp.mpf('16')/15*b - mp.mpf('12')/35*b**2
    gamma = mp.sqrt((b-1)/b)
    A = mp.acosh(mp.sqrt(b))
    return ((87*gamma)/b + 250*gamma - 376*b*gamma + 448*b
            + 144*b**2*gamma - 144*b**2 - 105*A/b**2) / 420

# ---- x_rho(β) and its derivative; and M_rho(β) ----
def x_rho_beta(beta):
    beta = mp.mpf(beta)
    if beta <= 1:
        return mp.mpf('1')/2*beta**2 - mp.mpf('1')/5*beta**3
    else:
        return 1 - 1/beta + mp.mpf('3')/(10*beta**2)

def M_rho_beta(beta):
    beta = mp.mpf(beta)
    if beta <= 1:
        return beta - mp.mpf('3')/10*beta**2
    else:
        return 1 - mp.mpf('1')/(2*beta**2) + mp.mpf('1')/(5*beta**3)

# ---- Robust inverse β = β(x) ----
def beta_of_x(x):
    x = mp.mpf(x)
    # join value:
    x_join = mp.mpf('3')/10  # 0.3
    if x <= x_join + mp.mpf('1e-30'):
        # monotone on [0,1]; bisection is trivial
        lo, hi = mp.mpf('0'), mp.mpf('1')
        for _ in range(100):
            mid = (lo+hi)/2
            xm = x_rho_beta(mid)
            if xm < x:
                lo = mid
            else:
                hi = mid
        return (lo+hi)/2
    else:
        # use quadratic in y = 1/β: 0.3 y^2 - y + (1-x) = 0
        disc = 1.2*x - 0.2  # must be in (0,1] for x in (0.3,1)
        # numerical guard:
        if disc < 0:
            disc = mp.mpf('0')
        y = (1 - mp.sqrt(disc)) / mp.mpf('0.6')  # pick the ≤1 branch
        # ensure y ∈ (0,1]:
        y = min(max(y, mp.mpf('0')), mp.mpf('1'))
        if y == 0:
            return mp.inf
        return 1/y

# ---- Δ(b) and its derivative using the key identity ----
def Delta_of_b(b):
    x = Xi_of_b(b)
    beta = beta_of_x(x)
    return Nu_of_b(b) - M_rho_beta(beta)

def Xi_prime(b):
    # analytic derivatives (you already have them in your notes);
    # for simplicity, use mp.diff with a tiny h here (good enough for printing):
    h = mp.mpf('1e-20')*max(1, abs(b))
    return (Xi_of_b(b+h) - Xi_of_b(b-h)) / (2*h)

def Delta_prime_of_b(b):
    # From the proof: Δ'(b) = Xi'(b) * (β(b)-b)/(b β(b))
    x = Xi_of_b(b)
    beta = beta_of_x(x)
    return Xi_prime(b) * (beta - b) / (b * beta)

# ---- demo sweep (safe printing via float cast) ----
def sweep_and_report():
    print("=== Sampling Δ(b) on (0,1] ===")
    for bv in [0.02, 0.05, 0.1, 0.25, 0.5, 0.75]:
        Db = float(Delta_of_b(bv))
        Dpb = float(Delta_prime_of_b(bv))
        print(f"b= {bv:5.3f}  Δ(b)={Db:.12f}   Δ'(b)≈{Dpb:.12f}")

    print("\n=== Around b=1 ===")
    for bv in [0.95, 0.99, 1.000001, 1.01, 1.1]:
        Db = float(Delta_of_b(bv))
        Dpb = float(Delta_prime_of_b(bv))
        print(f"b={bv:.6f}  Δ(b)={Db:.12f}   Δ'(b)≈{Dpb:.12f}")

    print("\n=== Sampling Δ(b) for larger b ===")
    for bv in [1.5, 2.0, 3.0, 5.0, 10.0, 30.0, 100.0]:
        Db = float(Delta_of_b(bv))
        Dpb = float(Delta_prime_of_b(bv))
        print(f"b={bv:6.2f}  Δ(b)={Db:.12f}   Δ'(b)≈{Dpb:.12f}")

    # endpoint checks
    print("\n=== Endpoint behavior ===")
    b_small = mp.mpf('1e-5')
    b_large = mp.mpf('1e5')
    print("Δ(0+) ~ Δ(1e-5) =", float(Delta_of_b(b_small)))
    print("Δ(+∞) ~ Δ(1e5)  =", float(Delta_of_b(b_large)))

if __name__ == "__main__":
    sweep_and_report()
