import numpy as np
import matplotlib.pyplot as plt

# >>> PARAMETERS ------------------------------------------------------------
a      = 0.01
delta  = 1/(2*a)              # = 50   (left half of every h_v vanishes)
v_grid = np.linspace(0.0, 1.0, 4001)   # high‑resolution v–grid
dv     = v_grid[1]-v_grid[0]

# helper: closed‑form integrals over t for ONE (v,b)–pair -------------------
def one_v(have_plateau, v, b):
    """Return (h^2 integral ,  2(1-t)h integral) for fixed v, b."""
    if have_plateau:                     # v > 1/(2b)
        s   = 2*v + 0.5/b
        a_v = s - 1/b                    # left end of ramp
        # plateau part:  t in [v, a_v] ,  h=1
        Lp  = a_v - v
        h2_plateau = Lp
        J_plateau  = 2*Lp - (a_v**2 - v**2)
        # ramp part:    t in [a_v, 1] ,  h = b(s-t)
        A = s - a_v
        B = s - 1.0
        h2_ramp = b**2 * (A**3 - B**3)/3
        Jramp   = 2*b*( s*(1-a_v) - (s+1)*(1 - a_v**2)/2 + (1 - a_v**3)/3 )
        return h2_plateau + h2_ramp , J_plateau + Jramp
    else:                                # v ≤ 1/(2b)  (no plateau)
        s = v + np.sqrt(2*v/b)
        if s <= 1.0:                     # ramp finishes before t=1
            A = s - v
            h2   = b**2 * A**3 / 3
            Jval = 2*b*( s*A - (s+1)*(s**2 - v**2)/2 + (s**3 - v**3)/3 )
            return h2 , Jval
        # ramp is truncated at t=1
        s = (v + 0.5*b*(1 - v**2)) / (b*(1 - v))
        A = s - v
        B = s - 1.0
        h2 = b**2 * (A**3 - B**3) / 3
        J  = 2*b*( s*(1-v) - (s+1)*(1 - v**2)/2 + (1 - v**3)/3 )
        return h2 , J

# main sweep over b ----------------------------------------------------------
b_vals  = np.concatenate( (np.linspace(0.20,1.00,90,endpoint=False),
                           np.linspace(1.00,4.00,90)) )
c_vals  = []
J_vals  = []

for b in b_vals:
    # vectorised classification of v–zones
    split = 0.5/b
    mask  = v_grid > split                  # plateau / no‑plateau flag
    h2, J = 0.0, 0.0
    for v,pl in zip(v_grid, mask):
        h2_i , J_i = one_v(pl, v, b)
        h2 += h2_i
        J  += J_i
    h2 *= dv
    J  *= dv
    c_vals.append( 6*a*h2 - 2 )
    J_vals.append( J )

c_vals = np.asarray(c_vals)
J_vals = np.asarray(J_vals)
order  = np.argsort(c_vals)
c_vals = c_vals[order]
J_vals = J_vals[order]

# >>> PLOT -------------------------------------------------------------------
plt.figure(figsize=(7,4))
plt.plot(c_vals, J_vals, lw=1.6)
plt.xlim(-0.5,1)
plt.ylim(-1,1.2)
plt.xlabel(r'$c$')
plt.ylabel(r'$\,\max J$')
plt.title(r'Maximal $J$ as a function of $c$   ($a=0.01$)')
plt.grid(True)
plt.tight_layout()
plt.show()
