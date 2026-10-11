Custom copulas
==============

Copulas that are not among the built-in families can be defined from a
symbolic expression. The result is an ordinary copula object: numerical and
symbolic evaluation, dependence measures, sampling, approximations and plots
work exactly as for the built-in families. Expressions are given in SymPy
syntax; symbols other than the variables are treated as parameters.

From a cumulative distribution function
---------------------------------------

The Gumbel--Barnett copula :math:`C(u,v) = uv\exp(-\theta\ln u\ln v)`,
:math:`\theta\in[0,1]`, defined from its cdf:

.. code-block:: python

   >>> import copul as cp
   >>> family = cp.from_cdf("u*v*exp(-theta*log(u)*log(v))")
   >>> C = family(theta=0.5)
   >>> C.validate_copula()          # margins, groundedness, 2-increasingness
   True
   >>> family(theta=1.5).validate_copula()   # not a copula for theta > 1
   False
   >>> C.cdf(0.3, 0.6)
   0.13234958214968093
   >>> C.spearmans_rho(), C.chatterjees_xi()
   (-0.30528865791601767, 0.059792277050704)
   >>> family.cond_distr_1()        # symbolic, theta stays free
   -theta*v*exp(-theta*log(u)*log(v))*log(v) + v*exp(-theta*log(u)*log(v))

``validate_copula`` checks the defining properties numerically on a grid; pass
``return_details=True`` to see which check failed.

Expressions in more variables (``u1, u2, u3, ...``) define multivariate copulas,
and fixing variables at one gives their margins:

.. code-block:: python

   >>> cdf = "max((u1**(-theta) + u2**(-theta) + u3**(-theta) - 2), 0)**(-1/theta)"
   >>> clayton_3d = cp.from_cdf(cdf)
   >>> clayton_3d(u3=1).cdf()       # bivariate margin
   Max(0, -1 + v**(-theta) + u**(-theta))**(-1/theta)

From a density
--------------

.. code-block:: python

   >>> fgm = cp.from_pdf("1 + theta*(1 - 2*u)*(1 - 2*v)")   # Farlie-Gumbel-Morgenstern
   >>> fgm(theta=0.5).cdf(0.3, 0.6)
   0.20519999999999997

From an Archimedean generator
-----------------------------

``cp.from_generator`` takes a generator :math:`\varphi(t)` with at most one
parameter, which is exposed as ``theta``; ``theta_interval`` sets its admissible
values.

.. code-block:: python

   >>> clayton = cp.from_generator("(t**(-theta) - 1)/theta")
   >>> round(clayton(theta=2).kendalls_tau(), 10)
   0.5

From a Pickands dependence function
-----------------------------------

.. code-block:: python

   >>> tawn = cp.from_pickands("1 - theta*t*(1 - t)")   # Tawn's mixed model
   >>> tawn(theta=0.5).cdf(0.3, 0.7)
   0.24097356574242804

From a mass matrix or from data
-------------------------------

``cp.from_matrix`` returns a checkerboard copula with the given (not necessarily
normalised) cell masses, and ``cp.from_data`` (alias ``cp.from_samples``) the
empirical checkerboard copula of a sample; see :doc:`approximating_copulas`.

.. code-block:: python

   >>> P = [[0, 9, 1], [1, 0, 9], [9, 1, 0]]
   >>> ccop = cp.from_matrix(P)
   >>> ccop.cdf(0.2, 1), ccop.pdf(0.2, 1)
   (0.2, 0.3)

Approximations when no closed form is known
-------------------------------------------

Every copula can be turned into an approximating copula with exact formulas for
its measures, for instance ``C.to_checkerboard(20)``, ``C.to_bernstein(10)`` or
``C.to_shuffle_of_min(10)``:

.. code-block:: python

   >>> C.to_checkerboard(20).spearmans_rho()
   -0.30326337421915284
