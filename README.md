# copul

**copul** is a package designed for mathematical computation and visualization of bivariate copula families.

# Install

Install the copul library using pip.

```bash
pip install copul
```

# Documentation

A guide and documentation is available at [https://copul.readthedocs.io/](https://copul.readthedocs.io/).

## Copula properties
For any of the bivariate copula families specified below, e.g. `copula = copul.Galambos()`, get the following properties (if applicable):
* Cumulative distribution function via `copula.cdf`
* Density function via `copula.pdf`
* Conditional distribution function via `copula.cond_distr_1` and `copula.cond_distr_2`
* Data sampling from the copula via `copula.rvs`. The number of samples can be specified as an argument, e.g. `copula.rvs(1000)`

The following measures of association and dependence are also added if closed-forms are known:
* `copula.lambda_L` - lower tail dependence coefficient
* `copula.lambda_U` - upper tail dependence coefficient
* `copula.tau` - Kendall's tau	
* `copula.rho` - Spearman's rho
* `copula.xi` - Chatterjee's xi

## Supported copula families:

### Archimedean Copulas
The 22 Archimedean copula families from the book "Nelsen - An Introduction to Copulas", accessible via
`copul.archimedean.Nelsen1`, `copul.archimedean.Nelsen2`, etc.
Let `copula` be any instance of those classes, e.g. `copula = copul.archimedean.Nelsen1()`.

For these families, the following properties are available:
* generator function is available via e.g. `copula.generator`
* inverse generator function is available via e.g. `copula.inverse_generator`
* CI char function is available via e.g. `copula.ci_char`
* the MTP2 char function is available via e.g. `copula.mtp2_char`

### Extreme Value Copulas
* BB5
* Cuadras-Augé
* Galambos
* Gumbel
* Husler-Reiss
* Joe
* Marshall-Olkin
* tEV
* tawn

Let `copula` be any instance of those classes, e.g. `copula = copul.extreme_value.Galambos()`.
Then, the Pickands function is available via e.g. `copula.pickands`.

### Elliptical Copulas
* Gaussian
* Student-t
* Laplace

### Other
* Farlie-Gumbel-Morgenstern
* Fréchet
* Mardia
* Plackett
* Raftery
