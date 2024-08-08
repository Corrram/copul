# copulas_in_systemic_risk

## Copula properties
For any of the copula families below, e.g. `copula = copul.Galambos()`, get the following properties:
* Cumulative distribution function via copula.cdf
* Density function via copula.pdf

## Supported copula families:

### Archimedean Copulas
The 22 copula families from Nelsen, accessible via
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
Then, the Pickand function is available via e.g. `copula.pickand`.

### Other
* Farlie-Gumbel-Morgenstern
* Fréchet
* Mardia
* Plackett
* Raftery

## Sample Usage
```
import copul

galambos = copul.extreme_value.Galambos()
params = galambos.sample_parameters(3)
galambos.plot_pickand(params)

clayton = copul.archimedean.Clayton()
clayton(theta=1.5).plot_cdf()
clayton(theta=2.5).plot_pdf()
```
