class CopulaGraphs:
    def __init__(self, copula, add_params=True):
        self._copula = copula
        self._add_params = add_params

    def get_copula_title(self):
        title = f"{type(self._copula).__name__}"
        if not title.endswith("Copula"):
            title += " Copula"
        param_dict = {s: getattr(self._copula, s) for s in self._copula.intervals}
        if param_dict and self._add_params:
            param_dict_str = ", ".join([f"{k}={v}" for k, v in param_dict.items()])
            title += f" ({param_dict_str})"
        return title
