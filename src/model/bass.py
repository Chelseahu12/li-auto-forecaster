"""Bass diffusion curve fitted to months 1-3, projected for months 4-12."""
import numpy as np
from scipy.optimize import minimize


def _bass_cumulative(t: np.ndarray, M: float, p: float, q: float) -> np.ndarray:
    e = np.exp(-(p + q) * t)
    return M * (1 - e) / (1 + (q / p) * e)


def _bass_incremental(t: np.ndarray, M: float, p: float, q: float) -> np.ndarray:
    cum = _bass_cumulative(t, M, p, q)
    return np.diff(np.concatenate([[0], cum]))


class BassBaseline:
    """Fit Bass diffusion to early months and predict later months."""

    def __init__(self):
        self.p_: float = 0.01
        self.q_: float = 0.3
        self.M_: float = 100000.0

    def fit(self, early_sales: dict[int, float]) -> "BassBaseline":
        months = np.array(sorted(early_sales.keys()), dtype=float)
        observed = np.array([early_sales[int(m)] for m in months])

        def loss(params):
            M, p, q = params
            if M <= 0 or p <= 0 or q <= 0:
                return 1e12
            pred = _bass_incremental(months, M, p, q)
            return float(np.sum((pred - observed) ** 2))

        x0 = [max(observed.sum() * 10, 50000), 0.01, 0.3]
        result = minimize(loss, x0, method="Nelder-Mead",
                          options={"maxiter": 5000, "xatol": 1.0, "fatol": 1.0})
        self.M_, self.p_, self.q_ = result.x
        return self

    def predict(self, months: list[int]) -> list[float]:
        t = np.array(months, dtype=float)
        return [max(0.0, v) for v in _bass_incremental(t, self.M_, self.p_, self.q_).tolist()]
