"""Evaluation metrics and incremental-value test."""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    """Mean pinball (quantile) loss."""
    errors = y_true - y_pred
    return float(np.mean(np.where(errors >= 0, q * errors, (q - 1) * errors)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error (%)."""
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def interval_coverage(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
) -> float:
    """Fraction of true values falling inside [y_lower, y_upper]."""
    inside = (y_true >= y_lower) & (y_true <= y_upper)
    return float(inside.mean())


def _orthogonalize(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Residualize X on Z (project out Z's influence from X)."""
    reg = LinearRegression().fit(Z, X)
    return X - reg.predict(Z)


def incremental_value_test(
    feature_groups: dict[str, np.ndarray],
    y: np.ndarray,
    n_estimators: int = 100,
    random_state: int = 42,
) -> dict[str, dict]:
    """For each group G, orthogonalize G against others, measure pinball drop.

    Args:
        feature_groups: dict mapping group name → 2D array (n_samples, n_features)
        y: target array (n_samples, n_months)

    Returns:
        dict mapping group name → {"delta_pinball": float, "full_pinball": float,
                                    "reduced_pinball": float}
    """
    group_names = list(feature_groups.keys())
    X_full = np.concatenate(list(feature_groups.values()), axis=1)

    def _fit_predict_median(X_tr: np.ndarray) -> np.ndarray:
        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        rf.fit(X_tr, y)
        tree_preds = np.stack([t.predict(X_tr) for t in rf.estimators_], axis=0)
        return np.quantile(tree_preds, 0.5, axis=0)

    full_median = _fit_predict_median(X_full)
    full_pinball = pinball_loss(y.ravel(), full_median.ravel(), q=0.5)

    results = {}
    for target_group in group_names:
        other_names = [g for g in group_names if g != target_group]
        Z = np.concatenate([feature_groups[g] for g in other_names], axis=1)
        G_ortho = _orthogonalize(feature_groups[target_group], Z)

        X_reduced = np.concatenate(
            [feature_groups[g] for g in other_names] + [G_ortho], axis=1
        )
        reduced_median = _fit_predict_median(X_reduced)
        reduced_pinball = pinball_loss(y.ravel(), reduced_median.ravel(), q=0.5)

        results[target_group] = {
            "full_pinball": full_pinball,
            "reduced_pinball": reduced_pinball,
            "delta_pinball": full_pinball - reduced_pinball,
        }
    return results
