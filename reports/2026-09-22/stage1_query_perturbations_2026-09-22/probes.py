"""Fixed-policy linear and additive nonlinear probes of query information."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer, StandardScaler


class Probe:
    def __init__(self, family, target, policy):
        self.family, self.target, self.policy = family, target, policy

    def basis(self, values, fit=False):
        if fit:
            self.scale = StandardScaler().fit(values)
        z = self.scale.transform(values)
        if self.family == "spline":
            if fit:
                self.spline = SplineTransformer(n_knots=self.policy["spline_knots"],
                                                degree=self.policy["spline_degree"],
                                                knots="quantile", include_bias=False).fit(z)
            z = self.spline.transform(z)
        return np.column_stack([np.ones(len(z)), z])

    def fit(self, values, t, y, u, v):
        b = self.basis(values, fit=True)
        if self.target == "effect":
            design = u[:, None] * b
            penalty = np.eye(b.shape[1]) * len(b) * self.policy["probe_effect_penalty_per_row"]
            penalty[0, 0] = 0
            self.coef = np.linalg.solve(design.T @ design + penalty, design.T @ v)
        else:
            self.model = LogisticRegression(C=self.policy["probe_logistic_C"], max_iter=1000,
                                            solver="lbfgs", tol=1e-7)
            self.model.fit(b[:, 1:], t if self.target == "treatment" else y)
        return self

    def predict(self, values):
        b = self.basis(values)
        if self.target == "effect":
            return b @ self.coef
        return self.model.predict_proba(b[:, 1:])[:, 1]


def loss(target, prediction, t, y, u, v):
    if target == "effect":
        return float(np.mean((v - u * prediction) ** 2))
    observed = t if target == "treatment" else y
    p = np.clip(prediction, 1e-8, 1 - 1e-8)
    return float(-np.mean(observed * np.log(p) + (1 - observed) * np.log1p(-p)))
