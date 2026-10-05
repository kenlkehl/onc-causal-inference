"""Honest sparse-text DINA nuisance and effect fits for Stage 1.

A fresh encoder and three binary nuisance models are fitted inside each fold.
Effect-fold validation labels are never used to fit a nuisance or an effect.
Text columns are singleton groups; every level/missingness column of a supplied
clinical variable shares its feature group. Ridge nuisances do not select levels.
"""

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from ..models import dina


@dataclass
class TextDesign:
    vectorizer: object
    specs: list
    means: dict
    stds: dict
    names: list
    groups: list

    def transform(self, texts, explicit=None):
        x = self.vectorizer.transform(list(texts))
        if self.specs:
            from ..models.explicit_feature_featurizer import get_raw_explicit_features

            if explicit is None:
                raise ValueError("DINA prediction requires the fitted explicit feature inputs")
            z, _ = get_raw_explicit_features(
                explicit,
                self.specs,
                continuous_means=self.means,
                continuous_stds=self.stds,
                role=None,
            )
            x = sparse.hstack([x, sparse.csr_matrix(z)], format="csr")
        return x


def encode(texts, *, view=None, explicit=None, specs=None):
    from ..config import BoWViewConfig
    from .multi_model_agentic_forest import _make_bow_vectorizer, _bow_vectorizer_params

    # HTR-only runs use this fixed, explicitly named auxiliary nuisance view.
    view = view or BoWViewConfig(name="dina_auxiliary", min_df=1, max_df=1.0)
    vectorizer = _make_bow_vectorizer(_bow_vectorizer_params(view))
    x = vectorizer.fit_transform(list(texts))
    names = vectorizer.get_feature_names_out().tolist()
    groups = ["text:" + name for name in names]
    means, stds = {}, {}
    specs = list(specs or []) if explicit is not None else []
    if specs:
        from ..models.explicit_feature_featurizer import get_raw_explicit_features

        z, zn = get_raw_explicit_features(
            explicit, specs, continuous_means=means, continuous_stds=stds, role=None
        )
        x = sparse.hstack([x, sparse.csr_matrix(z)], format="csr")
        for name in zn:
            matching = [spec.name for spec in specs if name.startswith(spec.name + "_")]
            if not matching:
                raise ValueError(f"no parent feature for DINA column {name}")
            groups.append("explicit:" + max(matching, key=len))
        names.extend("explicit:" + name for name in zn)
    return x, TextDesign(vectorizer, specs, means, stds, names, groups)


def splits(treatment, folds, seed):
    t = np.asarray(treatment)
    counts = np.bincount(t.astype(int), minlength=2)
    n = min(int(folds), int(counts.min()))
    if n < 2:
        raise ValueError("DINA cross-fitting requires at least two rows in each treatment arm")
    return list(StratifiedKFold(n, shuffle=True, random_state=seed).split(t, t))


def subset(values, rows):
    return None if values is None else [values[int(i)] for i in rows]


@dataclass
class TextNuisance:
    encoder: TextDesign
    models: list

    def predict(self, texts, explicit=None):
        x = self.encoder.transform(texts, explicit)
        preds = [
            np.full(len(texts), float(m)) if np.isscalar(m) else m.predict_proba(x)[:, 1]
            for m in self.models
        ]
        return dina.nuisances(*preds)


def fit_nuisance(texts, t, y, *, view=None, explicit=None, specs=None, seed=0):
    x, encoder = encode(texts, view=view, explicit=explicit, specs=specs)
    t, y = np.asarray(t), np.asarray(y)
    models = []
    for values, keep in ((t, np.ones(len(t), bool)), (y, t == 0), (y, t == 1)):
        if not keep.any():
            raise ValueError("DINA nuisance fit lacks a treatment arm")
        target = values[keep]
        if len(np.unique(target)) == 1:
            models.append(float(target.mean()))
        else:
            model = LogisticRegression(
                C=float(getattr(view, "logistic_c", 1.0)),
                max_iter=int(getattr(view, "logistic_max_iter", 1000)),
                solver="lbfgs",
                random_state=seed,
            )
            model.fit(x[keep], target)
            models.append(model)
    return TextNuisance(encoder, models)


def crossfit_nuisance(texts, t, y, *, folds=3, seed=0, view=None, explicit=None, specs=None):
    t, y = np.asarray(t), np.asarray(y)
    oof = {k: np.full(len(t), np.nan) for k in ("e", "mu0", "mu1", "a", "nu")}
    for fit, hold in splits(t, folds, seed):
        model = fit_nuisance(
            subset(texts, fit),
            t[fit],
            y[fit],
            view=view,
            explicit=subset(explicit, fit),
            specs=specs,
            seed=seed,
        )
        pred = model.predict(subset(texts, hold), subset(explicit, hold))
        for k in oof:
            oof[k][hold] = pred[k]
    full = fit_nuisance(texts, t, y, view=view, explicit=explicit, specs=specs, seed=seed)
    return oof, full


def fit_effect(
    texts,
    t,
    y,
    test_texts,
    *,
    folds=3,
    seed=0,
    view=None,
    explicit=None,
    test_explicit=None,
    specs=None,
):
    """One effect training set, with nested nuisances; no test labels accepted."""
    nt, nuisance = crossfit_nuisance(
        texts, t, y, folds=folds, seed=seed, view=view, explicit=explicit, specs=specs
    )
    nv = nuisance.predict(test_texts, test_explicit)
    x, encoder = encode(texts, view=view, explicit=explicit, specs=specs)
    xv = encoder.transform(test_texts, test_explicit)
    model = dina.fit(
        x,
        y,
        t,
        nt["a"],
        nt["nu"],
        groups=encoder.groups,
        regularization=float(getattr(view, "ridge_alpha", 10.0)) / len(t),
    )
    delta = model.predict(xv)
    mu0, mu1 = dina.counterfactuals(delta, nv["a"], nv["nu"])
    return {
        "tau": mu1 - mu0,
        "delta": delta,
        "mu0": mu0,
        "mu1": mu1,
        "model": model,
        "encoder": encoder,
        "training_nuisance": nt,
        "validation_nuisance": nv,
        "nuisance_model": nuisance,
    }


def crossfit_effect(
    texts, t, y, *, folds=3, seed=0, view=None, explicit=None, specs=None, test_texts=None
):
    t, y = np.asarray(t), np.asarray(y)
    test_texts = list(test_texts or [])
    result = {key: np.full(len(t), np.nan) for key in ("tau", "delta", "loss", "mu0", "mu1")}
    external = []
    for fit, hold in splits(t, folds, seed):
        prediction_texts = [*subset(texts, hold), *test_texts]
        if explicit is not None and test_texts:
            raise ValueError("explicit external predictions require the fit_effect API")
        f = fit_effect(
            subset(texts, fit),
            t[fit],
            y[fit],
            prediction_texts,
            folds=folds,
            seed=seed,
            view=view,
            explicit=subset(explicit, fit),
            test_explicit=subset(explicit, hold),
            specs=specs,
        )
        for key in ("tau", "delta", "mu0", "mu1"):
            result[key][hold] = f[key][: len(hold)]
        nv = f["validation_nuisance"]
        result["loss"][hold] = dina.loss(
            y[hold], t[hold], nv["a"][: len(hold)], nv["nu"][: len(hold)], f["delta"][: len(hold)]
        )
        external.append(f["tau"][len(hold) :])
    result["test_tau"] = np.mean(external, axis=0)
    return result


def prepare_neural_nuisance(runner, model, df, positions, seed):
    """Freeze independently cross-fitted nuisances for a neural effect fold."""
    positions = np.asarray(positions, int)
    train = df.iloc[positions]
    texts = train[runner.config.text_column].astype(str).tolist()
    dn, fitted = crossfit_nuisance(
        texts,
        train[runner.config.treatment_column].to_numpy(),
        train[runner.config.outcome_column].to_numpy(),
        folds=runner.avf_config.nuisance_folds,
        seed=seed,
    )
    model.dina_nuisance = fitted
    fields = {key: np.zeros(len(df), np.float32) for key in ("dina_a", "dina_nu", "dina_e")}
    for key in fields:
        fields[key][positions] = dn[key[5:]]
    return fields


def neural_predictions(runner, model, df, delta):
    dn = model.dina_nuisance.predict(df[runner.config.text_column].astype(str).tolist())
    mu0, mu1 = dina.counterfactuals(delta, dn["a"], dn["nu"])
    return {**dn, "tau": mu1 - mu0, "delta": np.asarray(delta), "mu0": mu0, "mu1": mu1}
