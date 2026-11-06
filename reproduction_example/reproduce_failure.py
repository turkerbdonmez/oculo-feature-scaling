"""Refit the seed-44, draw-13 scaling example on its bundled frozen features."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import warnings

import numpy as np
import scipy
from scipy.special import expit
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
TOL = {"rtol": 1e-9, "atol": 5e-10}


def checked_fixture():
    provenance = json.loads((HERE / "provenance.json").read_text())
    for name, expected in provenance["files_sha256"].items():
        with (HERE / name).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Fixture digest mismatch for {name}")
    return provenance


def metrics(y, raw):
    probability = expit(raw)
    return {"raw_auroc": float(roc_auc_score(y, raw)),
            "legacy_sigmoid_auroc": float(roc_auc_score(y, probability)),
            "sigmoid_exact_one": int(np.count_nonzero(probability == 1))}


def safe_sd(variance):
    return np.sqrt(np.where(variance > 0, variance, 1.0))


def run(inputs, reference, settings, coordinate):
    support = inputs["target_support_features"]
    support_ids = inputs["target_support_ids"]
    labels = inputs["target_support_labels"]
    assert support.shape == (8, 512) and np.bincount(labels).tolist() == [4, 4]
    assert len(np.unique(support_ids)) == 8
    assert not np.intersect1d(support_ids, inputs["erdes_test_ids"]).size
    assert not np.intersect1d(inputs["source_training_ids"], inputs["oculo_test_ids"]).size
    budget = len(labels)
    weights = budget / (2 * np.bincount(labels)[labels])
    draw_scaler = StandardScaler().fit(support, sample_weight=weights)
    source_scaler = StandardScaler().fit(inputs["source_training_features"])
    moments = {"draw_mean": draw_scaler.mean_, "draw_variance": draw_scaler.var_,
               "source_mean": source_scaler.mean_, "source_variance": source_scaler.var_}
    for name, values in moments.items():
        np.testing.assert_allclose(values, reference[name], **TOL, err_msg=name)

    c, alpha = settings["C"], settings["alpha"]
    source_coef = inputs["source_coefficient"]
    source_intercept = float(inputs["source_intercept"])
    source_predictions = {}
    test_cohorts = {
        "oculo": (inputs["oculo_test_features"], inputs["oculo_test_labels"]),
        "erdes": (inputs["erdes_test_features"], inputs["erdes_test_labels"]),
    }
    for domain, (features, y) in test_cohorts.items():
        raw = features @ source_coef + source_intercept
        np.testing.assert_allclose(raw, reference[f"{domain}_direct_logits"], **TOL)
        source_predictions[domain] = metrics(y, raw)
        moments[f"{domain}_direct_logits"] = raw

    scaling = {"draw": (draw_scaler.mean_, draw_scaler.scale_),
               "source": (source_scaler.mean_, source_scaler.scale_)}
    for name in settings["variants"][2:]:
        epsilon = float(name.split("_")[1])
        floor = np.maximum(draw_scaler.var_, epsilon * source_scaler.var_)
        scaling[name] = (draw_scaler.mean_, safe_sd(floor))
    index = coordinate["coordinate_index_zero_based"]
    epsilon = coordinate["floor_fraction"]
    assert (coordinate["seed"], coordinate["label_budget"], coordinate["draw_id"],
            coordinate["C"], coordinate["alpha"]) == (settings["seed"], budget, settings["draw"], c, alpha)
    assert index == 177
    coordinate_scale = draw_scaler.scale_.copy()
    coordinate_scale[index] = safe_sd(np.maximum(draw_scaler.var_[index], epsilon * source_scaler.var_[index]))
    assert np.count_nonzero(coordinate_scale != draw_scaler.scale_) == 1
    scaling["coordinate_177_only"] = (draw_scaler.mean_, coordinate_scale)

    report = {"seed": settings["seed"], "budget": budget, "draw": settings["draw"],
              "C": c, "alpha": alpha, "support_ids": support_ids.tolist(),
              "source_training_n": len(inputs["source_training_features"]),
              "software": {"python": platform.python_version(), "numpy": np.__version__,
                           "scipy": scipy.__version__, "scikit-learn": sklearn.__version__},
              "direct": source_predictions, "variants": {},
              "heldout_scope": settings["scope"]}
    outputs = dict(moments, source_coefficient=source_coef, source_intercept=np.asarray(source_intercept))
    for name, (mean, scale) in scaling.items():
        model = LogisticRegression(C=c, solver="lbfgs", max_iter=1000, tol=1e-4, random_state=20261105)
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=ConvergenceWarning)
            model.fit((support - mean) / scale, labels, sample_weight=weights)
        beta, intercept = model.coef_[0], float(model.intercept_[0])
        raw_coef = beta / scale
        raw_intercept = intercept - raw_coef @ mean
        anchor_coef = (1 - alpha) * source_coef + alpha * raw_coef
        anchor_intercept = (1 - alpha) * source_intercept + alpha * raw_intercept
        iterations = int(model.n_iter_[0])
        values = {
            f"{name}_standardized_coefficient": beta, f"{name}_standardized_intercept": np.asarray(intercept),
            f"{name}_raw_coefficient": raw_coef, f"{name}_raw_intercept": np.asarray(raw_intercept),
            f"{name}_anchor_coefficient": anchor_coef, f"{name}_anchor_intercept": np.asarray(anchor_intercept),
            f"{name}_mean": mean, f"{name}_scale": scale,
        }
        result = {
            "converged": True, "iterations": iterations, "coordinate_177_sd": float(scale[index]),
            "coordinate_177_standardized_coefficient": float(beta[index]),
            "coordinate_177_raw_coefficient": float(raw_coef[index]),
            "coordinate_177_fraction_of_squared_target_slope": float(raw_coef[index] ** 2 / (raw_coef @ raw_coef)),
            "target_slope_norm": float(np.linalg.norm(raw_coef)), "domains": {},
        }
        for domain, (features, y) in test_cohorts.items():
            raw = features @ raw_coef + raw_intercept
            anchored = features @ anchor_coef + anchor_intercept
            values[f"{name}_{domain}_target_logits"] = raw
            values[f"{name}_{domain}_anchor_logits"] = anchored
            result["domains"][domain] = {"target": metrics(y, raw), "anchor": metrics(y, anchored)}
        for key, value in values.items():
            if key in reference:
                np.testing.assert_allclose(value, reference[key], **TOL, err_msg=key)
        if name in settings["variants"]:
            assert iterations == reference[f"{name}_iterations"].item()
        report["variants"][name] = result
        outputs.update(values)
    return report, outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional directory for the refit summary and raw model outputs")
    args = parser.parse_args()
    settings = checked_fixture()
    coordinate = json.loads((HERE / "coordinate_intervention_protocol.json").read_text())
    with (np.load(HERE / "fixture.npz", allow_pickle=False) as inputs,
          np.load(HERE / "reference_outputs.npz", allow_pickle=False) as reference):
        report, outputs = run(inputs, reference, settings, coordinate)
    print("Raw-logit AUROC                       Oculo       ERDES")
    print(f"{'Frozen source':<36} {report['direct']['oculo']['raw_auroc']:.6f}    {report['direct']['erdes']['raw_auroc']:.6f}")
    for name, result in report["variants"].items():
        values = [result["domains"][domain]["anchor"]["raw_auroc"] for domain in ("oculo", "erdes")]
        print(f"{name:<36} {values[0]:.6f}    {values[1]:.6f}")
    print("Verified saved feature moments, five fits, and raw scores against the frozen references.")
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "refit_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        np.savez_compressed(args.output / "refitted_outputs.npz", **outputs)


if __name__ == "__main__":
    main()
