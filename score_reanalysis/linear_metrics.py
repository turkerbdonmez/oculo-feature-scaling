"""Discrimination, score-quality and fixed-threshold descriptive metrics."""

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_curve


def auc_rows(labels, scores):
    positive = labels == 1
    count = int(positive.sum())
    rank_sum = rankdata(scores, axis=-1, method="average")[..., positive].sum(axis=-1)
    return (rank_sum - count * (count + 1) / 2) / (count * (len(labels) - count))


def operating_point(labels, scores, cutoff):
    predicted = scores >= cutoff
    positive, negative = labels == 1, labels == 0
    true_positive = int((positive & predicted).sum())
    true_negative = int((negative & ~predicted).sum())
    return {"cutoff": float(cutoff), "sensitivity": true_positive / int(positive.sum()),
            "specificity": true_negative / int(negative.sum()),
            "tp": true_positive, "tn": true_negative, "fp": int(negative.sum()) - true_negative,
            "fn": int(positive.sum()) - true_positive}


def describe(labels, scores, thresholds, ranking_scores=None):
    decision = scores if ranking_scores is None else ranking_scores
    positives = int(labels.sum())
    result = {"n": len(labels), "rd": positives, "prevalence": positives / len(labels),
              "auroc": float(auc_rows(labels, decision)),
              "probability_auroc": float(auc_rows(labels, scores)),
              "average_precision": float(average_precision_score(labels, decision)),
              "probability_average_precision": float(average_precision_score(labels, scores)),
              "probability_zero": int((scores == 0).sum()), "probability_one": int((scores == 1).sum()),
              "brier": float(np.mean((scores - labels) ** 2))}
    for name, cutoff in thresholds.items():
        result.update({f"{name}_{field}": value for field, value in operating_point(labels, scores, cutoff).items()})
    fpr, tpr, cutoff = roc_curve(labels, decision, drop_intermediate=False)
    for nominal in (0.90, 0.95):
        first = int(np.flatnonzero(tpr >= nominal)[0])
        prefix = f"descriptive_test_roc_sensitivity_{round(100 * nominal)}"
        result[f"{prefix}_specificity"] = float(1 - fpr[first])
        result[f"{prefix}_attained_sensitivity"] = float(tpr[first])
        result[f"{prefix}_test_derived_ranking_cutoff"] = float(cutoff[first])
    return result


def concordance(labels, scores):
    positive = scores[..., labels == 1, None]
    negative = scores[..., None, labels == 0]
    return (positive > negative) + 0.5 * (positive == negative)


def draw_bootstrap_weights(labels, n_draws, repeats, seed):
    rng = np.random.default_rng(seed)
    counts = (int(labels.sum()), int((labels == 0).sum()), n_draws)
    return tuple(rng.multinomial(count, np.full(count, 1 / count), size=repeats) / count for count in counts)


def auc_percentile_interval(tables, weights):
    positive, negative, draw = weights
    samples = sum(np.sum((positive @ table) * negative, axis=1) * draw[:, i] for i, table in enumerate(tables))
    return np.quantile(samples, (0.025, 0.975)).tolist()
