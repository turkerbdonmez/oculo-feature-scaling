"""Regenerate Round 2 scaling contrasts, robustness tables and paired Brier estimates."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BUDGETS = (8, 16, 32)
SCALERS = {"draw": "Original scaling", "floor_0.001": r"0.1\% floor", "floor_0.01": r"1\% floor",
           "floor_0.1": r"10\% floor", "source": "Source-fixed"}
METHODS = ("floor_0.01", "source")
DOMAINS = ("oculo", "erdes")
SEED_DRAWS = [(seed, draw) for seed in (42, 43, 44) for draw in range(1, 16)]
TARGET = "retinal_detachment"


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def score_rows(index, budget, design="balanced", scaler="draw", reference=None):
    keep = index["budget"].eq(budget) & index["draw_design"].eq(design)
    if reference == "direct":
        keep &= index["model"].eq("direct_mean_probability")
    else:
        alpha = 1.0 if reference == "target" else 0.25
        keep &= (index["model"].eq("target_fit") & index["scaler"].eq(scaler) & index["C"].eq(0.3)
                 & index["alpha"].eq(alpha) & index["weight_policy"].eq("target_mass")
                 & index["target_logit_factor"].isna())
    selected = index.loc[keep].sort_values(["seed", "draw_id"])
    if list(zip(selected["seed"], selected["draw_id"], strict=True)) != SEED_DRAWS:
        raise ValueError(f"Incorrect run membership for {budget}/{design}/{scaler}/{reference}")
    return selected["score_row"].to_numpy()


def take_scores(archive, domain, rows):
    matrix = archive[domain][rows]
    return matrix.reshape(3, 15, matrix.shape[-1])


def case_inputs(directory, indexes, protocol):
    cases = {domain: pd.read_csv(directory / f"{domain}_cases.csv") for domain in DOMAINS}
    for domain, table in cases.items():
        expected = pd.read_csv(indexes / f"{domain}_test_index.csv")
        pd.testing.assert_frame_equal(table, expected)
    views = {domain: (domain, np.ones(len(cases[domain]), dtype=bool)) for domain in DOMAINS}
    views["oculo_me_positive"] = ("oculo", cases["oculo"]["membranous_echo"].eq(1).to_numpy())
    labels = {view: cases[domain].loc[keep, TARGET].to_numpy(dtype=int)
              for view, (domain, keep) in views.items()}
    for view, y in labels.items():
        expected = protocol["test_sets"][view]
        if len(y) != expected.get("cases", expected.get("clips")) or int(y.sum()) != expected["rd"]:
            raise ValueError(f"Unexpected case or class count for {view}")
    return cases, views, labels


def auroc_contrast(candidate, baseline, weights, metrics):
    difference = candidate - baseline
    ci = metrics.auc_percentile_interval(difference, weights)
    return {"candidate_mean": float(candidate.mean()), "reference_mean": float(baseline.mean()),
            "difference": float(difference.mean()), "ci_lower": ci[0], "ci_upper": ci[1]}


def brier_contrast(candidate, baseline, labels, weights):
    candidate_loss = ((candidate - labels) ** 2).mean(axis=0)
    baseline_loss = ((baseline - labels) ** 2).mean(axis=0)
    difference = candidate_loss - baseline_loss
    positive, negative, draw = weights
    mask = labels == 1
    positive_loss = positive @ difference[:, mask].T
    negative_loss = negative @ difference[:, ~mask].T
    per_draw = mask.mean() * positive_loss + (1 - mask.mean()) * negative_loss
    ci = np.quantile(np.sum(draw * per_draw, axis=1), (0.025, 0.975))
    return {"candidate_mean": float(candidate_loss.mean()), "reference_mean": float(baseline_loss.mean()),
            "difference": float(difference.mean()), "ci_lower": float(ci[0]), "ci_upper": float(ci[1])}


def derive(index, raw, probability, views, labels, metrics, protocol):
    boot = protocol["bootstrap"]
    weights = {view: metrics.draw_bootstrap_weights(y, 15, boot["replicates"], boot["seed"])
               for view, y in labels.items()}
    low, stability, subgroup, brier = [], [], [], []
    for budget in BUDGETS:
        original = score_rows(index, budget)
        direct = score_rows(index, budget, reference="direct")
        source = take_scores(raw, "oculo", direct)
        if not np.array_equal(source, np.broadcast_to(source[:, :1], source.shape)):
            raise ValueError("Frozen Oculo source scores unexpectedly depend on draw")
        source_auc = metrics.auc_rows(labels["oculo"], source)
        for design, scalers in (("balanced", SCALERS), ("unstratified", ("draw", *METHODS))):
            for scaler in scalers:
                rows = score_rows(index, budget, design, scaler)
                auc = {domain: metrics.auc_rows(labels[domain], take_scores(raw, domain, rows))
                       for domain in DOMAINS}
                stability.append({"budget": budget, "design": design, "scaler": scaler, "runs": len(rows),
                                  "oculo_mean": float(auc["oculo"].mean()), "erdes_mean": float(auc["erdes"].mean()),
                                  "oculo_min": float(auc["oculo"].min()),
                                  "oculo_below_0_50": int((auc["oculo"] < 0.5).sum()),
                                  "oculo_drop_gt_0_02": int((auc["oculo"] - source_auc < -0.02).sum())})
        for view, (domain, keep) in views.items():
            if view in DOMAINS and budget == 32:
                continue
            y = labels[view]
            baseline = metrics.concordance(y, take_scores(raw, domain, original)[..., keep]).mean(0)
            comparisons = {"draw": baseline}
            if view == "oculo_me_positive":
                comparisons["frozen_source"] = metrics.concordance(y, source[..., keep]).mean(0)
            for scaler in METHODS:
                current = take_scores(raw, domain, score_rows(index, budget, scaler=scaler))[..., keep]
                candidate = metrics.concordance(y, current).mean(0)
                for name, reference in comparisons.items():
                    table = subgroup if view == "oculo_me_positive" else low
                    table.append({"budget": budget, "view": view, "scaler": scaler, "reference": name,
                                  **auroc_contrast(candidate, reference, weights[view], metrics)})
        if budget == 32:
            target = score_rows(index, budget, reference="target")
            for domain in DOMAINS:
                current = take_scores(probability, domain, original)
                for name, rows in (("direct", direct), ("target_only", target)):
                    baseline = take_scores(probability, domain, rows)
                    brier.append({"budget": budget, "view": domain, "scaler": "draw", "reference": name,
                                  **brier_contrast(current, baseline, labels[domain], weights[domain])})
    return {"low_budget_auroc": low, "stability": stability, "me_positive_auroc": subgroup, "brier": brier}


def read_modern(report, focal):
    lookup = {(row["view"], row["model"], row["reference"]): row
              for row in focal["intervals"] if row["budget"] == 32}
    table = []
    for scaler, arm, reference in (("draw", "legacy_anchor", "direct_probability"),
                                   ("source", "source_scaled_anchor", "source_scaled_two_head_oracle")):
        contrast = lookup["oculo", arm, reference]
        table.append({"encoder": "ResNet-18", "training": "Oculo fine-tuned", "scaler": scaler,
                      "oculo_source": lookup["oculo", "direct_probability", None]["auroc"],
                      "oculo_anchor": lookup["oculo", arm, None]["auroc"], "difference": contrast["delta_auroc"],
                      "ci_lower": contrast["ci_95"][0], "ci_upper": contrast["ci_95"][1],
                      "erdes_source": lookup["erdes", "direct_probability", None]["auroc"],
                      "erdes_anchor": lookup["erdes", arm, None]["auroc"]})
    for model, name in (("usfm", "USFM"), ("convnext_tiny", "ConvNeXt-Tiny"),
                        ("resnet18_imagenet", "ImageNet ResNet-18")):
        for scaler, suffix in (("draw", ""), ("source", "_source_fixed")):
            arm = report["arms"][model + suffix]
            oculo = arm["oculo"]["32"]
            erdes = arm["erdes"]["32"]
            contrast = oculo["paired_95pct"]["source_logit_pool"]
            means = oculo["mean_run_auroc"]
            if (not np.isclose(contrast["candidate"], means["anchored"], atol=1e-12, rtol=0)
                    or not np.isclose(contrast["baseline"], means["source_logit_pool"], atol=1e-12, rtol=0)):
                raise ValueError(f"Inconsistent archived modern representation means for {model}/{scaler}")
            table.append({"encoder": name, "training": "Frozen + source probe", "scaler": scaler, "oculo_source": means["source_logit_pool"],
                          "oculo_anchor": means["anchored"], "difference": contrast["delta"],
                          "ci_lower": contrast["ci_95"][0], "ci_upper": contrast["ci_95"][1],
                          "erdes_source": erdes["mean_run_auroc"]["source_probability_pool"],
                          "erdes_anchor": erdes["mean_run_auroc"]["anchored"]})
    return table


def agreement_with_round1(tables, directory):
    prior = pd.read_csv(directory / "summary_metrics.csv")
    for row in tables["stability"]:
        match = prior.loc[prior["draw_design"].eq(row["design"]) & prior["budget"].eq(row["budget"])
                          & prior["model"].eq("target_fit") & prior["scaler"].eq(row["scaler"])
                          & prior["C"].eq(0.3) & prior["alpha"].eq(0.25)]
        if len(match) != 1:
            raise ValueError(f"Missing independent Round 1 summary for {row}")
        actual = [row[key] for key in ("oculo_mean", "erdes_mean", "oculo_min", "oculo_below_0_50", "oculo_drop_gt_0_02")]
        expected = match.iloc[0][["oculo_auroc", "erdes_auroc", "oculo_auc_min", "oculo_auroc_below_0_5",
                                  "oculo_source_drop_gt_0_02"]].to_numpy(dtype=float)
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)


def fmt(value):
    return f"{value:.4f}"


def interval(row):
    def signed(value):
        return f"{value:+.5f}" if 0 < abs(value) < 0.00005 else f"{value:+.4f}"

    return rf"${signed(row['difference'])}\ [{signed(row['ci_lower'])}, {signed(row['ci_upper'])}]$"


def tex_table(directory, name, label, title, spec, headers, rows, note, font=r"\small", column_padding=4):
    lines = [r"\begin{table}[!htbp]", r"\centering" + font, rf"\setlength{{\tabcolsep}}{{{column_padding}pt}}",
             rf"\caption{{{title}}}", rf"\label{{{label}}}", rf"\begin{{tabular}}{{@{{}}{spec}@{{}}}}",
             r"\toprule", " & ".join(headers) + r" \\", r"\midrule"]
    lines += [" & ".join(map(str, row)) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip", r"\begin{minipage}{\linewidth}\footnotesize",
              note, r"\end{minipage}", r"\end{table}"]
    (directory / name).write_text("\n".join(lines) + "\n")


def write_tex(output, tables):
    low = {(r["budget"], r["scaler"], r["view"]): r for r in tables["low_budget_auroc"]}
    rows = []
    for budget in (8, 16):
        for scaler in METHODS:
            o, e = [low[budget, scaler, domain] for domain in DOMAINS]
            rows.append((budget, SCALERS[scaler], fmt(o["candidate_mean"]), interval(o), fmt(e["candidate_mean"]), interval(e)))
    baselines = ["/".join(fmt(low[budget, METHODS[0], domain]["reference_mean"]) for domain in DOMAINS)
                 for budget in (8, 16)]
    tex_table(output, "tab_round2_low_budget.tex", "tab:r2-low-budget", "Paired low-budget effects of scaling on AUROC.",
              "rlrrrr", ("Labels", "Scaling", "Oculo mean", r"Oculo $\Delta$ [95\% CI]", "ERDES mean", r"ERDES $\Delta$ [95\% CI]"), rows,
              rf"Differences are candidate minus original interpolation. Original Oculo/ERDES means were {baselines[0]} with 8 labels and {baselines[1]} with 16. The 5,000 paired replicates resample matched draws and test items within class across the three fixed encoders.",
              font=r"\footnotesize")
    stability = {(r["budget"], r["design"], r["scaler"]): r for r in tables["stability"]}
    rows = []
    for design in ("balanced", "unstratified"):
        for scaler in ("draw", *METHODS):
            cells = [stability[budget, design, scaler] for budget in (8, 16)]
            rows.append((design.capitalize() if scaler == "draw" else "", SCALERS[scaler],
                         *(fmt(r["oculo_min"]) + f" ({r['oculo_below_0_50']}/{r['oculo_drop_gt_0_02']})" for r in cells)))
    tex_table(output, "tab_round2_minima.tex", "tab:r2-minima", "Minimum source AUROC and severe losses in low-budget fits.",
              "llrr", ("Draw design", "Scaling", "8 labels", "16 labels"), rows,
              r"Cells give the minimum Oculo AUROC, followed by the count below 0.50 and the count with a drop $>0.02$ versus the seed-matched source head. Each condition includes 45 fits with shared draws across three fixed encoders. The counts are descriptive.")
    rows = [(r["budget"], SCALERS[r["scaler"]], fmt(r["oculo_mean"]), fmt(r["erdes_mean"]), fmt(r["oculo_min"]),
             r["oculo_below_0_50"], r["oculo_drop_gt_0_02"]) for r in tables["stability"]
            if r["design"] == "balanced" and r["scaler"] != "source"]
    tex_table(output, "tab_round2_floors.tex", "tab:r2-floors", "Variance-floor sensitivity with balanced draws.",
              "rlrrrrr", ("Labels", "Scaling", "Oculo mean", "ERDES mean", "Oculo min", r"AUROC $<0.50$", r"Drop $>0.02$"), rows,
              r"Means and minima are for AUROC. A drop compares to the seed-matched source AUROC on the same cases. Each cell uses 45 fits. All three floors were retained as sensitivity analyses. The unstratified fits evaluated the 1\% floor only.")
    me = {(r["budget"], r["scaler"], r["reference"]): r for r in tables["me_positive_auroc"]}
    rows = []
    for budget in BUDGETS:
        old = me[budget, METHODS[0], "draw"]
        source = me[budget, METHODS[0], "frozen_source"]
        rows.append((budget, "Reference means", f"{fmt(source['reference_mean'])} / {fmt(old['reference_mean'])}", "--", "--"))
        for scaler in METHODS:
            old, source = me[budget, scaler, "draw"], me[budget, scaler, "frozen_source"]
            rows.append((budget, SCALERS[scaler], fmt(old["candidate_mean"]), interval(source), interval(old)))
    tex_table(output, "tab_round2_me_positive.tex", "tab:r2-me-positive", "Scaling interventions on the Oculo ME+ subgroup.",
              "rlrrr", ("Labels", "Method", "AUROC mean", r"$\Delta$ vs. source [95\% CI]", r"$\Delta$ vs. original [95\% CI]"), rows,
              r"The reference row gives the frozen source and original interpolated means, in that order. The post hoc subgroup contains 28 RD and 23 non-RD images. Intervals use the matched case/draw bootstrap on this subgroup.")
    rows = [("Oculo" if r["view"] == "oculo" else "ERDES",
             ("Frozen source" if r["view"] == "oculo" else "Direct transfer") if r["reference"] == "direct" else "Target-only",
             fmt(r["reference_mean"]), fmt(r["candidate_mean"]), interval(r)) for r in tables["brier"]]
    tex_table(output, "tab_round2_brier.tex", "tab:r2-brier", "Paired 32-label Brier loss contrasts.", "llrrr",
              ("Test", "Reference", "Reference loss", "Original anchor loss", r"$\Delta$ [95\% CI]"), rows,
              r"Negative differences favor the anchor. The matched case/draw bootstrap preserves each test set's observed class prevalence. Brier loss summarizes probability error and alone does not isolate calibration.")
    modern = {(r["encoder"], r["scaler"]): r for r in tables["modern_32"]}
    rows = []
    for encoder in dict.fromkeys(r["encoder"] for r in tables["modern_32"]):
        draw, fixed = (modern[encoder, scaler] for scaler in ("draw", "source"))
        rows.append((encoder, fixed["training"], fmt(fixed["oculo_source"]), fmt(draw["oculo_anchor"]),
                     fmt(fixed["oculo_anchor"]), interval(fixed),
                     f"{fmt(draw['erdes_anchor'])} / {fmt(fixed['erdes_anchor'])}"))
    tex_table(output, "tab_round2_modern_32.tex", "tab:r2-modern-32", "Oculo retention across representations at 32 labels.",
              "llrrrrr", ("Representation", "Source training", r"\shortstack[r]{Oculo\\source}",
                          r"\shortstack[r]{Oculo\\draw}", r"\shortstack[r]{Oculo\\fixed}",
                          r"\shortstack[r]{Oculo fixed\\$\Delta$ [95\% CI]}", r"\shortstack[r]{ERDES\\draw / fixed}"), rows,
              r"Entries are AUROC. Draw uses target draw scaling and fixed uses Oculo training scales. Differences compare the source-fixed adapted head with its own source head. The three frozen alternatives use one checkpoint and a fitted linear Oculo head. The Oculo fine-tuned row uses the three primary source encoders. All use the same 15 target draws.",
              font=r"\footnotesize", column_padding=3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--linear-input", type=Path, default=ROOT / "revision_round1/linear/cached_results")
    parser.add_argument("--modern-input", type=Path, default=ROOT / "revision_round1/imaging/modern_sensitivity_results.json")
    parser.add_argument("--test-indexes", type=Path, default=ROOT / "manuscript_preparation/data")
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    sys.path.insert(0, str(args.linear_input.resolve().parent))
    import linear_metrics as metrics

    protocol_path = HERE / "protocol.json"
    protocol = read_json(protocol_path)
    index = pd.read_csv(args.linear_input / "run_index.csv")
    np.testing.assert_array_equal(index["score_row"], np.arange(len(index)))
    cases, views, labels = case_inputs(args.linear_input, args.test_indexes, protocol)
    with (threadpool_limits(limits=2), np.load(args.linear_input / "test_ranking_scores.npz", allow_pickle=False) as r,
          np.load(args.linear_input / "test_scores.npz", allow_pickle=False) as p):
        raw, probability = dict(r), dict(p)
        for domain in DOMAINS:
            expected = (len(index), len(cases[domain]))
            if any(store[domain].shape != expected or not np.isfinite(store[domain]).all() for store in (raw, probability)):
                raise ValueError(f"Wrong shape or nonfinite archived scores for {domain}")
        tables = derive(index, raw, probability, views, labels, metrics, protocol)
    agreement_with_round1(tables, args.linear_input)
    focal_path = args.linear_input / "focal_auroc_intervals.json"
    tables["modern_32"] = read_modern(read_json(args.modern_input), read_json(focal_path))
    args.output.mkdir(parents=True, exist_ok=True)
    for key, values in tables.items():
        pd.DataFrame(values).to_csv(args.output / f"{key}.csv", index=False)
    write_tex(args.output, tables)
    files = ("run_index.csv", "test_ranking_scores.npz", "test_scores.npz", "summary_metrics.csv", "oculo_cases.csv", "erdes_cases.csv", "focal_auroc_intervals.json")
    provenance = {f"linear/{name}": sha256(args.linear_input / name) for name in files}
    provenance.update({f"case_indexes/{domain}_test_index.csv": sha256(args.test_indexes / f"{domain}_test_index.csv")
                       for domain in DOMAINS})
    provenance["modern_sensitivity_results.json"] = sha256(args.modern_input)
    (args.output / "results.json").write_text(json.dumps({"protocol_sha256": sha256(protocol_path),
                                                          "inputs_sha256": provenance, "tables": tables},
                                                         indent=2, allow_nan=False) + "\n")
    print("Saved", {key: len(values) for key, values in tables.items()})


if __name__ == "__main__":
    main()
