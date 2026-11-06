"""Regenerate this projection when the full upstream score archive is available."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reanalyze_scaling import BUDGETS, DOMAINS, METHODS, SCALERS, case_inputs, read_json, score_rows, sha256


HERE = Path(__file__).resolve().parent
INPUTS = HERE / "inputs"
INDEX_COLUMNS = ("budget", "draw_design", "model", "scaler", "C", "alpha", "weight_policy", "target_logit_factor",
                 "seed", "draw_id", "score_row")
SUMMARY_COLUMNS = ("draw_design", "budget", "model", "scaler", "C", "alpha", "oculo_auroc", "erdes_auroc",
                   "oculo_auc_min", "oculo_auroc_below_0_5", "oculo_source_drop_gt_0_02")


def write_json(name, data):
    (INPUTS / name).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def select_rows(index):
    raw, probability = set(), set()
    for budget in BUDGETS:
        raw.update(score_rows(index, budget, reference="direct"))
        for design, scalers in (("balanced", SCALERS), ("unstratified", ("draw", *METHODS))):
            for scaler in scalers:
                raw.update(score_rows(index, budget, design, scaler))
    for reference in (None, "direct", "target"):
        probability.update(score_rows(index, 32, reference=reference))
    return np.array(sorted(raw)), np.array(sorted(probability))


def project_modern(report):
    arms = {}
    for model in ("usfm", "convnext_tiny", "resnet18_imagenet"):
        for arm in (model, model + "_source_fixed"):
            oculo = report["arms"][arm]["oculo"]["32"]
            erdes = report["arms"][arm]["erdes"]["32"]
            arms[arm] = {"oculo": {"32": {"mean_run_auroc": oculo["mean_run_auroc"],
                                           "paired_95pct": {"source_logit_pool": oculo["paired_95pct"]["source_logit_pool"]}}},
                         "erdes": {"32": {"mean_run_auroc": erdes["mean_run_auroc"]}}}
    return {"arms": arms}


def project_focal(report):
    wanted = {(view, arm, None) for view in DOMAINS for arm in ("direct_probability", "legacy_anchor", "source_scaled_anchor")}
    wanted.update({("oculo", "legacy_anchor", "direct_probability"),
                   ("oculo", "source_scaled_anchor", "source_scaled_two_head_oracle")})
    rows = [row for row in report["intervals"] if row["budget"] == 32 and (row["view"], row["model"], row["reference"]) in wanted]
    if len(rows) != len(wanted):
        raise ValueError("Missing required focal summary rows")
    return {"intervals": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--linear-input", type=Path, required=True)
    parser.add_argument("--modern-input", type=Path, required=True)
    parser.add_argument("--test-indexes", type=Path, required=True)
    parser.add_argument("--original-results", type=Path, required=True)
    args = parser.parse_args()
    source = read_json(args.original_results / "results.json")
    protocol_path = HERE / "protocol.json"
    if sha256(protocol_path) != source["protocol_sha256"]:
        raise ValueError("Protocol does not match the original reanalysis")
    upstream = {name: (args.linear_input if name.startswith("linear/") else args.test_indexes) / Path(name).name
                for name in source["inputs_sha256"] if "/" in name}
    upstream["modern_sensitivity_results.json"] = args.modern_input
    for name, path in upstream.items():
        if sha256(path) != source["inputs_sha256"][name]:
            raise ValueError(f"Upstream archive digest mismatch for {name}")
    INPUTS.mkdir(exist_ok=True)
    index = pd.read_csv(upstream["linear/run_index.csv"])
    np.testing.assert_array_equal(index["score_row"], np.arange(len(index)))
    raw_rows, probability_rows = select_rows(index)
    index.loc[np.union1d(raw_rows, probability_rows), list(INDEX_COLUMNS)].to_csv(INPUTS / "run_index.csv", index=False)
    for rows, original, projected in ((raw_rows, "test_ranking_scores.npz", "rank_scores.npz"),
                                       (probability_rows, "test_scores.npz", "brier_scores.npz")):
        with np.load(args.linear_input / original, allow_pickle=False) as archive:
            np.savez_compressed(INPUTS / projected, score_rows=rows, **{domain: archive[domain][rows] for domain in DOMAINS})
    cases, _, labels = case_inputs(args.linear_input, args.test_indexes, read_json(protocol_path))
    np.savez_compressed(INPUTS / "labels.npz", oculo_y=labels["oculo"].astype(np.int8),
                        erdes_y=labels["erdes"].astype(np.int8), oculo_me=cases["oculo"]["membranous_echo"].eq(1).to_numpy())
    summary = pd.read_csv(upstream["linear/summary_metrics.csv"])
    keep = (summary["model"].eq("target_fit") & summary["C"].eq(0.3) & summary["alpha"].eq(0.25)
            & summary["budget"].isin(BUDGETS) & summary["target_logit_factor"].isna()
            & summary["weight_policy"].eq("target_mass")
            & ((summary["draw_design"].eq("balanced") & summary["scaler"].isin(SCALERS))
               | (summary["draw_design"].eq("unstratified") & summary["scaler"].isin(("draw", *METHODS)))))
    summary.loc[keep, list(SUMMARY_COLUMNS)].to_csv(INPUTS / "summary_metrics.csv", index=False)
    write_json("modern_32.json", project_modern(read_json(args.modern_input)))
    write_json("focal_32.json", project_focal(read_json(upstream["linear/focal_auroc_intervals.json"])))
    filenames = [*(path.name for path in sorted(args.original_results.glob("*.csv"))),
                 *(path.name for path in sorted(args.original_results.glob("tab_round2_*.tex"))), "results.json"]
    if len(filenames) != 12:
        raise ValueError("Unexpected number of original output artifacts")
    write_json("upstream_provenance.json", {"protocol_sha256": source["protocol_sha256"],
                                          "inputs_sha256": source["inputs_sha256"],
                                          "expected_output_sha256": {name: sha256(args.original_results / name) for name in filenames}})
    paths = [*sorted(INPUTS.glob("*")), *(HERE / name for name in ("run.py", "build_projection.py", "README.txt", "requirements.txt",
                                                                    "reanalyze_scaling.py", "linear_metrics.py", "protocol.json"))]
    (HERE / "SHA256SUMS").write_text("".join(f"{sha256(path)}  {path.relative_to(HERE).as_posix()}\n" for path in paths))
    print(f"Projected {len(raw_rows)} raw ranking rows and {len(probability_rows)} probability rows")


if __name__ == "__main__":
    main()
