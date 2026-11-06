"""Maintainer-only fixture extraction from the frozen study workspace.

The reviewer-facing reproduction script does not import this file or require
access to the original source model, workspace, or video/image releases.
"""
import hashlib
import json
import platform
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import scipy
import sklearn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LINEAR = ROOT / "revision_round1/linear"
sys.path.insert(0, str(LINEAR))
from linear_core import ECHO, TARGET, load_cohorts, load_seed

ARCHIVE = LINEAR / "cached_results"
SEED, BUDGET, DRAW = 44, 8, 13
C, ALPHA = 0.3, 0.25
VARIANTS = ("draw", "source", "floor_0.001", "floor_0.01", "floor_0.1")


def sha256(file):
    with file.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def extract():
    tables, masks, draws = load_cohorts()
    oculo, erdes, _ = tables
    cached, source_head, _, source_provenance = load_seed(SEED, tables)
    source_train = masks["oculo"]["train"]
    support = draws[BUDGET, DRAW]
    oculo_test, erdes_test = masks["oculo"]["test"], masks["erdes"]["test"]
    inputs = {
        "source_training_features": cached["oculo_oculo"][source_train],
        "source_training_ids": oculo.loc[source_train, "image_id"].astype(str).to_numpy(dtype=str),
        "source_coefficient": source_head.coef, "source_intercept": np.asarray(source_head.bias),
        "target_support_features": cached["oculo_erdes"][support],
        "target_support_ids": erdes.iloc[support]["clip_id"].to_numpy(dtype=str),
        "target_support_labels": erdes.iloc[support][TARGET].to_numpy(dtype=np.int8),
        "oculo_test_features": cached["oculo_oculo"][oculo_test],
        "oculo_test_ids": oculo.loc[oculo_test, "image_id"].astype(str).to_numpy(dtype=str),
        "oculo_test_labels": oculo.loc[oculo_test, TARGET].to_numpy(dtype=np.int8),
        "oculo_test_me_positive": oculo.loc[oculo_test, ECHO].to_numpy(dtype=bool),
        "erdes_test_features": cached["oculo_erdes"][erdes_test],
        "erdes_test_ids": erdes.loc[erdes_test, "clip_id"].to_numpy(dtype=str),
        "erdes_test_labels": erdes.loc[erdes_test, TARGET].to_numpy(dtype=np.int8),
        "erdes_test_strict": masks["strict"],
    }
    np.savez_compressed(HERE / "fixture.npz", **inputs)

    index_file = ARCHIVE / "run_index.csv"
    ranking_file = ARCHIVE / "test_ranking_scores.npz"
    fit_file = ARCHIVE / "fit_diagnostics.csv"
    slopes_file = ARCHIVE / "fit_slopes.npz"
    forensic_file = ARCHIVE / "seed44_draw13_arrays.npz"
    runs = pd.read_csv(index_file).query(
        "seed == @SEED and budget == @BUDGET and draw_id == @DRAW and draw_design == 'balanced'"
    )
    fits = pd.read_csv(fit_file).set_index("fit_id")
    ids = {"direct": int(runs.query("model == 'direct_matched_logit'")["score_row"].item())}
    references = {}
    with (np.load(ranking_file, allow_pickle=False) as scores,
          np.load(slopes_file, allow_pickle=False) as slopes,
          np.load(forensic_file, allow_pickle=False) as forensic):
        np.testing.assert_array_equal(forensic["target_support_features"], inputs["target_support_features"])
        for key in ("source_mean", "source_variance", "draw_mean", "draw_variance"):
            references[key] = forensic[key]
        score_arrays = {key: scores[key] for key in ("oculo", "erdes")}
        for key, matrix in score_arrays.items():
            references[f"{key}_direct_logits"] = matrix[ids["direct"]]
        for variant in VARIANTS:
            conditions = "model == 'target_fit' and scaler == @variant and C == @C"
            rows = runs.query(conditions)
            anchor = rows.loc[rows["alpha"].eq(ALPHA)].iloc[0]
            target = rows.loc[rows["alpha"].eq(1)].iloc[0]
            fit_id = int(anchor["fit_id"])
            assert fit_id == int(target["fit_id"])
            assert fits.loc[fit_id, "converged"]
            ids[variant] = {"fit_id": fit_id, "anchor": int(anchor["score_row"]),
                            "target": int(target["score_row"])}
            references[f"{variant}_raw_coefficient"] = slopes["coef"][fit_id]
            references[f"{variant}_raw_intercept"] = np.asarray(fits.loc[fit_id, "intercept"])
            references[f"{variant}_iterations"] = np.asarray(fits.loc[fit_id, "iterations"], dtype=int)
            for domain, matrix in score_arrays.items():
                references[f"{variant}_{domain}_target_logits"] = matrix[ids[variant]["target"]]
                references[f"{variant}_{domain}_anchor_logits"] = matrix[ids[variant]["anchor"]]
    np.savez_compressed(HERE / "reference_outputs.npz", **references)

    upstream = [index_file, ranking_file, fit_file, slopes_file, forensic_file,
                ROOT / "manuscript_preparation/data/target_label_draws.csv",
                ROOT / "manuscript_preparation/data/oculo_test_index.csv",
                ROOT / "manuscript_preparation/data/erdes_test_index.csv"]
    metadata = {
        "artifact_version": "1.0.0",
        "scope": "Single historical failure and fixed scaler interventions on already inspected tests",
        "seed": SEED, "budget": BUDGET, "draw": DRAW, "C": C, "alpha": ALPHA,
        "variants": VARIANTS,
        "datasets": {
            "oculo": {"url": "https://huggingface.co/datasets/SankaraEyeHospital/Oculo",
                      "revision": "306ba5d2901e520efdff11f46b561a84a3623c84", "license": "CC BY-SA 4.0"},
            "erdes": {"url": "https://doi.org/10.5281/zenodo.18644370", "license": "CC BY 4.0"},
        },
        "fixture_description": "Frozen 512-dimensional representations from the seed 44 grouped Oculo ResNet-18 refit. ERDES features average the original three support or five test frame embeddings. Release case IDs and binary labels only. No pixels, video or model checkpoint included.",
        "feature_and_encoder_provenance": source_provenance,
        "upstream_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in upstream},
        "archived_row_indices": ids,
        "python": platform.python_version(),
        "dependencies": {"numpy": np.__version__, "scipy": scipy.__version__, "scikit-learn": sklearn.__version__},
        "files_sha256": {name: sha256(HERE / name) for name in ("fixture.npz", "reference_outputs.npz")},
    }
    (HERE / "provenance.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print({name: (HERE / name).stat().st_size for name in metadata["files_sha256"]})


if __name__ == "__main__":
    extract()
