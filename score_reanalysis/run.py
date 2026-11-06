"""Reproduce the Round 2 score tables from the compact anonymous fixture."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import linear_metrics as metrics
from reanalyze_scaling import agreement_with_round1, derive, read_json, read_modern, sha256, write_tex


HERE = Path(__file__).resolve().parent
INPUTS = HERE / "inputs"


class ProjectedScores:
    """Map the original score-row keys to a compact score matrix."""

    def __init__(self, rows, values, case_count):
        if values.shape != (len(rows), case_count) or not np.isfinite(values).all():
            raise ValueError("Projected score shape or finiteness check failed")
        if rows.ndim != 1 or not np.all(np.diff(rows) > 0):
            raise ValueError("Projected score keys are not unique and ordered")
        self.rows = rows
        self.values = values

    def __getitem__(self, requested):
        location = np.searchsorted(self.rows, requested)
        if (location == len(self.rows)).any() or not np.array_equal(self.rows[location], requested):
            raise ValueError("Analysis requested a score row absent from the projection")
        return self.values[location]


def verify_manifest():
    for line in (HERE / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        path = (HERE / name).resolve()
        if not path.is_relative_to(HERE) or sha256(path) != expected:
            raise ValueError(f"Fixture checksum mismatch for {name}")


def load_labels(protocol):
    with np.load(INPUTS / "labels.npz", allow_pickle=False) as archive:
        oculo, erdes, me = (archive[key] for key in ("oculo_y", "erdes_y", "oculo_me"))
    if oculo.ndim != 1 or erdes.ndim != 1 or me.shape != oculo.shape or me.dtype != bool:
        raise ValueError("Unexpected shape or type for the ordered label vectors")
    views = {"oculo": ("oculo", np.ones(len(oculo), dtype=bool)),
             "erdes": ("erdes", np.ones(len(erdes), dtype=bool)),
             "oculo_me_positive": ("oculo", me)}
    labels = {"oculo": oculo, "erdes": erdes, "oculo_me_positive": oculo[me]}
    for view, values in labels.items():
        expected = protocol["test_sets"][view]
        if (not np.isin(values, (0, 1)).all() or len(values) != expected.get("cases", expected.get("clips"))
                or int(values.sum()) != expected["rd"]):
            raise ValueError(f"Unexpected case count or labels for {view}")
    return views, labels


def load_scores(name, labels):
    with np.load(INPUTS / name, allow_pickle=False) as archive:
        rows = archive["score_rows"]
        return {domain: ProjectedScores(rows, archive[domain], len(labels[domain]))
                for domain in ("oculo", "erdes")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path.cwd() / "round2_reproduced")
    args = parser.parse_args()
    verify_manifest()
    protocol_path = HERE / "protocol.json"
    protocol = read_json(protocol_path)
    evidence = read_json(INPUTS / "upstream_provenance.json")
    if sha256(protocol_path) != evidence["protocol_sha256"]:
        raise ValueError("Protocol differs from the original reanalysis")
    views, labels = load_labels(protocol)
    index = pd.read_csv(INPUTS / "run_index.csv")
    if not index["score_row"].is_unique:
        raise ValueError("Projected run index contains duplicate score keys")
    raw = load_scores("rank_scores.npz", labels)
    probability = load_scores("brier_scores.npz", labels)
    with threadpool_limits(limits=2):
        tables = derive(index, raw, probability, views, labels, metrics, protocol)
    agreement_with_round1(tables, INPUTS)
    tables["modern_32"] = read_modern(read_json(INPUTS / "modern_32.json"), read_json(INPUTS / "focal_32.json"))
    args.output.mkdir(parents=True, exist_ok=True)
    for key, values in tables.items():
        pd.DataFrame(values).to_csv(args.output / f"{key}.csv", index=False)
    write_tex(args.output, tables)
    report = {"protocol_sha256": evidence["protocol_sha256"], "inputs_sha256": evidence["inputs_sha256"], "tables": tables}
    (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for name, expected in evidence["expected_output_sha256"].items():
        if sha256(args.output / name) != expected:
            raise ValueError(f"Output differs from the original archived bytes: {name}")
    print(f"Reproduced all {len(evidence['expected_output_sha256'])} files byte-for-byte in {args.output.resolve()}")


if __name__ == "__main__":
    main()
