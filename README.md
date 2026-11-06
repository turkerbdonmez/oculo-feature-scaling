# Oculo–ERDES feature-scaling analysis

Version 1.0.0. [Versioned release](https://github.com/turkerbdonmez/oculo-feature-scaling/releases/tag/v1.0.0). Computational supplement to *Feature Scaling and Source Retention in Few-Shot Adaptation of Ocular Ultrasound Classifiers*.

This package reproduces a selected eight-label scale failure and its matched refits, then recomputes the Round 2 low-budget, ME+ and probability-loss analyses from frozen scores. Both datasets were inspected during earlier development. This is not independent patient-level or clinical validation. The single extreme fit accounts for much of the mean low-budget gain; source-fixed scaling also decreases 16-label target AUROC.

## Quick start

Use Python 3.12. From this repository's root, create an environment and install both pinned requirement lists:

```sh
python3.12 -m venv .venv
# macOS/Linux:
. .venv/bin/activate
# Windows PowerShell: .venv/Scripts/Activate.ps1
python -m pip install -r reproduction_example/requirements.txt -r score_reanalysis/requirements.txt
```

After installation, both commands work offline:

```sh
python reproduction_example/reproduce_failure.py --output reproduced_failure
python -B score_reanalysis/run.py --output reproduced_tables
```

The first command refits frozen-feature logistic heads and verifies the saved moments, coefficients and predictions. The second recomputes 12 files, including six LaTeX tables, and verifies their exact expected hashes. Source-model training, the wider comparator experiments, and the video-content audit require the separate full computational archive. See each folder's README for exact scope.

## Contents and attribution

- `reproduction_example/`: frozen feature fixture, source head, eight support IDs, source moments, reference predictions and refit code; no clinical media or neural-network weights.
- `score_reanalysis/`: anonymized ordered score arrays, binary labels and ME+ mask, run keys, pinned protocol and analysis code. The modern-representation table uses archived summary cells; the new paired analyses are recalculated from scores.
- `SHA256SUMS`: file-integrity manifest.
- `DATA_ATTRIBUTION.md` and `LICENSE-DATA.md`: source release attribution and CC BY-SA 4.0 terms for contributed derived data. Original upstream terms remain as stated.
- `LICENSE-CODE`: MIT terms for original Python code; this does not relicense upstream libraries or source data.

No manuscript text, reviewer submissions, response letters, clinical images/videos, patient-linkage keys, or model checkpoints are included. Release identifiers do not establish patient identities. No DOI has been assigned.

## Integrity and citation

The release includes a ZIP and its SHA-256 checksum. After extracting the ZIP, verify the included file manifest with `shasum -a 256 -c SHA256SUMS` (macOS) or `sha256sum -c SHA256SUMS` (Linux). See `CITATION.txt` for versioned software/data citation metadata. This GitHub release has no DOI.
