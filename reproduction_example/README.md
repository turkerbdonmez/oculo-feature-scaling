# Refit the eight-label scaling failure

This versioned example reproduces the seed-44, draw-13 failure and its scaling interventions from frozen features. It fits the target classifier from the eight labeled support clips and recomputes the Oculo-training and target-support moments. It verifies the original five fits against archived coefficients, intercepts, raw ranking scores, and iteration counts. The source encoder and classifier are fixed. No image, video, checkpoint, GPU, or original project directory is required.

Use Python 3.12. The example was verified with Python 3.12.12 and the pinned numerical libraries.

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce_failure.py --output refit_output
```

The `--output` option saves all refitted standardized and raw coefficients, intercepts, scaler parameters, case-level raw predictions, and a JSON summary. Omit it to run the verification without creating files. On systems with `shasum` you can verify the distributed files with `shasum -a 256 -c SHA256SUMS`. The Python script also checks both data file hashes.

| Scaling or reference | Oculo AUROC | ERDES AUROC |
|---|---:|---:|
| Frozen source, matched mean-frame logits | 0.984694 | 0.780192 |
| Historical draw scaling | 0.377551 | 0.789876 |
| Source-fixed scaling | 0.985023 | 0.792155 |
| 0.1% variance floor | 0.986669 | 0.787679 |
| 1% variance floor | 0.986175 | 0.788086 |
| 10% variance floor | 0.986175 | 0.790609 |
| 1% floor on coordinate 177 alone | 0.986669 | 0.787760 |

Each adaptation row is the anchored classifier at $C=0.3$ and $\alpha=0.25$. AUROC uses raw affine decision scores. The JSON separately reports legacy sigmoid-ranked AUROC and counts of exact-one sigmoid scores to show the effect of numerical ties. The coordinate-only follow-up holds the other 511 target scales and all target means fixed. It uses the coordinate selected in the earlier forensic analysis. These are retrospective results from one fit on previously inspected benchmarks.

- `fixture.npz` contains Oculo source-training features and IDs (1,187 rows), ERDES support features, clip IDs and labels (eight balanced clips), the frozen source affine head, and both grouped test feature arrays, IDs and labels (245 Oculo images and 224 ERDES clips). Each feature row has 512 dimensions. The Oculo ME-positive and ERDES strict-subset masks are provided for alignment.
- `reference_outputs.npz` contains upstream source/target moments, the five archived target-fit coefficient vectors and intercepts, raw target and interpolated test scores, and frozen-source scores. The example does not fit against any of these reference outputs.
- `provenance.json` identifies the releases, source encoder and feature cache, support-manifest and archive digests, source archive row indices, versions and the bundled data hashes.
- `coordinate_intervention_protocol.json` records the fixed scope of the single-coordinate follow-up before evaluation. `example_summary.json` records the verified run.
- `build_fixture_from_workspace.py` documents optional maintainer extraction from the original archival workspace. The standalone example has no dependency on this exporter.

Source datasets: [Oculo, Sankara Eye Hospital (CC BY-SA 4.0)](https://huggingface.co/datasets/SankaraEyeHospital/Oculo/tree/306ba5d2901e520efdff11f46b561a84a3623c84) and [ERDES, Ozkut and colleagues (CC BY 4.0)](https://doi.org/10.5281/zenodo.18644370). The fixture redistributes derived embeddings and linked release identifiers with attribution and contains no source media.
