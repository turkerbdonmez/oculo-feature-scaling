Round 2 portable score reanalysis

Use Python 3.12 with the pinned packages in requirements.txt. Install those packages once in your chosen environment. Then run this from the extracted directory.

  python -B run.py

This verifies the fixture's SHA-256 checksums, generates five CSV files, six LaTeX tables and one results JSON under round2_reproduced/, and verifies the 12 output hashes against the original archived analysis. Supply --output /path/to/new/directory to choose another destination. The run needs no network, full Round 1 archive, original project location, media, weights, or source model fitting. Exact output bytes were verified on Python 3.12.12 with the pinned numerical stack. Other numerical builds can differ in the last reported floating-point digits.

The fixture keeps the original row ordering for published test scores. It includes 1,215 raw ranking rows and 135 Brier probability rows, run keys, binary RD labels, ME+ membership, the archived stability cross-check rows, and the archived 32-label representation summary fields. No image, clip, encounter, or patient identifiers are included. The 32-label representation cells transcribe fixed Round 1 summaries and their archived intervals. The other tables are recomputed from case scores using the included original analysis and bootstrap code.

SHA256SUMS checks the integrity of the included files. The separate inputs/upstream_provenance.json preserves original full-archive file hashes and the expected output hashes. These upstream hashes identify files from which the fixture was projected. They do not claim that the full input files are inside this fixture. build_projection.py records and checks the projection against the full archive if it is available. That script is optional and is not used by run.py.

The resampling is retrospective and exploratory. It conditions on the observed public test cases and three fixed source fits, and it does not adjust for unknown patient clustering or multiplicity. The fixture reproduces tables and their uncertainty calculations. Training and clinical validation are outside its scope.
