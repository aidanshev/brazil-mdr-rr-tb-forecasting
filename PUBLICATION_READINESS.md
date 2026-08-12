# Publication readiness

## Current release class

`SOURCE_RICH_PUBLICATION_PAYLOAD`

This repository contains the materialized V16 analysis scripts, V15 lineage evaluators, contract tests, aggregate results, plotting data, source-resolution receipts, and pseudo-prospective release metadata used to document the Brazil RR/MDR-TB forecasting study.

## Reproducibility boundary

Raw and potentially sensitive surveillance records are intentionally excluded. Large public source binaries are not mirrored. `DATA_SOURCES.md` records source locations, while `manifests/` records identities, hashes, timing rules, and source disposition. The exhaustive sanitized acquisition inventory is represented by its SHA-256 anchor rather than being duplicated here.

The repository is suitable for code review and methods reproduction once the documented external data are obtained. It is not a redistribution archive for the underlying surveillance records.

## Publication safeguards

- `tools/public_release_audit.py` fails on common image binaries, restricted raw-data extensions, files larger than 25 MB, obvious private filesystem paths, and common token patterns.
- `.github/workflows/public-release-audit.yml` runs the public-tree audit and Python syntax compilation on pushes and pull requests.
- No raw third-party pictures or report screenshots are distributed.
- `CITATION.cff` contains repository citation metadata but intentionally has no release version or DOI until an immutable release exists.

## Items requiring author or external service action

1. Select an explicit software license after confirming that all included project and upstream code is eligible for that license.
2. After final manuscript/code reconciliation, create an immutable Git tag and GitHub Release.
3. Archive that release with a preservation service such as Zenodo and add the resulting DOI to `CITATION.cff`, `README.md`, and the manuscript code-availability statement.

Do not describe a GitHub branch alone as a permanent archived release.
