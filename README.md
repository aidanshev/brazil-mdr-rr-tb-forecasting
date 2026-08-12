# Brazil RR/MDR-TB Health-Region Forecasting

**Repository status:** `SOURCE_RICH_PUBLICATION_PAYLOAD`

Temporally validated forecasting and surveillance prioritization of detected/reported RR/MDR-positive burden in Brazil.

## Public-release policy

This repository is code/provenance-forward: raw patient-level surveillance data are excluded, third-party datasets are linked rather than mirrored, and image binaries are intentionally excluded. Source sites for visuals and data are recorded in `FIGURE_AND_IMAGE_PROVENANCE.md` and `DATA_SOURCES.md`.

Run before publishing:

```bash
python tools/public_release_audit.py
```

## Layout

- `code/` or `software/`: materialized analysis code
- `results/`: publication-safe aggregate results/receipts
- `manifests/`: identities and hashes without restricted raw data
- `docs/`: protocols/methods
- `REPOSITORY_STATUS.md`: completeness status

## Publication documentation

- `PUBLICATION_READINESS.md`: exact reproducibility and completeness boundary
- `CODE_AVAILABILITY.md`: manuscript-ready Code Availability language
- `RELEASE_CHECKLIST.md`: completed and remaining archival steps
- `CITATION.cff`: repository citation metadata

## Archival DOI

After the final code/manuscript reconciliation and license decision, create an immutable Git tag and GitHub Release, archive that release with Zenodo or an equivalent preservation service, and add the resulting DOI to this README, `CITATION.cff`, and the manuscript.
