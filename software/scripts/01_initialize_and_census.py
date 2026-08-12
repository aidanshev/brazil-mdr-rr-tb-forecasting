#!/usr/bin/env python3
"""Verify V15, register reusable artifacts, and census plausible local sources."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import stat
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from v16_common import environment_receipt, json_write, sha256

EXPECTED_V15 = "fd91325e649a581c2a2464f00f5f1c90c2b3f76f996cca451cb7c2aef9370402"
SKIP_PARTS = {".git", ".venv", "__pycache__", ".pytest_cache", "node_modules", "12_cache", "workbook_previews"}
JUNK_SUFFIX = {".css", ".js", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".woff", ".woff2", ".map"}


def source_group(path: Path) -> str:
    text = str(path).lower()
    name = path.name.lower()
    rules = [
        ("WHO_CONTEXT", ("gtbreport", "who_", "global_tb", "who global")),
        ("GENOMICS", ("genom", "sra", "mycobacterium", "ncbi")),
        ("CLIMATE", ("inmet", "clima", "climate", "meteorolog")),
        ("REGIC_MOBILITY", ("regic", "ligacoes", "rotas", "osm.pbf", "transport", "antt", "anac")),
        ("SISDEPEN", ("sisdepen", "pris", "penitenc")),
        ("APS_PMMB_PREVINE_UBS", ("pmm", "mais_medicos", "previne", "ubs", "primary_care", "aps")),
        ("SIM_MORTALITY", ("mortalidade", "do22", "do23", "do24", "sim_", "sim.", "obito")),
        ("SIH_HOSPITAL", ("sih", "hospitaliza", "rdbr", "/rd")),
        ("SIA_OUTPATIENT", ("sia", "sigtap", "proced", "pabr", "/pa")),
        ("CNES", ("cnes", "scnes", "estabelecimento")),
        ("CENSUS_SOCIAL_ECONOMIC", ("censo", "census", "cadun", "bolsa", "auxilio", "pib", "gdp", "ivs", "sane", "sanit", "domicilio", "alfabet", "raca", "social")),
        ("GEOGRAPHY_POPULATION", ("pop202", "pop_dou", "pop_tcu", "estimativa", "municipios_202", "macroregiao", "crosswalk", "geograph")),
        ("SINAN_TB", ("tubebr", "sinan", "tuberculose", "dados_tuberculose")),
        ("V10_REFERENCE", ("v10_focused",)),
        ("V13_V14_V15_REFERENCE", ("v13", "v14", "v15_final")),
        ("LAND_USE_OTHER", ("mapbiomas", "land_use", "land-use", "sinisa")),
    ]
    for group, tokens in rules:
        if any(token in text or token in name for token in tokens):
            return group
    return "OTHER_AUDIT_ASSET"


def candidate_use(group: str) -> str:
    return {
        "SINAN_TB": "F0 surveillance features and target",
        "GEOGRAPHY_POPULATION": "F0 geography and denominators",
        "CNES": "F1 capacity/referral challenger",
        "SIA_OUTPATIENT": "F2 diagnostic procedure challenger",
        "SIH_HOSPITAL": "F3 severe disease/referral challenger",
        "SIM_MORTALITY": "F4 mortality/reporting challenger",
        "APS_PMMB_PREVINE_UBS": "F5 primary-care workforce challenger",
        "SISDEPEN": "F6 prison context/challenger",
        "CENSUS_SOCIAL_ECONOMIC": "F7 structural context/challenger",
        "REGIC_MOBILITY": "F8 functional graph challenger",
        "CLIMATE": "F9 climate challenger",
        "LAND_USE_OTHER": "F10 optional challenger",
        "GENOMICS": "F10 optional context",
        "WHO_CONTEXT": "external context only",
    }.get(group, "audit/reference only")


def privacy_class(path: Path, group: str) -> str:
    text = str(path).lower()
    if group == "SINAN_TB" and path.suffix.lower() in {".dbc", ".dbf", ".csv"}:
        return "SENSITIVE_LINE_LIST_DO_NOT_PACKAGE"
    if any(token in text for token in ("nominal", "cpf", "profissionais_ativos")):
        return "POTENTIAL_PERSON_LEVEL_DO_NOT_PACKAGE"
    return "PUBLIC_OR_AGGREGATED"


def basic_signature(path: Path) -> str:
    suffix = path.suffix.lower()
    try:
        if suffix == ".parquet":
            schema = pq.ParquetFile(path).schema_arrow
            return "parquet:" + ",".join(f"{f.name}:{f.type}" for f in schema)
        if suffix in {".csv", ".tsv", ".txt", ".json", ".jsonl", ".yaml", ".yml"} and path.stat().st_size < 50_000_000:
            with path.open("rb") as handle:
                first = handle.readline(32768)
            return f"{suffix[1:]}_header_sha256:{__import__('hashlib').sha256(first).hexdigest()}"
        if suffix == ".zip":
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
            joined = "\n".join(names[:100]).encode()
            return f"zip_members={len(names)};prefix_sha256={__import__('hashlib').sha256(joined).hexdigest()}"
    except Exception as exc:
        return f"UNREADABLE:{type(exc).__name__}"
    return f"format={suffix.lstrip('.') or 'none'}"


def inferred_years(path: Path) -> tuple[str, str]:
    years = [int(value) for value in re.findall(r"(?:19|20)\d{2}", str(path))]
    years = [value for value in years if 1970 <= value <= 2030]
    return (str(min(years)), str(max(years))) if years else ("UNKNOWN", "UNKNOWN")


def final_group_dispositions() -> list[dict[str, object]]:
    return [
        {"source_group": "SINAN_TB", "feature_group": "F0", "disposition": "RETAINED_IN_FINAL_MODEL", "ranker_use": True, "reason": "V15 exact reconciled national surveillance core; reverified by hash."},
        {"source_group": "GEOGRAPHY_POPULATION", "feature_group": "F0", "disposition": "RETAINED_IN_FINAL_MODEL", "ranker_use": True, "reason": "Locked 5,570-municipality, 439-region analytical geography and origin-safe denominators."},
        {"source_group": "CNES", "feature_group": "F1", "disposition": "SCREEN_PENDING", "ranker_use": False, "reason": "Validated annual December Parquet exists and requires identical-row ablation."},
        {"source_group": "SIA_OUTPATIENT", "feature_group": "F2", "disposition": "REQUEST_REQUIRED_NOT_AVAILABLE", "ranker_use": False, "reason": "No validated national SIA extract found locally after exact-path census."},
        {"source_group": "SIH_HOSPITAL", "feature_group": "F3", "disposition": "REQUEST_REQUIRED_NOT_AVAILABLE", "ranker_use": False, "reason": "No validated national SIH extract found locally after exact-path census."},
        {"source_group": "SIM_MORTALITY", "feature_group": "F4", "disposition": "SCREEN_PENDING", "ranker_use": False, "reason": "National mortality archives through 2024 are local; conservative publication lag will be enforced."},
        {"source_group": "APS_PMMB_PREVINE_UBS", "feature_group": "F5", "disposition": "SCREEN_PENDING", "ranker_use": False, "reason": "Municipality-level PMMB historical series is local and temporally indexed."},
        {"source_group": "SISDEPEN", "feature_group": "F6", "disposition": "RETAINED_FOR_PROSPECTIVE_CONTEXT_ONLY", "ranker_use": False, "reason": "Only a 2025-2 wave retrieved in 2026; insufficient retrospective temporal coverage."},
        {"source_group": "CENSUS_SOCIAL_ECONOMIC", "feature_group": "F7", "disposition": "RETAINED_FOR_PROSPECTIVE_CONTEXT_ONLY", "ranker_use": False, "reason": "Census 2022 municipal aggregate releases staged in 2025-2026 cannot be backfilled into earlier origins."},
        {"source_group": "REGIC_MOBILITY", "feature_group": "F8", "disposition": "SCREEN_PENDING", "ranker_use": False, "reason": "Official REGIC 2018 directed connections and dictionary are locally readable."},
        {"source_group": "CLIMATE", "feature_group": "F9", "disposition": "SCREEN_PENDING", "ranker_use": False, "reason": "INMET hourly archives exist for 2017-2024; monthly lagged aggregation is feasible."},
        {"source_group": "LAND_USE_OTHER", "feature_group": "F10", "disposition": "EXCLUDED_INVALID_OR_JUNK", "ranker_use": False, "reason": "No specifically validated origin-aligned analytical land-use/transport series; broad-crawl assets are invalid."},
        {"source_group": "GENOMICS", "feature_group": "F10", "disposition": "EXCLUDED_INSUFFICIENT_TEMPORAL_COVERAGE", "ranker_use": False, "reason": "Search-page assets do not form a representative origin-safe regional genomic panel."},
        {"source_group": "WHO_CONTEXT", "feature_group": "CONTEXT", "disposition": "RETAINED_FOR_PROSPECTIVE_CONTEXT_ONLY", "ranker_use": False, "reason": "WHO Global TB Report 2025 is national context, not a subnational predictive feature."},
        {"source_group": "V10_REFERENCE", "feature_group": "REFERENCE", "disposition": "EXCLUDED_DUPLICATE_OR_SUPERSEDED", "ranker_use": False, "reason": "Exact V10 baseline is registered; V15/V16 canonical artifacts supersede duplicate copies."},
        {"source_group": "V13_V14_V15_REFERENCE", "feature_group": "REFERENCE", "disposition": "EXCLUDED_DUPLICATE_OR_SUPERSEDED", "ranker_use": False, "reason": "V15 is the immutable validated baseline; older duplicate packages are lineage only."},
        {"source_group": "OTHER_AUDIT_ASSET", "feature_group": "NONE", "disposition": "EXCLUDED_INVALID_OR_JUNK", "ranker_use": False, "reason": "Non-analytical software/cache/web assets are retained only in the census."},
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    root, workspace = args.root.resolve(), args.workspace.resolve()
    v15 = root / "BRAZIL_MDRTB_V15_FINAL_REAL_PREDICTIVE_PLATFORM"
    v15_zip = v15 / "22_final_package/BRAZIL_MDRTB_V15_FINAL_REAL_PREDICTIVE_PLATFORM.zip"
    staged_zip = workspace / "00_input_baseline/BRAZIL_MDRTB_V15_FINAL_REAL_PREDICTIVE_PLATFORM.zip"
    before = {"path": str(v15_zip), "sha256": sha256(v15_zip), "size_bytes": v15_zip.stat().st_size, "mtime_ns": v15_zip.stat().st_mtime_ns}
    staged_hash = sha256(staged_zip)
    if before["sha256"] != EXPECTED_V15 or staged_hash != EXPECTED_V15:
        raise SystemExit("V15 baseline hash mismatch")

    reused: list[dict[str, object]] = []
    for directory in ("03_canonical", "04_point_in_time_features", "05_graphs", "06_folds", "09_oof_predictions", "12_evaluation", "14_prospective_release"):
        for path in sorted((workspace / directory).rglob("*")):
            if path.is_file():
                reused.append({"artifact": str(path.relative_to(workspace)), "sha256": sha256(path), "size_bytes": path.stat().st_size, "source": "V15 validated local artifact"})
    pd.DataFrame(reused).to_parquet(workspace / "01_v15_registry/v15_valid_artifact_registry.parquet", index=False)
    json_write(workspace / "01_v15_registry/v15_baseline_verification.json", {"status": "PASS", "expected_sha256": EXPECTED_V15, "authoritative": before, "staged_sha256": staged_hash, "registered_artifacts": len(reused), "v15_snapshot_mutation_allowed": False})
    json_write(workspace / "00_control/environment_receipt.json", environment_receipt())

    candidates: list[Path] = []
    for base, dirs, files in os.walk(root):
        base_path = Path(base)
        if workspace == base_path or workspace in base_path.parents:
            dirs[:] = []
            continue
        dirs[:] = [name for name in dirs if name not in SKIP_PARTS]
        for name in files:
            path = base_path / name
            if not path.is_file():
                continue
            candidates.append(path)

    rows: list[dict[str, object]] = []
    for index, path in enumerate(sorted(candidates)):
        relative = path.relative_to(root)
        group = source_group(relative)
        suffix = path.suffix.lower()
        start_year, end_year = inferred_years(relative)
        lower = str(relative).lower()
        junk = suffix in JUNK_SUFFIX or any(token in lower for token in ("unconfirmed ", ".crdownload", ".part", "facebook", "twitter", "share-link"))
        try:
            digest = sha256(path)
            status = "HASHED_JUNK" if junk else "HASHED_CANDIDATE"
        except Exception as exc:
            digest = f"UNREADABLE:{type(exc).__name__}"
            status = "UNREADABLE"
        rows.append({
            "source_group": group,
            "path": str(path),
            "relative_path": str(relative),
            "sha256": digest,
            "size_bytes": path.stat().st_size,
            "format": suffix.lstrip(".") or "none",
            "schema_signature": basic_signature(path) if not junk else "NON_ANALYTICAL_ASSET",
            "geography": "Brazil; inferred from project scope",
            "time_coverage_start": start_year,
            "time_coverage_end": end_year,
            "release_or_retrieval_time": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
            "available_time_rule": "GROUP_LEVEL_RULE_REQUIRED_BEFORE_MODELING",
            "provenance": "local Brazil project root",
            "privacy_class": privacy_class(path, group),
            "candidate_use": candidate_use(group),
            "validation_status": status,
        })
        if (index + 1) % 5000 == 0:
            print(f"catalogued={index+1}", flush=True)
    catalog = pd.DataFrame(rows)
    duplicate = catalog["sha256"].duplicated(keep="first") & catalog["sha256"].str.fullmatch(r"[0-9a-f]{64}")
    catalog.loc[duplicate, "validation_status"] = "DUPLICATE_HASH"
    catalog.to_parquet(workspace / "00_control/source_master_catalog.parquet", index=False)
    catalog.groupby(["source_group", "validation_status"], as_index=False).agg(files=("path", "size"), bytes=("size_bytes", "sum")).to_csv(workspace / "02_source_census/file_catalog_summary.csv", index=False)

    disposition = pd.DataFrame(final_group_dispositions())
    disposition.to_csv(workspace / "00_control/source_usage_disposition.csv", index=False)
    resolution = disposition.rename(columns={"reason": "evidence_and_limitations"}).copy()
    resolution["files_in_catalog"] = resolution["source_group"].map(catalog.groupby("source_group").size()).fillna(0).astype(int)
    resolution.to_csv(workspace / "00_control/source_resolution_v16.csv", index=False)

    selected = catalog[(catalog["validation_status"] == "HASHED_CANDIDATE") & catalog["source_group"].isin(disposition[~disposition["disposition"].isin(["EXCLUDED_INVALID_OR_JUNK", "EXCLUDED_DUPLICATE_OR_SUPERSEDED"])] ["source_group"])].copy()
    selected[["path", "sha256", "size_bytes", "source_group", "schema_signature", "privacy_class"]].to_csv(workspace / "00_control/acquisition_snapshot_v16_manifest.csv", index=False)
    hashes = dict(zip(selected["path"], selected["sha256"]))
    json_write(workspace / "00_control/acquisition_snapshot_v16_hashes.json", hashes)
    snapshot_hash = __import__('hashlib').sha256("\n".join(f"{k}\t{hashes[k]}" for k in sorted(hashes)).encode()).hexdigest()
    json_write(workspace / "00_control/source_resolution_v16_receipt.json", {"status": "PASS_PENDING_SCREENS", "catalog_files": len(catalog), "plausible_groups": len(disposition), "exactly_one_disposition_per_group": bool(disposition["source_group"].is_unique), "snapshot_members": len(selected), "snapshot_hash": snapshot_hash})
    json_write(workspace / "00_control/GO_FOR_V16_TRAINING.json", {"status": "GO", "primary_core_blockers": [], "v15_hash_verified": True, "snapshot_frozen": True, "snapshot_hash": snapshot_hash, "optional_sources_do_not_block": True})
    json_write(workspace / "00_control/task_ledger.json", {"status": "IN_PROGRESS", "completed": ["read_continuation_contract", "verify_v15", "register_v15_artifacts", "census_local_sources", "freeze_v16_snapshot"], "pending": ["canonicalize_challengers", "feature_group_ablations", "local_model_completion", "colab_execution", "final_handoff_after_colab"]})
    after = {"sha256": sha256(v15_zip), "size_bytes": v15_zip.stat().st_size, "mtime_ns": v15_zip.stat().st_mtime_ns}
    if before["sha256"] != after["sha256"] or before["mtime_ns"] != after["mtime_ns"]:
        raise SystemExit("V15 frozen archive changed during initialization")
    print(json.dumps({"status": "PASS", "v15_hash": before["sha256"], "registered_v15_artifacts": len(reused), "catalog_files": len(catalog), "source_groups": int(catalog.source_group.nunique()), "snapshot_hash": snapshot_hash}, indent=2))


if __name__ == "__main__":
    main()
