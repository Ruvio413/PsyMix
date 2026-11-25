#!/usr/bin/env python3
"""Verify byte-level SHA-256 provenance for the PsyMix JCIM submission archive.

Pure Python standard-library verifier (zero third-party dependencies).
Reads ``manifests/PsyMix_Submission_SHA256_Manifest.csv`` (13 pre-outcome /
sealed lock entries + 98 archived source files = 111 total entries), recomputes
each file's SHA-256 digest on disk, and reports PASS / FAIL status by registry.

Usage:
    python verify_manifest.py
    python verify_manifest.py --verbose
    python verify_manifest.py --remap "C:\\zhuang_taste_pre=/mnt/archive"
    python verify_manifest.py --manifest path/to/PsyMix_Submission_SHA256_Manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


CHUNK_SIZE = 1024 * 1024  # 1 MiB streaming buffer


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def apply_remaps(raw_path: str, remaps: List[Tuple[str, str]]) -> Path:
    text = raw_path.strip()
    for old_prefix, new_prefix in remaps:
        if text.lower().startswith(old_prefix.lower()):
            suffix = text[len(old_prefix):].lstrip("\\/")
            return Path(new_prefix) / Path(suffix.replace("\\", os.sep))
    return Path(text)


def parse_args(argv: List[str]) -> argparse.Namespace:
    default_manifest = (
        Path(__file__).resolve().parent
        / "manifests"
        / "PsyMix_Submission_SHA256_Manifest.csv"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Verify all 13 pre-outcome/sealed lock entries and 98 archived "
            "source files in PsyMix_Submission_SHA256_Manifest.csv."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=default_manifest,
        help=f"Path to manifest CSV (default: {default_manifest})",
    )
    parser.add_argument(
        "--remap",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help=(
            "Optional path-prefix remapping (repeatable) when verifying a "
            "relocated Zenodo archive mirror."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-entry verification status for all 111 manifest rows.",
    )
    parser.add_argument(
        "--json-report",
        type=Path,
        default=None,
        help="Optional path to write a machine-readable JSON verification report.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    t0 = time.perf_counter()

    remaps: List[Tuple[str, str]] = []
    for item in args.remap:
        if "=" not in item:
            print(f"[ERROR] Invalid --remap value (expected OLD=NEW): {item}", file=sys.stderr)
            return 2
        old_p, new_p = item.split("=", 1)
        remaps.append((old_p, new_p))

    manifest_path: Path = args.manifest.resolve()
    if not manifest_path.is_file():
        print(f"[FAIL] Manifest file not found: {manifest_path}", file=sys.stderr)
        return 1

    manifest_digest = sha256_file(manifest_path)

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        required_cols = {"registry", "source", "expected_sha256"}
        if not reader.fieldnames or not required_cols.issubset(set(reader.fieldnames)):
            print(
                f"[FAIL] Manifest missing required columns {sorted(required_cols)}; "
                f"found {reader.fieldnames}",
                file=sys.stderr,
            )
            return 1
        rows = list(reader)

    stats: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"total": 0, "pass": 0, "mismatch": 0, "missing": 0, "error": 0}
    )
    failures: List[Dict[str, str]] = []
    results: List[Dict[str, str]] = []
    total_bytes = 0

    for idx, row in enumerate(rows, start=1):
        registry = (row.get("registry") or "unknown").strip()
        raw_source = (row.get("source") or "").strip()
        expected = (row.get("expected_sha256") or "").strip().lower()
        target_path = apply_remaps(raw_source, remaps)

        stats[registry]["total"] += 1
        status = "PASS"
        actual = ""
        detail = ""

        if not target_path.is_file():
            status = "MISSING"
            stats[registry]["missing"] += 1
            detail = "File not found on disk"
        else:
            try:
                total_bytes += target_path.stat().st_size
                actual = sha256_file(target_path)
                if actual == expected:
                    status = "PASS"
                    stats[registry]["pass"] += 1
                else:
                    status = "MISMATCH"
                    stats[registry]["mismatch"] += 1
                    detail = f"expected={expected[:16]}... actual={actual[:16]}..."
            except OSError as exc:
                status = "ERROR"
                stats[registry]["error"] += 1
                detail = str(exc)

        rec = {
            "index": str(idx),
            "registry": registry,
            "source": str(target_path),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "status": status,
            "detail": detail,
        }
        results.append(rec)
        if status != "PASS":
            failures.append(rec)
        if args.verbose:
            print(f"[{status:8s}] #{idx:03d} ({registry}) {target_path.name}  {actual[:12] or '-'}")

    elapsed = time.perf_counter() - t0
    total_entries = len(rows)
    total_pass = sum(s["pass"] for s in stats.values())
    total_fail = total_entries - total_pass

    print("=" * 78)
    print("PsyMix JCIM Submission — Byte-Level SHA-256 Provenance Verification")
    print("=" * 78)
    print(f"Manifest path   : {manifest_path}")
    print(f"Manifest SHA-256: {manifest_digest}")
    print(f"Entries audited : {total_entries} ({total_bytes / (1024 * 1024):.2f} MiB hashed in {elapsed:.2f} s)")
    print("-" * 78)
    print(f"{'Registry':<32} {'Pass':>6} {'Fail':>6} {'Miss':>6} {'Total':>7}  Status")
    print("-" * 78)
    for reg in sorted(stats.keys()):
        s = stats[reg]
        reg_fail = s["mismatch"] + s["error"]
        reg_ok = s["pass"] == s["total"]
        badge = "PASS" if reg_ok else "FAIL"
        print(
            f"{reg:<32} {s['pass']:>6d} {reg_fail:>6d} {s['missing']:>6d} "
            f"{s['total']:>7d}  [{badge}]"
        )
    print("-" * 78)
    overall = "PASS (100% VERIFIED)" if total_fail == 0 and total_entries > 0 else "FAIL"
    print(f"{'OVERALL TOTAL':<32} {total_pass:>6d} {total_fail:>6d} {'':>6} {total_entries:>7d}  [{overall}]")
    print("=" * 78)

    if failures:
        print("\nFailed / Missing Entries (showing up to 20):")
        for f in failures[:20]:
            print(f"  - [{f['status']}] #{f['index']} ({f['registry']}): {f['source']} :: {f['detail']}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more.")

    if args.json_report is not None:
        report = {
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_digest,
            "total_entries": total_entries,
            "total_pass": total_pass,
            "total_fail": total_fail,
            "elapsed_seconds": round(elapsed, 4),
            "by_registry": dict(stats),
            "overall_status": "PASS" if total_fail == 0 and total_entries > 0 else "FAIL",
            "failures": failures,
        }
        args.json_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"JSON report written to: {args.json_report}")

    return 0 if (total_fail == 0 and total_entries > 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
