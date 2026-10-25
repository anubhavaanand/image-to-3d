#!/usr/bin/env python3
"""Deterministic 1000-object GSO manifest builder.

Pool:      gso_catalog.txt (1046 objects, verified == live HF mirror 2026-10-22)
Forced:    handpicked.txt (100 objects; contains all of CANONICAL_10_GSO)
Fill:      random.Random(42) sample of the remaining valid pool
Gate:      optional validation report (kernel K0); objects absent or ok=false are excluded

Modes:
  dry-run (default, no --validation-report): assumes the whole pool validates;
          emits a PREVIEW manifest + the sampling drop-list.
  final   (--validation-report report.json): emits the run manifest.

Stdlib only; runs locally and on Kaggle.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import date
from pathlib import Path

SEED = 42
TARGET = 1000
MIRROR = "suvadityamuk/google-scanned-objects-raw"

# Ground truth from kaggle/kernel_batch1_triposr_lgm.ipynb (verified subset of handpicked.txt)
CANONICAL_10_GSO = [
    "ASICS_GELLinksmaster_WhiteCoffeeSand",
    "3D_Dollhouse_Happy_Brother",
    "3D_Dollhouse_Sofa",
    "Bradshaw_International_11642_7_Qt_MP_Plastic_Bowl",
    "Android_Figure_Panda",
    "Animal_Planet_Foam_2Headed_Dragon",
    "Black_and_Decker_TR3500SD_2Slice_Toaster",
    "Down_To_Earth_Orchid_Pot_Ceramic_Red",
    "Dog",
    "Dino_3",
]


def load_pool(catalog_path: Path) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for line in catalog_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s in seen:
            raise SystemExit(f"[selector] duplicate id in catalog: {s}")
        seen.add(s)
        ids.append(s)
    return ids


def load_forced(handpicked_path: Path, pool: set[str]) -> list[str]:
    hp = [l.strip() for l in handpicked_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    missing = [h for h in hp if h not in pool]
    if missing:
        raise SystemExit(f"[selector] handpicked objects absent from pool: {missing}")
    missing_canon = [c for c in CANONICAL_10_GSO if c not in hp]
    if missing_canon:
        raise SystemExit(f"[selector] CANONICAL_10 entries absent from handpicked: {missing_canon}")
    return hp


def valid_ids_from_report(report_path: Path) -> tuple[set[str], dict[str, dict]]:
    rep = json.loads(report_path.read_text(encoding="utf-8"))
    ok: set[str] = set()
    meta: dict[str, dict] = {}
    for o in rep.get("objects", []):
        oid = o.get("id")
        if not oid:
            continue
        meta[oid] = o
        if o.get("ok") is True:
            ok.add(oid)
    return ok, meta


def select(pool: list[str], valid: set[str], forced: list[str], seed: int) -> tuple[list[str], dict]:
    forced_valid = sorted(f for f in forced if f in valid)
    forced_dropped = sorted(f for f in forced if f not in valid)
    remaining = sorted(set(valid) - set(forced_valid))
    fill_n = TARGET - len(forced_valid)

    if fill_n <= 0:
        manifest = sorted(forced_valid)[:TARGET]
        fill: list[str] = []
        dropped_sampling: list[str] = []
    elif len(remaining) >= fill_n:
        rng = random.Random(seed)
        fill = sorted(rng.sample(remaining, fill_n))
        dropped_sampling = sorted(set(remaining) - set(fill))
        manifest = sorted(set(forced_valid) | set(fill))
    else:
        fill = remaining  # valid pool smaller than target: take everything
        dropped_sampling = []
        manifest = sorted(set(forced_valid) | set(fill))

    stats = {
        "pool": len(pool),
        "validated_ok": len(valid),
        "forced_valid": len(forced_valid),
        "fill": len(fill),
        "final": len(manifest),
        "dropped_validation": len(pool) - len(valid),
        "dropped_sampling": len(dropped_sampling),
        "forced_dropped_invalid": len(forced_dropped),
    }
    extras = {
        "forced_dropped_ids": forced_dropped,
        "dropped_sampling_ids": dropped_sampling,
    }
    return manifest, stats, extras


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    here = Path(__file__).resolve().parent
    repo = here.parent
    ap.add_argument("--catalog", type=Path, default=repo / "gso_catalog.txt")
    ap.add_argument("--handpicked", type=Path, default=repo / "handpicked.txt")
    ap.add_argument("--validation-report", type=Path, default=None,
                    help="validation_report.json from kernel K0; omit for dry-run preview")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    pool = load_pool(args.catalog)
    forced = load_forced(args.handpicked, set(pool))

    if args.validation_report:
        valid, meta = valid_ids_from_report(args.validation_report)
        uncovered = [p for p in pool if p not in meta]
        mode = "final"
        default_out = here / "manifest_1000.json"
    else:
        valid, meta, uncovered = set(pool), {}, []
        mode = "preview"
        default_out = here / "manifest_1000.preview.json"

    manifest, stats, extras = select(pool, valid, forced, args.seed)

    out = args.out or default_out
    doc = {
        "schema": "gso1000-manifest/v1",
        "mode": mode,
        "source": MIRROR,
        "catalog_verified_date": "2026-10-22",
        "seed": args.seed,
        "target": TARGET,
        "selection_rule": (
            "forced=handpicked_100 (incl. CANONICAL_10) if valid; "
            f"fill=Random({args.seed}) sample of remaining valid"
        ),
        "counts": stats,
        "objects": manifest,
        **extras,
    }
    if mode == "final":
        doc["validation_report"] = str(args.validation_report)
        doc["validation_uncovered_ids"] = uncovered  # absent from report -> excluded
    out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")

    print(f"[selector] mode={mode} pool={stats['pool']} valid={stats['validated_ok']} "
          f"forced={stats['forced_valid']} fill={stats['fill']} final={stats['final']}")
    if extras["forced_dropped_ids"]:
        print(f"[selector] forced dropped (failed validation): {extras['forced_dropped_ids']}")
    print(f"[selector] wrote {out} ({len(manifest)} objects)")
    if mode == "preview":
        print("[selector] PREVIEW ONLY - assumes whole pool validates; "
              "run kernel K0 and re-run with --validation-report for the final manifest")


if __name__ == "__main__":
    main()
