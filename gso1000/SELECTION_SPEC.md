# GSO-1000 Dataset Selection Spec

Deterministic selection of a 1000-object evaluation set from the Google Scanned Objects
mirror used throughout the benchmark. Extends the existing 10-object canonical run
(`CANONICAL_10_GSO`) to dataset scale for the ICIP 2027 submission.

## 1. Candidate pool

- Source: HF mirror `suvadityamuk/google-scanned-objects-raw` (flat layout, one ZIP per object).
- Pool size: **1046 objects**.
- In-repo inventory: `gso_catalog.txt` (1049 lines = 3 header comments + 1046 IDs).
- Verified **2026-10-22**: `gso_catalog.txt` set == live HF API sibling list (set-equal; only
  sort collation differs). Re-verify with:
  `curl -s https://huggingface.co/api/datasets/suvadityamuk/google-scanned-objects-raw | python3 -c "..."` (see git history of this spec for the one-liner) or diff against a fresh tree listing.
- Note: 5 IDs contain spaces (`Cardboard box`, `Door handle`, `Hinged door`, `Plastic Cup`,
  `Standard Toilet`). Safe for pathlib/JSON; never pass unquoted through shell.

## 2. Validation gate (mirrors `load_gso_samples` + metric needs)

An object enters the manifest only if ALL pass:

| ID  | Check                                                                    | Why |
|-----|--------------------------------------------------------------------------|-----|
| V1  | ZIP extracts to an object directory                                      | loader iterates dirs |
| V2  | >=1 input image: `images/000.png`, else any png/jpg in `images/`, `thumbnails/`, or object root | loader input image |
| V3  | mesh file exists: `meshes/*.obj|*.ply` or root `*.obj|*.ply`             | loader GT mesh |
| V4  | `trimesh.load(force="mesh")` succeeds, has faces, `sample_surface(mesh, 10000)` succeeds | GT points for chamfer/F-score |
| V5  | bounding-box extent > 1e-6 on every axis                                 | non-degenerate geometry |

Recorded but NOT gating: novel-view count (`images/001.png` etc.) — metrics render GT vs
prediction with the benchmark's own renderer, so GSO novel views are metadata only.

## 3. Selection rule (fully deterministic)

1. `forced` = `handpicked.txt` (100 objects; contains all of `CANONICAL_10_GSO`, verified).
   Forced objects are included **if they pass validation**; a forced object that fails V1-V5
   is dropped and reported (never force-invalid data into the run).
2. `fill` = uniform random sample, `random.Random(42)`, of size `1000 - |forced_valid|`
   from the remaining valid pool (sorted IDs before sampling for reproducibility).
3. If the valid pool < 1000, take all valid objects and record the shortfall.
4. Manifest list is emitted sorted by object ID.

Seed is fixed at **42** (matches the project's existing seed convention).

## 4. Manifest schema (`manifest_1000.json`)

```json
{
  "schema": "gso1000-manifest/v1",
  "source": "suvadityamuk/google-scanned-objects-raw",
  "catalog_verified_date": "2026-10-22",
  "seed": 42,
  "target": 1000,
  "selection_rule": "forced=handpicked_100 (incl. CANONICAL_10) if valid; fill=Random(42) sample of remaining valid",
  "counts": {
    "pool": 1046, "validated_ok": 0, "forced_valid": 0,
    "fill": 0, "final": 0,
    "dropped_validation": 0, "dropped_sampling": 0, "forced_dropped_invalid": 0
  },
  "objects": ["<sorted object ids>"]
}
```

Validation report (produced by kernel K0, consumed by the selector):

```json
{"source": "...", "checked_at": "...", "objects": [
  {"id": "...", "ok": true, "image": "images/000.png", "mesh": "meshes/x.obj",
   "faces": 12345, "bbox_extent": [x, y, z], "novel_views": 11}
]}
```

Objects absent from the report are treated as failed (conservative).

## 5. Compute plan (Kaggle T4, weekly quota 30 h)

Measured per-object inference (10-object run): TripoSR 3.59 s, LGM 10.15 s,
SPAR3D 33.26 s, TRELLIS 16.51 s -> 63.5 s/obj; metrics ~15 s/obj.

| Kernel | Hardware | Work | Est. wall |
|--------|----------|------|-----------|
| K0 validate | CPU | stream-download 1046 ZIPs (one at a time), run V1-V5, emit validation report + manifest | 1-2 h |
| K1 | T4 | TripoSR + LGM over manifest, metrics inline, JSONL checkpoint | ~8 h |
| K2 | 2xT4 | TRELLIS (staged CPU offload, proven config) over manifest | ~9 h |
| K3 | T4 | SPAR3D, objects 1-500 | ~6.7 h |
| K4 | T4 | SPAR3D, objects 501-1000 | ~6.7 h |

Total ~22 GPU-h <= 30 h weekly quota; every GPU kernel <= 9 h < 12 h session cap.
Kernels read the manifest; `load_gso_samples(object_ids=manifest)` raises on any missing
object, so the manifest must only contain validated IDs (guaranteed by the selector).

## 6. Disk strategy (per GPU kernel)

- Metrics: append-only JSONL, one line per object-engine (~1 MB total). Checkpoint after
  every object; resume = skip IDs already present.
- Qualitative renders (4 views/obj): chunk-zip every 100 objects.
- Predicted meshes: saved only for a 50-object qualitative subset (first 50 manifest IDs).
- Expected kernel output < 2 GB (well under the 20 GB working-dir limit).

## 7. Downstream

- Merge K1-K4 JSONL -> `results/unified_results_1000.csv` (same 8-metric schema as the
  10-object CSV; `engine` column distinguishes kernels).
- Paper: Table II (10-object, 4 engines) stays as the headline; the 1000-object run becomes
  the scale-validation table with bootstrap CIs, and the source of the qualitative figure
  grid (50-object subset).
