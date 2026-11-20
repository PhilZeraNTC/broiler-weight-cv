# Document Ingestion Synthesis

**Mode:** `merge`
**Precedence:** `ADR > SPEC > PRD > DOC`
**Cross-ref Cycle Check:** Passed (`0` cross-ref edges, `0` cycles, max depth `0`).

## Documents Synthesized (`2` total)
- `ADR`: `0`
- `SPEC`: `0`
- `PRD`: `1` (`docs/proposta_pdi.md.pdf`, confidence: `medium`)
- `DOC`: `1` (`docs/dataset/DOCUMENTACAO_DATASET_BROILER_COBB_AVIAN.md`, confidence: `high`)
- `UNKNOWN`: `0`

## Decisions Locked
- Count: `0` ingested locked ADRs (`decisions.md` contains 0 entries; existing project constraints in `.planning/PROJECT.md`, `.planning/REQUIREMENTS.md`, `.planning/STATE.md`, and `PRD.md` were enforced during merge-mode conflict detection).

## Requirements Extracted
- Count: `13` entries in `.planning/intel/requirements.md` (`5` non-conflicting requirements from `docs/proposta_pdi.md.pdf` + `4` competing-variant pairs preserving both proposal and existing `PRD.md` wording verbatim):
  - `REQ-classical-pdi-preprocessing`
  - `REQ-bird-litter-segmentation`
  - `REQ-geometric-feature-extraction`
  - `REQ-weight-regression-model`
  - `REQ-density-activity-extension`
  - `REQ-mvp-input-modality-proposal` / `REQ-mvp-input-modality-existing`
  - `REQ-watershed-output-proposal` / `REQ-watershed-output-existing`
  - `REQ-camera-geometry-proposal` / `REQ-camera-geometry-existing`
  - `REQ-dataset-coverage-proposal` / `REQ-dataset-coverage-existing`

## Constraints Extracted
- Count: `0` SPEC entries (`api-contract`: `0`, `schema`: `0`, `nfr`: `0`, `protocol`: `0`).

## Context Topics Extracted
- Count: `8` topics in `.planning/intel/context.md` (all attributed to `docs/dataset/DOCUMENTACAO_DATASET_BROILER_COBB_AVIAN.md`):
  1. Dataset Provenance & Data Card
  2. Published Metadata vs. Disk Audit Reconciliation
  3. Zootechnical Context & Field Collection Rules
  4. Directory Structure & Filename Encoding Patterns
  5. Image Integrity, Optics, EXIF Orientation & Pixel Statistics
  6. Weight Distribution, Coverage Gaps, Lens-Weight Confounding & Visual Scenarios
  7. Quality Anomalies, Quarantine Set & Clean Dataset Totals
  8. Reference Python Ingestion Pipeline & Audit Recommendations

## Conflict Summary
- **Blockers (`[BLOCKER]`):** `0`
- **Competing Variants (`[WARNING]`):** `4` — require explicit user resolution before routing:
  1. MVP input modality (`imagem ou vídeo` in `docs/proposta_pdi.md.pdf` vs. image-only upload + `PROIBIÇÃO 5` in `PRD.md`)
  2. Watershed multi-bird output contract (per-bird weight + flock mean vs. count + per-bird pixel area in `PDI-05`/`WEB-04`)
  3. Camera capture geometry (fixed ceiling camera vs. v1 handheld dual-lens EXIF calibration + v2 `BARN-01`)
  4. Dataset validation coverage (`45 dias de criação` in proposal vs. audited `~7–29 days` / `116–2,093 g` + v2 `BARN-02`)
- **Auto-Resolved & Alignments (`[INFO]`):** `6` (`4` `PRD > DOC` precedence wins — excluding YOLO-Seg/SAM/CNN/MLP, LCD OCR, CSV/root path overrides, and optional post-MVP extensions — plus `2` informational alignments on optical double-counting guard and literature $R^2$ / conceptual extension).

## Pointers
- Conflict Report: `.planning/INGEST-CONFLICTS.md`
- Decisions Intel: `.planning/intel/decisions.md`
- Requirements Intel: `.planning/intel/requirements.md`
- Constraints Intel: `.planning/intel/constraints.md`
- Context Intel: `.planning/intel/context.md`
