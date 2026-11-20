# Context (DOCs)

Extracted from classified DOC documents in `.planning/intel/classifications/` (`docs/dataset/DOCUMENTACAO_DATASET_BROILER_COBB_AVIAN.md`).

## Dataset Provenance & Data Card
- source: docs/dataset/DOCUMENTACAO_DATASET_BROILER_COBB_AVIAN.md
- Name: *Broiler weights of Cobb Avian*, Mendeley Data DOI `10.17632/zrs8kk9dvr.1` (Version 1, published 2024-03-25), License `CC BY 4.0`.
- Authors & Origin: Mohamed Fawzi Abdelshafie Abuhussein, Wael M. Elmessery, Gomaa G. Abd El-Wahhab, Ibrahim S. El-Soaly — Department of Poultry Production, Faculty of Agriculture, Kafrelsheikh University (Egypt) with Al-Azhar University.
- Sister Dataset: *Sick and Healthy: Thermal Images of Broiler Avian 48* (`10.17632/vwbb4dc9b7.1`).
- Population & Hardware: Commercial broiler strain **Cobb Avian 48** (mixed sex, reported 600-bird experimental flock), weighed on **SF-400** digital scale (`CAPACITY: 10000g x 1g`, nominal sensitivity `1 g`, circular platter `~14.5 cm`), captured on smartphone **Xiaomi Redmi 10** (`Make: Xiaomi`, `Model: 21061119AG`, proprietary tag `Redmi 10`).
- Disk Volume & Labeling: **1,753 JPEG RGB images** in **168 leaf folders** (**166 raw unique weights**, `116–2,093 g`), totaling **3.54 GiB** (`3,623.68 MiB`). Zero external annotation files (no CSV/JSON/XML/bboxes/masks); weight label in grams is encoded solely in the leaf folder name.

## Published Metadata vs. Disk Audit Reconciliation
- source: docs/dataset/DOCUMENTACAO_DATASET_BROILER_COBB_AVIAN.md
- Image count: Mendeley claims `1,772` images; disk contains `1,753` images (`-19` absent).
- Folder hierarchy: Mendeley claims `166` weights; disk has `165` root folders + `3` level-2 nested leaf folders (`171/173`, `190/191`, `190/193`) = `168` leaf folders and `166` unique raw weights (`173` and `191` appear in two folders each; `193` exists only in `190/193`).
- Collection span: Mendeley claims "daily for 45 days"; disk contains **9 unique dates** across **23 calendar days** (`2022-03-24` to `2022-04-15`, estimated **~7 to ~29 days of age**), with complete absence of days `1–6` (`<116 g`) and commercial slaughter days `30–45` (`>2,093 g`).
- Hardware note: Mendeley lists Snapdragon 678 / 8 GB RAM; EXIF confirms `Xiaomi 21061119AG` (Redmi 10 Global, commercially MediaTek Helio G88 with 50 MP `f/1.8` `4.25 mm` Wide and 8 MP `f/2.2` `1.65 mm` Ultrawide lenses).
- EXIF date corruption: `DateTime`, `DateTimeOriginal`, and `DateTimeDigitized` are literally `????:??:?? ??:??:??` in 100% (`1,753`) of files due to Arabic-locale digits replaced by `?` in ASCII EXIF writer.

## Zootechnical Context & Field Collection Rules
- source: docs/dataset/DOCUMENTACAO_DATASET_BROILER_COBB_AVIAN.md
- Plumage evolution across weights: `116–250 g` (~7–8 d) mostly yellow down with early white wing tips; `270–460 g` (~10–12 d) intense feather transition with white wings, yellowish back/head, pink skin patches; `480–800 g` (~14–17 d) mostly white body, pink wing/breast patches, slightly yellow head, incipient comb; `1,000–2,093 g` (~24–29 d) dense white plumage, developed pink/red comb and wattles, thick yellow legs.
- Inferred hatch date: `~2022-03-17` by aligning per-date weight medians (`185 g` on `03-24` ≈ Day 7 through `1,689 g` median / `2,093 g` max on `04-15` ≈ Day 29) against standard Cobb Avian 48 growth curve.
- Latent unrecorded variables: No `bird_id`, sex, or experimental treatment table, though colored leg rings appear visually (green rings e.g. `173, 190, 233, 423, 679, 719, 1340_, 1599, 2093`; red ring e.g. `537`; unringed e.g. `116, 171/173, 190/191, 190/193`).
- Primary key rule: Folder = weight in grams, NOT individual bird. Distinct birds with identical weight share a folder. Across 168 leaf folders there are **194 raw weighing sessions** (gap `> 10 min` within a folder): `20` leaf folders contain `2–4` sessions and `13` unique weights span multiple calendar dates (e.g. `719` spans `04-03` and `04-11`; weight `191` has `47` images across `4` sessions in `191/` and `190/191/`).
- Session profile: Mean `9.04` images/session (median `9`, Q1–Q3 `7–11`, max `24`) over `~91.5 s` median (`106.3 s` mean), often switching between Wide (`1.0x`) and Ultrawide (`0.6x`) seconds apart.
- Improvised scale cues & lighting: Ballpoint pen (`MY-TECH 0.7mm`) appears next to birds in some straw-floor sessions (`116 (6)–(8).jpg`); SF-400 circular platter (`~14.5 cm`) recurs in scale scenes; `91.3%` (`1,601` imgs) captured at night/early morning (`18h–05h`) under artificial barn lamps with `Flash = 0`, and `8.7%` (`152` imgs) in the afternoon (`15h–17h`, all on `2022-04-01`).

## Directory Structure & Filename Encoding Patterns
- source: docs/dataset/DOCUMENTACAO_DATASET_BROILER_COBB_AVIAN.md
- Structural anomalies: 3 level-2 nested leaf folders (`171/173`, `190/191`, `190/193`), 0 empty folders, non-numeric folder suffix `1340_` (real weight `1340 g`, confirmed on LCD).
- Filename pattern 1 (`1,628` files, `92.87%`, `154` folders): `IMG_YYYYMMDD_HHMMSS.jpg` with Eastern-Arabic Unicode digits `U+0660–U+0669` (`٠١٢٣٤٥٦٧٨٩`).
- Filename pattern 2 (`7` files, `0.40%`): `IMG_YYYYMMDD_HHMMSS_1.jpg` same-second bursts in folders `162, 171, 190, 207, 214, 236, 1583`.
- Filename pattern 3 (`97` files, `5.53%`, `12` folders): `<WEIGHT> (N).jpg` Windows batch-renamed (`116, 120, 139, 153, 157, 160, 229, 1599, 1689, 1779, 1986, 2093`), timestamp recoverable via `mtime - 14401 s`.
- Filename pattern 4 (`21` files, `1.20%`, `2` folders: `547/`, `562/`): UTF-8 Arabic digits corrupted by CP866 (DOS Cyrillic) decoding; losslessly reversible via `name.encode("cp866").decode("utf-8")`. (These 21 files carry March 2024 repackaging `mtime`, so timestamps must come from the decoded filename.)

## Image Integrity, Optics, EXIF Orientation & Pixel Statistics
- source: docs/dataset/DOCUMENTACAO_DATASET_BROILER_COBB_AVIAN.md
- Physical integrity: `1,753/1,753` valid JPEG RGB (8-bit, 0 corrupt/truncated), `0` exact MD5 duplicates, `24` near-duplicate burst pairs (dHash Hamming distance `<= 4`), `0` GPS metadata, file size `0.63–5.16 MiB` (mean `2.07 MiB`).
- Aspect ratio & dual-lens optics (fixed raw ratio `2.217:1`, ~`20:9`):
  - Wide (`4080 x 1840`, `7.51 MP`, `4.25 mm`, `f/1.8`): `1,251` imgs at `zoomMultiple = 1.00x` (`71.36%`) + `55` imgs with digital zoom `1.02x` (8), `1.06x` (41), `1.07x` (2), `2.00x` (4) (`3.14%`).
  - Ultrawide (`3264 x 1472`, `4.80 MP`, `1.65 mm`, `f/2.2`, 120° FOV): `435` imgs at `zoomMultiple = 0.60x` (`24.81%`) + `12` imgs with digital zoom `0.77x` (4), `0.85x` (3), `0.96x` (5) (`0.68%`).
  - `zoomMultiple` is stored in proprietary Xiaomi EXIF IFD `0x8769` JSON tag `39321` (`0x9999`). `114` of `168` folders (`67.9%`) mix Wide and Ultrawide lenses for the same bird; switching `4.25 mm` to `1.65 mm` alters linear scale `2.58x` and projected pixel area up to `6.6x` plus barrel distortion.
- EXIF orientation (`Tag 274`): `Orientation = 6` (`1,017` imgs, `58.01%`), `1` (`318`, `18.14%`), `3` (`299`, `17.06%`), `8` (`119`, `6.79%`). `81.86%` (`1,435` imgs) require `ImageOps.exif_transpose()`, yielding `1,136` Portrait (`64.8%`) and `617` Landscape (`35.2%`); accelerometer inconsistency in top-down/scale shots makes rotational invariance necessary.
- Exposure & RGB stats: `Flash = 0`, `Hdr = off` (100%), ISO `401.5 ± 164.7` (`116–1,073`), exposure `0.020–0.050 s` (mean `0.034 s` ~`1/29 s`), causing frequent motion blur (concentrated on `2022-03-28`). Global `256x256` RGB means: `R = 165.77 ± 26.75`, `G = 158.33 ± 25.77`, `B = 136.56 ± 25.07`, mean luminance `158.07 ± 25.66` (`82.20–221.06`).

## Weight Distribution, Coverage Gaps, Lens-Weight Confounding & Visual Scenarios
- source: docs/dataset/DOCUMENTACAO_DATASET_BROILER_COBB_AVIAN.md
- Target stats (raw `N = 1,753`, `166` unique weights): min `116 g`, image Q1 `207 g`, image median `368 g` (unique median `402.0 g`), image mean `475.4 ± 400.3 g`, image Q3 `542 g`, p90 `1,216 g`, p95 `1,520 g`, max `2,093 g`; images/weight range `1–47` (max `47` at `191 g`).
- Severe imbalance: `70.0%` of images are `<= 500 g`; only `10.7%` are `> 1,000 g` (`9` images of `1` bird at `2,093 g`).
- Lens-weight confounding: Ultrawide share jumps from `~16–32%` below `1,000 g` to `54.6%` (`1,001–1,500 g`), `56.1%` (`1,501–2,000 g`), and `77.8%` (`2,093 g`), creating spurious `lens/resolution -> heavy weight` shortcut risk.
- Major weight gaps: `801 -> 1,009 g` (`208 g` real gap because `823` is 4 chicks) and `1,009 -> 1,216 g` (`207 g` gap — only `6` valid images of `1` bird across `415 g`), plus `1,779 -> 1,986 g` (`207 g`), `1,340 -> 1,454 g` (`114 g`, since `1,351` is mislabeled), `1,992 -> 2,093 g` (`101 g`), `1,599 -> 1,689 -> 1,779 g` (`90 g` each).
- Collection timeline hiatus: 8-day gap between `2022-04-03` (~17 d, median `719 g`) and `2022-04-11` (~25 d, median `1,288 g`).
- Four visual scenarios: (A) bird on SF-400 scale over straw litter with LCD and buttons frequently visible; (B) top-down white plastic bucket (`~353` raw / `~351` clean imgs, `~20.1%`, severe white-on-white contrast drop for `>1,000 g` birds); (C) free single bird on straw litter / oblique/lateral angles (occasionally with `MY-TECH 0.7mm` pen); (D) multi-bird group scenes (`13` imgs in `417, 757, 823, 1520`).

## Quality Anomalies, Quarantine Set & Clean Dataset Totals
- source: docs/dataset/DOCUMENTACAO_DATASET_BROILER_COBB_AVIAN.md
- Critical visual leakage: SF-400 LCD displays exact gram weight in hundreds of images; platter and pen act as artificial field-absent cues.
- Multi-bird quarantine (`4` folders / `13` imgs): `417` (4 imgs, 2 chicks in red feeder), `757` (1 img, 2 birds), `823` (4 imgs, 4 chicks), `1520` (4 imgs, 2 birds) — folder names equal group weight sum.
- Severe single-bird mislabels (`2` folders / `2` imgs): `1465` (1 img, `2022-03-29`, ~380–465 g chick labeled `1,465 g`) and `1351` (1 img, `2022-04-03`, ~650–720 g bird labeled `1,351 g`).
- Confirmed LCD label typo (`1` folder / `6` imgs): `367` (`2022-03-29`) displays `267 g` on the SF-400 LCD (`+100 g` typo); §9 hardcodes `LCD_CORRECTIONS = {"367": 267}` while §8.1 notes verifying all 6 photos.
- Live-scale aleatoric noise (`±3–5 g`): `1599/1599 (1).jpg` reads `1596 g` on LCD; `719/IMG_20220411_210111.jpg` reads `716 g`.
- Clean single-bird totals: **1,738 images**, **160 unique weights**, **188 sessions** (`15` quarantined images across `6` folders/sessions). Domain breakdown: `~351` clean dorsal bucket images and `~1,387` clean box/scale images.

## Reference Python Ingestion Pipeline & Audit Recommendations
- source: docs/dataset/DOCUMENTACAO_DATASET_BROILER_COBB_AVIAN.md
- Section 9 reference rules: recursive `os.walk` leaf-folder crawl; `raw_weight = int(re.search(r"\d+", dir_p.name).group(0))`; `LCD_CORRECTIONS = {"367": 267}`; CP866 reversal (`encode("cp866").decode("utf-8")`) + Eastern-Arabic translation (`str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")`); `capture_dt = fn_dt or (mtime_dt - timedelta(seconds=14401))`; Pillow EXIF `274` (`Orientation`), IFD `0x8769` `37386` (`FocalLength`), `39321` JSON `zoomMultiple`; mandatory `ImageOps.exif_transpose(im).convert("RGB")`; `session_id = f"{fc}__s{sid}"` incrementing whenever sequential gap within `folder_chain` exceeds `600 s`.
- Validation & modeling guidance retained: 5-fold `StratifiedGroupKFold` grouped by `session_id` and stratified by weight band; report MAE (g), RMSE (g), MAPE (%), R², and Mean Bias Error globally and stratified by weight band (`<500 g`, `500–1000 g`, `>1000 g`), lens (`4.25 mm` vs. `1.65 mm`), and scenario; Huber / log-weight target transform and inverse-density / LDS reweighting for `>1,000 g` tail; preserve aspect ratio (`letterbox`, never stretch `20:9` to square); ban scale-altering random crops (`RandomResizedCrop`) while allowing rotation/flip/mild photometric transforms.
- Precedence note (auto-resolved in `.planning/INGEST-CONFLICTS.md`): DOC recommendations in §7/§8 for Deep Learning (`YOLO-Seg`, `SAM 2`, `Mask R-CNN`, `ConvNeXt`, `EfficientNetV2`, `EVA-02/DINOv2`, `MLP`), SF-400 LCD OCR (`PaddleOCR`/`EasyOCR`/`VLM`), CSV manifest output, `CatBoost`/`XGBoost`, bucket-rim `mm/px` calibration, blur-gate filtering, and Leave-One-Date-Out CV are superseded in v1 by higher-precedence `PRD.md` (`PROIBIÇÃO 1`, `PROIBIÇÃO 2`, `REQ-01`, `REQ-08`) and `.planning/REQUIREMENTS.md` (`HARD-01`–`HARD-04`).
