---
type: reference
title: TB-E3 dataset availability — what actually exists, where, and with which strings attached
status: current
as_of: 2026-09-01
source: "web verification, 2026-09-01 (three research passes: synthetic / tampered / camera-real)"
tags: [track-b, curation, datasets, tb-e3]
links: [specs/tb-e3-curation-ladder, snapshots/2026-09-01-backbone-grid]
---

# TB-E3 dataset availability (verified 2026-09-01)

Runbook step R1 of `specs/tb-e3-curation-ladder.md`. Verified via official pages the same day —
academic hosts rot, re-check anything older than a few months.

## Synthetic environments

| dataset | alive | access | license | size | labels | caveat that matters |
|---|---|---|---|---|---|---|
| GenImage | yes (GDrive verified) | direct | none stated (ImageNet reals ⇒ research-only) | ~2.7M imgs, ~500 GB | per-generator folders, 8 generators | **documented JPEG+size confound** — detectors learn format, not generation; use unbiased-genimage.org bias-controlled splits or let gate G1 force matched compression |
| Community Forensics | yes (HF, ungated) | HF streaming | CC-BY-4.0 research-only | 2.76M fakes, **4,803 generators**, 1.08 TB (Small: 278 GB) | per-image generator + prompt | reals ship as pointers (LAION/ImageNet/CelebA), must fetch separately — except the redistributable `-Small` |
| WildFake | **unverified** | ModelScope (CN), JS-gated | not found | 3.69M imgs | hierarchical generator taxonomy | only host is ModelScope; license and download mechanics unconfirmed — deprioritize or email authors |
| ArtiFact | yes (Kaggle) | Kaggle account | mixed, research-only in aggregate | 2.5M imgs, all **200×200 px** | folder-level generator id | pre-JPEG'd + downsampled; 200px < CLIP's 224 input — uniform upsampling artifact, note in captions |
| Synthbuster | yes (Zenodo, checksummed) | direct | CC BY-NC-SA 4.0 | 9k fakes, 12.4 GB | per-generator folders + prompts.csv | test-scale; undegraded on purpose; RAISE-1k reals are a separate portal download |
| ELSA D3 | yes (HF) | HF streaming | undeclared on card | ~9.2M fakes, 2.63 TB | per-variant generator id (4 diffusion gens) | reals are **LAION URLs** — expect link-rot loss in 2026; license unconfirmed |
| Chameleon | repo alive, data email-gated | author request | academic-only | 26k imgs | category labels, generator community-reported | **eval-only by design** (as planned); no reliable per-generator labels; lead time for the email |

**R1 synthetic verdict:** primary = Community Forensics (generator diversity, per-image labels,
ungated) + GenImage (scale; bias-controlled splits mandatory). Synthbuster = small clean env or
eval. ELSA D3 = optional scale, budget for link-rot. WildFake = out unless authors answer.
Chameleon = request now, eval-only.

## Tampered environments

| dataset | alive | access | license | size | labels | caveat that matters |
|---|---|---|---|---|---|---|
| AUDITS (supersedes MAGIC) | yes (HF, ungated) | HF direct | MIT card (atop VisualNews/COCO sources) | 530k imgs, 37.2 GB | GT masks, **11 diffusion methods** labeled, train/val/test | MAGIC itself was never released — AUDITS is its usable form; COCO/VisualNews-derived ⇒ cross-dedup |
| TGIF → **TGIF2** | yes (GitHub links) | direct, no reg | CC BY-SA 4.0 | TGIF 75k + TGIF2 adds 197k FLUX.1 ⇒ ~272k fakes | seg+bbox masks; method in filename (SD2/SDXL/Firefly/FLUX.1) | get TGIF2: random rectangular masks added because semantic-mask-only sets let detectors cheat on object boundaries; JPEG **and WebP** variants |
| SAGI-D | yes (Kaggle) | Kaggle account | CC BY 4.0 | >95k inpainted | masks, per-model provenance, human realism ratings | actual models: BrushNet, PowerPaint, HD-Painter, ControlNet, Inpaint-/Remove-Anything; **sources include RAISE** ⇒ dedup vs our reals |
| DEFACTO | yes (Kaggle mirrors) | Kaggle | none of its own (COCO terms) | ~190–229k | probe + donor masks, per-category folders | fully COCO-derived (leakage vs everything COCO-based); automated ⇒ some implausible forgeries; murky redistribution |
| IMD2020 | yes (TLS cert broken — curl -k) | direct zips | none stated, cite WACVW'20 | 2,010 real-life manip. + 35k GAN-inpaint + 35k camera-real | manual masks on the 2,010 | the only *human-made in-the-wild* edits — small but irreplaceable; inpainting half is one pre-diffusion 2018 GAN |
| NIST MFC | info pages yes; **portal DOWN** (Cloudflare 524) | signup + 2 signed agreements + email mfc_poc@nist.gov; **days–weeks** | Data Use Agreement | ~100k manip. imgs, ~176k pristine | richest: masks + per-operation manipulation-history journals | start the request NOW or drop it; program "no longer actively updated" |
| CASIA v2 | official dead; community mirrors | GDrive via mirrors | never licensed; academic citation | 12.6k imgs, 2.6 GB | third-party GT (use SunnyHaze corrected) | **metadata alone reaches ~0.92 AUC** (TIFF-vs-JPEG split, single quantization table) — gate G1 poster child; re-encode to uniform JPEG or exclude |
| AutoSplice | yes (GitHub→GDrive) | direct | academic-only | 5.9k imgs | masks + captions, local/global labels | QF mismatch shortcut (fakes QF100/90 vs reals 75); authors ship a JPEG-75 fake variant as the control — use it |

**R1 tampered verdict:** primary = AUDITS + TGIF2 (scale, masks, method labels, clean licenses);
SAGI-D adds tool diversity (dedup vs RAISE). IMD2020 in as the only human-made in-the-wild env.
DEFACTO optional (COCO leakage tax). NIST MFC: fire the request today, treat as a later bonus.
CASIA v2 only re-encoded and never for headline numbers. AutoSplice only via its JPEG-75 control.
L4 candidates (held-out tools): reserve e.g. Firefly (TGIF2) or one AUDITS method never trained on.

## Overall R1 verdict

Confirmed-and-clean core for the hybrid: **SID-Set + Community Forensics + GenImage(unbiased
splits) [synthetic] + AUDITS + TGIF2 + SAGI-D + IMD2020 [tampered] + VISION (+RAISE) [reals]**.
Eval-only: So-Fake-OOD, Chameleon (email now). Dropped: WildFake (unverifiable), Dresden
(host dead), MAGIC (never released), FakeClue (policy). Every "caveat" column above is a G1/G2
work item, not trivia — CASIA and GenImage prove the shortcut risk is real, at ~0.9 AUC.

## Camera-native reals

| dataset | alive | access | license | size | devices | caveat that matters |
|---|---|---|---|---|---|---|
| VISION | yes (open nginx index) | direct, wget lists | **CC BY-SA 4.0** | 34k imgs + 1.9k videos, ~134 GB (±30%, self-estimated) | 35 smartphones, 11 brands | cleanest option; native + Facebook/WhatsApp/YouTube re-shared versions = free "web-recompressed real" env; 2019 errata: 6 D03/D19 videos misplaced |
| RAISE | yes (form-gated, immediate) | lightweight form | custom, non-commercial | 8,156 RAW, ~350 GB (RAISE-1k/2k/4k subsets exist) | **3 Nikon DSLRs only** | RAW-only (NEF) — you develop the JPEGs yourself; great pixel purity, poor device diversity |
| Dresden | **DEAD** (host refused/timeout 2026-09-01) | unofficial Kaggle mirror only (53.2 GB, JPEG-only) | original "free for science"; mirror's CC0 tag is uploader-applied, not authoritative | >14k imgs | 73 devices, 25 models | mirror provenance unverified (byte-identity unknown, RAW/flat-field sets absent) — use only if device diversity is worth the provenance asterisk |

**R1 reals verdict:** primary = VISION (diversity + license + its social-media re-shares cover the
web-recompressed cell natively). RAISE = RAW purity + it's Synthbuster's paired real source.
Dresden = skip unless 73-device diversity becomes necessary; mirror is unofficial.
