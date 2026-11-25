# PsyMix — Cross-Scale Taste Identifiability Workbench & JCIM Reproducibility Archive

[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-0072B2.svg)](LICENSE)
[![Workbench: 100% Client-Side Offline](https://img.shields.io/badge/Workbench-Zero--Backend%20%7C%20Offline%20Ready-1B7F6B.svg)](docs/index.html)
[![SHA-256 Provenance: 111/111 Verified](https://img.shields.io/badge/SHA--256%20Manifest-13%20Locks%20%2B%2098%20Sources-2C3E50.svg)](manifests/PsyMix_Submission_SHA256_Manifest.csv)

Official reproducibility repository and offline-capable interactive identifiability workbench accompanying:

> **From Molecular Taste Labels to Human Sensory Responses: Contributions of Chemical Information, Models, and Sensory Calibration in Mixtures and Foods**  
> Dedong Chen, Miao Zhang, Hua Zhu,\* Liba Xu,\* and Jiahao Lu\*  
> *Guangxi Key Laboratory of Zhuang and Yao Ethnic Medicine, Guangxi University of Chinese Medicine, Nanning 530200, Guangxi, China*  
> *Journal of Chemical Information and Modeling* (American Chemical Society, 2026)  
> **Repository:** `[https://github.com/Ruvio413/PsyMix]`  
> **Live Workbench (GitHub Pages):** `[https://ruvio413.github.io/PsyMix/]`  
> **Permanent Archive (Zenodo):** `[Zenodo DOI]`

---

## 1. Quick Start

### Option A — Launch the Static Identifiability Workbench (Zero Backend, 100% Offline)

No server, Python environment, or internet connection is required.

1. Download or unzip this repository (e.g., from Zenodo or GitHub).
2. Double-click **[`docs/index.html`](docs/index.html)** in any modern desktop browser (Chrome, Edge, Firefox, or Safari).
3. All frozen audit slices (`44`), model configurations (`153`), scored contrasts (`706`), curated literature thresholds (`25`), and interactive SVG simulators load locally from [`docs/assets/psymix_data_bundle.js`](docs/assets/psymix_data_bundle.js) with **zero `fetch()` calls, zero external CDN requests, and zero data uploads**.
4. To deploy on **GitHub Pages**, go to *Settings → Pages → Build and deployment → Deploy from a branch* and select `main` + `/docs`.

#### Workbench Modules (`docs/index.html`)
- **Tab 1 — 44-Slice & 706-Contrast Explorer:** Filter all 44 audited evaluation slices (`T1a`–`T6b`), A1–A6 audit states (`Pass` / `Warning` / `N/A`), 153 archived configurations across four model families, and 706 locked-factor contrasts (`information`, `model`, `calibration`, `descriptive`) strictly within native study scales (with explicit `No CI` badges for small-sample `n = 2–3` slices).
- **Tab 2 — Pre-Experiment Identifiability Workbench:** Load six one-click paper presets (`Apple Sugar`, `Tomato Sugar`, `Yu Green Tea k=4`, `Darjeeling Tea k=9`, `Orange Juice V09 α=37.17`, `Grape → Yogurt @1`) or drop a local formulation CSV ([sample template](docs/assets/sample_design_template.csv)) to run:
  - **A2 Collinearity & Dirichlet Dispersion Designer** (`r ≥ 0.98` alert, VIF, live seeded Dirichlet share simulation & inline SVG scatter);
  - **A3 Exact Combinatorial Derangement Resolution Floor** (`BigInt` subfactorial `D(k)`, `p_min = 1 / (1 + D(k))`, red alert at `k < 5`, prospective prescription `k ≥ 5`);
  - **A6 Single-Anchor Geometry & `@1` Error Amplification Simulator** (`DOT-log` curvature ray plot + `Source-affine` zero-crossing `-b/a` diagnostic);
  - **One-Click Pre-Registration Audit Receipt Export** (`.json` + `.md` with UTC timestamp and genuine client-side FIPS-180-4 SHA-256 digest working both online and via `file://`).
- **Tab 3 — Curated Parameters & Protocol Provenance:** Frozen curvature constants (`DOT-log` `K = 2.33647` / `1.27707`, `DOT-rat` `K = 7.16283` / `7.62473`), searchable 25-entry medium-specific threshold dictionary with DOIs (including the rutin / kaempferol-3-O-rutinoside / myricetin-3-O-rutinoside astringency cliffs), and the `T1b` pharmacopoeial flavor harmonization table.

### Option B — Verify Cryptographic SHA-256 Provenance (One Command)

Uses Python 3.10+ standard library only (zero third-party packages):

```bash
python verify_manifest.py
```

Expected verification summary on the frozen archive:
- `pre_outcome_or_sealed_lock`: **13 / 13 PASS**
- `submission_source_archive`: **98 / 98 PASS**
- **OVERALL TOTAL: 111 / 111 PASS (100% VERIFIED)**

If verifying a relocated archive mirror, remap root path prefixes via `--remap OLD_PREFIX=NEW_PREFIX` (repeatable) and optionally export a JSON proof with `--json-report verification_report.json`.

---

## 2. Manuscript-to-Code Crosswalk

Every main figure, supplementary figure, main table, and supporting table in the JCIM submission maps directly to archived scripts and frozen evaluation tables below:

| Manuscript Item | Description | Script(s) | Frozen Input Table(s) / Provenance | Output Artifact(s) |
|---|---|---|---|---|
| **Figure 1** | Hard-gate qualification funnel, 5-tier `(S, P, Q, W, A)` contrast design (706 contrasts), and 44-slice A1–A6 audit matrix | `scripts/build_batch1_fig1_fig2.py` | `tables/Table_S3_44slice_identifiability_audit.csv`, `tables/PsyMix_Model_Information_Comparison_Registry.csv` | `PsyMix_Figure1.{pdf,svg,png}` |
| **Figure 2** | Qualitative molecular (`T1a`, canonical vs Murcko scaffold) & botanical (`T1b`, Zhuang645 + CP2025) classification and probability saturation on sweeteners / tea | `scripts/build_batch1_fig1_fig2.py` | `tables/Table_S2_full_model_mapping.csv`, `tables/T2_3_tea_omission_paired.csv` | `PsyMix_Figure2.{pdf,svg,png}` |
| **Figure 3** | Fruit analytical signal gains (`TREE-pres` vs `TREE-conc-all`), potency weighting vs `SUM-mass`, A2 collinearity (`r = 0.982–0.999`), and Dirichlet dispersion blueprint (`13.3–28.9 pp`) | `scripts/build_batch2_fig3_fig4.py` | `tables/T2_2_fruit_weighting_identifiability.csv`, `tables/A2_fruit_collinearity.csv`, `tables/A2_sugar_design_requirements.csv` | `PsyMix_Figure3.{pdf,svg,png}` |
| **Figure 4** | Binary sweetener network (`T3`), dual-endpoint ranking inversion (`TREE-conc` vs `EQS×ID`), pair residual reproducibility (`T4a`, 86.5% var.), and ternary transfer (`T4b`, `n = 64`) | `scripts/build_batch2_fig3_fig4.py` | `tables/Table_S4_sweetener_controls_and_adapters.csv`, `tables/PsyMix_Model_Information_Comparison_Registry.csv` | `PsyMix_Figure4.{pdf,svg,png}` |
| **Figure 5** | Cross-crop calibration (`@0 → @5 → @10`, `T5`), grape→yogurt `-b/a = 34.613 g L⁻¹` algebraic retention, and orange juice `V09` (`α = 37.17`) vs `H10` ray diagram (A6) | `scripts/build_batch3_fig5_fig6.py` | `tables/T6_real_food_perturbations.csv`, `tables/T6_all_model_metrics.csv`, `tables/PsyMix_Model_Information_Comparison_Registry.csv` | `PsyMix_Figure5.{pdf,svg,png}` |
| **Figure 6** | 17-slice real-food perturbation `RER` forest (`T6`), A3 tea derangement ranks (`k = 4, 9`), rutin/kaempferol-3-O-rutinoside cliff, orange juice suppression, yogurt, Ancha tea, and locked coffee diagnostic | `scripts/build_batch3_fig5_fig6.py` | `tables/T6_real_food_perturbations.csv`, `tables/T6_all_model_metrics.csv`, `tables/T6_paired_contrasts.csv`, `tables/T2_3_tea_omission_paired.csv` | `PsyMix_Figure6.{pdf,svg,png}` |
| **TOC Graphic** | 3.25 × 1.75 in ACS Graphical Abstract summarizing the quantitative sensory bottleneck & identifiability gate | `scripts/build_toc_and_figs1.py` | Frozen headline invariants (`153` configs, `44` slices, `A1–A6`) | `PsyMix_TOC_Graphic.{pdf,svg,png}` |
| **Figure S1** | 4-family architecture taxonomy (`SET-LM+comp-LR`, 768-bit ECFP4 + CatBoost, Set-attention / Set Transformer, continuous-dose adapters) & `T1b`/`T3` capacity ceiling test | `scripts/build_toc_and_figs1.py`, `scripts/qa_and_export_toc_s1.py` | `tables/Table_S2_full_model_mapping.csv`, `tables/Table_S4_sweetener_controls_and_adapters.csv` | `PsyMix_FigureS1.{pdf,svg,png}` |
| **Figure S2** | 2×2 diagnostic disconnects (S2a: plant-milk FoodAtlas priors & 1,335 FlavorDB bags; S2b: Pingyin rose & licorice matrices; S2c: 3%+5%→7% SE upward dose extrapolation failure; S2d: RDKit 2D activity cliffs & in vitro vs oral decoupling) | `scripts/build_figs2.py` | Curated monomer threshold metadata (Supporting Information §S3) + RDKit 2D structures | `PsyMix_FigureS2.{pdf,svg,png}` |
| **Table 1–3** | Task overview (`Table 1`, T1–T6), unseen binary sweetener benchmarks (`Table 2`, T3, 55 pairs), and real compositional perturbations (`Table 3`, T6, 17 slices) | Manuscript master + `docs/assets/build_data_bundle.py` | `tables/PsyMix_Model_Information_Comparison_Registry.csv`, `tables/T6_all_model_metrics.csv`, `tables/T6_real_food_perturbations.csv` | Main Manuscript (`Tables 1–3`) |
| **Table 4 & Table S3** | Condensed & complete 44-slice `A1–A6` attribution identifiability audit and prospective wet-lab design prescriptions | `docs/assets/build_data_bundle.py` | `tables/Table_S3_44slice_identifiability_audit.csv`, `tables/A2_fruit_collinearity.csv`, `tables/A2_sugar_design_requirements.csv` | Main Manuscript (`Table 4`), SI (`Table S3`), Workbench Tab 1 |
| **Tables S1–S2** | Complete dataset/source provenance history (`S1`) and 153-configuration task–input–model crosswalk (`S2`: 98 main + 55 SI) | `docs/assets/build_data_bundle.py` | `tables/Table_S1_analysis_history.csv`, `tables/Table_S2_full_model_mapping.csv` | Supporting Information (`Tables S1–S2`), Workbench Tab 1 |
| **Table S4** | Sweetener fixed-input controls, continuous-dose adapters (`ADPT-pool/set/dyn`), and graph-completion ablation ledger | `scripts/build_toc_and_figs1.py` | `tables/Table_S4_sweetener_controls_and_adapters.csv` | Supporting Information (`Table S4`), Figure S1 |
| **Tables S5–S9** | FART 5-label benchmark & overlap audit (`S5`), exploratory response-exponent grid (`DOT-log(η)@1`, `S6`), exploratory caffeine–rutin bitterness modulation (`DOT-log+CR@1`, `S7`), ternary cold-ingredient proxy & yogurt digitization sensitivities (`S8`), and Darjeeling probability weighting / herbal infusion / plant-milk prior ablations (`S9`) | `docs/assets/build_data_bundle.py` | `tables/T6_real_food_perturbations.csv`, `tables/T2_3_tea_omission_paired.csv`, Supporting Information archived CSV ledgers | Supporting Information (`Tables S5–S9`), Workbench Tab 3 |
| **SHA-256 Lock Audit** | Byte-level integrity audit of 13 pre-outcome/sealed locks + 98 frozen archive source files (`111/111`) | `verify_manifest.py` | `manifests/PsyMix_Submission_SHA256_Manifest.csv` | Console / JSON verification report |

---

## 3. Repository Structure

```text
PsyMix_Release_Package/
├── README.md                                 # Overview, quick start, and Manuscript-to-Code Crosswalk
├── LICENSE                                   # MIT code license + third-party data provenance notice
├── CITATION.cff                              # Citation File Format v1.2.0 metadata
├── verify_manifest.py                        # Pure-stdlib SHA-256 verifier (13 locks + 98 sources = 111)
├── docs/                                     # GitHub Pages root (also 100% offline via file://)
│   ├── index.html                            # Zero-backend static identifiability workbench (Tabs 1–3)
│   └── assets/
│       ├── build_data_bundle.py              # Rebuilds psymix_data_bundle.js from frozen CSV tables
│       ├── psymix_data_bundle.js             # Pre-compiled offline JS data bundle (44 slices, 153 models, 706 contrasts)
│       └── sample_design_template.csv        # Sample local formulation CSV for Tab 2 upload
├── manifests/
│   └── PsyMix_Submission_SHA256_Manifest.csv # 111 byte-level SHA-256 records (13 locks + 98 sources)
├── tables/                                   # PsyMix-derived summary & audit CSVs (no third-party raw tables)
│   ├── A2_fruit_collinearity.csv
│   ├── A2_sugar_design_requirements.csv
│   ├── PsyMix_Model_Information_Comparison_Registry.csv
│   ├── T2_2_fruit_weighting_identifiability.csv
│   ├── T2_3_tea_omission_paired.csv
│   ├── T6_all_model_metrics.csv
│   ├── T6_paired_contrasts.csv
│   ├── T6_real_food_perturbations.csv
│   ├── Table_S1_analysis_history.csv
│   ├── Table_S2_full_model_mapping.csv
│   ├── Table_S3_44slice_identifiability_audit.csv
│   └── Table_S4_sweetener_controls_and_adapters.csv
└── scripts/                                  # Publication figure-generation & QA scripts (PDF/SVG/PNG)
    ├── build_batch1_fig1_fig2.py
    ├── build_batch2_fig3_fig4.py
    ├── build_batch3_fig5_fig6.py
    ├── build_toc_and_figs1.py
    ├── build_figs2.py
    └── qa_and_export_toc_s1.py
```

---

## 4. Environment Requirements & Data Licensing

- **Workbench (`docs/index.html`)**: Any modern browser (Chromium/Edge, Firefox, Safari). Works completely offline (`file://`) with zero external dependencies.
- **Manifest Verification (`verify_manifest.py`)**: Python `>= 3.10` standard library only.
- **Figure Regeneration (`scripts/*.py`)**: Python `>= 3.10` with `numpy`, `pandas`, `scipy`, `matplotlib >= 3.8`, `pillow`, `pymupdf` (and `rdkit` for 2D chemical structures in `build_figs2.py`). All figures export with embedded TrueType fonts (`pdf.fonttype = 42`, `ps.fonttype = 42`, `svg.fonttype = 'none'`).
- **Third-Party Data Notice**: This release distributes only PsyMix-owned code, audit summaries, derived evaluation metrics, and curated literature parameters with DOI provenance. Restricted or independently published raw analytical/sensory datasets remain under their original authors' citation and licensing terms (see [`LICENSE`](LICENSE) and Supporting Information Table S1).

---

## 5. Authors, Correspondence & Funding

- **Authors**: Dedong Chen, Miao Zhang, Hua Zhu,\* Liba Xu,\* and Jiahao Lu\*
- **Affiliations**: Guangxi University of Chinese Medicine; Guangxi Key Laboratory of Zhuang and Yao Ethnic Medicine; University Engineering Research Center of Development and Industrialization of Zhuang and Yao Ethnic Medicinal Materials, Guangxi; The Collaborative Innovation Center of Zhuang and Yao Ethnic Medicine; Guangxi Engineering Research Center of Ethnic Medicine Resources and Application, Nanning 530200, Guangxi, China.
- **Corresponding Authors**: Hua Zhu (`zhuhuagx@163.com`), Liba Xu (`xulb@gxtcmu.edu.cn`), and Jiahao Lu (`lujiahao2022@126.com`).
- **Funding**: Supported by the Guangxi Key Laboratory of Zhuang and Yao Ethnic Medicine Capacity Building Project (Guike LT2600640036), the University Engineering Research Center of Development and Industrialization of Zhuang and Yao Ethnic Medicinal Materials, Guangxi (Gui Jiao Ke Yan [2024] No. 10), the Guangxi Engineering Research Center of Ethnic Medicine Resources and Application (Gui Fa Gai Gao Ji Han [2020] No. 2605), the Guangxi Key Discipline of Zhuang Pharmacy (GZXK-Z-20-64), and the Guangxi First-Class Discipline of Chinese Materia Medica (Ethnic Pharmacy) (Gui Jiao Ke Yan [2018] No. 12).
