# -*- coding: utf-8 -*-
"""
Build PsyMix_Release_Package/docs/assets/psymix_data_bundle.js directly from frozen source CSVs.
Zero external fetch() calls required at runtime: works 100% offline via file:// (Zenodo ZIP)
and live on GitHub Pages. Never bundles restricted third-party raw datasets.

All 153 models are exported in clean, standardized manuscript English with exact 4-family
classification (matching Figure S1), explicit per-model task arrays, zero internal engineering
codes, parsed human-readable audit notes (zero raw JSON fragments), strict small_n_no_ci flags
for all n <= 3 slices (7) and contrasts (82), harmonized coffee lock evidence tier, calibrated
tea presets, and DOI-linked Rutin / Kaempferol-3-O-rutinoside / Myricetin-3-O-rutinoside entries.
"""

import json
import math
import os
import re
import shutil
import numpy as np
import pandas as pd

REL_ROOT = r'c:\gemi-antigravity\PsyMix_Release_Package'
DOCS_ASSETS = os.path.join(REL_ROOT, 'docs', 'assets')
TABLES_DIR = os.path.join(REL_ROOT, 'tables')
MANIFEST_DIR = os.path.join(REL_ROOT, 'manifests')
SCRIPTS_DIR = os.path.join(REL_ROOT, 'scripts')

for d in [DOCS_ASSETS, TABLES_DIR, MANIFEST_DIR, SCRIPTS_DIR]:
    os.makedirs(d, exist_ok=True)

# Frozen source files (read-only)
CSV_AUDIT44  = r'C:\zhuang_taste_pre\psymix_final_tables_for_writing\Table_4_identifiability_audit.csv'
CSV_S1       = r'C:\zhuang_taste_pre\psymix_final_tables_for_writing\Table_S1_analysis_history.csv'
CSV_S2       = r'C:\zhuang_taste_pre\psymix_final_tables_for_writing\Table_S2_full_model_mapping.csv'
CSV_REG      = r'C:\zhuang_taste_pre\PsyMix_Bench_Audit_20261123\PsyMix_Model_Information_Comparison_Registry.csv'
CSV_A2_COLL  = r'C:\zhuang_taste_pre\psymix_final_tables_for_writing\A2_fruit_collinearity.csv'
CSV_A2_REQ   = r'C:\zhuang_taste_pre\psymix_final_tables_for_writing\A2_sugar_design_requirements.csv'
CSV_T6_PERT  = r'C:\zhuang_taste_pre\psymix_final_tables_for_writing\T6_real_food_perturbations.csv'
CSV_C1_MAIN  = r'c:\gemi-antigravity\psymix_paper_freeze_20261122\T1_C1_controls\C1_main_table_v2.csv'
CSV_TEA_POT  = r'C:\zhuang_taste_pre\phase30_independent_validation_20261116\tea\components_potency.csv'
CSV_MANIFEST = r'c:\gemi-antigravity\00_PsyMix_Submission_Master\03_Provenance_and_Audit_Ledger\PsyMix_Submission_SHA256_Manifest.csv'


def safe_val(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        fv = float(v)
        if math.isnan(fv):
            return None
        if math.isinf(fv):
            return 'inf'
        return int(fv) if fv.is_integer() and abs(fv) < 1e12 else round(fv, 6)
    s = str(v).strip()
    if s == '' or s.lower() in ('nan', 'none', 'null'):
        return None
    return s


def to_num_or_none(v):
    """Parse integer or float safely (including numeric strings like '2' or '3.0')."""
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        fv = float(v)
        if math.isnan(fv) or math.isinf(fv):
            return None
        return int(fv) if fv.is_integer() else round(fv, 6)
    s = str(v).strip()
    try:
        fv = float(s)
        if math.isnan(fv) or math.isinf(fv):
            return None
        return int(fv) if fv.is_integer() else round(fv, 6)
    except (ValueError, TypeError):
        return None


def sanitize_text(s):
    """Remove internal engineering tags and normalize model labels cleanly."""
    if not s:
        return ''
    out = str(s).strip()
    replacements = [
        ('RIDGE-conc-fixed8', 'RIDGE-conc'),
        ('A2aET', 'CLS-FPDescLM-ET'),
        ('A2aLR', 'CLS-FPDescLM-LR'),
        ('A2a', '104-d FP+Desc+LM'),
        ('B2_log', 'DOT-log'),
        ('B1_linear', 'DOT-lin'),
        ('M0_sum', 'SUM-mass'),
        ('M1_correct', 'DOT-lin'),
        ('M2_correct', 'DOT-log'),
        ('archive:M3-graph', 'SELF+GRAPH'),
        ('archive:M3-local', 'SELF+LOCAL-SIM'),
        ('archive:M2', 'SELF+ID'),
        ('archive:Additive Null', 'ADD-nominal'),
        ('archive:Pair-wise Level Extrapolation', 'PAIR-level-extrap'),
        ('archive:PsyMix-id', 'SELF+ID (SE3+5→7)'),
        ('archive:PsyMix-pair', 'SELF+PAIR (SE3+5→7)'),
        ('archive:', ''),
        ('equal_mgL', 'SUM-mass'),
        ('fold_fits alpha', 'ridge regularization penalty'),
    ]
    for old, new in replacements:
        out = out.replace(old, new)
    # Remove any phaseNN_... directory/run markers
    out = re.sub(r'\bphase\d+[a-z0-9_/-]*\b', 'archived protocol', out, flags=re.IGNORECASE)
    out = re.sub(r'\s+', ' ', out).strip()
    return out


def sanitize_model_name(name):
    s = str(name).strip()
    direct = {
        'RIDGE-conc-fixed8': 'RIDGE-conc',
        'archive:M2': 'SELF+ID',
        'archive:M3-graph': 'SELF+GRAPH',
        'archive:M3-local': 'SELF+LOCAL-SIM',
        'archive:Additive Null': 'ADD-nominal',
        'archive:Pair-wise Level Extrapolation': 'PAIR-level-extrap',
        'archive:PsyMix-id': 'SELF+ID (SE3+5→7)',
        'archive:PsyMix-pair': 'SELF+PAIR (SE3+5→7)',
    }
    if s in direct:
        return direct[s]
    return sanitize_text(s)


def classify_task_group(task_str):
    t = str(task_str)
    if t.startswith('T1/P0') or t.startswith('T1/FART'):
        return 'T1a'
    if t.startswith('T1/'):
        return 'T1b'
    if t.startswith('T2/'):
        return 'T2'
    if t.startswith('T3/'):
        return 'T3'
    if t.startswith('T4/'):
        return 'T4'
    if t.startswith('T5/'):
        return 'T5'
    if any(k in t for k in ['McClure', 'ancha', 'coffee']):
        return 'T6b'
    if t.startswith('T6/'):
        return 'T6a'
    return 'Other'


def derive_audit_states(row):
    """Derive clean Pass / Warning / N/A status for A1-A6 strictly matching Figure 1c / Table 4."""
    t = str(row['task'])
    if 'FART fixed' in t or 'Zhuang random' in t:
        a1 = 'Warning'
    else:
        a1 = 'Pass'
    r_val = row['A2_r']
    if pd.isna(r_val):
        a2 = 'Warning' if 'Ahmad' in t else 'N/A'
    else:
        a2 = 'Warning' if float(r_val) >= 0.98 else 'Pass'
    k_val = row['A3_k_molecules']
    obs_p = row['A3_observed_p']
    if pd.isna(k_val):
        a3 = 'N/A'
    else:
        a3 = 'Warning' if (float(k_val) < 5 or (not pd.isna(obs_p) and float(obs_p) > 0.05)) else 'Pass'
    cov = row['A4_count_fraction']
    if 'coffee' in t or 'Dea/bitterness/acid' in t or 'Dea/bitterness/sugar' in t or 'grape/sourness' in t or 'ancha' in t:
        a4 = 'Warning'
    elif not pd.isna(cov) and float(cov) < 0.35:
        a4 = 'Warning'
    elif t.startswith('T1/Zhuang') or t.startswith('T1/Pharmacopeia'):
        a4 = 'Warning'
    else:
        a4 = 'Pass'
    a5_det = str(row.get('A5_details', ''))
    if 'Zhuang' in t or 'Pharmacopeia' in t:
        a5 = 'Warning'
    elif 'N/A' in a5_det and len(a5_det) < 10:
        a5 = 'N/A'
    else:
        a5 = 'Pass'
    alpha = row['A6_alpha_I0_over_H0']
    if 'Dea' in t or 'ancha/bitter' in t or 'Ahmad' in t or 'McClure' in t or 'coffee' in t:
        a6 = 'Warning'
    elif pd.isna(alpha):
        a6 = 'Pass' if t.startswith('T5/') else 'N/A'
    else:
        a6 = 'Warning' if float(alpha) >= 10.0 else 'Pass'
    return {'A1': a1, 'A2': a2, 'A3': a3, 'A4': a4, 'A5': a5, 'A6': a6}


def infer_input_tuple(task_grp, task_str):
    mapping = {
        'T1a': 'I = (S, —, —, —, A_class)',
        'T1b': 'I = (S, P_set, —, —, A_multi)',
        'T2':  'I = (S, P, Q, W_proxy, A_train)',
        'T3':  'I = (S, P, Q, W_EQS, A_train)',
        'T4':  'I = (S, P, Q, W_EQS/SELF, A_train)',
        'T5':  'I = (S, P, Q, W_proxy, A_@0/@5/@10)',
        'T6a': 'I = (S, P, Q, W_DoT, A_@1)',
        'T6b': 'I = (S, P, Q, W_DoT/none, A_@0/@1)',
    }
    return mapping.get(task_grp, 'I = (S, P, Q, W, A)')


def format_audit_note(col_key, raw_cell):
    """Parse JSON cell from Table_4_identifiability_audit.csv into a crisp plain-English summary."""
    if pd.isna(raw_cell):
        return ''
    s = str(raw_cell).strip()
    if not s:
        return ''
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            items = [x for x in parsed if isinstance(x, dict)]
            if not items:
                return sanitize_text(str(parsed))
            item = items[0]
        elif isinstance(parsed, dict):
            items = [parsed]
            item = parsed
        else:
            return sanitize_text(str(parsed))
    except Exception:
        return sanitize_text(s)

    parts = []
    if col_key == 'A2':
        status = item.get('status')
        r_v = item.get('r')
        vif_v = item.get('VIF')
        ci_lo = item.get('CI95_low')
        ci_hi = item.get('CI95_high')
        base = item.get('baseline')
        cand = item.get('candidate')
        analytes = item.get('analytes')
        if r_v is not None:
            r_str = f"r = {float(r_v):.3f}"
            if ci_lo is not None and ci_hi is not None:
                r_str += f" (95% CI [{float(ci_lo):.3f}, {float(ci_hi):.3f}])"
            if vif_v is not None:
                vf = float(vif_v)
                r_str += f", VIF = {'∞' if (math.isinf(vf) or vf > 1e7) else f'{vf:.1f}'}"
            parts.append(r_str)
            if base and cand:
                parts.append(f"comparing {base} vs {cand}")
            if analytes:
                parts.append(f"analytes: {str(analytes).replace(';', ', ')}")
            if status:
                parts.append(f"flag: {status}")
        elif status:
            parts.append(str(status))
            if base and cand:
                parts.append(f"({base} vs {cand})")

    elif col_key == 'A3':
        k = item.get('k_molecules')
        d = item.get('D_labeled_molecule_derangements')
        min_p = item.get('minimum_add_one_p_all_labeled_derangements')
        obs_p = item.get('observed_p')
        actual_null = item.get('actual_null')
        note = item.get('note')
        if k is not None:
            k_i = int(round(float(k)))
            seg = f"k = {k_i} active molecules"
            if d is not None:
                seg += f", D({k_i}) = {d} derangements"
            if min_p is not None:
                seg += f", min add-one p = {float(min_p):.3f}"
            if obs_p is not None:
                seg += f", observed p = {float(obs_p):.3f}"
            parts.append(seg)
            if actual_null:
                parts.append(f"null: {actual_null}")
        elif actual_null:
            parts.append(str(actual_null))
        if note:
            parts.append(str(note))

    elif col_key == 'A4':
        rep = item.get('reported_components')
        mat = item.get('matched_components')
        frac = item.get('count_fraction')
        denom = item.get('coverage_denominator')
        proc = item.get('processing_products')
        mass_mean = item.get('matched_mass_fraction_of_compatible_panel_mean')
        fresh_mean = item.get('matched_mass_fraction_of_fresh_tissue_mean')
        note = item.get('note')
        if mat is not None and rep is not None:
            seg = f"Matched {int(round(float(mat)))}/{int(round(float(rep)))} reported components"
            if frac is not None:
                seg += f" ({float(frac)*100:.1f}%)"
            parts.append(seg)
        elif mat is not None:
            parts.append(f"Matched {int(round(float(mat)))} threshold-compatible analytes")
        if mass_mean is not None:
            mm = f"mean panel mass coverage {float(mass_mean)*100:.1f}%"
            if fresh_mean is not None:
                mm += f" ({float(fresh_mean)*100:.1f}% fresh tissue)"
            parts.append(mm)
        if denom:
            parts.append(str(denom))
        if proc:
            parts.append(f"unmodeled/processing: {proc}")
        if note:
            parts.append(str(note))

    elif col_key == 'A6':
        # Prefer DOT-log / DOT-meas records over unscaled SUM-molar entries;
        # when multiple anchors exist (e.g., Dea orange juice V09 & H10), sort by highest alpha and show all.
        pref = [
            it for it in items
            if any(tok in str(it.get('model') or '') for tok in ('DOT-log', 'DOT-meas'))
        ]
        selected = pref if pref else [item]
        selected = sorted(
            selected,
            key=lambda it: float(it['alpha_I0_over_H0']) if it.get('alpha_I0_over_H0') is not None else -1.0,
            reverse=True,
        )
        if len(selected) == 1:
            sel = selected[0]
            model = sel.get('model')
            budget = sel.get('budget')
            i0 = sel.get('I0')
            h0 = sel.get('H0')
            alpha = sel.get('alpha_I0_over_H0')
            ref = sel.get('reference')
            note = sel.get('note')
            if model or budget:
                seg = f"Calibration: {budget or 'N/A'}"
                if model:
                    seg += f" ({model})"
                parts.append(seg)
            if i0 is not None and h0 is not None and alpha is not None:
                parts.append(f"anchor I0 = {float(i0):.2f}, H0 = {float(h0):.4f}, multiplier α = {float(alpha):.2f}")
            if ref:
                parts.append(f"reference: {ref}")
            if note:
                parts.append(str(note))
        else:
            lead = selected[0]
            parts.append(f"Calibration: {lead.get('budget') or '@1'} ({lead.get('model') or 'DOT-log@1'})")
            for sel in selected:
                ref_lbl = sel.get('reference') or sel.get('model') or 'anchor'
                i0, h0, alpha = sel.get('I0'), sel.get('H0'), sel.get('alpha_I0_over_H0')
                if i0 is not None and h0 is not None and alpha is not None:
                    parts.append(f"{ref_lbl}: I0 = {float(i0):.2f}, H0 = {float(h0):.4f}, α = {float(alpha):.2f}")
            if lead.get('note'):
                parts.append(str(lead.get('note')))

    text = '; '.join(p.strip().rstrip(';.') for p in parts if p and str(p).strip())
    return sanitize_text(text)


# ==============================================================================
# CURATED 100% ENGLISH 153-MODEL CATALOG (ID 1..153, exact Figure S1 families & tasks)
# ==============================================================================
# Each entry: id -> (clean_model, clean_context, tasks_list, clean_definition)
MODEL_CATALOG_EN = {
    1:  ('CLS-FPDescLM-ET', 'T1a Molecular taste (P0 fixed & Murcko scaffold)', ['T1a'],
         '104-d FP+Desc+LM representation (32-d projected 2048-bit fingerprint + 40 descriptors + 32-d projected generic ChemBERTa CLS); 160 ExtraTrees (min_samples_leaf=1, max_features=sqrt).'),
    2:  ('CLS-FPDescLM-LR', 'T1a Molecular taste (P0 fixed & Murcko scaffold)', ['T1a'],
         '104-d FP+Desc+LM representation (32-d projected fingerprint + 40 descriptors + 32-d generic ChemBERTa CLS); per-label standardized logistic regression (C=1).'),
    3:  ('CLS-UF-DLMTF', 'T1a Molecular taste (P0 fixed & Murcko scaffold)', ['T1a'],
         'Local reproduction of official Uni-Flavor dual-branch neural architecture (768-d Uni-Mol2 + 300-d Mordred, product interaction, label queries, auxiliary head; 5-seed frozen average).'),
    4:  ('CLS-LM-FT', 'T1a Molecular taste (P0 canonical/scaffold)', ['T1a'],
         'ChemBERTa-initialized SMILES language model with 6-label linear head (2 head-only warmup epochs followed by full encoder fine-tuning; max_length=128; BCE loss).'),
    5:  ('CLS-SMILES-TR', 'T1a Molecular taste (P0 canonical/scaffold)', ['T1a'],
         'Scratch-trained 6-layer SMILES Transformer (8 attention heads, hidden=256, FFN=1024, masked mean pooling, 6-label BCE loss).'),
    6:  ('CLS-ECFP-CB', 'T1a Molecular taste (P0 canonical/scaffold)', ['T1a'],
         '768-bit ECFP4 Morgan fingerprint with 6-label CatBoost MultiLogloss classifier (240 trees, depth=5, learning_rate=0.1, L2=3).'),
    7:  ('CLS-SMILES-TR-budget8', 'T1a Molecular taste (P0 SI reference)', ['T1a'],
         'Early 8-epoch budget-limited SMILES Transformer reference checkpoint retained as an archived SI capacity control.'),
    8:  ('NULL-label-rate', 'T1b Botanicals (Zhuang645 & CP2025)', ['T1b'],
         'Smoothed training-plant label prevalence baseline (pos + 0.5)/(n + 1); unseen botanical families use the 5-fold mean prior.'),
    9:  ('SET-FPDescLM-att', 'T1b Botanicals (Zhuang645 & CP2025)', ['T1b'],
         '104-d FP+Desc+LM constituent features with per-label gated attention exp(3·tanh(x·g)) pooling linear molecular logits (7 outputs; 6 evaluated excluding salty; non-Transformer).'),
    10: ('SET-comp-LR', 'T1b Botanicals (Zhuang645 & CP2025)', ['T1b'],
         '33-d equal-weight botanical composition statistics (log N, 5 descriptor means, 17 chemical-class shares, 10 q90/q98 tail offsets) with per-label logistic regression (C=0.1).'),
    11: ('SET-simweight-LR', 'T1b Botanicals (Zhuang645 & CP2025)', ['T1b'],
         'Same 33-d composition statistics weighted by sensory-reference fingerprint similarity w_i = 0.05 + s_active,i (normalized per plant) with per-label logistic regression (C=0.1).'),
    12: ('SET-LM+comp-LR', 'T1b Botanicals (Zhuang645 & CP2025)', ['T1b'],
         '65-d linear botanical set baseline combining equal-weight mean 32-d generic ChemBERTa embeddings with 33-d composition statistics (StandardScaler + per-label LR, C=0.1).'),
    13: ('SET-FPDescLM-MIL', 'T1b Botanicals (Zhuang645 & CP2025)', ['T1b'],
         '104-d FP+Desc+LM features passed through a 16-d GELU node MLP and 7 label-gated molecular softmax MIL pooling heads (training-selected early-stop epoch).'),
    14: ('SET-FPDescLM-TR', 'T1b Botanicals (Zhuang645 & CP2025)', ['T1b'],
         '104-d FP+Desc+LM features with a 16-d node MLP, single-layer 2-head Set Transformer self-attention (16→32→16 FFN), and scalar attention pooling (7 outputs).'),
    15: ('SET-FARTprior-MIL', 'T1b Botanicals (Zhuang645 & CP2025)', ['T1b'],
         '110-d FART-augmented constituent features (104-d FART-LM variant + sweet indicator + 5 FART probabilities) with 16-d GELU MLP and label-gated MIL softmax pooling.'),
    16: ('SET-FARTprior-TR', 'T1b Botanicals (Zhuang645 & CP2025)', ['T1b'],
         '110-d FART-augmented constituent features with 16-d node MLP, single-layer 2-head Set Transformer self-attention, and scalar attention pooling.'),
    17: ('SET-LM-LR', 'T1b Botanicals (Zhuang645 & CP2025)', ['T1b'],
         'Equal-weight mean of 32-d generic ChemBERTa molecular embeddings per botanical set with training-fold StandardScaler and per-label logistic regression (C=0.1).'),
    18: ('SET-LM+FARTprob-LR', 'T1b Botanicals (Zhuang645 & CP2025)', ['T1b'],
         'Mean-pooled 32-d generic ChemBERTa embeddings concatenated with 5 FART taste probabilities and per-label logistic regression (C=0.1).'),
    19: ('SET-FART-LM-LR', 'T1b Botanicals (Zhuang645 & CP2025)', ['T1b'],
         'Mean-pooled 32-d FART fine-tuned CLS projections per botanical set with standardized per-label logistic regression (C=0.1).'),
    20: ('SET-FARTprob-LR', 'T1b Botanicals (Zhuang645 & CP2025)', ['T1b'],
         'Mean-pooled 5-class FART taste probabilities per botanical set with standardized per-label logistic regression (C=0.1).'),
    21: ('SET-LM+hybridprob-LR', 'T1b Botanicals (Zhuang645 & CP2025)', ['T1b'],
         'Mean-pooled 32-d generic ChemBERTa embeddings plus 6 local hybrid molecular probabilities (distinct from official DLMTF) with per-label LR (C=0.1).'),
    22: ('SET-hybridprob-LR', 'T1b Botanicals (Zhuang645 & CP2025)', ['T1b'],
         'Mean-pooled 6 local hybrid molecular taste probabilities per botanical set with per-label logistic regression (C=0.1).'),
    23: ('SET-FartRF-LR', 'T1b Botanicals (Zhuang645 & CP2025)', ['T1b'],
         'Mean-pooled 32-d generic LM embeddings plus 5 FART random-forest probabilities (training-mean imputation for missing values) with per-label LR (C=0.1).'),
    24: ('SET-UniDirect-LR', 'T1b Botanicals (Zhuang645 & CP2025)', ['T1b'],
         'Mean-pooled 32-d generic LM embeddings plus 6 official Uni-Flavor outputs and 1 coverage flag (training-mean imputation) with per-label LR (C=0.1).'),
    25: ('SET-UniCoverage-LR', 'T1b Botanicals (Zhuang645 & CP2025)', ['T1b'],
         'Mean-pooled 32-d generic LM embeddings plus Uni-Flavor structural coverage flag (ablation without probabilities) with per-label LR (C=0.1).'),
    26: ('SET-DmcScores-LR', 'T1b Botanicals (Zhuang645 & CP2025)', ['T1b'],
         'Mean-pooled 32-d generic LM embeddings plus 5 local DmcFlavor molecular outputs and 1 coverage flag with per-label LR (C=0.1).'),
    27: ('SET-DmcCoverage-LR', 'T1b Botanicals (Zhuang645 & CP2025)', ['T1b'],
         'Mean-pooled 32-d generic LM embeddings plus DmcFlavor structural coverage flag (ablation without probabilities) with per-label LR (C=0.1).'),
    28: ('SET-UniOwn-LR', 'T1b Botanicals (Zhuang645 & CP2025)', ['T1b'],
         'Mean-pooled 32-d generic LM embeddings plus 6 local CLS-FPDescLM-LR probabilities with per-label logistic regression (C=0.1).'),
    29: ('SET-UniET-LR', 'T1b Botanicals (Zhuang645 & CP2025)', ['T1b'],
         'Mean-pooled 32-d generic LM embeddings plus 6 local CLS-FPDescLM-ET probabilities with per-label logistic regression (C=0.1).'),
    30: ('SET-UniET5-LR', 'T1b Botanicals (Zhuang645 & CP2025)', ['T1b'],
         'Mean-pooled 32-d generic LM embeddings plus 5-run averaged CLS-FPDescLM-ET probabilities (6-d) with per-label LR (C=0.1).'),
    31: ('SET-UniOwn5-LR', 'T1b Botanicals (Zhuang645 & CP2025)', ['T1b'],
         'Mean-pooled 32-d generic LM embeddings plus 5-run averaged CLS-FPDescLM-LR probabilities (6-d) with per-label LR (C=0.1).'),
    32: ('SET-FartRefit-LR', 'T1b Botanicals (Zhuang645 & CP2025)', ['T1b'],
         'Mean-pooled 32-d generic LM embeddings plus 5 canonical FART refit probabilities with per-label logistic regression (C=0.1).'),
    33: ('SET-FartOwn-LR', 'T1b Botanicals (Zhuang645 & CP2025)', ['T1b'],
         'Mean-pooled 32-d generic LM embeddings plus 5 multiclass LR probabilities trained on 104-d FP+Desc+LM FART features with per-label LR (C=0.1).'),
    34: ('SET-FartHead32-LR', 'T1b Botanicals (Zhuang645 & CP2025)', ['T1b'],
         'Mean-pooled 32-d generic LM embeddings plus 5 multiclass LR probabilities trained on 32-d generic LM FART features with per-label LR (C=0.1).'),
    35: ('SET-FartFullFP-LR', 'T1b Botanicals (Zhuang645 & CP2025)', ['T1b'],
         'Mean-pooled 32-d generic LM embeddings plus 5 multiclass LR probabilities trained on 2120-d full FP+Desc+LM FART features with per-label LR (C=0.1).'),
    36: ('SET-LM-att', 'T1b Botanicals (Zhuang645 & CP2025)', ['T1b'],
         '32-d generic ChemBERTa molecular features with label-gated attention pooling over linear molecular logits (non-Transformer).'),
    37: ('SET-LM+FARTprob-att', 'T1b Botanicals (Zhuang645 & CP2025)', ['T1b'],
         '32-d generic LM embeddings plus 5 FART probabilities with label-gated attention pooling over linear molecular logits (non-Transformer).'),
    38: ('SET-FartFullFP-att', 'T1b Botanicals (Zhuang645 & CP2025)', ['T1b'],
         '32-d generic LM embeddings plus 5 full-FP FART multiclass LR probabilities with label-gated attention pooling (non-Transformer).'),
    39: ('SET-FartHead32-att', 'T1b Botanicals (Zhuang645 & CP2025)', ['T1b'],
         '32-d generic LM embeddings plus 5 32-d LM FART multiclass LR probabilities with label-gated attention pooling (non-Transformer).'),
    40: ('SET-LM+hybridprob-att', 'T1b Botanicals (Zhuang645 & CP2025)', ['T1b'],
         '32-d generic LM embeddings plus 6 local hybrid molecular probabilities with label-gated attention pooling (non-Transformer).'),
    41: ('SET-UniET-att', 'T1b Botanicals (Zhuang645 & CP2025)', ['T1b'],
         '32-d generic LM embeddings plus 6 local CLS-FPDescLM-ET probabilities with label-gated attention pooling (non-Transformer).'),
    42: ('SET-UniOwn-att', 'T1b Botanicals (Zhuang645 & CP2025)', ['T1b'],
         '32-d generic LM embeddings plus 6 local CLS-FPDescLM-LR probabilities with label-gated attention pooling (non-Transformer).'),
    43: ('CLS-FART-LM-LR', 'T1a Molecular taste (FART 5-class)', ['T1a'],
         '32-d generic ChemBERTa features with median imputation, standardization, and multiclass LR (C ∈ {0.01, 0.1, 1} tuned by validation macro-F1).'),
    44: ('CLS-FART-FPDescLM-LR', 'T1a Molecular taste (FART 5-class)', ['T1a'],
         '104-d FP+Desc+LM features with median imputation, standardization, and multiclass LR (C ∈ {0.01, 0.1, 1} tuned by validation macro-F1).'),
    45: ('CLS-FART-fullFPDescLM-LR', 'T1a Molecular taste (FART 5-class)', ['T1a'],
         '2120-d full molecular representation (2048-bit FP + 40 descriptors + 32-d ChemBERTa) with standardized multiclass LR (C ∈ {0.01, 0.1, 1}).'),
    46: ('CLS-FART-LM', 'T1a Molecular taste (FART 5-class)', ['T1a'],
         'Official FART fine-tuned 5-class SMILES language model evaluated separately on per-string, full canonical (n=2,254), and accepted subsets.'),
    47: ('CLS-FART-LM-vote10', 'T1a Molecular taste (FART 5-class)', ['T1a'],
         'Official FART fine-tuned 5-class SMILES language model aggregating probabilities across 10 randomized SMILES augmentations per molecule.'),
    48: ('CLS-FART-LM-refit', 'T1a Molecular taste (FART 5-class)', ['T1a'],
         'ChemBERTa-initialized RoBERTa 5-class classifier fine-tuned independently on canonical SMILES train/validation splits with validation-NLL early stopping.'),
    49: ('SET-comp-LR', 'T1b Botanicals (Zhuang645 & CP2025, SI alias)', ['T1b'],
         '33-d equal-weight composition statistics (background-only reference arm) with per-label logistic regression (C=0.1).'),
    50: ('SET-comp-LR', 'T1b Botanicals (Zhuang645 & CP2025, SI weight-0)', ['T1b'],
         '33-d equal-weight composition statistics (uniform weight control prior to similarity weighting) with per-label logistic regression (C=0.1).'),
    51: ('SET-simweight-LR', 'T1b Botanicals (Zhuang645 & CP2025, main arm)', ['T1b'],
         '33-d botanical composition statistics weighted by sensory-reference similarity w_i = 0.05 + s_active,i (normalized per plant) with per-label LR (C=0.1).'),
    52: ('SET-FPDescLM-MIL', 'T1b Botanicals (Zhuang645 & CP2025, selected)', ['T1b'],
         '104-d FP+Desc+LM features with 16-d GELU node MLP and 7 label-gated molecular softmax MIL pooling heads (selected training checkpoint).'),
    53: ('SET-FPDescLM-TR', 'T1b Botanicals (Zhuang645 & CP2025, selected)', ['T1b'],
         '104-d FP+Desc+LM features with 16-d node MLP, single-layer 2-head Set Transformer self-attention, and scalar attention pooling (selected checkpoint).'),
    54: ('RIDGE-pres', 'T2 Fruit information ladder (5 cohorts)', ['T2'],
         'Ridge regression on binary chemical detection indicators plus explicit missingness flags (separating measured zero from unmeasured entries).'),
    55: ('RIDGE-pres+prior', 'T2 Fruit information ladder (5 cohorts)', ['T2'],
         'Ridge regression augmenting binary presence indicators with mean qualitative taste priors over structure-mapped constituents.'),
    56: ('RIDGE-conc-all', 'T2 Fruit information ladder (5 cohorts)', ['T2'],
         'Ridge regression on all eligible quantitative analytes (log1p-transformed after training positive-median scaling, imputation, and status indicators).'),
    57: ('RIDGE-conc-mapped', 'T2 Fruit information ladder (5 cohorts)', ['T2'],
         'Ridge regression restricted strictly to structure-mapped quantitative chemical measurements.'),
    58: ('RIDGE-conc-named', 'T2 Fruit information ladder (5 cohorts)', ['T2'],
         'Ridge regression using structure-mapped plus named quantitative analytical measurements.'),
    59: ('RIDGE-conc-all+matrix', 'T2 Fruit information ladder (5 cohorts)', ['T2'],
         'Ridge regression combining all quantitative analytes with routine bulk matrix indices (°Brix, pH, titratable acidity) and missingness flags.'),
    60: ('TREE-pres', 'T2 Fruit information ladder (5 cohorts)', ['T2'],
         'ExtraTrees regression on binary chemical detection indicators plus explicit missingness flags (separating measured zero from unmeasured entries).'),
    61: ('TREE-pres+prior', 'T2 Fruit information ladder (5 cohorts)', ['T2'],
         'ExtraTrees regression augmenting binary detection indicators with mean qualitative taste priors over structure-mapped constituents.'),
    62: ('TREE-conc-all', 'T2 Fruit information ladder (5 cohorts)', ['T2'],
         'ExtraTrees regression on all eligible quantitative analytes (log1p-scaled by training positive medians with imputation and missingness flags).'),
    63: ('TREE-conc-mapped', 'T2 Fruit information ladder (5 cohorts)', ['T2'],
         'ExtraTrees regression restricted strictly to structure-mapped quantitative analytical measurements.'),
    64: ('TREE-conc-named', 'T2 Fruit information ladder (5 cohorts)', ['T2'],
         'ExtraTrees regression using structure-mapped plus named quantitative analytical measurements.'),
    65: ('TREE-conc-all+matrix', 'T2 Fruit information ladder (5 cohorts)', ['T2'],
         'ExtraTrees regression combining all quantitative analytes with routine bulk matrix indices (°Brix, pH, titratable acidity) and missingness flags.'),
    66: ('SUM-mass', 'T2 Within-cohort fruit panels', ['T2'],
         'Unweighted sum of measured mass concentrations with training-fold nonnegative slope and intercept readout.'),
    67: ('MATRIX-Brix/TA', 'T2 Within-cohort fruit panels', ['T2'],
         'Routine bulk matrix reference using °Brix (sweetness) or titratable acidity (sourness) only; distinct from named-sugar or threshold loads.'),
    68: ('DOT-lin', 'T2 Grape juice volumetric cohort', ['T2'],
         'Linear volumetric dose-over-threshold load U = Σ c [g L⁻¹] × 1000 / (MW × T [mmol L⁻¹]) with training-fold nonnegative slope and intercept.'),
    69: ('DOT-log', 'T2 Grape juice volumetric cohort', ['T2'],
         'Log-compressed volumetric dose-over-threshold response log(1 + U/K) with K = 1 and training-fold nonnegative slope and intercept.'),
    70: ('DOT-log-Ktrain', 'T2 Grape juice volumetric cohort (SI)', ['T2'],
         'Volumetric log-DoT sensitivity arm setting curvature K to the training-fold median of positive U values.'),
    71: ('DOT-lin-mass-proxy', 'T2 Fresh-mass fruit cohorts (tomato, blueberry, apple)', ['T2'],
         'Fresh-mass relative potency load weighting each sugar/acid by MW_ref·T_ref / (MW_i·T_i); mass-basis proxy rather than aqueous molar DoT.'),
    72: ('DOT-log-mass-proxy', 'T2 Fresh-mass fruit cohorts (tomato, blueberry, apple)', ['T2'],
         'Log-compressed fresh-mass relative potency load with K set to the training-fold positive load median; mass-basis proxy rather than aqueous DoT.'),
    73: ('SUM-molar', 'T2 Grape juice volumetric cohort', ['T2'],
         'Equal-weight molar concentration sum Σ c × 1000 / MW fitted with training-fold affine readout.'),
    74: ('SUM-mass', 'T2 Fresh-mass fruit cohorts (equal-weight control)', ['T2'],
         'Equal-mass unweighted concentration sum on tissue fresh-mass cohorts (identical to SUM-mass linear control).'),
    75: ('SUM-log-molar / SUM-log-mass', 'T2 Within-cohort fruit panels (SI)', ['T2'],
         'Equal-weight log-compressed molar (juice) or mass (tissue) total; kept strictly distinct from threshold-weighted DOT-log.'),
    76: ('RIDGE-conc', 'T2 Named-sugar/acid fruit mini-panel', ['T2'],
         'Ridge regression on the matched named-analyte panel with inner 4-fold selection over α ∈ {0.1, 1, 10, 100, 1000}.'),
    77: ('TREE-conc', 'T2 Named-sugar/acid fruit mini-panel', ['T2'],
         'ExtraTrees regression (256 trees, min_samples_leaf=4) on the matched named-analyte panel.'),
    78: ('RIDGE-conc-all+matrix / TREE-conc-all+matrix', 'T2 Within-cohort fruit panels (full-panel comparison)', ['T2'],
         'Full-information analytical-plus-matrix predictions aligned on shared test rows to separate feature breadth from mini-panel models.'),
    79: ('DOT-lin-perm / DOT-log-perm', 'T2 Fruit potency-weight negative controls (SI)', ['T2'],
         'Non-identity threshold-weight permutations (5 permutations for 3 sugars; 1 swap for 2 sugars), averaging metrics across individual permutations.'),
    80: ('SUM-mass@0 / @5 / @10', 'T5 Cross-fruit calibration transfer (6 routes)', ['T5'],
         'Unweighted compatible fresh-mass sum evaluated at @0 (no target ratings) and @5/@10 out-of-test target anchors with intercept and slope-to-1 shrinkage (λ=5).'),
    81: ('DOT-lin-mass-proxy@0 / @5 / @10', 'T5 Cross-fruit calibration transfer (6 routes)', ['T5'],
         'Relative-potency fresh-mass linear load transferred at @0, @5, and @10 target anchors with shrunk-affine calibration (single malic acid equals SUM-mass).'),
    82: ('DOT-log-mass-proxy@0 / @5 / @10', 'T5 Cross-fruit calibration transfer (6 routes)', ['T5'],
         'Log-compressed relative-potency mass load with source-frozen positive-median K transferred across crops at @0, @5, and @10 shrunk-affine anchors.'),
    83: ('SUM-log-mass@0 / @5 / @10', 'T5 Cross-fruit calibration transfer (6 routes)', ['T5'],
         'Equal-weight log-compressed fresh-mass load using the same source-frozen K rule transferred at @0, @5, and @10 shrunk-affine anchors.'),
    84: ('RIDGE-conc@0 / @5 / @10', 'T5 Cross-fruit calibration transfer (6 routes)', ['T5'],
         'Source-trained common-panel Ridge predictor transferred across crops at @0, @5, and @10 shrunk-affine target anchors.'),
    85: ('TREE-conc@0 / @5 / @10', 'T5 Cross-fruit calibration transfer (6 routes)', ['T5'],
         'Source-trained common-panel ExtraTrees predictor transferred across crops at @0, @5, and @10 shrunk-affine target anchors.'),
    86: ('NULL-mean', 'T3 Unseen binary sweeteners (55 pairs / 165 cells)', ['T3'],
         'Training-fold mean sweetness across all 132 outer-training mixture ratings under 5-fold whole-pair holdout.'),
    87: ('RIDGE-id', 'T3 Unseen binary sweeteners (55 pairs / 165 cells)', ['T3'],
         '11-d two-hot ingredient identity Ridge regression (5-fold whole-pair holdout; inner 4-fold GroupKFold over α ∈ {1, 4, 16}).'),
    88: ('RIDGE-anon', 'T3 Unseen binary sweeteners (55 pairs / 165 cells)', ['T3'],
         'Anonymous sorted pair doses log1p(g L⁻¹) with Ridge regression (5-fold whole-pair holdout; inner GroupKFold α ∈ {1, 4, 16}).'),
    89: ('RIDGE-conc', 'T3 Unseen binary sweeteners (55 pairs / 165 cells)', ['T3'],
         '22-slot named concentration representation (11 presence + 11 log1p g L⁻¹ dose slots) with Ridge regression (α ∈ {1, 4, 16}).'),
    90: ('EQS', 'T3 Unseen binary sweeteners (55 pairs / 165 cells)', ['T3'],
         '3-level equi-sweet indicator (3%, 5%, 7% SE one-hot) with intercept Ridge regression (distinct from nominal 2×level prediction).'),
    91: ('EQS+ID', 'T3 Unseen binary sweeteners (55 pairs / 165 cells)', ['T3'],
         'Additive main effects of equi-sweet level (3/5/7% SE) plus 11-d ingredient identity via Ridge regression (no interaction terms).'),
    92: ('EQS×ID', 'T3 Unseen binary sweeteners (55 pairs / 165 cells)', ['T3'],
         'Equi-sweet level and ingredient main effects plus full ingredient × level indicator interactions via Ridge regression (ingredient-specific slopes).'),
    93: ('RIDGE-conc+level', 'T3 Unseen binary sweeteners (55 pairs / 165 cells)', ['T3'],
         '22 named concentration slots augmented with scalar nominal equi-sweet level under Ridge regression (α ∈ {1, 4, 16}).'),
    94: ('TREE-id', 'T3 Unseen binary sweeteners (55 pairs / 165 cells)', ['T3'],
         '11-d ingredient identity ExtraTrees regression (256 trees; inner GroupKFold min_samples_leaf ∈ {2, 4, 8}).'),
    95: ('TREE-anon', 'T3 Unseen binary sweeteners (55 pairs / 165 cells)', ['T3'],
         'Anonymous sorted pair log1p(g L⁻¹) doses with ExtraTrees regression (256 trees; inner leaf ∈ {2, 4, 8}).'),
    96: ('TREE-conc', 'T3 Unseen binary sweeteners (55 pairs / 165 cells)', ['T3'],
         '22-slot named presence and log1p(g L⁻¹) dose representation with ExtraTrees regression (256 trees; inner leaf ∈ {2, 4, 8}).'),
    97: ('TREE-conc+level', 'T3 Unseen binary sweeteners (55 pairs / 165 cells)', ['T3'],
         '22 named concentration slots plus scalar nominal equi-sweet level with ExtraTrees regression (256 trees; inner leaf ∈ {2, 4, 8}).'),
    98: ('TREE-conc+level-fixed4', 'T3 Unseen binary sweeteners (55 pairs / 165 cells)', ['T3'],
         '22 named concentration slots plus nominal level with ExtraTrees regression fixed at min_samples_leaf=4 (256 trees).'),
    99: ('ADD-nominal', 'T3 Unseen binary sweeteners (nominal reference)', ['T3'],
         'Nominal linear additivity reference predicting 2 × level (6, 10, 14 SE) without measured monomer self-mixture anchors.'),
    100: ('NULL-mean', 'T3 Unseen binary sweeteners (direct-intensity control)', ['T3'],
          'Outer training-fold mean sweetness reference for direct-intensity fixed-hyperparameter controls.'),
    101: ('RIDGE-conc-fixed4', 'T3 Unseen binary sweeteners (direct-intensity control)', ['T3'],
          '22-slot named concentration representation with StandardScaler and fixed-penalty Ridge regression (α = 4).'),
    102: ('TREE-conc-fixed4', 'T3 Unseen binary sweeteners (direct-intensity control)', ['T3'],
          '22-slot named concentration representation with ExtraTrees regression fixed at 256 trees and min_samples_leaf=4.'),
    103: ('RIDGE-conc-fixed4+GRAPH', 'T3 Unseen binary sweeteners (direct-intensity control)', ['T3'],
          '22-slot Ridge (α = 4) followed by structural graph-kernel correction on raw intensity residuals without monomer self-mixture anchors.'),
    104: ('SELF-mean', 'T4a Binary sweetener residuals (136 self-referenced cells)', ['T4'],
          'Mean of measured same-level monomer self-mixture intensities (zero interaction residual) evaluated on the 136 self-referenced binary subset.'),
    105: ('SELF+ID', 'T4a Binary sweetener residuals (136 self-referenced cells)', ['T4'],
          'Per-level two-hot ingredient identity Ridge model fitted to self-centered interaction residuals (observed minus SELF-mean).'),
    106: ('SELF+GRAPH', 'T4a Binary sweetener residuals (136 self-referenced cells)', ['T4'],
          'SELF+ID residual baseline augmented with optional structural graph-kernel completion on the 136 self-referenced binary subset.'),
    107: ('SELF+DIRECT', 'T4a Binary sweetener residuals (136 self-referenced cells)', ['T4'],
          '11-d aggregated dose and pure-reference feature rule refitted to binary self-centered interaction residuals on the 136-cell subset.'),
    108: ('TREE-conc+level-profilecontrol', 'T3 Binary sweetener matched-input readout (SI)', ['T3'],
          '23-slot named concentration-plus-level ExtraTrees control matched to the dose-profile grid without subtracting training level means.'),
    109: ('TREE-conc+doseprofile', 'T3 Binary sweetener dose-profile adapter (SI)', ['T3'],
          'Level-demeaned 23-slot ExtraTrees adapter fitting within-level residuals and restoring training level means at prediction (leaf ∈ {2, 4, 8}).'),
    110: ('ADPT-set', 'T2–T6 Continuous-dose neural adapter (SI ceiling test)', ['T2', 'T3', 'T4', 'T5', 'T6a', 'T6b'],
          'Local set-attention continuous regressor operating over RDKit molecular descriptors paired with continuous dose tokens (distinct from official DmcFlavor).'),
    111: ('ADPT-pool', 'T2–T6 Continuous-dose neural adapter (SI ceiling test)', ['T2', 'T3', 'T4', 'T5', 'T6a', 'T6b'],
          'DeepSets permutation-invariant pooling regressor operating over identical RDKit descriptor + continuous dose tokens.'),
    112: ('ADPT-dyn', 'T2–T6 Continuous-dose neural adapter (SI ceiling test)', ['T2', 'T3', 'T4', 'T5', 'T6a', 'T6b'],
          'Physics/dynamics-inspired continuous-dose latent interaction taste regressor evaluated across fruit, sweetener, and food benchmarks.'),
    113: ('ADPT-dyn-off', 'T2–T6 Continuous-dose neural adapter (SI ablation)', ['T2', 'T3', 'T4', 'T5', 'T6a', 'T6b'],
          'No-dynamics ablation of ADPT-dyn removing latent interaction updates while keeping identical input tokens and training budget.'),
    114: ('EXT-UF-[representation]-[head] / EXT-DMC-score-[head]', 'T2/T3/T6 External frozen embedding controls (SI)', ['T2', 'T3', 'T6a'],
          'Frozen official Uni-Flavor (3D/Mordred/fused/6-prob) and DmcFlavor (5-prob) readouts paired with Ridge or ExtraTrees heads (where full structural coverage exists).'),
    115: ('SELF-mean', 'T4b Ternary sweetener transfer (64 core / 192 cold)', ['T4'],
          'Unfitted arithmetic mean of the three target monomer self-mixture sweetness ratings (distinct from binary training mean).'),
    116: ('TER-direct', 'T4b Ternary sweetener transfer (64 core / 192 cold)', ['T4'],
          '11-d aggregated dose, SE3-relative dose, and pure-reference statistics fitted by Ridge to binary residuals and added back to ternary SELF-mean (no endpoint one-hot).'),
    117: ('TER-direct-desc', 'T4b Ternary sweetener transfer (64 core / 192 cold)', ['T4'],
          '27-d extension of TER-direct adding mean, SD, dose-weighted mean, and missingness of MW, LogP, TPSA, HBD, and HBA descriptors.'),
    118: ('TER-id', 'T4b Ternary sweetener transfer (64 core / 192 cold)', ['T4'],
          'Endpoint-identity Ridge trained on binary mixtures predicting the three masked constituent binary edges, averaged and added to ternary SELF-mean.'),
    119: ('TER-graph', 'T4b Ternary sweetener transfer (64 core / 192 cold)', ['T4'],
          'TER-id augmented with inner-CV-selectable structural similarity graph-kernel completion across masked binary edges (may select graph_off per fold).'),
    120: ('NULL-mean', 'T4b Ternary sweetener transfer (binary training null)', ['T4'],
          'Mean measured binary sweetness over all remaining non-masked binary pairs after withholding all three constituent edges of each target ternary recipe.'),
    121: ('{model}[pure-observed] / {model}[pure-proxy]', 'T4b Ternary transfer condition modifiers', ['T4'],
          'Condition modifiers: pure-observed retains target monomer self-references; pure-proxy substitutes nearest structural donor curves; cold further masks all edges of cold monomers.'),
    122: ('SUM-molar@1', 'T6a/T6b Tea omissions, Ancha drying & coffee roasting', ['T6a', 'T6b'],
          'Equal-weight molar sum S = Σ c [µmol L⁻¹] scaled via single-anchor zero-intercept ratio ΔÎ = I0·(S1/S0 − 1) on the shared full/reference preparation.'),
    123: ('DOT-lin@1', 'T6a/T6b Tea omissions, Ancha drying & coffee roasting', ['T6a', 'T6b'],
          'Linear dose-over-threshold load U = Σ c/T scaled via single-anchor zero-intercept ratio ΔÎ = I0·(U1/U0 − 1).'),
    124: ('SUM-log-molar-matched@1', 'T6a/T6b Tea omissions, Ancha drying & coffee roasting', ['T6a', 'T6b'],
          'Equal-molar log control matched to full-preparation DoT load U0: ΔÎ = I0·{ln[1 + (S1/S0)·U0/K] / ln(1 + U0/K) − 1}.'),
    125: ('DOT-log@1', 'T6a/T6b Tea omissions, Ancha drying & coffee roasting', ['T6a', 'T6b'],
          'Frozen single-compound log-DoT response ΔÎ = I0·{ln(1 + U1/K) / ln(1 + U0/K) − 1} anchored on one full/reference preparation.'),
    126: ('DOT-log^η@1', 'T6a/T6b Tea, Ancha & coffee exploratory curvature (SI)', ['T6a', 'T6b'],
          'Exploratory power-law extension ΔÎ = I0·(r^η − 1) with frozen η = 0.5 applied to compressed log-DoT load ratio r.'),
    127: ('SUM-molar@1', 'T6a Sweetened yogurt formulation (Ahmad)', ['T6a'],
          'Molar sum of four quantified sugars (mmol L⁻¹) anchored proportionally to the independent commercial AGY yogurt sweetness rating.'),
    128: ('SUM-mass@1', 'T6a Sweetened yogurt formulation (Ahmad)', ['T6a'],
          'Zero-intercept mass sum of four quantified sugars (g L⁻¹) anchored proportionally to the independent commercial AGY rating.'),
    129: ('SUM-mass-source-affine@1', 'T5/T6a Grape-to-yogurt @1 transfer (Ahmad)', ['T5', 'T6a'],
          'Grape-trained total-sugar affine regression retaining extrapolated zero crossing −b/a = 34.613 g L⁻¹ and rescaled proportionally to AGY.'),
    130: ('DOT-lin@1', 'T6a Sweetened yogurt formulation (Ahmad)', ['T6a'],
          'Linear four-sugar DoT load using aqueous threshold midpoints (39, 24.5, 33.5, 35 mmol L⁻¹) anchored to AGY.'),
    131: ('DOT-log@1', 'T6a Sweetened yogurt formulation (Ahmad)', ['T6a'],
          'Log-compressed four-sugar DoT load ln(1 + Σ c/T) with K = 1 anchored proportionally to the commercial AGY rating.'),
    132: ('NULL-zero@1', 'T6a Sweetened yogurt formulation (Ahmad)', ['T6a'],
          'Constant anchor-only baseline predicting identical AGY sweetness across formulations (predicted sensory change Δ = 0).'),
    133: ('SUM-mass@1', 'T6a Orange juice spiking & background (Dea)', ['T6a'],
          'Unweighted limonin + nomilin mass sum (mg L⁻¹) anchored proportionally to each native orange juice bitterness rating (H10 or V09).'),
    134: ('DOT-lin@1', 'T6a Orange juice spiking & background (Dea)', ['T6a'],
          'Linear bitterant DoT load Σ c/T using sugar–acid matrix thresholds (limonin 7.0 mg L⁻¹, nomilin 8.4 mg L⁻¹) anchored to native juice.'),
    135: ('DOT-log@1', 'T6a Orange juice spiking & background (Dea)', ['T6a'],
          'Log-compressed bitterant DoT load with frozen aqueous bitterness curvature K = 2.33647 anchored to each native juice rating.'),
    136: ('NULL-zero@1', 'T6a Orange juice spiking & background (Dea)', ['T6a'],
          'Native-juice constant baseline predicting zero bitterness change (Δ = 0) across spiking and sweet/acid background treatments.'),
    137: ('NULL-mean', 'T6b Ghana cocoa roasting (absolute intensity)', ['T6b'],
          'Leave-raw-and-roast-pair-out training mean over the remaining 6 cocoa roast states (zero difference across the held-out pair).'),
    138: ('RIDGE-SUM-mass', 'T6b Ghana cocoa roasting (6 chemical markers)', ['T6b'],
          'Standardized Ridge regression (α = 8) on the total mass (mg g⁻¹) of six quantified cocoa chemical markers trained on the other 6 roast states (@0).'),
    139: ('RIDGE-conc', 'T6b Ghana cocoa roasting (6 chemical markers)', ['T6b'],
          'Standardized multivariate Ridge regression (α = 8) on all six individual cocoa marker concentrations trained on the other 6 roast states (@0).'),
    140: ('SUM-count@1', 'T6a Tea omission panels (Darjeeling & Yu)', ['T6a'],
          'Active-compound count fraction within the threshold-matched subset scaled proportionally by the full reconstruction rating.'),
    141: ('DOT-raw', 'T6a Tea omission panels (uncalibrated SI reference)', ['T6a'],
          'Uncalibrated raw ΔΣ(c/T) chemical change (α = 1); reported strictly as a chemical reference, not sensory-scale error.'),
    142: ('DOT-lin@1', 'T6a Tea omission panels (full threshold coverage)', ['T6a'],
          'Linear DoT load over the full threshold-matched tea compound panel anchored proportionally to the full reconstruction.'),
    143: ('DOT-log@1', 'T6a Tea omission panels (full threshold coverage)', ['T6a'],
          'Frozen external-K log-DoT model over the full threshold-matched tea compound panel anchored to the full reconstruction.'),
    144: ('DOT-rat@1', 'T6a Tea omission panels (rational saturation)', ['T6a'],
          'Rational saturation DoT model U / (K + U) using independently calibrated monomer curvature K (7.16283 bitter; 7.62473 astringent) anchored at @1.'),
    145: ('DOT-hill@1', 'T6a Tea omission panels (SI Hill sensitivity)', ['T6a'],
          'Hill-function saturation sensitivity arm (U/K)^h / [1 + (U/K)^h] with independently fitted Hill exponent h ≈ 1.'),
    146: ('DOT-log-Krutin@1', 'T6a Tea astringency sensitivity (SI)', ['T6a'],
          'Astringency curvature sensitivity check using K = 12,034.4 derived from the published monomer rutin psychophysical curve.'),
    147: ('DOT-log+CR@1', 'T6a Tea omission mechanism check (SI)', ['T6a'],
          'Log-DoT bitterness response multiplied by literature caffeine–rutin co-enhancement factor [1 + f_caf × 0.75 × min(c_rutin/11 µM, 1)] before @1 anchoring.'),
    148: ('DOT-log+sucrose@1', 'T6a Tea omission mechanism check (SI)', ['T6a'],
          'Log-DoT response modulated on the caffeine fraction by a literature sucrose-release scaling factor before @1 anchoring.'),
    149: ('DOT-equal@1', 'T6a Tea common-subset threshold control', ['T6a'],
          'Structure-matched common subset assigning uniform unit thresholds (T = 1 µmol L⁻¹) prior to log compression (distinct from linear SUM-molar).'),
    150: ('DOT-meas@1', 'T6a Tea common-subset measured-threshold arm', ['T6a'],
          'Structure-matched common subset combining measured literature thresholds with frozen log-DoT compression (kept separate from full-coverage DOT-log@1).'),
    151: ('DOT-struct@1', 'T6a Tea common-subset structural-donor proxy', ['T6a'],
          'Common-subset threshold substitution using nearest non-identical structural donor thresholds (geometric mean on ties).'),
    152: ('DOT-perm@1', 'T6a Tea common-subset derangement null (SI)', ['T6a'],
          'Fixed-point-free derangement reassignment of measured aqueous thresholds within the common tea subset under frozen log compression.'),
    153: ('NULL-zero', 'T6b Ghana cocoa roasting (change ΔI readout)', ['T6b'],
          'Zero-change readout of the 6-state training-mean baseline predicting identical endpoints for raw and roast cocoa (ΔÎ = 0).'),
}


def infer_model_family_clean(clean_code):
    """Exact 4-family assignment strictly matching Figure S1 (zero classical ML leaks into Family 3)."""
    c = clean_code.strip()
    cl = c.lower()
    # Family 4: Continuous-dose neural adapters & structural graph completion
    if c.startswith('ADPT-') or 'GRAPH' in c or c == 'TER-graph':
        return '4. Continuous-dose adapters & graph'
    # Family 2: Classical tabular & molecular ML (all -LR, -ET, -CB, RIDGE, TREE, EXT, tabular residual learners)
    if (
        c.endswith('-LR') or c.endswith('-ET') or c.endswith('-CB')
        or c.startswith('RIDGE-') or c.startswith('TREE-') or c.startswith('EXT-')
        or c in ('SELF+ID', 'SELF+DIRECT', 'TER-direct', 'TER-direct-desc', 'TER-id')
    ):
        return '2. Classical tabular & molecular ML'
    # Family 3: True neural molecular encoders & botanical set attention/MIL/Transformer architectures
    if (
        c in ('CLS-UF-DLMTF', 'CLS-LM-FT', 'CLS-SMILES-TR', 'CLS-SMILES-TR-budget8',
              'CLS-FART-LM', 'CLS-FART-LM-vote10', 'CLS-FART-LM-refit')
        or c.endswith('-att') or c.endswith('-MIL') or c.endswith('-TR')
    ):
        return '3. Botanical set & molecular NNs'
    # Family 1: Psychophysical, DoT, equi-sweet, matrix & null baselines
    return '1. Psychophysical, DoT & null'


def make_calibrated_fruit_preset(seed, n, weights, total_mean, total_cv, share_means, share_sds):
    """Generate a reproducible synthetic-calibrated cohort matching frozen A2 summary stats (no raw 3rd-party data)."""
    rng = np.random.default_rng(seed)
    totals = np.clip(rng.normal(total_mean, total_mean * total_cv, size=n), total_mean * 0.45, total_mean * 1.75)
    k = len(weights)
    shares = np.zeros((n, k))
    for j in range(k - 1):
        shares[:, j] = np.clip(rng.normal(share_means[j], share_sds[j], size=n), 0.03, 0.90)
    rem = 1.0 - shares[:, :k - 1].sum(axis=1)
    rem = np.clip(rem, 0.03, 0.90)
    shares[:, k - 1] = rem
    shares = shares / shares.sum(axis=1, keepdims=True)
    conc = np.round(totals[:, None] * shares, 3)
    return conc.tolist()


def main():
    # 1. Load & verify 44 audit slices
    df44 = pd.read_csv(CSV_AUDIT44)
    assert len(df44) == 44, f'Expected 44 slices, got {len(df44)}'
    slices = []
    for idx, r in df44.iterrows():
        t_grp = classify_task_group(r['task'])
        states = derive_audit_states(r)
        n_val = to_num_or_none(r['n'])
        g_val = to_num_or_none(r['groups'])
        small_n = (n_val is not None and n_val <= 3)
        slices.append({
            'id': idx + 1,
            'task_group': t_grp,
            'slice': sanitize_text(str(r['task'])),
            'holdout_unit': sanitize_text(str(r['holdout_unit'])) if not pd.isna(r['holdout_unit']) else 'See Table S3',
            'n': n_val,
            'groups': g_val,
            'small_n_no_ci': small_n,
            'input_tuple': infer_input_tuple(t_grp, r['task']),
            'A2_r': safe_val(r['A2_r']),
            'A2_VIF': safe_val(r['A2_VIF']),
            'A3_k': to_num_or_none(r['A3_k_molecules']),
            'A3_D': safe_val(r['A3_D_labeled_molecule_derangements']),
            'A3_min_p': safe_val(r['A3_minimum_add_one_p_all_labeled_derangements']),
            'A3_obs_p': safe_val(r['A3_observed_p']),
            'A4_reported': to_num_or_none(r['A4_reported_components']),
            'A4_matched': to_num_or_none(r['A4_matched_components']),
            'A4_cov': safe_val(r['A4_count_fraction']),
            'A6_budget': sanitize_text(str(r['A6_budget'])) if not pd.isna(r['A6_budget']) else None,
            'A6_I0': safe_val(r['A6_I0']),
            'A6_H0': safe_val(r['A6_H0']),
            'A6_alpha': safe_val(r['A6_alpha_I0_over_H0']),
            'states': states,
            'notes': {
                'A2': format_audit_note('A2', r['A2_details']),
                'A3': format_audit_note('A3', r['A3_details']),
                'A4': format_audit_note('A4', r['A4_details']),
                'A6': format_audit_note('A6', r['A6_details']),
            },
        })

    # 2. Load & verify 153 model configurations (Table S2 — 100% English, exact families & tasks)
    df_s2 = pd.read_csv(CSV_S2)
    assert len(df_s2) == 153, f'Expected 153 configs, got {len(df_s2)}'
    assert len(MODEL_CATALOG_EN) == 153, f'Catalog size mismatch: {len(MODEL_CATALOG_EN)}'
    models = []
    for idx, r in df_s2.iterrows():
        mid = idx + 1
        clean_model, clean_ctx, tasks_list, clean_def = MODEL_CATALOG_EN[mid]
        fam = infer_model_family_clean(clean_model)
        models.append({
            'id': mid,
            'task': tasks_list[0] if len(tasks_list) == 1 else '/'.join(tasks_list),
            'tasks': tasks_list,
            'context': clean_ctx,
            'model': clean_model,
            'family': fam,
            'placement': str(r['placement']).strip(),
            'definition': clean_def,
        })

    # 3. Load & verify 706 scored registry contrasts
    df_reg = pd.read_csv(CSV_REG)
    scored = df_reg[df_reg['status'] == 'Verified'].copy()
    assert len(scored) == 706, f'Expected 706 scored contrasts, got {len(scored)}'
    type_map = {'A1': 'Information', 'A2': 'Model', 'A3': 'Calibration', 'Mixed': 'Multi-factor (descriptive)'}
    contrasts = []
    for _, r in scored.iterrows():
        n_u = to_num_or_none(r['n_test_unique'])
        small_n = (n_u is not None and n_u <= 3)
        ci_lo = None if small_n else safe_val(r['ci95_low'])
        ci_hi = None if small_n else safe_val(r['ci95_high'])
        dataset = sanitize_text(str(r['dataset']))
        tier = str(r['candidate_evidence_tier']).strip() if not pd.isna(r['candidate_evidence_tier']) else 'Retrospective frozen'
        if dataset.lower() == 'coffee' and tier == 'Prospective external (project-level)':
            tier = 'Internal pre-outcome lock'
        contrasts.append({
            'id': str(r['comparison_id']).strip(),
            'task': str(r['task_id']).strip(),
            'dataset': dataset,
            'endpoint': sanitize_text(str(r['endpoint'])),
            'axis': type_map.get(str(r['comparison_type']), str(r['comparison_type'])),
            'ref_model': sanitize_model_name(r['reference_model']),
            'cand_model': sanitize_model_name(r['candidate_model']),
            'metric': sanitize_text(str(r['metric'])),
            'scale': sanitize_text(str(r['scale_or_units'])) if not pd.isna(r['scale_or_units']) else 'native',
            'n': n_u,
            'groups': to_num_or_none(r['n_groups']),
            'ref_val': safe_val(r['reference_value']),
            'cand_val': safe_val(r['candidate_value']),
            'delta': safe_val(r['delta_candidate_minus_reference']),
            'ci_lo': ci_lo,
            'ci_hi': ci_hi,
            'small_n_no_ci': small_n,
            'tier': tier,
            'slice': sanitize_text(str(r['audit_slice_id'])) if not pd.isna(r['audit_slice_id']) else '',
        })

    # 4. Curated parameters & medium-specific threshold dictionary (Tab 3 — verified literature DOIs only)
    df_tea = pd.read_csv(CSV_TEA_POT)
    curated_thresholds = []
    key_tea_names = [
        'caffeine', 'epigallocatechin-3-gallate', 'epicatechin-3-gallate', 'catechin',
        'epigallocatechin', 'theaflavin', 'theaflavin-3,3′-digallate',
        'quercetin-3-O-β-D-glucopyranoside', 'myricetin-3-O-β-D-galactopyranoside', 'L-theanine'
    ]
    for _, r in df_tea.iterrows():
        nm = str(r['name']).strip()
        if any(k.lower() == nm.lower() or (k.lower() in nm.lower() and len(k) > 8) for k in key_tea_names):
            if not pd.isna(r['bitter_threshold']):
                curated_thresholds.append({
                    'compound': nm, 'quality': 'Bitterness', 'threshold': float(r['bitter_threshold']),
                    'unit': 'µmol L⁻¹', 'matrix': 'Aqueous tea / coffee / Ancha',
                    'medium': 'Bottled water (pH 5.5–6.0, 22 °C)',
                    'ref': 'Scharbert et al. (2004, 2005)', 'doi': '10.1021/jf050294d'
                })
            if not pd.isna(r['astringent_threshold']):
                curated_thresholds.append({
                    'compound': nm, 'quality': 'Astringency', 'threshold': float(r['astringent_threshold']),
                    'unit': 'µmol L⁻¹', 'matrix': 'Aqueous tea / Ancha',
                    'medium': 'Bottled water (pH 5.5–6.0, 22 °C)',
                    'ref': 'Scharbert et al. (2004, 2005)', 'doi': '10.1021/jf049802u'
                })

    # Read exact frozen flavonol-3-O-rutinoside thresholds directly from components_potency.csv (no hardcoding)
    def lookup_tea_astringent(substr):
        sub = df_tea[df_tea['name'].str.contains(substr, case=False, na=False, regex=False)]
        vals = sub['astringent_threshold'].dropna()
        assert len(vals) >= 1, f'Missing astringent_threshold for {substr}'
        return float(vals.iloc[0])

    t_rutin = lookup_tea_astringent('quercetin-3-O-[α-L-rhamnopyranosyl-(1→6)-β-D-glucopyranoside]')
    t_kaemp = lookup_tea_astringent('kaempferol-3-O-[α-L-rhamnopyranosyl-(1→6)-β-D-glucopyranoside]')
    t_myric = lookup_tea_astringent('myricetin-3-O-[α-L-rhamnopyranosyl-(1→6)-β-D-glucopyranoside]')

    # Add signature flavonol-3-O-rutinoside astringency cliffs (echoing Figure 6b & Figure S2d) + non-tea matrices (§2.3)
    curated_thresholds.extend([
        {'compound': 'Rutin (quercetin-3-O-rutinoside)', 'quality': 'Astringency', 'threshold': t_rutin,
         'unit': 'µmol L⁻¹', 'matrix': 'Aqueous tea (Fig. 6b & S2d cliff)',
         'medium': f'Bottled water (pH 5.5–6.0, 22 °C); {t_kaemp/t_rutin:.0f}-fold below kaempferol analog (0.001 µmol L⁻¹ in Scharbert 2004)',
         'ref': 'Scharbert et al. (2004, 2005)', 'doi': '10.1021/jf049802u'},
        {'compound': 'Kaempferol-3-O-rutinoside', 'quality': 'Astringency', 'threshold': t_kaemp,
         'unit': 'µmol L⁻¹', 'matrix': 'Aqueous tea (Fig. 6b cliff)',
         'medium': f'Bottled water (pH 5.5–6.0, 22 °C); single 3′-H vs 3′-OH cliff ({t_kaemp/t_rutin:.0f}× vs rutin)',
         'ref': 'Scharbert et al. (2004)', 'doi': '10.1021/jf049802u'},
        {'compound': 'Myricetin-3-O-rutinoside', 'quality': 'Astringency', 'threshold': t_myric,
         'unit': 'µmol L⁻¹', 'matrix': 'Aqueous tea (Fig. S2d cliff)',
         'medium': f'Bottled water (pH 5.5–6.0, 22 °C); {t_myric/t_rutin:,.0f}× vs 2005 rutin ({t_rutin} µmol L⁻¹) / 10,500× vs 2004 rutin (0.001 µmol L⁻¹)',
         'ref': 'Scharbert et al. (2004, 2005)', 'doi': '10.1021/jf049802u'},
        {'compound': 'Limonin', 'quality': 'Bitterness', 'threshold': 7.0, 'unit': 'mg L⁻¹',
         'matrix': 'Orange juice', 'medium': 'Water + 6% sucrose + 0.15% citric acid',
         'ref': 'Dea et al. (2013)', 'doi': '10.1111/joss.12046'},
        {'compound': 'Nomilin', 'quality': 'Bitterness', 'threshold': 8.4, 'unit': 'mg L⁻¹',
         'matrix': 'Orange juice', 'medium': 'Water + 6% sucrose + 0.15% citric acid',
         'ref': 'Dea et al. (2013)', 'doi': '10.1111/joss.12046'},
        {'compound': 'Sucrose', 'quality': 'Sweetness', 'threshold': 24.5, 'unit': 'mmol L⁻¹',
         'matrix': 'Yogurt supernatant', 'medium': 'Aqueous threshold midpoint (fruit ref: 24.0 mmol L⁻¹)',
         'ref': 'Ahmad et al. (2026)', 'doi': '10.1016/j.foodchem.2026.148351'},
        {'compound': 'Glucose', 'quality': 'Sweetness', 'threshold': 39.0, 'unit': 'mmol L⁻¹',
         'matrix': 'Yogurt supernatant', 'medium': 'Aqueous threshold midpoint (grape juice ref: 90 mmol L⁻¹)',
         'ref': 'Ahmad et al. (2026)', 'doi': '10.1016/j.foodchem.2026.148351'},
        {'compound': 'Galactose', 'quality': 'Sweetness', 'threshold': 33.5, 'unit': 'mmol L⁻¹',
         'matrix': 'Yogurt supernatant', 'medium': 'Aqueous threshold midpoint',
         'ref': 'Ahmad et al. (2026)', 'doi': '10.1016/j.foodchem.2026.148351'},
        {'compound': 'Lactose', 'quality': 'Sweetness', 'threshold': 35.0, 'unit': 'mmol L⁻¹',
         'matrix': 'Yogurt supernatant', 'medium': 'Aqueous threshold midpoint',
         'ref': 'Ahmad et al. (2026)', 'doi': '10.1016/j.foodchem.2026.148351'},
        {'compound': 'D-Fructose', 'quality': 'Sweetness', 'threshold': 52.0, 'unit': 'mmol L⁻¹',
         'matrix': 'Grape juice / Fruits', 'medium': 'Frozen aqueous recognition threshold (rel. mass wt = 0.877)',
         'ref': 'Cai et al. (2026) / PsyMix frozen dict', 'doi': '10.1002/fsn3.71824'},
        {'compound': 'D-Glucose', 'quality': 'Sweetness', 'threshold': 90.0, 'unit': 'mmol L⁻¹',
         'matrix': 'Grape juice / Fruits', 'medium': 'Frozen aqueous recognition threshold (rel. mass wt = 0.507)',
         'ref': 'Cai et al. (2026) / PsyMix frozen dict', 'doi': '10.1002/fsn3.71824'},
        {'compound': 'L-Malic acid', 'quality': 'Sourness', 'threshold': 3.7, 'unit': 'mmol L⁻¹',
         'matrix': 'Grape juice / Fruits', 'medium': 'Frozen aqueous recognition threshold (rel. mass wt = 1.000)',
         'ref': 'Cai et al. (2026) / Rotzoll et al. (2005)', 'doi': '10.1021/jf050056i'},
        {'compound': 'Citric acid', 'quality': 'Sourness', 'threshold': 2.6, 'unit': 'mmol L⁻¹',
         'matrix': 'Grape juice / Fruits', 'medium': 'Frozen aqueous recognition threshold (rel. mass wt = 0.993)',
         'ref': 'Cai et al. (2026) / Rotzoll et al. (2005)', 'doi': '10.1021/jf050056i'},
    ])
    seen_th = set()
    uniq_th = []
    for row in curated_thresholds:
        key = (row['compound'].lower(), row['quality'], row['matrix'])
        if key not in seen_th:
            seen_th.add(key)
            uniq_th.append(row)

    # 5. Calibrated Tab 2 Presets (reproducing exact frozen A2/A3/A6 paper benchmarks!)
    presets = {
        'apple_sugar': {
            'label': 'Apple Sugar Cohort (T2, n = 40; r = 0.982)',
            'description': 'Calibrated to frozen A2 apple summary (Glucose/Fructose/Sucrose; weights 0.507 / 0.877 / 1.000; observed r = 0.982, VIF = 27.94).',
            'analytes': ['Glucose', 'Fructose', 'Sucrose'],
            'weights': [0.5067, 0.8769, 1.0000],
            'unit': 'g kg⁻¹ FW',
            'rows': make_calibrated_fruit_preset(202611, 40, [0.5067, 0.8769, 1.0], 115.0, 0.147, [0.16, 0.52], [0.04455, 0.04860]),
            'a3_k': 3,
            'a6': {'mode': 'dot_log', 'cA': 12.0, 'IA': 52.0, 'K': 2.33647, 'c1': 16.0, 'scale_max': 100, 'note': 'Within-cohort affine fit (train-fold calibrated)'},
        },
        'tomato_sugar': {
            'label': 'Tomato Sugar Cohort (T2, n = 60 repr.; r = 0.999)',
            'description': 'Calibrated to frozen A2 tomato summary (Glucose/Fructose; weights 0.507 / 0.877; observed r = 0.999, VIF = 582.1).',
            'analytes': ['Glucose', 'Fructose'],
            'weights': [0.5067, 0.8769],
            'unit': 'mg g⁻¹ FW',
            'rows': make_calibrated_fruit_preset(202612, 60, [0.5067, 0.8769], 28.0, 0.327, [0.46], [0.024]),
            'a3_k': 2,
            'a6': {'mode': 'dot_log', 'cA': 10.0, 'IA': 35.0, 'K': 2.33647, 'c1': 14.0, 'scale_max': 100, 'note': 'Within-cohort affine fit (train-fold calibrated)'},
        },
        'yu_green_tea': {
            'label': 'Yu Green Tea Bitterness Omission (T6a, k = 4)',
            'description': 'Primary A3/A6 benchmark (k = 4 active bitterants → D(4) = 9, min p = 0.100; α = 3.22 on 0–4 scale) with illustrative omission matrix calibrated to observed A2 Δload r ≈ 0.999 (Table 4).',
            'analytes': ['Caffeine', 'EGCG', 'ECG', 'EGC'],
            'weights': [2.000, 2.632, 3.846, 1.923],
            'unit': 'mmol L⁻¹ (w = 1/T)',
            'rows': [
                [2.75, 0.35, 0.08, 0.14],
                [0.00, 0.35, 0.08, 0.14],
                [2.75, 0.00, 0.08, 0.14],
                [2.75, 0.35, 0.00, 0.14],
                [2.75, 0.35, 0.08, 0.00],
                [0.00, 0.00, 0.00, 0.14],
            ],
            'a3_k': 4,
            'a6': {'mode': 'dot_log', 'cA': 4.10, 'IA': 3.25, 'K': 2.33647, 'c1': 2.20, 'scale_max': 4, 'note': 'Full tea reconstruction anchor (0–4 scale, α ≈ 3.22)'},
        },
        'darjeeling_tea': {
            'label': 'Darjeeling Black Tea Omission (T6a, k = 9 bitter)',
            'description': 'Primary A3/A6 benchmark (k = 9 active bitterants → D(9) = 133,496, obs p = 0.540; α = 1.83 on 0–3 scale) with illustrative omission matrix calibrated to observed A2 Δload r ≈ 0.995 (Table 4).',
            'analytes': ['Caffeine', 'EGCG', 'ECG', 'Theaflavin', 'Theaflavin-3G'],
            'weights': [2.000, 2.632, 3.846, 62.50, 66.67],
            'unit': 'mmol L⁻¹ (w = 1/T)',
            'rows': [
                [2.05, 0.32, 0.10, 0.0012, 0.0008],
                [0.00, 0.32, 0.10, 0.0012, 0.0008],
                [2.05, 0.00, 0.00, 0.0012, 0.0008],
                [2.05, 0.32, 0.10, 0.0000, 0.0000],
                [2.05, 0.00, 0.00, 0.0000, 0.0000],
                [0.00, 0.32, 0.10, 0.0000, 0.0000],
                [0.00, 0.00, 0.00, 0.0012, 0.0008],
                [1.025, 0.16, 0.05, 0.0006, 0.0004],
            ],
            'a3_k': 9,
            'a6': {'mode': 'dot_log', 'cA': 2.96, 'IA': 1.50, 'K': 2.33647, 'c1': 0.98, 'scale_max': 3, 'note': 'Full Darjeeling reconstruction anchor (0–3 scale, α = 1.83)'},
        },
        'orange_juice_v09': {
            'label': 'Orange Juice Valencia V09 Near-Floor Anchor (T6a, α = 37.17)',
            'description': 'Near-floor single-anchor amplification hazard (A6): Valencia V09 native bitterness I0 = 1.31 at H0 = 0.0352 produces α = 37.17 and severe overshoot (+12.99 pred vs +4.65 obs).',
            'analytes': ['Limonin', 'Nomilin'],
            'weights': [0.1429, 0.1190],
            'unit': 'mg L⁻¹ (w = 1/T)',
            'rows': [
                [0.60, 0.00],
                [8.60, 1.50],
                [4.60, 0.75],
                [12.0, 2.00],
                [2.00, 0.40],
                [6.00, 1.20],
            ],
            'a3_k': 2,
            'a6': {'mode': 'dot_log', 'cA': 0.083814, 'IA': 1.31, 'K': 2.33647, 'c1': 1.096275, 'scale_max': 15, 'note': 'Valencia V09 near-floor anchor (H0 = 0.0352, α = 37.17); compare Hamlin H10 (cA=0.31, IA=3.04, α=24.44)'},
        },
        'grape_to_yogurt': {
            'label': 'Grape → Yogurt @1 Source-Affine Retention (T5/T6a, −b/a = 34.61 g L⁻¹)',
            'description': 'Algebraic retention of extrapolated grape zero crossing (−b/a = 34.613 g L⁻¹) under single commercial AGY anchor (167.6 g L⁻¹, rating 9.98 on 0–10 scale).',
            'analytes': ['Glucose', 'Fructose', 'Sucrose', 'Lactose'],
            'weights': [0.5067, 0.8769, 1.0000, 0.3500],
            'unit': 'g L⁻¹',
            'rows': [
                [5.0, 0.0, 0.0, 35.0],
                [5.0, 0.0, 100.0, 35.0],
                [6.0, 0.0, 0.0, 36.0],
                [6.0, 0.0, 100.0, 36.0],
                [12.0, 8.0, 112.0, 35.6],
            ],
            'a3_k': 4,
            'a6': {'mode': 'source_affine', 'cA': 167.569, 'IA': 9.979, 'zero_cross': 34.613, 'c0': 38.0, 'c1': 144.6, 'scale_max': 10, 'note': 'AGY commercial yogurt anchor (167.57 g L⁻¹, 9.98) with retained grape zero crossing −b/a = 34.613 g L⁻¹'},
        },
    }

    # Curated monograph / T1b harmonization & curvature tables for Tab 3
    curvatures = [
        {'channel': 'Bitterness (aqueous)', 'model': 'DOT-log', 'formula': 'H(U) = ln(1 + U/K)', 'K': 2.33647,
         'basis': 'Caffeine suprathreshold gLMS geometric means (1, 2, 4 mmol L⁻¹; T = 0.5 mmol L⁻¹)',
         'ref': 'Webb et al. (2015), Chemosens. Percept. 8 (1), 11–18', 'doi': '10.1007/s12078-015-9183-x'},
        {'channel': 'Astringency (aqueous)', 'model': 'DOT-log', 'formula': 'H(U) = ln(1 + U/K)', 'K': 1.27707,
         'basis': '10-step aqueous EGCG reference ladder (0.19–10.0 mmol L⁻¹; T = 0.19 mmol L⁻¹)',
         'ref': 'Hofmann et al. (2006), J. Agric. Food Chem. 54 (25), 9503–9509', 'doi': '10.1021/jf062272c'},
        {'channel': 'Bitterness (aqueous)', 'model': 'DOT-rat', 'formula': 'H(U) = U / (K + U)', 'K': 7.16283,
         'basis': 'Same independent caffeine suprathreshold gLMS curve (source amplitude discarded)',
         'ref': 'Webb et al. (2015), Table 4', 'doi': '10.1007/s12078-015-9183-x'},
        {'channel': 'Astringency (aqueous)', 'model': 'DOT-rat', 'formula': 'H(U) = U / (K + U)', 'K': 7.62473,
         'basis': 'Same independent 10-step EGCG ladder (source amplitude discarded)',
         'ref': 'Hofmann et al. (2006), Methods', 'doi': '10.1021/jf062272c'},
    ]

    harmonization = [
        {'zh': '苦 / 微苦', 'en': 'Bitter', 'core4': 'Yes', 'eval6': 'Yes', 'rule': 'Mild modifiers (微苦) retained as positive binary label'},
        {'zh': '辛 / 辣 / 微辛', 'en': 'Pungent', 'core4': 'Yes', 'eval6': 'Yes', 'rule': '辛 and 辣 harmonized jointly to pungent; mild modifiers retained'},
        {'zh': '甘 / 甜 / 微甘', 'en': 'Sweet', 'core4': 'Yes', 'eval6': 'Yes', 'rule': 'Pharmacopoeial 甘 mapped deterministically to sweet'},
        {'zh': '酸 / 微酸', 'en': 'Sour', 'core4': 'Yes', 'eval6': 'Yes', 'rule': 'Direct mapping; mild modifiers retained as positive'},
        {'zh': '涩 / 微涩', 'en': 'Astringent', 'core4': 'No', 'eval6': 'Yes', 'rule': 'Included in 6-attribute evaluation; chirality sensitive (AUROC 0.555 → 0.473)'},
        {'zh': '淡 / 微淡', 'en': 'Bland', 'core4': 'No', 'eval6': 'Yes', 'rule': 'Included in 6-attribute evaluation across Zhuang645 & CP2025'},
        {'zh': '咸', 'en': 'Salty', 'core4': 'No', 'eval6': 'Excluded (n=7)', 'rule': 'Only 7 positives in Zhuang645; retained in 7-output training head, excluded from 6-attribute macro-AP/AUROC'},
    ]

    bundle = {
        'meta': {
            'version': 'PsyMix-JCIM-2026.2',
            'n_configs': len(models),
            'n_slices': len(slices),
            'n_contrasts_scored': len(contrasts),
            'n_contrasts_total': len(df_reg),
            'n_manifest_locks': 13,
            'n_manifest_sources': 98,
            'n_references': 37,
        },
        'slices': slices,
        'models': models,
        'contrasts': contrasts,
        'presets': presets,
        'curvatures': curvatures,
        'thresholds': uniq_th,
        'harmonization': harmonization,
    }

    out_js = os.path.join(DOCS_ASSETS, 'psymix_data_bundle.js')
    with open(out_js, 'w', encoding='utf-8') as f:
        f.write('/* Auto-generated offline data bundle for PsyMix Static Identifiability Workbench */\n')
        f.write('window.PSYMIX_DATA = ')
        json.dump(bundle, f, ensure_ascii=False, separators=(',', ':'))
        f.write(';\n')

    # Print built-in QA summary
    n_small_slices = sum(1 for s in slices if s['small_n_no_ci'])
    n_small_contr = sum(1 for c in contrasts if c['small_n_no_ci'])
    fam_counts = {}
    for m in models:
        fam_counts[m['family']] = fam_counts.get(m['family'], 0) + 1
    print(f'[OK] Built psymix_data_bundle.js ({os.path.getsize(out_js)/1024:.1f} KB): '
          f'{len(slices)} slices ({n_small_slices} small-n), {len(models)} models, '
          f'{len(contrasts)} scored contrasts ({n_small_contr} small-n), {len(uniq_th)} thresholds')
    print('[OK] Model family breakdown:', fam_counts)
    for pk in ['apple_sugar', 'tomato_sugar', 'yu_green_tea', 'darjeeling_tea']:
        p = presets[pk]
        arr = np.array(p['rows'], dtype=float)
        w = np.array(p['weights'], dtype=float)
        r_p = float(np.corrcoef(arr.sum(axis=1), arr @ w)[0, 1])
        vif_p = 1.0 / (1.0 - r_p * r_p)
        print(f'     Preset {pk:<16s}: r = {r_p:.4f}, VIF = {vif_p:.1f}')

    # Sync curated PsyMix-owned summary CSVs (zero raw third-party datasets)
    copies = [
        (CSV_AUDIT44, os.path.join(TABLES_DIR, 'Table_S3_44slice_identifiability_audit.csv')),
        (CSV_S1,      os.path.join(TABLES_DIR, 'Table_S1_analysis_history.csv')),
        (CSV_S2,      os.path.join(TABLES_DIR, 'Table_S2_full_model_mapping.csv')),
        (CSV_A2_COLL, os.path.join(TABLES_DIR, 'A2_fruit_collinearity.csv')),
        (CSV_A2_REQ,  os.path.join(TABLES_DIR, 'A2_sugar_design_requirements.csv')),
        (CSV_T6_PERT, os.path.join(TABLES_DIR, 'T6_real_food_perturbations.csv')),
        (CSV_C1_MAIN, os.path.join(TABLES_DIR, 'Table_S4_sweetener_controls_and_adapters.csv')),
        (CSV_MANIFEST, os.path.join(MANIFEST_DIR, 'PsyMix_Submission_SHA256_Manifest.csv')),
    ]
    for src, dst in copies:
        if os.path.exists(src):
            shutil.copy2(src, dst)

    # Sample CSV template for Tab 2 upload
    sample_csv = os.path.join(DOCS_ASSETS, 'sample_design_template.csv')
    with open(sample_csv, 'w', encoding='utf-8') as f:
        f.write('sample_id,Glucose_w0.5067,Fructose_w0.8769,Sucrose_w1.0000\n')
        for i, row in enumerate(presets['apple_sugar']['rows'][:15], 1):
            f.write(f'S{i:02d},{row[0]},{row[1]},{row[2]}\n')


if __name__ == '__main__':
    main()
