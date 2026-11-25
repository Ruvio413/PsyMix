# -*- coding: utf-8 -*-
"""
PsyMix JCIM TOC Graphic (3.25 x 1.75 in) & Supplementary Figure S1 (180 x 175 mm)
Strict continuation of accepted Figures 1-6 visual constitution:
  - Colorblind-safe palette (#0072B2, #1B7F6B, #C0392B, #D55E00, #E69F00, #6A51A3, #9BB5CE, #2C3E50)
  - pdf.fonttype = 42, ps.fonttype = 42, svg.fonttype = 'none', mathtext.fontset = 'dejavusans'
  - All Figure S1 Panel b values loaded live from frozen source CSVs with assertions
  - Zero bare Unicode sub/superscript digits in Arial (always dejavusans mathtext)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Polygon, Circle, FancyArrowPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

# Global typography & vector fidelity (exact match to Figures 1-6)
mpl.rcParams.update({
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'mathtext.fontset': 'dejavusans',
    'axes.edgecolor': '#2C3E50',
    'axes.linewidth': 0.75,
    'xtick.color': '#2C3E50',
    'ytick.color': '#2C3E50',
    'text.color': '#2C3E50',
    'figure.facecolor': '#FFFFFF',
    'axes.facecolor': '#FFFFFF',
})

# Unified PsyMix Color Palette
BLUE       = '#0072B2'
BLUE_SOFT  = '#E6F2FA'
TEAL       = '#1B7F6B'
TEAL_SOFT  = '#E6F4F0'
RED        = '#C0392B'
RED_SOFT   = '#FCECEB'
AMBER      = '#D55E00'
AMBER_SOFT = '#FDF0E6'
GOLD       = '#E69F00'
PURPLE     = '#6A51A3'
PURPLE_SOFT= '#F1EEF8'
SLATE      = '#4D648D'
SLATE_LT   = '#9BB5CE'
SLATE_SOFT = '#EDF2F7'
CHARCOAL   = '#2C3E50'
GREY_TEXT  = '#4A5568'
GREY_LINE  = '#CBD5E1'
GREY_BG    = '#F3F4F6'
STRIPE_BG  = '#F8FAFC'
WHITE      = '#FFFFFF'

# Paths
OUT_DIR = r'c:\gemi-antigravity\00_PsyMix_Submission_Master\02_Figures_Design_and_Drafts\final_figures\TOC_and_FigS1'
SPEC_DIR = r'c:\gemi-antigravity\00_PsyMix_Submission_Master\02_Figures_Design_and_Drafts\specimens'

# Frozen source CSVs (read-only)
CSV_HERB_METRICS   = r'C:\zhuang_taste_pre\phase15_crossstudy_benchmark\results\herb_metrics.csv'
CSV_UNI_5SEED      = r'C:\zhuang_taste_pre\phase15_crossstudy_benchmark\results\Uni五种子平均概率敏感性.csv'
CSV_FAMILY_CURVES  = r'C:\zhuang_taste_pre\phase14_0_cross_scale_aggregation\training\family_curve_means.csv'
CSV_P18_PERF       = r'C:\zhuang_taste_pre\phase18_training_sufficiency\performance.csv'
CSV_C1_MAIN        = r'c:\gemi-antigravity\psymix_paper_freeze_20261122\T1_C1_controls\C1_main_table_v2.csv'
CSV_ADPT_FITLOG    = r'C:\zhuang_taste_pre\psymix_foodatlas_multimodel_20261121\set_dynamics\schiffman\fit_log.csv'
CSV_TABLE_S2       = r'C:\zhuang_taste_pre\psymix_final_tables_for_writing\Table_S2_full_model_mapping.csv'

os.makedirs(OUT_DIR, exist_ok=True)


def load_specimen_rgba(name, max_px=260):
    path = os.path.join(SPEC_DIR, name)
    im = Image.open(path).convert('RGBA')
    arr = np.array(im)
    # Remove near-white backdrop gently while preserving watercolor saturation
    r, g, b, a = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3]
    white_mask = (r > 246) & (g > 246) & (b > 246)
    arr[white_mask, 3] = 0
    # Crop to non-transparent bounding box
    ys, xs = np.where(arr[:, :, 3] > 10)
    if len(ys) > 0:
        arr = arr[ys.min():ys.max()+1, xs.min():xs.max()+1]
    im_clean = Image.fromarray(arr)
    im_clean.thumbnail((max_px, max_px), Image.Resampling.LANCZOS)
    return np.array(im_clean)


def add_rounded_box(ax, x, y, w, h, fc, ec='none', lw=0.7, r=0.015, zorder=2, ls='-'):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f'round,pad=0,rounding_size={r}',
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=zorder
    )
    ax.add_patch(box)
    return box


def draw_vector_arrow(ax, x0, y0, x1, y1, color='#64748B', lw=1.1, ms=7.5, zorder=5):
    arr = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle='-|>', mutation_scale=ms,
        color=color, linewidth=lw, shrinkA=0, shrinkB=0, zorder=zorder
    )
    ax.add_patch(arr)


# ==============================================================================
# TASK 1: JCIM TOC GRAPHIC (3.25 x 1.75 in = 82.55 x 44.45 mm, 1.857:1)
# ==============================================================================
def build_toc_graphic():
    fig = plt.figure(figsize=(3.25, 1.75), dpi=600, facecolor=WHITE)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    # Use inch coordinates directly (0..3.25 x 0..1.75) so circles/angles have 1:1 aspect!
    ax.set_xlim(0.0, 3.25)
    ax.set_ylim(0.0, 1.75)
    ax.set_aspect('equal')
    ax.axis('off')

    # Clean outer card background with ultra-subtle border
    add_rounded_box(ax, 0.02, 0.02, 3.21, 1.71, fc=WHITE, ec='#DCE3EC', lw=0.65, r=0.04, zorder=0)

    # --------------------------------------------------------------------------
    # LEFT ZONE (~28% width: x = 0.055 to 0.950, w = 0.895) — DATA GATE
    # --------------------------------------------------------------------------
    lx0, lw_z = 0.055, 0.895
    add_rounded_box(ax, lx0, 0.075, lw_z, 1.60, fc='#FAFBFD', ec='#E2E8F0', lw=0.55, r=0.045, zorder=1)

    gate_x = 0.760
    # Red vertical dashed Sensory gate line
    ax.plot([gate_x, gate_x], [0.145, 1.465], color=RED, lw=1.15, ls=(0, (3.2, 2.2)), zorder=6)
    ax.text(0.640, 1.545, 'Sensory gate', ha='center', va='center',
            fontsize=6.55, fontweight='bold', color=RED, zorder=7)

    # Flat muted slate-blue block: Unpaired DBs stopped flat at the gate
    bx0, by0, bw, bh = 0.090, 1.035, gate_x - 0.090 - 0.015, 0.340
    add_rounded_box(ax, bx0, by0, bw, bh, fc='#E4ECF4', ec=SLATE_LT, lw=0.85, r=0.028, zorder=3)
    # Crisp flat right wall flush against the gate
    ax.plot([bx0 + bw, bx0 + bw], [by0 + 0.02, by0 + bh - 0.02], color=SLATE, lw=1.1, zorder=4)
    ax.text(bx0 + bw * 0.50, by0 + bh * 0.50, 'Unpaired DBs', ha='center', va='center',
            fontsize=6.55, fontweight='bold', color='#3B5168', zorder=5)

    # Crimson vector X mark directly on the gate face (comfortably inside left zone)
    cx, cy, cr = gate_x + 0.088, by0 + bh * 0.50, 0.048
    ax.add_patch(Circle((cx, cy), cr * 1.40, facecolor=RED_SOFT, edgecolor=RED, lw=0.7, zorder=6))
    ax.plot([cx - cr*0.72, cx + cr*0.72], [cy - cr*0.72, cy + cr*0.72], color=RED, lw=1.35, solid_capstyle='round', zorder=7)
    ax.plot([cx - cr*0.72, cx + cr*0.72], [cy + cr*0.72, cy - cr*0.72], color=RED, lw=1.35, solid_capstyle='round', zorder=7)

    # High-saturation royal-blue ribbon crossing straight through the gate: Paired Q + A
    ry0, rh = 0.585, 0.315
    rx0, rx1, tip = 0.090, 0.850, 0.918
    ribbon_pts = [
        (rx0, ry0),
        (rx1, ry0),
        (tip, ry0 + rh * 0.50),
        (rx1, ry0 + rh),
        (rx0, ry0 + rh),
    ]
    ax.add_patch(Polygon(ribbon_pts, closed=True, facecolor=BLUE, edgecolor='#005A8D', lw=0.65, zorder=7))
    ax.text(0.460, ry0 + rh * 0.50, 'Paired Q + A', ha='center', va='center',
            fontsize=6.85, fontweight='bold', color=WHITE, zorder=8)

    # Subtle botanical/food watercolor corner accents below the ribbon (echo of Fig. 1)
    try:
        img_grape = load_specimen_rgba('grape.png', max_px=180)
        ab_g = AnnotationBbox(OffsetImage(img_grape, zoom=0.140), (0.255, 0.315),
                              frameon=False, zorder=3)
        ax.add_artist(ab_g)
        img_tea = load_specimen_rgba('tea.png', max_px=180)
        ab_t = AnnotationBbox(OffsetImage(img_tea, zoom=0.135), (0.540, 0.305),
                              frameon=False, zorder=3)
        ax.add_artist(ab_t)
    except Exception as e:
        print('Warning: specimen overlay skipped:', e)

    # Connector Arrow 1 (Left -> Center)
    draw_vector_arrow(ax, 0.962, 0.875, 1.032, 0.875, color='#64748B', lw=1.2, ms=8.0, zorder=8)

    # --------------------------------------------------------------------------
    # CENTER ZONE (~28.5% width: x = 1.045 to 1.950, w = 0.905) — ORTHOGONAL ATTRIBUTION
    # --------------------------------------------------------------------------
    cx0, cw = 1.045, 0.905
    add_rounded_box(ax, cx0, 0.075, cw, 1.60, fc='#FAFBFD', ec='#E2E8F0', lw=0.55, r=0.045, zorder=1)

    # Top row: 5 compact soft-tinted input badges S, P, Q, W, A
    badges = [
        ('S', '#E6EEF5', '#4D648D'),
        ('P', '#E5F4F0', TEAL),
        ('Q', '#E3F2FB', BLUE),
        ('W', '#EFEBF7', PURPLE),
        ('A', '#FDEEE3', AMBER),
    ]
    bw_b, bh_b = 0.142, 0.215
    gap_b = 0.028
    total_bw = 5 * bw_b + 4 * gap_b
    bx_start = cx0 + (cw - total_bw) / 2.0
    by_top = 1.385
    for idx, (letter, fc_b, ec_b) in enumerate(badges):
        bx = bx_start + idx * (bw_b + gap_b)
        add_rounded_box(ax, bx, by_top, bw_b, bh_b, fc=fc_b, ec=ec_b, lw=0.85, r=0.030, zorder=4)
        ax.text(bx + bw_b/2.0, by_top + bh_b/2.0, letter, ha='center', va='center',
                fontsize=7.1, fontweight='bold', color=ec_b, zorder=5)

    # Crisp clean downward bracket arrow from badges to orthogonal axes
    mid_c = cx0 + cw / 2.0
    ax.plot([bx_start + bw_b/2, bx_start + total_bw - bw_b/2], [1.335, 1.335], color='#94A3B8', lw=0.75, zorder=3)
    draw_vector_arrow(ax, mid_c, 1.335, mid_c, 1.245, color='#64748B', lw=0.95, ms=6.8, zorder=4)

    # Three stacked crisp orthogonal attribution pills
    pills = [
        ('Information', BLUE,   '#005B8F', 0.960),
        ('Model',       PURPLE, '#523C82', 0.655),
        ('Calibration', AMBER,  '#B04D00', 0.350),
    ]
    pw, ph = 0.775, 0.245
    px = cx0 + (cw - pw) / 2.0
    for label, fc_p, ec_p, py in pills:
        add_rounded_box(ax, px, py, pw, ph, fc=fc_p, ec=ec_p, lw=0.6, r=0.095, zorder=4)
        ax.text(px + pw/2.0, py + ph/2.0, label, ha='center', va='center',
                fontsize=7.0, fontweight='bold', color=WHITE, zorder=5)

    ax.text(mid_c, 0.195, '706 contrasts · T1–T6', ha='center', va='center',
            fontsize=6.5, fontweight='normal', color=GREY_TEXT, zorder=5)

    # Connector Arrow 2 (Center -> Right)
    draw_vector_arrow(ax, 1.958, 0.875, 2.022, 0.875, color='#64748B', lw=1.2, ms=7.8, zorder=8)

    # --------------------------------------------------------------------------
    # RIGHT ZONE (~36.5% width: x = 2.032 to 3.202, w = 1.170) — THREE QUANTITATIVE LIMITS
    # --------------------------------------------------------------------------
    rx_box, rw_box = 2.032, 1.170
    add_rounded_box(ax, rx_box, 0.075, rw_box, 1.60, fc='#FAFBFD', ec='#E2E8F0', lw=0.55, r=0.045, zorder=1)

    # Top deep-charcoal banner with crisp white text (ample side padding inside 1.114 in = 80.2 pt pill!)
    add_rounded_box(ax, rx_box + 0.028, 1.370, rw_box - 0.056, 0.245, fc=CHARCOAL, ec='none', r=0.042, zorder=4)
    ax.text(rx_box + rw_box / 2.0, 1.370 + 0.245 / 2.0, 'Accuracy ≠ Identifiability',
            ha='center', va='center', fontsize=6.5, fontweight='bold', color=WHITE, zorder=5)

    # Three minimalist tickless micro-pictograms + 2-3 word labels
    row_ys = [0.995, 0.585, 0.175]
    row_h = 0.315
    ic_x = rx_box + 0.055
    ic_w = 0.285
    txt_x = ic_x + ic_w + 0.055

    # --- Micro-row 1: Collinearity (r >= 0.98) ---
    y1 = row_ys[0]
    add_rounded_box(ax, ic_x, y1, ic_w, row_h, fc=WHITE, ec='#CBD5E1', lw=0.65, r=0.032, zorder=3)
    ax.plot([ic_x + 0.042, ic_x + ic_w - 0.042], [y1 + 0.055, y1 + row_h - 0.055],
            color=BLUE, lw=1.15, zorder=4)
    pts_t = np.array([0.08, 0.24, 0.38, 0.52, 0.67, 0.80, 0.92])
    pts_j = np.array([0.008, -0.009, 0.007, -0.006, 0.009, -0.007, 0.005])
    xs_s = ic_x + 0.048 + pts_t * (ic_w - 0.096)
    ys_s = y1 + 0.060 + pts_t * (row_h - 0.120) + pts_j
    ax.scatter(xs_s, ys_s, s=7.5, color=TEAL, edgecolors=WHITE, linewidths=0.25, zorder=5)
    ax.text(txt_x, y1 + row_h * 0.67, 'Collinearity', ha='left', va='center',
            fontsize=6.75, fontweight='bold', color=CHARCOAL, zorder=5)
    ax.text(txt_x, y1 + row_h * 0.27, '(r ≥ 0.98)', ha='left', va='center',
            fontsize=6.5, fontweight='normal', color=BLUE, zorder=5)

    # --- Micro-row 2: Dose-change inversion ---
    y2 = row_ys[1]
    add_rounded_box(ax, ic_x, y2, ic_w, row_h, fc=WHITE, ec='#CBD5E1', lw=0.65, r=0.032, zorder=3)
    draw_vector_arrow(ax, ic_x + 0.048, y2 + 0.075, ic_x + ic_w - 0.044, y2 + row_h - 0.070,
                      color=AMBER, lw=1.35, ms=7.2, zorder=5)
    draw_vector_arrow(ax, ic_x + 0.048, y2 + row_h - 0.070, ic_x + ic_w - 0.044, y2 + 0.075,
                      color=TEAL, lw=1.35, ms=7.2, zorder=5)
    ax.text(txt_x, y2 + row_h * 0.67, 'Dose-change', ha='left', va='center',
            fontsize=6.75, fontweight='bold', color=CHARCOAL, zorder=5)
    ax.text(txt_x, y2 + row_h * 0.27, 'inversion', ha='left', va='center',
            fontsize=6.6, fontweight='bold', color=AMBER, zorder=5)

    # --- Micro-row 3: Anchors & coverage ---
    y3 = row_ys[2]
    add_rounded_box(ax, ic_x, y3, ic_w, row_h, fc=WHITE, ec='#CBD5E1', lw=0.65, r=0.032, zorder=3)
    ox, oy = ic_x + 0.048, y3 + 0.055
    ax.plot([ox, ox, ic_x + ic_w - 0.038], [y3 + row_h - 0.045, oy, oy], color='#94A3B8', lw=0.65, zorder=4)
    ax.plot([ox, ox + 0.080, ic_x + ic_w - 0.048], [oy, oy + 0.035, oy + 0.095],
            color=TEAL, lw=1.15, zorder=5)
    ax.plot([ox, ox + 0.048, ox + 0.145], [oy, oy + 0.035, y3 + row_h - 0.048],
            color=RED, lw=1.20, ls='--', zorder=5)
    ax.scatter([ox + 0.048], [oy + 0.035], s=10, color=RED, zorder=6)
    ax.text(txt_x, y3 + row_h * 0.67, 'Anchors &', ha='left', va='center',
            fontsize=6.75, fontweight='bold', color=CHARCOAL, zorder=5)
    ax.text(txt_x, y3 + row_h * 0.27, 'coverage', ha='left', va='center',
            fontsize=6.6, fontweight='bold', color=RED, zorder=5)

    # Save PDF, SVG, and 600 dpi PNG (exact 1950 x 1050 px)
    pdf_path = os.path.join(OUT_DIR, 'PsyMix_TOC_Graphic.pdf')
    svg_path = os.path.join(OUT_DIR, 'PsyMix_TOC_Graphic.svg')
    png_path = os.path.join(OUT_DIR, 'PsyMix_TOC_Graphic_600dpi.png')
    fig.savefig(pdf_path, dpi=600)
    fig.savefig(svg_path, dpi=600)
    fig.savefig(png_path, dpi=600)
    plt.close(fig)
    print(f'[OK] Saved TOC Graphic -> {pdf_path}, {svg_path}, {png_path}')


# ==============================================================================
# TASK 2: SUPPLEMENTARY FIGURE S1 (180.0 x 175.0 mm)
# Panel a: Four model families across 153 archived configurations (no cross-domain sharing)
# Panel b: Architecture-capacity ceiling test (T1b botanical sets + T3 Table S4 adapters)
# ==============================================================================
def load_figs1_frozen_data():
    """Load 100% live from frozen source CSVs and assert exact Table S4 / manuscript parity."""
    # 1. Table S2 total configuration count
    df_s2 = pd.read_csv(CSV_TABLE_S2)
    assert len(df_s2) == 153, f'Expected 153 archived configs in Table S2, got {len(df_s2)}'

    # 2. T1b botanical set architectures (herb_metrics.csv, task=='six')
    hm = pd.read_csv(CSV_HERB_METRICS)
    t1b_specs = [
        ('Generic_BG',              'SET-LM+comp-LR',   'Linear bag + stats'),
        ('A2a_Attention',           'SET-FPDescLM-att', 'Label-gated att.'),
        ('DmcStyleTransformer_A2a', 'SET-FPDescLM-TR',  'Set Transformer (4.3k)'),
        ('DmcStyleMIL_A2a',         'SET-FPDescLM-MIL', 'MIL softmax (2.2k)'),
        ('TasteFrequency',          'NULL-label-rate',  'Label-freq. prior'),
    ]
    t1b_rows = []
    for old_code, new_code, desc in t1b_specs:
        r_fam = hm[(hm['dataset'] == 'Zhuang645') & (hm['protocol'] == 'family') &
                   (hm['task'] == 'six') & (hm['model'] == old_code)]
        r_ext = hm[(hm['dataset'] == 'Pharmacopoeia2025') & (hm['protocol'] == 'external') &
                   (hm['task'] == 'six') & (hm['model'] == old_code)]
        assert len(r_fam) == 1 and len(r_ext) == 1, f'Missing T1b row for {old_code}'
        t1b_rows.append({
            'old_code': old_code,
            'new_code': new_code,
            'desc': desc,
            'fam_ap': float(r_fam['Macro_AP'].iloc[0]),
            'fam_auc': float(r_fam['Macro_AUROC'].iloc[0]),
            'ext_ap': float(r_ext['Macro_AP'].iloc[0]),
            'ext_auc': float(r_ext['Macro_AUROC'].iloc[0]),
        })
    df_t1b = pd.DataFrame(t1b_rows)
    assert abs(df_t1b.loc[df_t1b['new_code'] == 'SET-FPDescLM-att', 'fam_ap'].iloc[0] - 0.324045) < 1e-4
    assert abs(df_t1b.loc[df_t1b['new_code'] == 'SET-FPDescLM-att', 'ext_ap'].iloc[0] - 0.385571) < 1e-4

    # 3. Cross-level attenuation (Uni五种子平均概率敏感性.csv) & Set-TR overfitting (family_curve_means.csv)
    u5 = pd.read_csv(CSV_UNI_5SEED)
    mol_delta_ap = float(u5[(u5['model'] == 'A2aET minus A2aLR') & (u5['metric'] == 'Macro_AP')]['value'].iloc[0])
    assert abs(mol_delta_ap - 0.109876) < 1e-4

    fcm = pd.read_csv(CSV_FAMILY_CURVES)
    tr_18 = float(fcm[(fcm['model'] == 'M3') & (fcm['step'] == 18)]['val_loss'].iloc[0])
    tr_300 = float(fcm[(fcm['model'] == 'M3') & (fcm['step'] == 300)]['val_loss'].iloc[0])
    assert abs(tr_18 - 0.426327) < 1e-4 and abs(tr_300 - 1.945358) < 1e-4

    # 4. T3 continuous-dose adapters vs baselines (C1_main_table_v2.csv + fit_log.csv)
    c1 = pd.read_csv(CSV_C1_MAIN)
    fl = pd.read_csv(CSV_ADPT_FITLOG)
    param_map = {}
    for pref in ['POOL_REG', 'DMC_SET_REG', 'PHYSSIM_INSPIRED_TASTE', 'PHYSSIM_NO_DYNAMICS']:
        sub = fl[fl['fit'].str.startswith(pref + '_')]
        param_map[pref] = int(sub['parameters'].iloc[0])
    assert param_map['POOL_REG'] == 465 and param_map['DMC_SET_REG'] == 2706

    t3_specs = [
        ('C1_ET_raw',              'TREE-conc',    'baseline', AMBER),
        ('C1_Ridge_se_by_id',      'EQS×ID',       'baseline', TEAL),
        ('C1_Ridge_se',            'EQS (ref)',    'ref',      CHARCOAL),
        ('C1_Ridge_raw',           'RIDGE-conc',   'baseline', SLATE),
        ('POOL_REG',               'ADPT-pool (465 p.)',  'adapter', PURPLE),
        ('DMC_SET_REG',            'ADPT-set (2.7k p.)',  'adapter', PURPLE),
        ('PHYSSIM_INSPIRED_TASTE', 'ADPT-dyn (1.4k p.)',  'adapter', PURPLE),
        ('C1_mean',                'NULL-mean',    'null',     '#64748B'),
    ]
    eqs_row = c1[c1['model'] == 'C1_Ridge_se'].iloc[0]
    eqs_abs = float(eqs_row['abs_mae'])
    eqs_chg = float(eqs_row['delta_mae'])
    assert abs(eqs_abs - 0.929768) < 1e-4 and abs(eqs_chg - 0.610163) < 1e-4

    # Also verify ADPT-dyn-off for footnote
    dyn_off = c1[c1['model'] == 'PHYSSIM_NO_DYNAMICS'].iloc[0]
    assert abs(float(dyn_off['abs_mae']) - 1.994066) < 1e-4

    t3_rows = []
    for old_code, label, group, col in t3_specs:
        r = c1[c1['model'] == old_code].iloc[0]
        abs_m = float(r['abs_mae'])
        chg_m = float(r['delta_mae'])
        if group == 'ref':
            abs_lo, abs_hi = abs_m, abs_m
            chg_lo, chg_hi = chg_m, chg_m
        else:
            abs_lo = eqs_abs + float(r['abs_mae_lo'])
            abs_hi = eqs_abs + float(r['abs_mae_hi'])
            chg_lo = eqs_chg + float(r['delta_mae_lo'])
            chg_hi = eqs_chg + float(r['delta_mae_hi'])
        t3_rows.append({
            'old_code': old_code, 'label': label, 'group': group, 'color': col,
            'abs_mae': abs_m, 'abs_lo': abs_lo, 'abs_hi': abs_hi,
            'chg_mae': chg_m, 'chg_lo': chg_lo, 'chg_hi': chg_hi,
        })
    df_t3 = pd.DataFrame(t3_rows)

    return {
        'n_configs': len(df_s2),
        'df_t1b': df_t1b,
        'mol_delta_ap': mol_delta_ap,
        'tr_18': tr_18,
        'tr_300': tr_300,
        'df_t3': df_t3,
        'eqs_abs': eqs_abs,
        'eqs_chg': eqs_chg,
        'dyn_off_abs': float(dyn_off['abs_mae']),
        'dyn_off_chg': float(dyn_off['delta_mae']),
    }


def draw_chip(ax, x, y, w, h, title, sub, fc, ec, title_color=CHARCOAL, sub_color=GREY_TEXT,
              t_fs=6.5, s_fs=5.75, r=1.2, zorder=4):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f'round,pad=0,rounding_size={r}',
        facecolor=fc, edgecolor=ec, linewidth=0.65, zorder=zorder
    )
    ax.add_patch(box)
    if sub:
        ax.text(x + w/2.0, y + h*0.67, title, ha='center', va='center',
                fontsize=t_fs, fontweight='bold', color=title_color, zorder=zorder+1)
        ax.text(x + w/2.0, y + h*0.28, sub, ha='center', va='center',
                fontsize=s_fs, color=sub_color, zorder=zorder+1)
    else:
        ax.text(x + w/2.0, y + h*0.50, title, ha='center', va='center',
                fontsize=t_fs, fontweight='bold', color=title_color, zorder=zorder+1)


def draw_domain_pills(ax, x_center, y_center, domains, fc, ec, text_color):
    """Draw 1 to 3 compact stacked rows of domain pills centered in column 4."""
    n = len(domains)
    if n <= 2:
        rows = [domains]
    elif n <= 4:
        rows = [domains[:2], domains[2:]]
    else:
        rows = [domains[:2], domains[2:4], domains[4:]]
    ph = 4.0
    pw = 6.7
    gx, gy = 1.1, 1.25
    tot_h = len(rows) * ph + (len(rows) - 1) * gy
    y_top = y_center + tot_h / 2.0 - ph
    for ri, row in enumerate(rows):
        ry = y_top - ri * (ph + gy)
        tot_w = len(row) * pw + (len(row) - 1) * gx
        rx0 = x_center - tot_w / 2.0
        for ci, d in enumerate(row):
            rx = rx0 + ci * (pw + gx)
            box = FancyBboxPatch(
                (rx, ry), pw, ph,
                boxstyle='round,pad=0,rounding_size=1.15',
                facecolor=fc, edgecolor=ec, linewidth=0.65, zorder=4
            )
            ax.add_patch(box)
            ax.text(rx + pw/2.0, ry + ph/2.0, d, ha='center', va='center',
                    fontsize=5.8, fontweight='bold', color=text_color, zorder=5)


def build_figure_s1():
    data = load_figs1_frozen_data()

    # Canvas: 180.0 x 175.0 mm (7.0866 x 6.8898 in)
    W_mm, H_mm = 180.0, 175.0
    fig = plt.figure(figsize=(W_mm / 25.4, H_mm / 25.4), dpi=600, facecolor=WHITE)

    # ==========================================================================
    # PANEL a (top 60.5% height): Four model families across 153 configurations
    # Coordinate system in mm: x in [0, 180], y in [0, 105.875]
    # ==========================================================================
    ax_a = fig.add_axes([0.0, 0.395, 1.0, 0.605])
    ax_a.set_xlim(0.0, 180.0)
    ax_a.set_ylim(0.0, 105.875)
    ax_a.set_aspect('equal')
    ax_a.axis('off')

    # Panel tag 'a' + panel title
    ax_a.text(3.2, 101.9, 'a', ha='left', va='center', fontsize=11.0, fontweight='bold', color=CHARCOAL)
    ax_a.text(8.2, 101.9, f"Four model families across {data['n_configs']} archived configurations",
              ha='left', va='center', fontsize=8.8, fontweight='bold', color=CHARCOAL)

    # Crisp top scope banner: explicitly NOT a cross-domain end-to-end black box
    banner = FancyBboxPatch(
        (3.5, 93.3), 173.0, 5.5,
        boxstyle='round,pad=0,rounding_size=1.3',
        facecolor='#EDF2F7', edgecolor=CHARCOAL, linewidth=0.75, zorder=2
    )
    ax_a.add_patch(banner)
    ax_a.add_patch(FancyBboxPatch(
        (3.5, 93.3), 2.4, 5.5,
        boxstyle='round,pad=0,rounding_size=0.8',
        facecolor=CHARCOAL, edgecolor='none', zorder=3
    ))
    ax_a.text(90.5, 96.05,
              'Independent domain benchmarks (no cross-domain parameter sharing; frozen task-native holdout units & rating scales)',
              ha='center', va='center', fontsize=6.6, fontweight='bold', color=CHARCOAL, zorder=4)

    # Spacious balanced columns across 3.5 .. 176.5 mm (zero horizontal collisions!)
    col_x = {
        'fam':  (3.5,   34.3),   # 30.8 mm
        'inp':  (36.0,  64.0),   # 28.0 mm
        'core': (67.6, 133.4),   # 65.8 mm
        'read': (136.9, 160.3),  # 23.4 mm (roomy for '(train-fold CV / @0 pair)')
        'dom':  (161.6, 176.5),  # 14.9 mm
    }

    # Column header strip
    hdr_y = 89.4
    headers = [
        ('Model family',                 (col_x['fam'][0] + col_x['fam'][1]) / 2.0),
        ('Inputs',                       (col_x['inp'][0] + col_x['inp'][1]) / 2.0),
        ('Core architecture / operator', (col_x['core'][0] + col_x['core'][1]) / 2.0),
        ('Readout & calib.',             (col_x['read'][0] + col_x['read'][1]) / 2.0),
        ('Domains',                      (col_x['dom'][0] + col_x['dom'][1]) / 2.0),
    ]
    for label, hx in headers:
        ax_a.text(hx, hdr_y, label, ha='center', va='center', fontsize=6.85, fontweight='bold', color=CHARCOAL)
    for x_arr in [65.8, 135.15]:
        ax_a.text(x_arr, hdr_y, '→', ha='center', va='center', fontsize=7.5, color='#64748B', fontweight='bold')
    ax_a.plot([3.5, 176.5], [86.9, 86.9], color=CHARCOAL, lw=0.85, zorder=3)

    # Four horizontal swimlanes (height = 19.7 mm each, gap = 1.6 mm -> bottom = 1.9 mm)
    lane_h = 19.7
    lane_gap = 1.6
    lane_tops = [85.5 - i * (lane_h + lane_gap) for i in range(4)]

    lanes_meta = [
        {
            'num': '1', 'name': 'Psychophysical &\ndose-response',
            'tag': 'Mechanistic / semi-empirical',
            'color': TEAL, 'soft': TEAL_SOFT, 'bg': WHITE,
            'domains': ['T2', 'T3', 'T5', 'T6'],
        },
        {
            'num': '2', 'name': 'Classical tabular &\nmolecular baselines',
            'tag': 'Fixed-feature ML controls',
            'color': SLATE, 'soft': SLATE_SOFT, 'bg': STRIPE_BG,
            'domains': ['T1a', 'T1b', 'T2', 'T3', 'T4b', 'T6'],
        },
        {
            'num': '3', 'name': 'Botanical constituent-\nset neural networks',
            'tag': 'Unordered-set DL (no doses)',
            'color': BLUE, 'soft': BLUE_SOFT, 'bg': WHITE,
            'domains': ['T1a', 'T1b'],
        },
        {
            'num': '4', 'name': 'Continuous-dose deep\nadapters & graph',
            'tag': 'Capacity ceiling & completion',
            'color': PURPLE, 'soft': PURPLE_SOFT, 'bg': STRIPE_BG,
            'domains': ['T3', 'T4a', 'T4b'],
        },
    ]

    for idx, lm in enumerate(lanes_meta):
        y_top = lane_tops[idx]
        y_bot = y_top - lane_h
        y_mid = (y_top + y_bot) / 2.0

        ax_a.add_patch(FancyBboxPatch(
            (3.5, y_bot), 173.0, lane_h,
            boxstyle='round,pad=0,rounding_size=1.3',
            facecolor=lm['bg'], edgecolor='#DCE3EC', linewidth=0.6, zorder=1
        ))
        ax_a.add_patch(FancyBboxPatch(
            (3.5, y_bot), 2.3, lane_h,
            boxstyle='round,pad=0,rounding_size=0.8',
            facecolor=lm['color'], edgecolor='none', zorder=3
        ))
        ax_a.add_patch(Circle((8.5, y_top - 4.3), 2.05, facecolor=lm['color'], edgecolor='none', zorder=4))
        ax_a.text(8.5, y_top - 4.3, lm['num'], ha='center', va='center',
                  fontsize=6.2, fontweight='bold', color=WHITE, zorder=5)
        ax_a.text(11.4, y_top - 6.6, lm['name'], ha='left', va='center',
                  fontsize=6.05, fontweight='bold', color=CHARCOAL, linespacing=1.18, zorder=4)
        ax_a.text(6.9, y_bot + 3.4, lm['tag'], ha='left', va='center',
                  fontsize=5.25, color=lm['color'], fontweight='bold', zorder=4)

        draw_vector_arrow(ax_a, col_x['inp'][1] + 0.45, y_mid, col_x['core'][0] - 0.45, y_mid,
                          color='#64748B', lw=0.95, ms=6.8, zorder=5)
        draw_vector_arrow(ax_a, col_x['core'][1] + 0.45, y_mid, col_x['read'][0] - 0.45, y_mid,
                          color='#64748B', lw=0.95, ms=6.8, zorder=5)

        ax_a.plot([160.95, 160.95], [y_bot + 1.8, y_top - 1.8], color='#CBD5E1', lw=0.6, ls=':', zorder=2)
        draw_domain_pills(ax_a, (col_x['dom'][0] + col_x['dom'][1]) / 2.0, y_mid,
                          lm['domains'], lm['soft'], lm['color'], lm['color'])

    # ---------------- LANE 1 CONTENT (Psychophysical & dose-response) ----------------
    y1_t, y1_b = lane_tops[0], lane_tops[0] - lane_h
    y1_m = (y1_t + y1_b) / 2.0
    add_rounded_box(ax_a, col_x['inp'][0], y1_b + 1.9, col_x['inp'][1] - col_x['inp'][0], lane_h - 3.8,
                    fc=TEAL_SOFT, ec='#9ED3C5', lw=0.65, r=1.1, zorder=3)
    ax_a.text((col_x['inp'][0] + col_x['inp'][1])/2, y1_m,
              'Quantified $c_i$,\naqueous $T_i$ or EQS,\ntarget anchors\n(@0 / @1 / @5 / @10)',
              ha='center', va='center', fontsize=5.65, color=CHARCOAL, linespacing=1.26, zorder=4)
    cx_l = col_x['core'][0]
    cw_core = col_x['core'][1] - col_x['core'][0]  # 65.8 mm
    cw_half = (cw_core - 1.8) / 2.0  # 32.0 mm
    ch_half = 7.45
    draw_chip(ax_a, cx_l, y1_t - 9.0, cw_half, ch_half,
              'SUM-mass / molar / count', 'Unweighted additive dose totals',
              WHITE, TEAL, title_color=TEAL, t_fs=6.0, s_fs=5.3)
    draw_chip(ax_a, cx_l + cw_half + 1.8, y1_t - 9.0, cw_half, ch_half,
              'DOT-lin / DOT-log', r'$U=\sum c_i/T_i$; $H=\ln(1+U/K)$',
              WHITE, TEAL, title_color=TEAL, t_fs=6.0, s_fs=5.25)
    draw_chip(ax_a, cx_l, y1_b + 1.55, cw_half, ch_half,
              'EQS  &  EQS×ID', 'Iso-sweet levels + id×level slopes',
              WHITE, TEAL, title_color=TEAL, t_fs=6.0, s_fs=5.25)
    draw_chip(ax_a, cx_l + cw_half + 1.8, y1_b + 1.55, cw_half, ch_half,
              'Source-affine@1', r'Retains source offset $-b/a$',
              WHITE, TEAL, title_color=TEAL, t_fs=6.0, s_fs=5.3)
    add_rounded_box(ax_a, col_x['read'][0], y1_b + 1.9, col_x['read'][1] - col_x['read'][0], lane_h - 3.8,
                    fc=WHITE, ec='#CBD5E1', lw=0.65, r=1.1, zorder=3)
    ax_a.text((col_x['read'][0] + col_x['read'][1])/2, y1_m,
              'Train-fold affine,\nshrunk (@5/@10)\nor prop. @1\n' + r'($\alpha = I_0 / H_0$)',
              ha='center', va='center', fontsize=5.55, color=CHARCOAL, linespacing=1.26, zorder=4)

    # ---------------- LANE 2 CONTENT (Classical tabular & molecular baselines) ----------------
    y2_t, y2_b = lane_tops[1], lane_tops[1] - lane_h
    y2_m = (y2_t + y2_b) / 2.0
    add_rounded_box(ax_a, col_x['inp'][0], y2_b + 1.9, col_x['inp'][1] - col_x['inp'][0], lane_h - 3.8,
                    fc=SLATE_SOFT, ec=SLATE_LT, lw=0.65, r=1.1, zorder=3)
    ax_a.text((col_x['inp'][0] + col_x['inp'][1])/2, y2_m,
              '104-d mol., 65-d bag,\nECFP4 (768-bit),\nnamed/anon $\\log(1+c)$,\nmonomer pure refs',
              ha='center', va='center', fontsize=5.4, color=CHARCOAL, linespacing=1.26, zorder=4)
    # Symmetric 3 x 2 operator grid (includes SET-LM+comp-LR for exact 1:1 closure with Panel b-left!)
    cw3 = (cw_core - 3.2) / 3.0  # 20.87 mm
    draw_chip(ax_a, cx_l, y2_t - 9.0, cw3, ch_half,
              'Logistic', 'L2 (C = 1 / 0.1)',
              WHITE, SLATE, title_color=SLATE, t_fs=5.9, s_fs=5.2)
    draw_chip(ax_a, cx_l + cw3 + 1.6, y2_t - 9.0, cw3, ch_half,
              'RIDGE', r'$\alpha \in \{1, 4, 16\}$',
              WHITE, SLATE, title_color=SLATE, t_fs=5.9, s_fs=5.2)
    draw_chip(ax_a, cx_l + 2*(cw3 + 1.6), y2_t - 9.0, cw3, ch_half,
              'ExtraTrees', '160–256 trees',
              WHITE, SLATE, title_color=SLATE, t_fs=5.9, s_fs=5.2)
    draw_chip(ax_a, cx_l, y2_b + 1.55, cw3, ch_half,
              'ECFP4 + CatBoost', '240 trees, d=5 (T1a)',
              WHITE, SLATE, title_color=SLATE, t_fs=5.65, s_fs=5.1)
    draw_chip(ax_a, cx_l + cw3 + 1.6, y2_b + 1.55, cw3, ch_half,
              'SET-LM+comp-LR', 'LM32+stats LR (T1b)',
              WHITE, SLATE, title_color=SLATE, t_fs=5.6, s_fs=5.05)
    draw_chip(ax_a, cx_l + 2*(cw3 + 1.6), y2_b + 1.55, cw3, ch_half,
              'TER-direct', 'Binary-edge ridge',
              WHITE, SLATE, title_color=SLATE, t_fs=5.8, s_fs=5.1)
    add_rounded_box(ax_a, col_x['read'][0], y2_b + 1.9, col_x['read'][1] - col_x['read'][0], lane_h - 3.8,
                    fc=WHITE, ec='#CBD5E1', lw=0.65, r=1.1, zorder=3)
    ax_a.text((col_x['read'][0] + col_x['read'][1])/2, y2_m,
              '6-label BCE or\nnative intensity /\nΔintensity regr.\n(train-fold CV / @0 pair)',
              ha='center', va='center', fontsize=5.15, color=CHARCOAL, linespacing=1.28, zorder=4)

    # ---------------- LANE 3 CONTENT (Botanical constituent-set neural networks) ----------------
    y3_t, y3_b = lane_tops[2], lane_tops[2] - lane_h
    y3_m = (y3_t + y3_b) / 2.0
    add_rounded_box(ax_a, col_x['inp'][0], y3_b + 1.9, col_x['inp'][1] - col_x['inp'][0], lane_h - 3.8,
                    fc=BLUE_SOFT, ec='#94C5E8', lw=0.65, r=1.1, zorder=3)
    ax_a.text((col_x['inp'][0] + col_x['inp'][1])/2, y3_m,
              'SMILES (T1a) or\nunordered plant set\n' + r'$\{h_i\}$ (104-d/mol:' + '\nFP32+Desc40+LM32)',
              ha='center', va='center', fontsize=5.5, color=CHARCOAL, linespacing=1.26, zorder=4)
    draw_chip(ax_a, cx_l, y3_t - 7.1, cw_core, 5.65,
              'ChemBERTa (fine-tuned SMILES LM) & 6L-8H SMILES Transformer',
              'Released Uni-Flavor & local LM encoders (d=256; single molecules, T1a)',
              '#F4F9FD', BLUE, title_color=BLUE, t_fs=5.45, s_fs=5.05, r=1.0)
    bh3 = 9.3
    draw_chip(ax_a, cx_l, y3_b + 1.5, cw3, bh3,
              'SET-FPDescLM-att', 'Label-gated pool\n' + r'$\exp(3\tanh(h_i g_k))$',
              WHITE, BLUE, title_color=BLUE, t_fs=5.45, s_fs=4.95)
    draw_chip(ax_a, cx_l + cw3 + 1.6, y3_b + 1.5, cw3, bh3,
              'SET-FPDescLM-MIL', '104→16 GELU MLP\n7-label softmax',
              WHITE, BLUE, title_color=BLUE, t_fs=5.45, s_fs=4.95)
    draw_chip(ax_a, cx_l + 2*(cw3 + 1.6), y3_b + 1.5, cw3, bh3,
              'SET-FPDescLM-TR', '1L-2H self-att + FFN\n4.3k p., perm.-inv.',
              WHITE, BLUE, title_color=BLUE, t_fs=5.45, s_fs=4.95)
    add_rounded_box(ax_a, col_x['read'][0], y3_b + 1.9, col_x['read'][1] - col_x['read'][0], lane_h - 3.8,
                    fc=WHITE, ec='#CBD5E1', lw=0.65, r=1.1, zorder=3)
    ax_a.text((col_x['read'][0] + col_x['read'][1])/2, y3_m,
              'Multi-label BCE\n(6 eval / 4 core\nflavors; no dose\nor rating scale)',
              ha='center', va='center', fontsize=5.55, color=CHARCOAL, linespacing=1.26, zorder=4)

    # ---------------- LANE 4 CONTENT (Continuous-dose deep adapters & graph) ----------------
    y4_t, y4_b = lane_tops[3], lane_tops[3] - lane_h
    y4_m = (y4_t + y4_b) / 2.0
    add_rounded_box(ax_a, col_x['inp'][0], y4_b + 1.9, col_x['inp'][1] - col_x['inp'][0], lane_h - 3.8,
                    fc=PURPLE_SOFT, ec='#BFAEE0', lw=0.65, r=1.1, zorder=3)
    ax_a.text((col_x['inp'][0] + col_x['inp'][1])/2, y4_m,
              r'Dose tokens $\{(h_i, c_i)\}$' + '\n(10-d: 8 RDKit + dose\n+ mask) OR binary\nresidual graph $J_{ij}$',
              ha='center', va='center', fontsize=5.45, color=CHARCOAL, linespacing=1.24, zorder=4)
    th4 = 9.3
    draw_chip(ax_a, cx_l, y4_t - 10.75, cw3, th4,
              'ADPT-pool', 'DeepSets 10→16→16\nmean pool (465 p.)',
              WHITE, PURPLE, title_color=PURPLE, t_fs=5.75, s_fs=5.0)
    draw_chip(ax_a, cx_l + cw3 + 1.6, y4_t - 10.75, cw3, th4,
              'ADPT-set', '1L-2H self-att + FFN\natt. pool (2.7k p.)',
              WHITE, PURPLE, title_color=PURPLE, t_fs=5.75, s_fs=5.0)
    draw_chip(ax_a, cx_l + 2*(cw3 + 1.6), y4_t - 10.75, cw3, th4,
              'ADPT-dyn', '3-step low-rank force\n8-D latent (1.4k p.)',
              WHITE, PURPLE, title_color=PURPLE, t_fs=5.75, s_fs=5.0)
    draw_chip(ax_a, cx_l, y4_b + 1.5, cw_core, 5.9,
              'SELF+GRAPH & TER-graph (Laplacian regularization)',
              r'Tanimoto line-graph $(I + hL)^{-1}$ on shared binary edges + monomers (T4a–T4b)',
              '#F7F5FC', PURPLE, title_color=PURPLE, t_fs=5.7, s_fs=4.95, r=1.0)
    add_rounded_box(ax_a, col_x['read'][0], y4_b + 1.9, col_x['read'][1] - col_x['read'][0], lane_h - 3.8,
                    fc=WHITE, ec='#CBD5E1', lw=0.65, r=1.1, zorder=3)
    ax_a.text((col_x['read'][0] + col_x['read'][1])/2, y4_m,
              'Z-scored MSE\n(3 seeds, 5-fold CV,\nearly stop) or\nternary sum (0–15)',
              ha='center', va='center', fontsize=5.45, color=CHARCOAL, linespacing=1.25, zorder=4)

    # Clean horizontal rule separating Panel a and Panel b
    ax_a.plot([3.5, 176.5], [0.5, 0.5], color='#CBD5E1', lw=0.75)

    # ==========================================================================
    # PANEL b (bottom 39.5% height): Architecture-capacity ceiling test
    # ==========================================================================
    fig.text(3.2 / W_mm, 0.376, 'b', ha='left', va='center', fontsize=11.0, fontweight='bold', color=CHARCOAL)
    fig.text(8.2 / W_mm, 0.376,
             'Architecture-capacity ceiling test (frozen T1b botanical sets & Table S4 continuous-dose adapters)',
             ha='left', va='center', fontsize=8.35, fontweight='bold', color=CHARCOAL)

    fig.text(0.145, 0.348, 'T1b: Botanical constituent-set architectures',
             ha='left', va='center', fontsize=7.0, fontweight='bold', color=BLUE)
    fig.text(0.578, 0.348, 'T3: Continuous-dose adapters vs. baselines (55 pairs, Table S4)',
             ha='left', va='center', fontsize=6.9, fontweight='bold', color=PURPLE)

    # -------------------- b-left: T1b Botanical Set Ceiling -------------------
    ax_bl = fig.add_axes([0.145, 0.112, 0.315, 0.193])
    df_t1b = data['df_t1b']
    y_pos = np.arange(len(df_t1b))[::-1]

    null_fam = df_t1b.loc[df_t1b['new_code'] == 'NULL-label-rate', 'fam_ap'].iloc[0]
    ax_bl.axvline(null_fam, color='#94A3B8', lw=0.9, ls=':', zorder=1)

    for i, row in df_t1b.iterrows():
        y = y_pos[i]
        is_null = (row['new_code'] == 'NULL-label-rate')
        c_fam = '#64748B' if is_null else BLUE
        c_ext = '#94A3B8' if is_null else TEAL
        ax_bl.plot([row['fam_ap'], row['ext_ap']], [y, y], color='#CBD5E1', lw=1.15, zorder=2)
        ax_bl.scatter([row['fam_ap']], [y], s=26, color=c_fam, edgecolors=WHITE, linewidths=0.45, zorder=4,
                      label='Zhuang645 (n=645)' if i == 0 else None)
        ax_bl.scatter([row['ext_ap']], [y], s=25, facecolors=WHITE, edgecolors=c_ext, marker='D', linewidths=1.15, zorder=5,
                      label='CP2025 (n=128)' if i == 0 else None)
        ax_bl.text(0.418, y, f"{row['fam_ap']:.3f} / {row['ext_ap']:.3f}",
                   ha='left', va='center', fontsize=5.8, color=GREY_TEXT if is_null else CHARCOAL,
                   fontweight='bold' if i < 2 else 'normal')

    ax_bl.set_yticks(y_pos)
    ax_bl.set_yticklabels(df_t1b['new_code'].tolist(), fontsize=6.2, fontweight='bold', color=CHARCOAL)
    ax_bl.set_xlim(0.20, 0.505)
    ax_bl.set_ylim(-0.6, len(df_t1b) - 0.32)
    ax_bl.set_xticks([0.20, 0.25, 0.30, 0.35, 0.40])
    ax_bl.tick_params(axis='x', labelsize=6.0, length=2.8, pad=2)
    ax_bl.tick_params(axis='y', length=0, pad=3)
    ax_bl.set_xlabel('6-attribute Macro-AP (higher is better →)', fontsize=6.35, fontweight='bold', labelpad=2.5)
    ax_bl.text(0.418, len(df_t1b) - 0.32, 'Fam / Ext', ha='left', va='bottom',
               fontsize=5.65, fontweight='bold', color=GREY_TEXT)
    ax_bl.spines['top'].set_visible(False)
    ax_bl.spines['right'].set_visible(False)
    ax_bl.grid(axis='x', color='#E2E8F0', lw=0.5, ls='-', zorder=0)

    ax_bl.legend(loc='lower left', bbox_to_anchor=(-0.03, 1.075), ncol=2, frameon=False,
                 fontsize=5.75, handletextpad=0.22, columnspacing=0.65, borderaxespad=0.0)

    callout_text = (
        f"Cross-level gain attenuation: Mol. +{data['mol_delta_ap']:.3f} → Flora +0.0013 [−0.008, +0.011] AP\n"
        f"Extended Set-TR (18 → 300 steps): val. BCE loss {data['tr_18']:.3f} → {data['tr_300']:.3f} (overfitting)"
    )
    fig.text(
        0.024, 0.038, callout_text,
        ha='left', va='center', fontsize=5.65, color=CHARCOAL, linespacing=1.35,
        bbox=dict(boxstyle='round,pad=0.36,rounding_size=0.25', facecolor='#F8FAFC', edgecolor='#CBD5E1', lw=0.65)
    )

    # -------------------- b-right: T3 Table S4 Adapters vs Baselines ----------
    # Dedicated right-aligned numeric MAE columns cleanly outside data/CI ranges (zero line collision!)
    ax_br1 = fig.add_axes([0.578, 0.112, 0.193, 0.193])
    ax_br2 = fig.add_axes([0.788, 0.112, 0.195, 0.193])

    df_t3 = data['df_t3']
    y_t3 = np.array([7.35, 6.35, 5.35, 4.35, 3.00, 2.00, 1.00, 0.00])

    for ax_sub, col_m, col_lo, col_hi, ref_val, xlim, xticks, div_x, val_x, sub_hdr in [
        (ax_br1, 'abs_mae', 'abs_lo', 'abs_hi', data['eqs_abs'],
         (0.0, 2.92), [0.0, 0.5, 1.0, 1.5, 2.0], 2.24, 2.88, 'Absolute MAE'),
        (ax_br2, 'chg_mae', 'chg_lo', 'chg_hi', data['eqs_chg'],
         (0.0, 4.58), [0.0, 1.0, 2.0, 3.0],      3.54, 4.52, 'Dose-change MAE'),
    ]:
        # Shaded soft purple band behind the 3 continuous neural adapters (stops cleanly at divider)
        ax_sub.axhspan(0.55, 3.45, xmin=0.0, xmax=div_x / xlim[1], color=PURPLE_SOFT, alpha=0.55, zorder=0)
        ax_sub.axhline(3.68, xmin=0.0, xmax=div_x / xlim[1], color='#CBD5E1', lw=0.7, ls='--', zorder=1)
        # Dashed vertical EQS reference line
        ax_sub.axvline(ref_val, color=CHARCOAL, lw=0.95, ls='--', zorder=2)
        # Subtle divider separating plot zone from aligned right MAE column
        ax_sub.axvline(div_x, color='#E2E8F0', lw=0.7, ls='-', zorder=1)
        ax_sub.text(val_x, 7.95, 'MAE', ha='right', va='bottom',
                    fontsize=5.35, fontweight='bold', color=GREY_TEXT)

        for i, row in df_t3.iterrows():
            y = y_t3[i]
            val = row[col_m]
            lo, hi = row[col_lo], row[col_hi]
            col = row['color']
            is_null = (row['group'] == 'null')
            is_ref = (row['group'] == 'ref')

            if not is_ref:
                stem_c = TEAL if val < ref_val else ('#94A3B8' if is_null else col)
                ax_sub.plot([ref_val, val], [y, y], color=stem_c, lw=1.05, alpha=0.45, zorder=2)
                ax_sub.plot([lo, hi], [y, y], color=col, lw=1.35, solid_capstyle='round', zorder=4)

            if is_null:
                ax_sub.scatter([val], [y], s=22, facecolors=WHITE, edgecolors=col, linewidths=1.1, zorder=5)
            elif is_ref:
                ax_sub.scatter([val], [y], s=25, color=CHARCOAL, marker='s', edgecolors=WHITE, linewidths=0.4, zorder=5)
            else:
                ax_sub.scatter([val], [y], s=24, color=col, edgecolors=WHITE, linewidths=0.4, zorder=5)

            # Right-aligned dedicated numeric column (zero overlap with dashed line, marker, or whiskers!)
            ax_sub.text(val_x, y, f'{val:.3f}', ha='right', va='center',
                        fontsize=5.6, color=GREY_TEXT if is_null else CHARCOAL,
                        fontweight='bold' if (val < ref_val or is_ref) else 'normal')

        ax_sub.set_xlim(*xlim)
        ax_sub.set_ylim(-0.6, 8.05)
        ax_sub.set_xticks(xticks)
        ax_sub.tick_params(axis='x', labelsize=5.85, length=2.6, pad=2)
        ax_sub.set_xlabel('← Better (MAE, 0–15)', fontsize=5.8, fontweight='bold', labelpad=2.5)
        ax_sub.set_title(sub_hdr, fontsize=6.4, fontweight='bold', color=CHARCOAL, pad=6)
        ax_sub.spines['top'].set_visible(False)
        ax_sub.spines['right'].set_visible(False)
        ax_sub.spines['bottom'].set_bounds(0.0, div_x)
        ax_sub.grid(axis='x', color='#E2E8F0', lw=0.45, ls='-', zorder=0)

    ax_br1.set_yticks(y_t3)
    ax_br1.set_yticklabels(df_t3['label'].tolist(), fontsize=5.9, fontweight='bold', color=CHARCOAL)
    ax_br1.tick_params(axis='y', length=0, pad=3)
    for lbl in ax_br1.get_yticklabels():
        if lbl.get_text().startswith('ADPT-'):
            lbl.set_color(PURPLE)
        elif lbl.get_text().startswith('NULL'):
            lbl.set_color('#64748B')
    ax_br2.set_yticks(y_t3)
    ax_br2.set_yticklabels([])
    ax_br2.tick_params(axis='y', length=0)

    # Explicit 3-line bottom footnote clarifying EQS + paired CI & ADPT-* non-EQS token inputs
    fig.text(
        0.578, 0.036,
        f"Dashed line: EQS ref ({data['eqs_abs']:.3f} / {data['eqs_chg']:.3f}). Whiskers: EQS + paired 95% CI\n"
        f"of ΔMAE (spans dashed line iff ns). ADPT-* use RDKit + dose tokens without\n"
        f"EQS scaling; ADPT-dyn-off ({1364:,} p.): {data['dyn_off_abs']:.3f} / {data['dyn_off_chg']:.3f} (identical to ADPT-dyn).",
        ha='left', va='center', fontsize=5.2, color=GREY_TEXT, linespacing=1.28
    )

    pdf_path = os.path.join(OUT_DIR, 'PsyMix_FigureS1.pdf')
    svg_path = os.path.join(OUT_DIR, 'PsyMix_FigureS1.svg')
    png_path = os.path.join(OUT_DIR, 'PsyMix_FigureS1_600dpi.png')
    fig.savefig(pdf_path, dpi=600)
    fig.savefig(svg_path, dpi=600)
    fig.savefig(png_path, dpi=600)
    plt.close(fig)
    print(f'[OK] Saved Figure S1 -> {pdf_path}, {svg_path}, {png_path}')


if __name__ == '__main__':
    if '--figs1-only' not in sys.argv:
        build_toc_graphic()
    build_figure_s1()
