# -*- coding: utf-8 -*-
"""
PsyMix Supplementary Figure S2 (180.0 x 165.0 mm, 2x2):
  a: Cross-database prior failure (15 commercial plant milks FoodAtlas/FDC + 1,335 FlavorDB/FlavorGraph bags)
  b: Dry-material vs brewed-cup & whole-extract matrix disconnects (Pingyin rose + licorice)
  c: Upward dose extrapolation failure on 55 sweetener pairs (3%+5% -> 7% SE)
  d: RDKit 2D vector structure-activity cliffs & in vitro vs human-oral decoupling
Strict continuation of Figures 1-6 & S1 visual constitution.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from scipy.stats import spearmanr

sys.path.insert(0, r'C:\zhuang_taste_pre\psymix_foodatlas_multimodel_20261121\.deps\uni_mordred')
sys.path.insert(0, r'C:\zhuang_taste_pre\psymix_foodatlas_multimodel_20261121\.deps')
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdDepictor, rdFingerprintGenerator
import fitz

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

BLUE        = '#0072B2'
BLUE_SOFT   = '#E6F2FA'
TEAL        = '#1B7F6B'
TEAL_SOFT   = '#E6F4F0'
RED         = '#C0392B'
RED_SOFT    = '#FCECEB'
AMBER       = '#D55E00'
AMBER_SOFT  = '#FDF0E6'
GOLD        = '#E69F00'
PURPLE      = '#6A51A3'
PURPLE_SOFT = '#F1EEF8'
SLATE       = '#4D648D'
SLATE_LT    = '#9BB5CE'
CHARCOAL    = '#2C3E50'
GREY_TEXT   = '#4A5568'
GREY_LINE   = '#CBD5E1'
STRIPE_BG   = '#F8FAFC'
WHITE       = '#FFFFFF'

OUT_DIR = r'c:\gemi-antigravity\00_PsyMix_Submission_Master\02_Figures_Design_and_Drafts\final_figures\TOC_and_FigS1'

CSV_FA_INC   = r'C:\zhuang_taste_pre\psymix_foodatlas_multimodel_20261121\results\foodatlas_increment.csv'
CSV_UPWARD   = r'C:\zhuang_taste_pre\psymix_foodatlas_multimodel_20261121\set_dynamics\schiffman\dose\upward_oos.csv'
CSV_ROSE_C   = r'C:\zhuang_taste_pre\psymix_herbal_infusions_20261122\rose\chemistry_sample_means.csv'
CSV_HERB_H   = r'C:\zhuang_taste_pre\psymix_herbal_infusions_20261122\human_observations.csv'
CSV_LIC_H    = r'C:\zhuang_taste_pre\psymix_herbal_infusions_20261122\licorice\human.csv'


def load_figs2_data():
    # a1: 15 plant milks FoodAtlas increment (product_line overall)
    fa = pd.read_csv(CSV_FA_INC)
    fa_sub = fa[(fa['protocol'] == 'product_line') & (fa['matrix'] == 'overall') &
                (fa['plus'] == 'label+identity+FA') & (fa['reference'] == 'label+identity')]
    et_row = fa_sub[fa_sub['model_id'] == 'ExtraTrees'].iloc[0]
    ri_row = fa_sub[fa_sub['model_id'] == 'Ridge'].iloc[0]
    ep_row = fa_sub[fa_sub['model_id'] == 'Encoded_pool_Ridge'].iloc[0]
    assert abs(float(et_row['delta_MAE']) - 0.336861) < 1e-4
    assert abs(float(ri_row['delta_MAE']) - 0.119468) < 1e-4

    # b1: Rose dry sugar/acid vs cup sweet/sour
    rc = pd.read_csv(CSV_ROSE_C)
    hh = pd.read_csv(CSV_HERB_H)
    rh = hh[hh['system'].str.contains('rose', case=False, na=False)]
    sug = rc[rc['source_compound'].isin(['Glucose', 'D-Fructose', 'Sucrose'])].groupby('sample_id')['mean_amount'].sum()
    acd = rc[rc['panel'] == 'organic acid panel'].groupby('sample_id')['mean_amount'].sum() / 1e6
    samples = ['VFD-RB', 'LTD-RC', 'VFD-RC', 'LTD-RP']
    sweet_obs = {r['sample_id']: float(r['observed']) for _, r in rh[rh['endpoint'] == 'sweetness'].iterrows()}
    sour_obs = {r['sample_id']: float(r['observed']) for _, r in rh[rh['endpoint'] == 'sourness'].iterrows()}
    rho_sw = float(spearmanr([sug[s] for s in samples], [sweet_obs[s] for s in samples]).statistic)
    rho_so = float(spearmanr([acd[s] for s in samples], [sour_obs[s] for s in samples]).statistic)
    assert abs(rho_sw - (-0.40)) < 0.01 and abs(rho_so - (-0.73786)) < 0.02

    # b2: Licorice whole root extract vs dose-matched glycyrrhizin
    lh = pd.read_csv(CSV_LIC_H)
    lic_sw_d = float(lh[(lh['sample_id'] == 'root_extract') & (lh['endpoint'] == 'sweetness')]['reported_delta'].iloc[0])
    lic_bt_d = float(lh[(lh['sample_id'] == 'root_extract') & (lh['endpoint'] == 'bitterness')]['reported_delta'].iloc[0])
    lic_lc_d = float(lh[(lh['sample_id'] == 'root_extract') & (lh['endpoint'] == 'licorice')]['reported_delta'].iloc[0])
    assert abs(lic_sw_d - (-0.7)) < 1e-3 and abs(lic_bt_d - 1.0) < 1e-3

    # c: Upward dose extrapolation (3%+5% -> 7% SE)
    up = pd.read_csv(CSV_UPWARD)
    up['ae'] = (up['observed'] - up['prediction']).abs()
    up_mae = up.groupby('model')['ae'].mean().to_dict()
    assert abs(up_mae['CARRY_FORWARD_SE5'] - 1.896909) < 1e-4
    assert abs(up_mae['RAW_ID_ET'] - 2.283924) < 1e-4

    return {
        'fa': [
            ('ExtraTrees', float(et_row['delta_MAE']), float(et_row['ci_low']), float(et_row['ci_high']), AMBER),
            ('Encoded-pool Ridge', float(ep_row['delta_MAE']), float(ep_row['ci_low']), float(ep_row['ci_high']), SLATE),
            ('Ridge', float(ri_row['delta_MAE']), float(ri_row['ci_low']), float(ri_row['ci_high']), BLUE),
        ],
        'rose': {
            'samples': samples,
            'sug': [float(sug[s]) for s in samples],
            'sweet': [sweet_obs[s] for s in samples],
            'acd': [float(acd[s]) for s in samples],
            'sour': [sour_obs[s] for s in samples],
            'rho_sw': rho_sw, 'rho_so': rho_so,
        },
        'licorice': [('Sweetness', lic_sw_d, BLUE), ('Licorice', lic_lc_d, SLATE), ('Bitterness', lic_bt_d, RED)],
        'upward': [
            ('Carry-fwd 5% obs', up_mae['CARRY_FORWARD_SE5'], CHARCOAL, True),
            ('RIDGE-conc',       up_mae['RAW_ID_RIDGE'],      SLATE,    False),
            ('TREE-conc',        up_mae['RAW_ID_ET'],         AMBER,    False),
            ('ADPT-pool',        up_mae['POOL_REG'],          PURPLE,   False),
            ('ADPT-dyn',         up_mae['PHYSSIM_INSPIRED_TASTE'], PURPLE, False),
            ('ADPT-set',         up_mae['DMC_SET_REG'],       PURPLE,   False),
        ],
    }


def draw_rdkit_vector(ax, smiles, cx, cy, box_w, box_h, highlight_oh_idx=None, hl_color=RED, bond_lw=0.85, fs=5.2):
    """Draw a crisp 100% vector 2D skeletal structure from RDKit 2D coordinates inside (cx, cy, box_w, box_h)."""
    mol = Chem.MolFromSmiles(smiles)
    rdDepictor.Compute2DCoords(mol)
    conf = mol.GetConformer()
    pts = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y] for i in range(mol.GetNumAtoms())])

    # Normalize & center cleanly inside target box
    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-6)
    scale = min(box_w / span[0], box_h / span[1]) * 0.88
    mid = (min_xy + max_xy) / 2.0
    xy = (pts - mid) * scale + np.array([cx, cy])

    # Identify highlighted atom(s) if requested (e.g., extra phenolic OH)
    hl_atoms = set()
    if highlight_oh_idx is not None:
        hl_atoms = set(highlight_oh_idx)

    # Draw bonds
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        p1, p2 = xy[i], xy[j]
        vec = p2 - p1
        dist = np.linalg.norm(vec)
        if dist < 1e-6:
            continue
        u = vec / dist
        perp = np.array([-u[1], u[0]])
        is_hl = (i in hl_atoms or j in hl_atoms)
        bcol = hl_color if is_hl else CHARCOAL
        blw = bond_lw * (1.45 if is_hl else 1.0)
        btype = bond.GetBondType()
        if btype == Chem.rdchem.BondType.DOUBLE:
            off = perp * (0.075 * scale)
            ax.plot([p1[0] + off[0], p2[0] + off[0]], [p1[1] + off[1], p2[1] + off[1]], color=bcol, lw=blw*0.85, zorder=3)
            ax.plot([p1[0] - off[0], p2[0] - off[0]], [p1[1] - off[1], p2[1] - off[1]], color=bcol, lw=blw*0.85, zorder=3)
        elif btype == Chem.rdchem.BondType.AROMATIC:
            off = perp * (0.065 * scale)
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=bcol, lw=blw, zorder=3)
            ax.plot([p1[0] + off[0], p2[0] + off[0]], [p1[1] + off[1], p2[1] + off[1]], color=bcol, lw=blw*0.6, ls='--', zorder=3)
        else:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=bcol, lw=blw, zorder=3)

    # Draw heteroatom labels with clean white mask halo
    for idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(idx)
        sym = atom.GetSymbol()
        if sym == 'C':
            continue
        n_h = atom.GetTotalNumHs()
        lbl = sym + ('' if n_h == 0 else ('H' if n_h == 1 else f'H{n_h}'))
        is_hl = (idx in hl_atoms)
        if is_hl:
            ax.add_patch(Circle((xy[idx, 0], xy[idx, 1]), 0.42 * scale, facecolor=RED_SOFT, edgecolor=hl_color, lw=0.7, zorder=4))
        tcol = hl_color if is_hl else (RED if sym == 'O' else BLUE)
        ax.text(xy[idx, 0], xy[idx, 1], lbl, ha='center', va='center',
                fontsize=fs * (1.12 if is_hl else 1.0),
                fontweight='bold' if is_hl else 'normal',
                color=tcol, zorder=5,
                bbox=dict(boxstyle='round,pad=0.05', facecolor=RED_SOFT if is_hl else WHITE, edgecolor='none', alpha=0.92))


def find_extra_phenolic_oh(smiles, core_pattern_smarts):
    mol = Chem.MolFromSmiles(smiles)
    matches = mol.GetSubstructMatches(Chem.MolFromSmarts(core_pattern_smarts))
    if matches:
        return [matches[0][-1]]
    return None


def build_figure_s2():
    data = load_figs2_data()
    W_mm, H_mm = 180.0, 165.0
    fig = plt.figure(figsize=(W_mm / 25.4, H_mm / 25.4), dpi=600, facecolor=WHITE)

    # ==========================================================================
    # PANEL a (top-left): Cross-database prior failure (Plant milks + FlavorDB)
    # ==========================================================================
    fig.text(0.018, 0.972, 'a', ha='left', va='center', fontsize=11.0, fontweight='bold', color=CHARCOAL)
    fig.text(0.044, 0.972, 'Cross-database category priors degrade or collapse predictions',
             ha='left', va='center', fontsize=7.6, fontweight='bold', color=CHARCOAL)

    # a-top: 15 commercial plant milks (FoodAtlas/FDC sugar vectors)
    ax_a1 = fig.add_axes([0.165, 0.745, 0.285, 0.175])
    ax_a1.axvline(0.0, color=CHARCOAL, lw=0.9, ls='--', zorder=2)
    ys = np.arange(len(data['fa']))[::-1]
    for i, (name, dm, lo, hi, col) in enumerate(data['fa']):
        y = ys[i]
        ax_a1.plot([lo, hi], [y, y], color=col, lw=1.45, solid_capstyle='round', zorder=3)
        ax_a1.scatter([dm], [y], s=28, color=col, edgecolors=WHITE, linewidths=0.45, zorder=4)
        ax_a1.text(0.84, y, f'+{dm:.3f}', ha='left', va='center', fontsize=5.8,
                   fontweight='bold' if i == 0 else 'normal', color=col if i == 0 else CHARCOAL)
    ax_a1.set_yticks(ys)
    ax_a1.set_yticklabels([x[0] for x in data['fa']], fontsize=6.0, fontweight='bold')
    ax_a1.set_xlim(-0.75, 1.25)
    ax_a1.set_ylim(-0.6, 2.6)
    ax_a1.set_xticks([-0.5, 0.0, 0.5])
    ax_a1.tick_params(axis='x', labelsize=5.8, length=2.6, pad=2)
    ax_a1.tick_params(axis='y', length=0, pad=3)
    ax_a1.set_xlabel('ΔMAE adding FoodAtlas sugars (15 plant milks; >0 worse)', fontsize=5.65, fontweight='bold', labelpad=2)
    ax_a1.set_title('Plant milks: category sugars vs. label+identity', fontsize=6.25, fontweight='bold', color=AMBER, loc='left', pad=4)
    ax_a1.spines['top'].set_visible(False)
    ax_a1.spines['right'].set_visible(False)
    ax_a1.spines['bottom'].set_bounds(-0.75, 0.78)
    ax_a1.grid(axis='x', color='#E2E8F0', lw=0.45, zorder=0)
    ax_a1.text(0.84, 2.48, 'ΔMAE', ha='left', va='bottom', fontsize=5.35, fontweight='bold', color=GREY_TEXT)

    # a-bottom: 1,335 FlavorDB/FlavorGraph bags (656 distinct inputs) top-1 pungent collapse
    ax_a2 = fig.add_axes([0.055, 0.545, 0.405, 0.115])
    cats = [('Pungent\n(77.3%, n=507)', 77.3, RED),
            ('Sweet\n(21.0%)', 21.0, BLUE),
            ('Other\n(1.7%)', 1.7, SLATE_LT)]
    left = 0.0
    for lbl, pct, col in cats:
        ax_a2.barh([0.5], [pct], left=left, height=0.44, color=col, edgecolor=WHITE, lw=0.8, zorder=3)
        if pct >= 18:
            ax_a2.text(left + pct/2.0, 0.5, lbl, ha='center', va='center',
                       fontsize=5.8, fontweight='bold', color=WHITE, linespacing=1.15, zorder=4)
        left += pct
    ax_a2.annotate('Bitter 1.1%, Sour 0.6%', xy=(99.1, 0.72), xytext=(86.5, 0.96),
                   ha='center', va='bottom', fontsize=5.3, color=GREY_TEXT,
                   arrowprops=dict(arrowstyle='->', color='#94A3B8', lw=0.65, shrinkA=0, shrinkB=1))
    ax_a2.text(0.0, 0.06, 'Examples ranked pungent #1: Beer (0.560), Milk (0.530), Coffee (0.675–0.759)',
               ha='left', va='center', fontsize=5.45, style='italic', color=RED)
    ax_a2.set_xlim(0, 100)
    ax_a2.set_ylim(-0.08, 1.15)
    ax_a2.axis('off')
    ax_a2.set_title('Frozen botanical set models on 1,335 FlavorDB/FlavorGraph food bags (656 distinct)',
                    fontsize=6.05, fontweight='bold', color=CHARCOAL, loc='left', pad=2)

    # ==========================================================================
    # PANEL b (top-right): Dry-petal vs brewed-cup & licorice extract disconnects
    # ==========================================================================
    fig.text(0.515, 0.972, 'b', ha='left', va='center', fontsize=11.0, fontweight='bold', color=CHARCOAL)
    fig.text(0.541, 0.972, 'Dry-tissue & whole-extract matrix disconnects in infusions',
             ha='left', va='center', fontsize=7.6, fontweight='bold', color=CHARCOAL)

    # b-left: Pingyin rose (dry petal sugars/acids vs brewed cup sweet/sour)
    ax_b1 = fig.add_axes([0.565, 0.755, 0.175, 0.165])
    rs = data['rose']
    ax_b1.scatter(rs['sug'], rs['sweet'], s=30, color=BLUE, edgecolors=WHITE, lw=0.5, zorder=4)
    # Trend line for visual orientation
    z = np.polyfit(rs['sug'], rs['sweet'], 1)
    xs_l = np.linspace(85, 215, 30)
    ax_b1.plot(xs_l, np.polyval(z, xs_l), color=BLUE, lw=0.95, ls='--', alpha=0.65, zorder=2)
    offsets_sw = {'VFD-RB': (5, -9), 'LTD-RC': (-2, 6), 'VFD-RC': (-16, 6), 'LTD-RP': (-18, 7)}
    for s, xv, yv in zip(rs['samples'], rs['sug'], rs['sweet']):
        dx, dy = offsets_sw[s]
        ax_b1.annotate(s, (xv, yv), xytext=(dx, dy), textcoords='offset points', fontsize=5.1, color=GREY_TEXT)
    ax_b1.text(0.05, 0.08, f"Spearman ρ = −{abs(rs['rho_sw']):.2f}", transform=ax_b1.transAxes,
               fontsize=5.7, fontweight='bold', color=RED,
               bbox=dict(boxstyle='round,pad=0.22', facecolor=RED_SOFT, edgecolor='none'))
    ax_b1.set_xlim(78, 222)
    ax_b1.set_ylim(3.95, 7.2)
    ax_b1.tick_params(labelsize=5.6, length=2.4, pad=2)
    ax_b1.set_xlabel('Dry-petal sugars (mg/g)', fontsize=5.65, fontweight='bold', labelpad=2)
    ax_b1.set_ylabel('Cup sweetness', fontsize=5.65, fontweight='bold', labelpad=2)
    ax_b1.set_title('Pingyin rose (n=4): sugars', fontsize=6.05, fontweight='bold', color=BLUE, loc='left', pad=4)
    ax_b1.spines['top'].set_visible(False)
    ax_b1.spines['right'].set_visible(False)
    ax_b1.grid(color='#E2E8F0', lw=0.4, zorder=0)

    # b-right-top: Rose organic acids vs cup sourness
    ax_b2 = fig.add_axes([0.795, 0.755, 0.175, 0.165])
    ax_b2.scatter(rs['acd'], rs['sour'], s=30, color=TEAL, edgecolors=WHITE, lw=0.5, zorder=4)
    z2 = np.polyfit(rs['acd'], rs['sour'], 1)
    xs_l2 = np.linspace(1.18, 2.20, 30)
    ax_b2.plot(xs_l2, np.polyval(z2, xs_l2), color=TEAL, lw=0.95, ls='--', alpha=0.65, zorder=2)
    offsets_so = {'LTD-RC': (-3, -12), 'LTD-RP': (5, 2), 'VFD-RB': (-18, 7), 'VFD-RC': (-12, 7)}
    for s, xv, yv in zip(rs['samples'], rs['acd'], rs['sour']):
        dx, dy = offsets_so[s]
        ax_b2.annotate(s, (xv, yv), xytext=(dx, dy), textcoords='offset points', fontsize=5.1, color=GREY_TEXT)
    ax_b2.text(0.05, 0.08, f"Spearman ρ = −{abs(rs['rho_so']):.2f}", transform=ax_b2.transAxes,
               fontsize=5.75, fontweight='bold', color=RED,
               bbox=dict(boxstyle='round,pad=0.22', facecolor=RED_SOFT, edgecolor='none'))
    ax_b2.set_xlim(1.12, 2.28)
    ax_b2.set_ylim(0.5, 5.8)
    ax_b2.tick_params(labelsize=5.6, length=2.4, pad=2)
    ax_b2.set_xlabel('Dry-petal acids (mg/g)', fontsize=5.65, fontweight='bold', labelpad=2)
    ax_b2.set_ylabel('Cup sourness', fontsize=5.65, fontweight='bold', labelpad=2)
    ax_b2.set_title('Pingyin rose (n=4): acids', fontsize=6.05, fontweight='bold', color=TEAL, loc='left', pad=4)
    ax_b2.spines['top'].set_visible(False)
    ax_b2.spines['right'].set_visible(False)
    ax_b2.grid(color='#E2E8F0', lw=0.4, zorder=0)

    # b-bottom: Whole licorice root extract vs dose-matched glycyrrhizin (0-5 scale)
    ax_b3 = fig.add_axes([0.625, 0.555, 0.345, 0.115])
    ax_b3.axvline(0.0, color=CHARCOAL, lw=0.85, zorder=2)
    ys_l = np.arange(len(data['licorice']))[::-1]
    for i, (attr, dv, col) in enumerate(data['licorice']):
        y = ys_l[i]
        ax_b3.barh([y], [dv], height=0.55, color=col, edgecolor='none', alpha=0.88, zorder=3)
        x_t = dv + (0.06 if dv >= 0 else -0.06)
        ha_t = 'left' if dv >= 0 else 'right'
        ax_b3.text(x_t, y, f'{dv:+.1f}', ha=ha_t, va='center', fontsize=5.85, fontweight='bold', color=col)
    ax_b3.set_yticks(ys_l)
    ax_b3.set_yticklabels([x[0] for x in data['licorice']], fontsize=5.9, fontweight='bold')
    ax_b3.set_xlim(-1.15, 1.45)
    ax_b3.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax_b3.tick_params(axis='x', labelsize=5.6, length=2.4, pad=2)
    ax_b3.tick_params(axis='y', length=0, pad=3)
    ax_b3.set_xlabel('Δ rating (0–5): whole licorice extract minus glycyrrhizin ref', fontsize=5.55, fontweight='bold', labelpad=2)
    ax_b3.set_title('Licorice matrix shift at matched glycyrrhizin dose (Schmid et al., 2021)',
                    fontsize=5.45, fontweight='bold', color=CHARCOAL, loc='right', pad=3)
    ax_b3.spines['top'].set_visible(False)
    ax_b3.spines['right'].set_visible(False)
    ax_b3.grid(axis='x', color='#E2E8F0', lw=0.45, zorder=0)

    # ==========================================================================
    # PANEL c (bottom-left): Upward dose extrapolation failure (3%+5% -> 7% SE)
    # ==========================================================================
    fig.text(0.018, 0.468, 'c', ha='left', va='center', fontsize=11.0, fontweight='bold', color=CHARCOAL)
    fig.text(0.044, 0.468, 'Upward dose extrapolation failure (3% + 5% → 7% SE, 55 pairs)',
             ha='left', va='center', fontsize=7.5, fontweight='bold', color=CHARCOAL)

    ax_c = fig.add_axes([0.165, 0.090, 0.295, 0.325])
    up_rows = data['upward']
    y_c = np.arange(len(up_rows))[::-1]
    ref_cf = up_rows[0][1]
    ax_c.axvline(ref_cf, color=CHARCOAL, lw=1.0, ls='--', zorder=2)
    ax_c.axvline(3.82, color='#E2E8F0', lw=0.7, zorder=1)
    ax_c.axhspan(-0.45, 2.45, xmin=0.0, xmax=3.82 / 4.55, color=PURPLE_SOFT, alpha=0.5, zorder=0)

    for i, (lbl, mae, col, is_ref) in enumerate(up_rows):
        y = y_c[i]
        if is_ref:
            ax_c.scatter([mae], [y], s=32, marker='s', color=CHARCOAL, edgecolors=WHITE, lw=0.4, zorder=5)
        else:
            ax_c.plot([ref_cf, mae], [y, y], color=col, lw=1.25, alpha=0.45, zorder=2)
            ax_c.scatter([mae], [y], s=30, color=col, edgecolors=WHITE, lw=0.45, zorder=5)
        ax_c.text(4.48, y, f'{mae:.3f}', ha='right', va='center', fontsize=5.85,
                  fontweight='bold' if (is_ref or lbl == 'TREE-conc') else 'normal',
                  color=col if not is_ref else CHARCOAL)

    ax_c.text(4.48, len(up_rows) - 0.35, 'MAE', ha='right', va='bottom', fontsize=5.4, fontweight='bold', color=GREY_TEXT)
    ax_c.annotate('No learner beats\ncopying 5% obs', xy=(ref_cf, 4.9), xytext=(2.42, 4.75),
                  fontsize=5.45, fontweight='bold', color=RED, ha='left', va='center',
                  bbox=dict(boxstyle='round,pad=0.22', facecolor=RED_SOFT, edgecolor=RED, lw=0.55),
                  arrowprops=dict(arrowstyle='->', color=RED, lw=0.75))

    ax_c.set_yticks(y_c)
    ax_c.set_yticklabels([x[0] for x in up_rows], fontsize=5.95, fontweight='bold')
    for tl in ax_c.get_yticklabels():
        if tl.get_text().startswith('ADPT-'):
            tl.set_color(PURPLE)
    ax_c.set_xlim(0.0, 4.55)
    ax_c.set_ylim(-0.6, len(up_rows) - 0.25)
    ax_c.set_xticks([0.0, 1.0, 2.0, 3.0])
    ax_c.tick_params(axis='x', labelsize=5.8, length=2.6, pad=2)
    ax_c.tick_params(axis='y', length=0, pad=3)
    ax_c.set_xlabel('← Better (7% SE out-of-range MAE, 0–15)', fontsize=5.8, fontweight='bold', labelpad=2.5)
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)
    ax_c.spines['bottom'].set_bounds(0.0, 3.82)
    ax_c.grid(axis='x', color='#E2E8F0', lw=0.45, zorder=0)
    fig.text(0.032, 0.025,
             'Trained on 3% and 5% SE pairs only; evaluated on unseen 7% SE (55 pairs). ADPT-dyn-off = 3.434 MAE.',
             ha='left', va='center', fontsize=5.35, color=GREY_TEXT)

    # ==========================================================================
    # PANEL d (bottom-right): RDKit 2D vector structure-activity cliffs
    # ==========================================================================
    fig.text(0.515, 0.468, 'd', ha='left', va='center', fontsize=11.0, fontweight='bold', color=CHARCOAL)
    fig.text(0.541, 0.468, 'Steep activity cliffs & in vitro vs. human-oral decoupling',
             ha='left', va='center', fontsize=7.5, fontweight='bold', color=CHARCOAL)

    ax_d = fig.add_axes([0.515, 0.045, 0.468, 0.398])
    ax_d.set_xlim(0.0, 100.0)
    ax_d.set_ylim(0.0, 82.0)
    ax_d.set_aspect('equal')
    ax_d.axis('off')

    # --- Top card: Rutin vs Myricetin-3-O-rutinoside (10,500x astringency cliff) ---
    ax_d.add_patch(FancyBboxPatch((0.5, 42.5), 99.0, 38.5, boxstyle='round,pad=0,rounding_size=2.2',
                                  facecolor=STRIPE_BG, edgecolor=GREY_LINE, lw=0.7, zorder=1))
    ax_d.text(3.0, 77.6, 'Astringency cliff: +1 –OH → 10,500× shift',
              ha='left', va='center', fontsize=5.95, fontweight='bold', color=RED, zorder=3)
    ax_d.text(97.0, 77.6, 'Chiral Tanimoto = 0.864', ha='right', va='center',
              fontsize=5.45, fontweight='bold', color=BLUE, zorder=3)

    # Draw full rutin vs myricetin-3-O-rutinoside
    smi_rut = 'C[C@@H]1O[C@@H](OC[C@H]2O[C@@H](Oc3c(-c4ccc(O)c(O)c4)oc4cc(O)cc(O)c4c3=O)[C@H](O)[C@@H](O)[C@@H]2O)[C@H](O)[C@H](O)[C@H]1O'
    smi_myr = 'C[C@@H]1O[C@@H](OC[C@H]2O[C@@H](Oc3c(-c4cc(O)c(O)c(O)c4)oc4cc(O)cc(O)c4c3=O)[C@H](O)[C@@H](O)[C@@H]2O)[C@H](O)[C@H](O)[C@H]1O'
    m_myr = Chem.MolFromSmiles(smi_myr)
    pyro_match = m_myr.GetSubstructMatch(Chem.MolFromSmarts('c1cc(O)c(O)c(O)c1'))
    hl_myr = [pyro_match[7]] if len(pyro_match) >= 8 else None

    ax_d.add_patch(FancyBboxPatch((2.5, 48.0), 42.0, 26.2, boxstyle='round,pad=0,rounding_size=1.5',
                                  facecolor=WHITE, edgecolor='#E2E8F0', lw=0.55, zorder=2))
    draw_rdkit_vector(ax_d, smi_rut, 23.5, 61.1, 39.5, 24.0, highlight_oh_idx=None, bond_lw=0.68, fs=4.15)
    ax_d.text(23.5, 45.2, 'Rutin (catechol B-ring)', ha='center', va='center', fontsize=5.4, fontweight='bold', color=CHARCOAL)

    ax_d.annotate('', xy=(54.5, 61.1), xytext=(45.5, 61.1),
                  arrowprops=dict(arrowstyle='-|>', color=RED, lw=1.2, mutation_scale=8))
    ax_d.text(50.0, 66.2, '+OH', ha='center', va='center', fontsize=5.6, fontweight='bold', color=RED)
    ax_d.text(50.0, 56.0, '10,500×', ha='center', va='center', fontsize=5.35, fontweight='bold', color=RED)

    ax_d.add_patch(FancyBboxPatch((55.5, 48.0), 42.0, 26.2, boxstyle='round,pad=0,rounding_size=1.5',
                                  facecolor=WHITE, edgecolor='#F2B8B5', lw=0.65, zorder=2))
    draw_rdkit_vector(ax_d, smi_myr, 76.5, 61.1, 39.5, 24.0, highlight_oh_idx=hl_myr, hl_color=RED, bond_lw=0.68, fs=4.15)
    ax_d.text(76.5, 45.2, 'Myricetin-3-O-rutinoside (pyrogallol)', ha='center', va='center', fontsize=5.3, fontweight='bold', color=RED)

    # --- Bottom card: Daidzein vs Genistein (in vitro TAS2R14/39 vs human oral inversion) ---
    ax_d.add_patch(FancyBboxPatch((0.5, 0.8), 99.0, 39.2, boxstyle='round,pad=0,rounding_size=2.2',
                                  facecolor=STRIPE_BG, edgecolor=GREY_LINE, lw=0.7, zorder=1))
    ax_d.text(3.0, 36.8, 'In vitro–oral decoupling (opposing shift)',
              ha='left', va='center', fontsize=5.95, fontweight='bold', color=AMBER, zorder=3)
    ax_d.text(97.0, 36.8, 'Chiral Tanimoto = 0.632', ha='right', va='center',
              fontsize=5.4, fontweight='bold', color=SLATE, zorder=3)

    smi_dai = 'O=c1c(-c2ccc(O)cc2)coc2cc(O)ccc12'
    smi_gen = 'O=c1c(-c2ccc(O)cc2)coc2cc(O)cc(O)c12'
    m_gen = Chem.MolFromSmiles(smi_gen)
    # Explicitly locate the 5-OH oxygen atom (second OH on the A-ring adjacent to carbonyl C=O)
    hl_gen = None
    for atom in m_gen.GetAtoms():
        if atom.GetSymbol() == 'O' and atom.GetTotalNumHs() == 1:
            nbr_c = atom.GetNeighbors()[0]
            # Check if this phenolic C is 2 bonds away from C=O carbon
            for c2 in nbr_c.GetNeighbors():
                if c2.GetSymbol() == 'C' and any(b.GetBondType() == Chem.rdchem.BondType.DOUBLE and b.GetOtherAtom(c2).GetSymbol() == 'O' for b in c2.GetBonds()):
                    pass
                for c3 in c2.GetNeighbors():
                    if c3.GetSymbol() == 'C' and any(b.GetBondType() == Chem.rdchem.BondType.DOUBLE and b.GetOtherAtom(c3).GetSymbol() == 'O' for b in c3.GetBonds()):
                        hl_gen = [atom.GetIdx()]
                        break

    ax_d.add_patch(FancyBboxPatch((2.5, 11.2), 34.0, 22.0, boxstyle='round,pad=0,rounding_size=1.5',
                                  facecolor=WHITE, edgecolor='#E2E8F0', lw=0.55, zorder=2))
    draw_rdkit_vector(ax_d, smi_dai, 19.5, 22.2, 31.0, 19.5, highlight_oh_idx=None, bond_lw=0.82, fs=5.0)
    ax_d.text(19.5, 8.6, 'Daidzein', ha='center', va='center', fontsize=5.5, fontweight='bold', color=CHARCOAL)

    ax_d.annotate('', xy=(44.2, 22.2), xytext=(37.4, 22.2),
                  arrowprops=dict(arrowstyle='-|>', color=AMBER, lw=1.15, mutation_scale=8))
    ax_d.text(40.8, 26.8, '+5-OH', ha='center', va='center', fontsize=5.15, fontweight='bold', color=AMBER)

    ax_d.add_patch(FancyBboxPatch((44.9, 11.2), 34.0, 22.0, boxstyle='round,pad=0,rounding_size=1.5',
                                  facecolor=WHITE, edgecolor='#F5CBA7', lw=0.65, zorder=2))
    draw_rdkit_vector(ax_d, smi_gen, 61.9, 22.2, 31.0, 19.5, highlight_oh_idx=hl_gen, hl_color=AMBER, bond_lw=0.82, fs=5.0)
    ax_d.text(61.9, 8.6, 'Genistein', ha='center', va='center', fontsize=5.5, fontweight='bold', color=AMBER)

    # Compact opposing direction box on the right of bottom card (ample bottom margin!)
    ax_d.add_patch(FancyBboxPatch((80.2, 8.8), 18.0, 24.4, boxstyle='round,pad=0,rounding_size=1.5',
                                  facecolor=WHITE, edgecolor=GREY_LINE, lw=0.6, zorder=2))
    ax_d.text(89.2, 29.2, 'In vitro\nTAS2R14/39', ha='center', va='center', fontsize=4.75, fontweight='bold', color=TEAL, linespacing=1.12)
    ax_d.text(89.2, 23.6, '500 → 4–8 µM\n(↑ potent)', ha='center', va='center', fontsize=4.6, color=TEAL, linespacing=1.12)
    ax_d.plot([81.8, 96.6], [20.2, 20.2], color=GREY_LINE, lw=0.55)
    ax_d.text(89.2, 16.8, 'Human oral\nthreshold', ha='center', va='center', fontsize=4.75, fontweight='bold', color=RED, linespacing=1.12)
    ax_d.text(89.2, 11.6, '2,921 → 4,006\nµM (↓ potent)', ha='center', va='center', fontsize=4.5, color=RED, linespacing=1.12)

    ax_d.text(50.0, 3.6, 'Opposing receptor vs. oral shifts preclude direct in vitro EC50 → human intensity transfer',
              ha='center', va='center', fontsize=4.9, style='italic', color=GREY_TEXT)

    pdf_p = os.path.join(OUT_DIR, 'PsyMix_FigureS2.pdf')
    svg_p = os.path.join(OUT_DIR, 'PsyMix_FigureS2.svg')
    png_p = os.path.join(OUT_DIR, 'PsyMix_FigureS2_600dpi.png')
    fig.savefig(pdf_p, dpi=600)
    fig.savefig(svg_p, dpi=600)
    fig.savefig(png_p, dpi=600)
    plt.close(fig)
    print(f'[OK] Saved Figure S2 -> {pdf_p}, {svg_p}, {png_p}')

    # Quick PyMuPDF audit + 1200 dpi zooms
    doc = fitz.open(pdf_p)
    page = doc[0]
    rect = page.rect
    fonts = sorted(set(f[3] for f in page.get_fonts(full=True)))
    ftypes = sorted(set(f[2] for f in page.get_fonts(full=True)))
    td = page.get_text('dict')
    clipped, tags, bad = [], [], []
    all_t = []
    for b in td['blocks']:
        if b.get('type') != 0: continue
        for l in b.get('lines', []):
            for sp in l.get('spans', []):
                t = sp['text'].strip()
                if not t: continue
                all_t.append(t)
                x0, y0, x1, y1 = sp['bbox']
                if x0 < -0.5 or y0 < -0.5 or x1 > rect.width + 0.5 or y1 > rect.height + 0.5:
                    clipped.append((t, (round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1))))
                if sp['size'] >= 10.0 and t in ('a', 'b', 'c', 'd', 'A', 'B', 'C', 'D'):
                    tags.append(t)
                for ch in t:
                    if ch in ('\ufffd', '□', 'ㅓ', '┤'):
                        bad.append(ch)
    full_t = ' '.join(all_t)
    print(f'PDF Audit FigS2: {rect.width*25.4/72:.2f} x {rect.height*25.4/72:.2f} mm | Fonts: {ftypes} ({len(fonts)}) | Tags: {tags} | Clipped: {len(clipped)} | Bad: {len(bad)}')
    assert 'Type3' not in ftypes and len(clipped) == 0 and len(bad) == 0 and tags == ['a', 'b', 'c', 'd'], \
        f'Audit failed: ftypes={ftypes}, clipped={clipped}, bad={bad}, tags={tags}'

    mat = fitz.Matrix(1200 / 72.0, 1200 / 72.0)
    page.get_pixmap(matrix=mat, clip=fitz.Rect(0, 0, rect.width, rect.height * 0.52), alpha=False).save(
        os.path.join(OUT_DIR, 'zoom_1200dpi_FigS2_top_ab.png'))
    page.get_pixmap(matrix=mat, clip=fitz.Rect(0, rect.height * 0.48, rect.width, rect.height), alpha=False).save(
        os.path.join(OUT_DIR, 'zoom_1200dpi_FigS2_bot_cd.png'))
    doc.close()
    print('[PASS] PsyMix_FigureS2 passed ALL automated PDF checks!')


if __name__ == '__main__':
    build_figure_s2()
