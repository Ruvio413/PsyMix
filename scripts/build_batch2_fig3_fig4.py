# -*- coding: utf-8 -*-
"""
PsyMix Batch 2 Publication Figures — Figure 3 (a-d) & Figure 4 (a-d)
  - PsyMix Figure 3 (a-d): 180.0 mm x 202.0 mm (Task T2: Fruits, potency loads, collinearity & Dirichlet design)
  - PsyMix Figure 4 (a-d): 180.0 mm x 206.0 mm (Tasks T3 & T4: Controlled binary & ternary sweetener mixtures)

Strict compliance with VISUAL_SPEC.md (Batch 1 accepted standard) & User's 6 Hard Rules:
  1. Global continuity: Colorblind-safe palette (#0072B2, #1B7F6B, #D55E00/#E69F00, #6A51A3, #9BB5CE, #2C3E50),
     strict bold lowercase a/b/c/d, pdf.fonttype=42, ps.fonttype=42, svg.fonttype='none', mathtext.fontset='dejavusans',
     asymmetric top rows (Fig. 3 ~[1.35, 0.95], Fig. 4 ~[0.82, 1.38]), zero overlap/occlusion/truncated bars.
  2. Fig. 3a: 5 fruits x 2 rows (Sweet/Sour) sharing 1 watercolor specimen per fruit (5 icons total); hollow circle
     ○ NULL-mean (% vs TREE-pres) on %ΔRMSE axis; genuine archived 95% CIs only; aligned right native RMSE table.
  3. Fig. 3b: Unified %ΔRMSE vs SUM-mass X-axis (zero cross-scale mixing) + Apple sourness 1-acid log-curvature
     control row + clean right aligned Native Δ [95% CI] column; explicit volumetric U (Grape) vs proxy L distinction.
  4. Fig. 3c-d: Explicit Vol. load U (Grape) vs Proxy load L (Apple/Blueberry/Tomato) Y-axes + acid collinearity footer
     r = 1.000 (VIF > 1.6×10^5); 3d top warning band r >= 0.98 (VIF >= 25) + bottom 13.3-28.9 pp band at r = 0.90.
  5. Fig. 4a-b: Anti-hairball 1-fold network (44 train edges lw=0.6 alpha=0.25, 11 test edges lw=1.5 solid #D55E00;
     3 extended complex sweeteners as dashed outer nodes without extra edges; log10 potency node scaling; horizontal
     labels). 4b locked to [-0.36, +0.88] with clean right-edge overflow arrows (zero '//' broken axis), separating
     fixed-input model gain (TREE-conc - RIDGE-conc) from dual-endpoint inversion (TREE-conc - EQS) and EQSxID gain.
  6. Fig. 4c-d: 4c left exact 3% vs 7% r = 0.745 + all dose pairs r = 0.745-0.875 + Pair constant 86.5% vs TREE-conc
     8.1%; 4c right strictly split into 2 vertical sub-axes (upper: Method II residual vs 2D Tanimoto, n=136, ρ=0.002;
     lower: dedicated MAE axis for local neighbor borrowing 1.074 -> 0.921, Δ=-0.154 [-0.283, -0.014]); 4d clear
     SELF-mean (pure-ref) vs NULL-mean (binary train) + TER-direct - TER-graph = -0.022 [-0.128, 0.081].
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from PIL import Image
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, Polygon, FancyArrowPatch
from matplotlib.lines import Line2D

# Strict VISUAL_SPEC.md §1.1 configuration (identical to Batch 1 accepted scripts)
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "svg.hashsalt": "PsyMix-Batch2-Final",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 7.6,
    "axes.labelsize": 8.0,
    "axes.titlesize": 8.4,
    "xtick.labelsize": 7.2,
    "ytick.labelsize": 7.3,
    "legend.fontsize": 6.8,
    "axes.unicode_minus": True,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

OUT_DIR = Path(r"c:\gemi-antigravity\00_PsyMix_Submission_Master\02_Figures_Design_and_Drafts\final_figures\Batch2_Fig3_Fig4")
SPEC_DIR = Path(r"c:\gemi-antigravity\00_PsyMix_Submission_Master\02_Figures_Design_and_Drafts\specimens")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Unified Colorblind-Safe Palette (aligned 100% with Figures 1-2)
CHARCOAL = "#2C3E50"
INK = "#1E293B"
SUB = "#4A5568"
MUTED = "#718096"
LINE = "#CBD5E0"
HAIR = "#E6ECF2"

BLUE = "#0072B2"
BLUE_LT = "#E8F3FA"
TEAL = "#1B7F6B"
TEAL_LT = "#DDF3EC"
AMBER = "#D55E00"
GOLD = "#E69F00"
AMBER_LT = "#FEF2E8"
PURPLE = "#6A51A3"
PURPLE_LT = "#F2EFFA"
SLATE_BLUE = "#9BB5CE"
SLATE_BG = "#EDF3F8"
CRIMSON = "#C0392B"
CRIMSON_LT = "#FDECEC"
GREY = "#8C98A4"
STRIPE_BG = "#F8FAFC"


def load_cropped_rgba(path, pad=4):
    im = Image.open(path).convert("RGBA")
    arr = np.array(im)
    non_bg = (arr[:, :, 3] > 20) & ((arr[:, :, 0] < 248) | (arr[:, :, 1] < 248) | (arr[:, :, 2] < 248))
    if non_bg.any():
        ys, xs = np.where(non_bg)
        y0, y1 = max(0, ys.min() - pad), min(arr.shape[0], ys.max() + pad)
        x0, x1 = max(0, xs.min() - pad), min(arr.shape[1], xs.max() + pad)
        im = im.crop((x0, y0, x1, y1))
    return im


def place_image_mm(fig, W, H, img, x_mm, y_mm, w_mm, h_mm, gid=None, alpha=1.0, zorder=5):
    """Place an RGBA image preserving aspect ratio centered inside (x_mm, y_mm, w_mm, h_mm) [y downward]."""
    iw, ih = img.size
    box_aspect = w_mm / h_mm
    img_aspect = iw / ih
    if img_aspect > box_aspect:
        rw = w_mm
        rh = w_mm / img_aspect
    else:
        rh = h_mm
        rw = h_mm * img_aspect
    rx = x_mm + (w_mm - rw) / 2.0
    ry = y_mm + (h_mm - rh) / 2.0
    ax_im = fig.add_axes([rx / W, 1.0 - (ry + rh) / H, rw / W, rh / H], zorder=zorder)
    ax_im.imshow(img, interpolation="lanczos", alpha=alpha)
    ax_im.set_axis_off()
    ax_im.patch.set_alpha(0.0)
    if gid:
        ax_im.set_gid(gid)
    return ax_im


def add_rbox(ax, x, y, w, h, fc="white", ec=LINE, lw=0.7, r=1.5, ls="-", alpha=1.0, zorder=2):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, alpha=alpha, zorder=zorder
    )
    ax.add_patch(patch)
    return patch


def style_clean_axes(ax, left_lw=0.65, bottom_lw=0.65):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(left_lw)
    ax.spines["bottom"].set_linewidth(bottom_lw)
    ax.spines["left"].set_color(CHARCOAL)
    ax.spines["bottom"].set_color(CHARCOAL)
    ax.tick_params(colors=INK, width=0.65, length=2.6, pad=2.2)


# =========================================================================
# FIGURE 3: 180.0 mm x 202.0 mm (Task T2 — Fruit Cohorts & Identifiability)
# =========================================================================
def build_figure3():
    W, H = 180.0, 202.0
    fig = plt.figure(figsize=(W / 25.4, H / 25.4), dpi=600)

    ax_bg = fig.add_axes([0, 0, 1, 1], xlim=(0, W), ylim=(H, 0), zorder=0)
    ax_bg.set_axis_off()
    ax_bg.set_gid("Canvas_Background")
    ax_bg.add_patch(Rectangle((0, 0), W, H, facecolor="white", edgecolor="none", zorder=0))
    # Horizontal tier divider + subtle top vertical separator
    ax_bg.plot([4.0, W - 4.0], [108.5, 108.5], color=HAIR, lw=0.85, zorder=1)
    ax_bg.plot([103.5, 103.5], [4.5, 105.5], color=HAIR, lw=0.75, zorder=1)
    ax_bg.plot([93.0, 93.0], [111.5, 198.5], color=HAIR, lw=0.75, zorder=1)

    ax_hdr = fig.add_axes([0, 0, 1, 1], xlim=(0, W), ylim=(H, 0), zorder=10)
    ax_hdr.set_axis_off()
    ax_hdr.patch.set_alpha(0.0)
    ax_hdr.set_gid("Panel_Headers")

    # ---------------------------------------------------------------------
    # PANEL a (Left top, wider: x = 4.0 .. 101.5 mm, ~1.35 ratio)
    # ---------------------------------------------------------------------
    ax_hdr.text(4.0, 6.6, "a", fontsize=11.0, fontweight="bold", color=CHARCOAL, ha="left", va="center")
    ax_hdr.text(9.0, 6.6, "Analytical signal gain over presence & NULL-mean (T2)", fontsize=8.4, fontweight="bold", color=CHARCOAL, ha="left", va="center")

    f3a = pd.read_csv(r"C:\zhuang_taste_pre\psymix_manuscript_release\Figures_round1_20261124\source_data\Figure3A.csv")
    n3a = pd.read_csv(r"C:\zhuang_taste_pre\psymix_A5_null_baseline_audit_20261124\PsyMix_T2_Null_Baseline_Comparisons.csv")

    # Lookup NULL-mean RMSE and whether TREE-conc significantly beats NULL-mean (delta_rmse_hi < 0)
    null_rmse_map = {}
    beats_null_map = {}
    for _, r in n3a.iterrows():
        key = (r["crop"], r["endpoint"])
        if r["model"] == "NULL-mean" and r["reference"] == "NULL-mean":
            null_rmse_map[key] = float(r["rmse"])
        if r["model"] == "ExtraTrees-quantity" and r["reference"] == "NULL-mean":
            beats_null_map[key] = float(r["delta_rmse_hi"]) < 0.0

    # Assert exact counts: 5 beat both, 8 beat TREE-pres
    assert sum(beats_null_map.values()) == 5, beats_null_map
    assert (f3a["hi"] < 0).sum() == 8

    # Grouped 5 fruits x 2 rows (Sweet, Sour) with generous inter-group spacing
    fruit_order = [
        ("grape", "Grape", "n=41", "grape.png"),
        ("tomato", "Tomato", "n=209", "tomato.png"),
        ("blueberry", "Blueberry", "n=244", "blueberry.png"),
        ("apple", "Apple", "n=40", "apple.png"),
        ("strawberry", "Strawberry*", "n=207", "strawberry.png"),
    ]
    # y positions in data coordinates (top=0 downward)
    y_rows = []
    group_centers = []
    cur_y = 0.0
    for fi, (crop, disp, ntxt, img_fn) in enumerate(fruit_order):
        y_sw = cur_y
        y_so = cur_y + 0.92
        group_centers.append((fi, crop, disp, ntxt, img_fn, 0.5 * (y_sw + y_so), y_sw - 0.44, y_so + 0.44))
        for ep, yp in [("sweetness", y_sw), ("sourness", y_so)]:
            row = f3a[(f3a["crop"] == crop) & (f3a["taste"] == ep)].iloc[0]
            pres = float(row["baseline"])
            conc = float(row["candidate"])
            null_r = null_rmse_map[(crop, ep)]
            pct = float(row["delta"])
            lo = float(row["lo"])
            hi = float(row["hi"])
            null_pct = 100.0 * (null_r - pres) / pres
            b_null = beats_null_map[(crop, ep)]
            b_pres = hi < 0.0
            y_rows.append(dict(
                crop=crop, ep="Sweet" if ep == "sweetness" else "Sour",
                yp=yp, pres=pres, conc=conc, null_r=null_r,
                pct=pct, lo=lo, hi=hi, null_pct=null_pct,
                beats_null=b_null, beats_pres=b_pres
            ))
        cur_y += 2.28

    # Plot axes for 3a: shifted right to x = 35.5 mm so fruit names & Sweet/Sour have 3.5 mm clean air!
    x_3a_L, x_3a_W = 35.5, 38.2
    y_3a_T, y_3a_H = 16.5, 67.0
    ax_3a = fig.add_axes([x_3a_L / W, 1.0 - (y_3a_T + y_3a_H) / H, x_3a_W / W, y_3a_H / H], zorder=3)
    ax_3a.set_gid("Panel_3a_Forest")

    y_min_a, y_max_a = -0.58, cur_y - 0.86
    ax_3a.set_ylim(y_max_a, y_min_a)
    ax_3a.set_xlim(-67.0, 9.0)

    # Helper: map data y in ax_3a to figure mm y
    def ya_to_mm(yd):
        return y_3a_T + (yd - y_min_a) / (y_max_a - y_min_a) * y_3a_H

    # Alternating group stripes spanning fruit icons -> plot -> right RMSE columns!
    for fi, crop, disp, ntxt, img_fn, yc, y0, y1 in group_centers:
        y0_mm, y1_mm = ya_to_mm(y0), ya_to_mm(y1)
        if fi % 2 == 0:
            ax_bg.add_patch(Rectangle((4.0, y0_mm), 97.8, y1_mm - y0_mm, facecolor=STRIPE_BG, edgecolor="none", zorder=1))
        # Place ONE watercolor specimen per fruit centered vertically on its 2-row group!
        im_fruit = load_cropped_rgba(SPEC_DIR / img_fn)
        yc_mm = ya_to_mm(yc)
        place_image_mm(fig, W, H, im_fruit, 4.1, yc_mm - 4.3, 7.2, 8.6, gid=f"Specimen_3a_{crop}", zorder=6)
        # Fruit name + sample size cleanly beside specimen (zero touch with Sweet/Sour ticks!)
        ax_hdr.text(11.9, yc_mm - 1.5, disp, fontsize=6.75, fontweight="bold", color=INK, ha="left", va="center")
        ax_hdr.text(11.9, yc_mm + 2.1, f"({ntxt})", fontsize=6.25, color=MUTED, ha="left", va="center")

    # Right aligned native RMSE numeric table header (aligned with Fig. 2a style)
    col_pres_x, col_conc_x, col_null_x = 83.0, 91.9, 100.8
    hdr_y_mm = 13.6
    ax_hdr.text(col_pres_x, hdr_y_mm, "Pres", fontsize=6.75, fontweight="bold", color=SUB, ha="right", va="center")
    ax_hdr.text(col_conc_x, hdr_y_mm, "Conc", fontsize=6.75, fontweight="bold", color=BLUE, ha="right", va="center")
    ax_hdr.text(col_null_x, hdr_y_mm, "NULL", fontsize=6.75, fontweight="bold", color=MUTED, ha="right", va="center")
    ax_hdr.text(91.9, 10.6, "Native RMSE", fontsize=6.7, fontweight="bold", color=CHARCOAL, ha="center", va="center")
    ax_bg.plot([75.6, 101.4], [15.3, 15.3], color=LINE, lw=0.6, zorder=2)

    ax_3a.axvline(0.0, color=CHARCOAL, lw=0.85, ls="--", zorder=2)
    ax_3a.grid(axis="x", color=HAIR, lw=0.5, zorder=0)

    for rd in y_rows:
        yp = rd["yp"]
        if rd["beats_null"]:
            col, mk, ms = BLUE, "o", 4.8
        elif rd["beats_pres"]:
            col, mk, ms = TEAL, "D", 4.3
        else:
            col, mk, ms = MUTED, "s", 4.3

        # Genuine archived 95% CI from Figure3A.csv
        ax_3a.plot([rd["lo"], rd["hi"]], [yp, yp], color=col, lw=1.45, solid_capstyle="round", zorder=3)
        ax_3a.plot(rd["pct"], yp, marker=mk, ms=ms, color=col, markeredgecolor="white", markeredgewidth=0.45, zorder=5)

        # Hollow circle ○ NULL-mean (% vs TREE-pres) on the SAME %ΔRMSE axis!
        ax_3a.plot(
            rd["null_pct"], yp, marker="o", ms=4.1,
            markerfacecolor="white", markeredgecolor=CHARCOAL, markeredgewidth=1.05, zorder=4
        )

        # Right-aligned native RMSE numbers (2 decimals cleanly tabular)
        y_mm = ya_to_mm(yp)
        ax_hdr.text(col_pres_x, y_mm, f"{rd['pres']:.2f}", fontsize=6.7, color=SUB, ha="right", va="center")
        ax_hdr.text(col_conc_x, y_mm, f"{rd['conc']:.2f}", fontsize=6.7, fontweight="bold", color=col, ha="right", va="center")
        ax_hdr.text(col_null_x, y_mm, f"{rd['null_r']:.2f}", fontsize=6.7, color=MUTED, ha="right", va="center")

    ax_3a.set_yticks([r["yp"] for r in y_rows])
    ax_3a.set_yticklabels([r["ep"] for r in y_rows], fontsize=6.8, color=INK)
    ax_3a.set_xticks([-60, -40, -20, 0])
    ax_3a.set_xticklabels(["−60%", "−40%", "−20%", "0%"], fontsize=6.9)
    ax_3a.set_xlabel("%ΔRMSE vs TREE-pres (95% group-bootstrap CI)", fontsize=7.3, fontweight="bold", color=INK, labelpad=2.5)
    style_clean_axes(ax_3a)
    ax_3a.patch.set_alpha(0.0)

    # Compact legend cleanly below Panel a X-axis (zero data occlusion!)
    leg_a_handles = [
        Line2D([0], [0], marker="o", color=BLUE, lw=1.3, ms=4.6, markeredgecolor="white", markeredgewidth=0.4, label="Beats TREE-pres & NULL (5/10)"),
        Line2D([0], [0], marker="D", color=TEAL, lw=1.3, ms=4.2, markeredgecolor="white", markeredgewidth=0.4, label="Beats TREE-pres only (3/10)"),
        Line2D([0], [0], marker="s", color=MUTED, lw=1.3, ms=4.2, markeredgecolor="white", markeredgewidth=0.4, label="95% CI spans 0 (2/10)"),
        Line2D([0], [0], marker="o", color="none", ms=4.2, markerfacecolor="white", markeredgecolor=CHARCOAL, markeredgewidth=1.05, label="NULL-mean (% vs TREE-pres)"),
    ]
    ax_3a.legend(
        handles=leg_a_handles, loc="upper center", bbox_to_anchor=(0.48, -0.155),
        ncol=2, frameon=False, fontsize=6.35, columnspacing=1.15, handletextpad=0.62, handlelength=1.45, labelspacing=0.34
    )
    ax_hdr.text(4.2, 105.0, "*Strawberry: 87 relative GC–MS volatile peaks (no °Brix/pH/TA).", fontsize=6.1, color=MUTED, ha="left", va="center")

    # ---------------------------------------------------------------------
    # PANEL b (Right top, narrower: x = 105.5 .. 176.0 mm, ~0.95 ratio)
    # ---------------------------------------------------------------------
    ax_hdr.text(105.5, 6.6, "b", fontsize=11.0, fontweight="bold", color=CHARCOAL, ha="left", va="center")
    ax_hdr.text(110.5, 6.6, "Potency load vs SUM-mass (same panel)", fontsize=8.4, fontweight="bold", color=CHARCOAL, ha="left", va="center")

    p3b = pd.read_csv(r"C:\zhuang_taste_pre\psymix_fruit_validation_20261117\within\paired.csv")
    m3b = pd.read_csv(r"C:\zhuang_taste_pre\psymix_fruit_validation_20261117\within\metrics.csv")

    b_specs = [
        ("grape", "sweetness", "Grape sweet", "vol. U (n=41)", 0.0, "vol"),
        ("blueberry", "sweetness", "Blueb. sweet", "proxy L (n=219)", 1.12, "proxy"),
        ("tomato", "sweetness", "Tomato sweet", "proxy L (n=209)", 2.24, "proxy"),
        ("apple", "sweetness", "Apple sweet", "proxy L (n=40)", 3.36, "proxy"),
        ("apple", "sourness", "Apple sour", "1 acid: malic\n(lin = SUM)", 4.92, "ctrl"),
    ]

    # Plot axes for 3b: x = 124.2 .. 154.0 mm (29.8 mm), right native text column x = 155.2 .. 176.2 mm (21.0 mm)
    x_3b_L, x_3b_W = 124.2, 29.8
    y_3b_T, y_3b_H = 16.5, 67.0
    ax_3b = fig.add_axes([x_3b_L / W, 1.0 - (y_3b_T + y_3b_H) / H, x_3b_W / W, y_3b_H / H], zorder=3)
    ax_3b.set_gid("Panel_3b_PotencyLoad")

    y_min_b, y_max_b = -0.68, 5.56
    ax_3b.set_ylim(y_max_b, y_min_b)
    ax_3b.set_xlim(-12.2, 7.2)

    def yb_to_mm(yd):
        return y_3b_T + (yd - y_min_b) / (y_max_b - y_min_b) * y_3b_H

    # Group shading: volumetric U top band, proxy L middle, curvature control bottom band
    y_vol0, y_vol1 = yb_to_mm(-0.52), yb_to_mm(0.54)
    ax_bg.add_patch(Rectangle((105.5, y_vol0), 71.0, y_vol1 - y_vol0, facecolor=BLUE_LT, alpha=0.50, edgecolor="none", zorder=1))
    y_ctl0, y_ctl1 = yb_to_mm(4.32), yb_to_mm(5.48)
    ax_bg.add_patch(Rectangle((105.5, y_ctl0), 71.0, y_ctl1 - y_ctl0, facecolor=PURPLE_LT, alpha=0.65, edgecolor="none", zorder=1))
    ax_3b.axhline(4.14, color=LINE, lw=0.75, ls="--", zorder=2)

    # Headers for 3b (explicit Native ΔRMSE to distinguish clearly from manuscript MAE!)
    ax_hdr.text(139.1, 13.6, "Named-sugar panel", fontsize=6.7, fontweight="bold", color=CHARCOAL, ha="center", va="center")
    ax_hdr.text(176.0, 10.6, "Native ΔRMSE (DOT-log*)", fontsize=6.45, fontweight="bold", color=CHARCOAL, ha="right", va="center")
    ax_hdr.text(176.0, 13.6, "point [95% CI]", fontsize=6.25, color=MUTED, ha="right", va="center")
    ax_bg.plot([153.5, 176.2], [15.3, 15.3], color=LINE, lw=0.6, zorder=2)

    ax_3b.axvline(0.0, color=CHARCOAL, lw=0.85, ls="--", zorder=2)
    ax_3b.grid(axis="x", color=HAIR, lw=0.5, zorder=0)

    for crop, ep, lbl1, lbl2, yp, kind in b_specs:
        m0 = float(m3b[(m3b["crop"] == crop) & (m3b["endpoint"] == ep) & (m3b["model"] == "M0_sum")]["rmse"].iloc[0])
        r1 = p3b[(p3b["crop"] == crop) & (p3b["endpoint"] == ep) & (p3b["baseline"] == "M0_sum") & (p3b["candidate"] == "M1_correct") & (p3b["metric"] == "rmse")].iloc[0]
        r2 = p3b[(p3b["crop"] == crop) & (p3b["endpoint"] == ep) & (p3b["baseline"] == "M0_sum") & (p3b["candidate"] == "M2_correct") & (p3b["metric"] == "rmse")].iloc[0]

        # Note: in paired.csv, gain = RMSE(M0_sum) - RMSE(candidate), so Δ = RMSE(cand) - RMSE(M0) = -gain, CI = [-hi, -lo]
        d1, lo1, hi1 = -float(r1["gain"]), -float(r1["hi"]), -float(r1["lo"])
        d2, lo2, hi2 = -float(r2["gain"]), -float(r2["hi"]), -float(r2["lo"])
        p1, plo1, phi1 = 100.0 * d1 / m0, 100.0 * lo1 / m0, 100.0 * hi1 / m0
        p2, plo2, phi2 = 100.0 * d2 / m0, 100.0 * lo2 / m0, 100.0 * hi2 / m0

        if crop == "grape" and ep == "sweetness":
            assert abs(d2 - (-0.0362)) < 0.001
        if crop == "apple" and ep == "sourness":
            assert abs(d2 - (-0.5051)) < 0.001

        # DOT-lin* (upper slight offset) & DOT-log* (lower slight offset)
        # Note: for Apple sour (1 acid: malic), linear weighted load == SUM-mass identically (0%), so omit redundant grey marker!
        c_lin = GOLD
        c_log = BLUE if kind == "vol" else (TEAL if kind == "proxy" else PURPLE)

        if kind != "ctrl":
            if abs(phi1 - plo1) > 1e-6:
                ax_3b.plot([plo1, phi1], [yp - 0.16, yp - 0.16], color=c_lin, lw=1.35, solid_capstyle="round", zorder=3)
            ax_3b.plot(p1, yp - 0.16, marker="s", ms=4.0, color=c_lin, markeredgecolor="white", markeredgewidth=0.4, zorder=4)
            y_log = yp + 0.16
        else:
            y_log = yp + 0.14

        # Plot DOT-log*
        ax_3b.plot([plo2, phi2], [y_log, y_log], color=c_log, lw=1.45, solid_capstyle="round", zorder=3)
        ax_3b.plot(p2, y_log, marker="o", ms=4.5, color=c_log, markeredgecolor="white", markeredgewidth=0.45, zorder=5)

        # Left row label cleanly outside spine (wrapped 2-line sub-label for ctrl stays 5 mm inside divider!)
        y_mm = yb_to_mm(yp)
        if kind == "ctrl":
            ax_hdr.text(122.8, y_mm - 2.3, lbl1, fontsize=6.8, fontweight="bold", color=PURPLE, ha="right", va="center")
            ax_hdr.text(122.8, y_mm + 2.1, lbl2, fontsize=5.8, color=MUTED, ha="right", va="center", linespacing=1.18)
        else:
            ax_hdr.text(122.8, y_mm - 1.5, lbl1, fontsize=6.8, fontweight="bold", color=INK, ha="right", va="center")
            ax_hdr.text(122.8, y_mm + 1.9, lbl2, fontsize=6.05, color=BLUE if kind == "vol" else MUTED, ha="right", va="center")

        # Right two-line Native Δ [95% CI] cleanly aligned (zero clipping!)
        sgn = "+" if d2 > 0 else "−"
        s_lo = "+" if lo2 > 0 else "−"
        s_hi = "+" if hi2 > 0 else "−"
        ax_hdr.text(176.0, y_mm - 1.5, f"{sgn}{abs(d2):.3f}", fontsize=6.8, fontweight="bold", color=c_log, ha="right", va="center")
        ax_hdr.text(176.0, y_mm + 1.9, f"[{s_lo}{abs(lo2):.3f}, {s_hi}{abs(hi2):.3f}]", fontsize=5.95, color=SUB, ha="right", va="center")

    # Small label on Apple sourness curvature control with subtle matching halo (dashed 0% line never cuts letters!)
    ax_3b.text(
        -11.6, 4.35, "Log-curvature control (1 acid)",
        fontsize=5.85, fontweight="bold", color=PURPLE, ha="left", va="top",
        bbox=dict(boxstyle="square,pad=0.05", facecolor="#F6F4FC", edgecolor="none", alpha=0.95), zorder=6
    )

    ax_3b.set_yticks([])
    ax_3b.set_xticks([-10, -5, 0, 5])
    ax_3b.set_xticklabels(["−10%", "−5%", "0%", "+5%"], fontsize=6.9)
    ax_3b.set_xlabel("%ΔRMSE vs SUM-mass", fontsize=7.3, fontweight="bold", color=INK, labelpad=2.5)
    style_clean_axes(ax_3b)
    ax_3b.patch.set_alpha(0.0)

    # Compact 2-row legend below Panel b X-axis
    leg_b_handles = [
        Line2D([0], [0], marker="s", color=GOLD, lw=1.3, ms=4.1, markeredgecolor="white", markeredgewidth=0.4, label="DOT-lin / -mass-proxy"),
        Line2D([0], [0], marker="o", color=BLUE, lw=1.3, ms=4.5, markeredgecolor="white", markeredgewidth=0.4, label="DOT-log (vol. U, Grape)"),
        Line2D([0], [0], marker="o", color=TEAL, lw=1.3, ms=4.5, markeredgecolor="white", markeredgewidth=0.4, label="DOT-log-mass-proxy (L)"),
        Line2D([0], [0], marker="o", color=PURPLE, lw=1.3, ms=4.5, markeredgecolor="white", markeredgewidth=0.4, label="DOT-log (1-acid control)"),
    ]
    ax_3b.legend(
        handles=leg_b_handles, loc="upper center", bbox_to_anchor=(0.48, -0.155),
        ncol=2, frameon=False, fontsize=6.1, columnspacing=0.85, handletextpad=0.55, handlelength=1.25, labelspacing=0.34
    )
    ax_hdr.text(105.6, 105.0, "All 4 sugar cohorts span 0; native units differ across scales.", fontsize=6.0, color=MUTED, ha="left", va="center")

    # ---------------------------------------------------------------------
    # PANEL c (Bottom left: x = 4.0 .. 90.5 mm, 2x2 collinearity small-multiples)
    # ---------------------------------------------------------------------
    ax_hdr.text(4.0, 113.5, "c", fontsize=11.0, fontweight="bold", color=CHARCOAL, ha="left", va="center")
    ax_hdr.text(9.0, 113.5, "Near-unity sugar collinearity (identifiability gate A2)", fontsize=8.4, fontweight="bold", color=CHARCOAL, ha="left", va="center")

    inp = pd.read_csv(r"C:\zhuang_taste_pre\psymix_fruit_validation_20261117\within\inputs.csv")
    wt = pd.read_csv(r"C:\zhuang_taste_pre\psymix_fruit_validation_20261117\within\chemistry_weights.csv")

    c_panels = [
        ("apple", "Apple (n = 40)", [("raw_Glucose", "Glucose"), ("raw_Fructose", "Fructose"), ("raw_Sucrose", "Sucrose")],
         "Proxy load L", "SUM-mass (mg/g)", AMBER, 0, 0),
        ("blueberry", "Blueberry (n = 219)", [("raw_glucose", "glucose"), ("raw_fructose", "fructose"), ("raw_sucrose", "sucrose")],
         "Proxy load L", "SUM-mass (mg/g)", PURPLE, 0, 1),
        ("grape", "Grape juice (n = 41)", [("raw_Glucose", "Glucose"), ("raw_Fructose", "Fructose")],
         "Vol. load U", "SUM-mass (g/L)", BLUE, 1, 0),
        ("tomato", "Tomato (n = 209)", [("raw_glucose", "glucose"), ("raw_fructose", "fructose")],
         "Proxy load L", "SUM-mass (mg/g)", TEAL, 1, 1),
    ]

    # 2x2 box positions (generous spacing so zero axis-label clash)
    col_x = [14.8, 57.2]
    row_y = [122.2, 157.8]
    cw, ch = 31.8, 25.0

    for crop, title, cols, ylab, xlab, col_c, ri, ci in c_panels:
        sub = inp[(inp["crop"] == crop) & (inp["endpoint"] == "sweetness")].dropna(subset=[c for c, _ in cols])
        wsub = wt[(wt["crop"] == crop) & (wt["endpoint"] == "sweetness") & (wt["assignment"] == "M1_correct")]
        wmap = dict(zip(wsub["analyte"], wsub["multiplier"]))
        S = sum(sub[c].values for c, _ in cols)
        L = sum(sub[c].values * wmap[a] for c, a in cols)
        r_val = pearsonr(S, L)[0]
        vif_val = 1.0 / (1.0 - r_val * r_val)

        if crop == "apple":
            assert abs(r_val - 0.9819) < 0.0005 and abs(vif_val - 27.94) < 0.1
        elif crop == "blueberry":
            assert abs(r_val - 0.9973) < 0.0005 and abs(vif_val - 188.09) < 0.2
        elif crop == "grape":
            assert abs(r_val - 0.9989) < 0.0005 and abs(vif_val - 473.79) < 0.3
        elif crop == "tomato":
            assert abs(r_val - 0.9991) < 0.0005 and abs(vif_val - 582.06) < 0.4

        ax_c = fig.add_axes([col_x[ci] / W, 1.0 - (row_y[ri] + ch) / H, cw / W, ch / H], zorder=3)
        ax_c.set_gid(f"Panel_3c_{crop}")

        # OLS trend line + scatter
        slope, intercept = np.polyfit(S, L, 1)
        xs_line = np.linspace(S.min(), S.max(), 50)
        ax_c.plot(xs_line, slope * xs_line + intercept, color=CHARCOAL, lw=0.9, ls="--", alpha=0.75, zorder=2)
        ax_c.scatter(S, L, s=11 if len(S) > 100 else 17, color=col_c, alpha=0.68, edgecolors="white", linewidths=0.32, zorder=3)

        # Clean upper-left stat callout (always opposite diagonal points!)
        ax_c.text(
            0.04, 0.95, f"r = {r_val:.3f}\nVIF = {vif_val:.1f}",
            transform=ax_c.transAxes, fontsize=6.4, fontweight="bold", color=col_c,
            ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.82), zorder=5
        )

        ax_c.set_title(title, fontsize=7.1, fontweight="bold", color=CHARCOAL, pad=2.5)
        ax_c.set_ylabel(ylab, fontsize=6.5, fontweight="bold" if crop == "grape" else "normal", color=BLUE if crop == "grape" else SUB, labelpad=1.8)
        ax_c.set_xlabel(xlab, fontsize=6.4, color=SUB, labelpad=1.6)
        ax_c.locator_params(axis="x", nbins=3)
        ax_c.locator_params(axis="y", nbins=3)
        ax_c.tick_params(labelsize=6.3, pad=1.6, length=2.2)
        ax_c.grid(color=HAIR, lw=0.45, zorder=0)
        style_clean_axes(ax_c, left_lw=0.6, bottom_lw=0.6)

    # Acid collinearity footer pill (Rule 4 mandatory!)
    add_rbox(ax_hdr, 7.5, 191.5, 81.5, 6.8, fc=STRIPE_BG, ec=LINE, lw=0.6, r=1.4, zorder=3)
    ax_hdr.text(
        48.25, 194.9, r"Acid panels (malic + citric): $r = 1.000$ ($\mathrm{VIF} > 1.6 \times 10^{5}$)",
        fontsize=6.6, fontweight="bold", color=CHARCOAL, ha="center", va="center", zorder=4
    )

    # ---------------------------------------------------------------------
    # PANEL d (Bottom right: x = 96.0 .. 176.0 mm, Dirichlet design curves)
    # ---------------------------------------------------------------------
    ax_hdr.text(96.0, 113.5, "d", fontsize=11.0, fontweight="bold", color=CHARCOAL, ha="left", va="center")
    ax_hdr.text(101.0, 113.5, "Dirichlet sugar-share dispersion needed for r = 0.90", fontsize=8.4, fontweight="bold", color=CHARCOAL, ha="left", va="center")

    dr = pd.read_csv(r"C:\zhuang_taste_pre\psymix_final_tables_for_writing\A2_sugar_design_requirements.csv").set_index("crop")
    dd = pd.read_csv(r"C:\zhuang_taste_pre\psymix_final_tables_for_writing\A2_sugar_Dirichlet_design.csv")
    fd = pd.read_csv(r"C:\zhuang_taste_pre\psymix_final_tables_for_writing\A2_sugar_fraction_dispersion.csv")

    ax_3d = fig.add_axes([107.2 / W, 1.0 - 183.2 / H, 67.8 / W, 61.0 / H], zorder=3)
    ax_3d.set_gid("Panel_3d_Dirichlet")

    ax_3d.set_xlim(0.0, 34.0)
    ax_3d.set_ylim(0.845, 1.007)

    # Rule 4: Top subtle warning band r >= 0.98 (operational collinearity flag, VIF >= 25)
    ax_3d.axhspan(0.98, 1.007, facecolor=CRIMSON_LT, alpha=0.80, zorder=0)
    ax_3d.axhline(0.98, color=CRIMSON, lw=0.75, ls=":", zorder=1)
    ax_3d.text(33.3, 0.9935, "r ≥ 0.98 (operational collinearity flag, VIF ≥ 25)",
               fontsize=5.95, fontweight="bold", color=CRIMSON, ha="right", va="center", zorder=5)

    # Rule 4 + P0 Fix #1: Anchor bottom shaded zone EXACTLY to the 4 major-sugar r = 0.90 crossings [14.58, 28.91 pp]
    # (zero hanging gap!), while noting Apple's minor sugar (glucose, 13.3 pp) to match Table 4's full non-trace range.
    band_lo, band_hi = 14.5812, 28.9065
    ax_3d.fill_between([band_lo, band_hi], 0.845, 0.90, facecolor=BLUE_LT, alpha=0.85, zorder=0)
    ax_3d.axvline(band_lo, color=SLATE_BLUE, lw=0.65, ls=":", zorder=1)
    ax_3d.axvline(band_hi, color=SLATE_BLUE, lw=0.65, ls=":", zorder=1)
    ax_3d.axhline(0.90, color=CHARCOAL, lw=0.80, ls="--", zorder=1)

    ax_3d.annotate(
        "Major-sugar r = 0.90:\n14.6–28.9 pp (VIF = 5.3)\n(Apple glucose: 13.3 pp)",
        xy=(band_lo, 0.866), xytext=(0.9, 0.866),
        fontsize=5.65, fontweight="bold", color=BLUE, ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor=SLATE_BLUE, lw=0.6),
        arrowprops=dict(arrowstyle="-|>,head_length=0.4,head_width=0.22", color=BLUE, lw=0.8),
        zorder=7
    )

    # Analytical closed-form Dirichlet curves (matching A2_sugar_Dirichlet_design.csv to 4 decimals!)
    w_std = {"glucose": 0.506667, "fructose": 0.876923, "sucrose": 1.0}
    w_grp = {"glucose": 0.061675, "fructose": 0.106745}

    d_configs = [
        ("apple", "Apple (sucrose)", AMBER, "Sucrose", w_std, (17.25, 0.9045, "left", "bottom")),
        ("blueberry", "Blueberry (fructose)", PURPLE, "fructose", w_std, (13.65, 0.8945, "right", "top")),
        ("grape", "Grape (fructose)", BLUE, "Fructose", w_grp, (21.35, 0.9045, "left", "bottom")),
        ("tomato", "Tomato (fructose)", TEAL, "fructose", w_std, (29.55, 0.9045, "left", "bottom")),
    ]

    alphas = np.geomspace(0.35, 3000.0, 400)
    for crop, lbl, col_d, maj_sugar, wdict, ann_cfg in d_configs:
        sub_dd = dd[dd["crop"] == crop]
        mu = np.array([float(r["mean_mass_fraction"]) for _, r in sub_dd.iterrows()])
        ws = np.array([wdict[r["sugar"].lower()] for _, r in sub_dd.iterrows()])
        mu_w = float(np.dot(mu, ws))
        cv_s = float(dr.loc[crop, "observed_total_CV"])
        var_w_num = float(np.sum(mu * (ws - mu_w) ** 2))

        maj_row_dd = sub_dd[sub_dd["sugar"].str.lower() == maj_sugar.lower()].iloc[0]
        mu_maj = float(maj_row_dd["mean_mass_fraction"])
        sd_target_pp = float(maj_row_dd["SD_percentage_points"])

        # Closed-form r(S, L) under P ~ Dirichlet(alpha0 * mu), S independent of P
        var_w = var_w_num / (alphas + 1.0)
        cv_w2 = var_w / (mu_w ** 2)
        r_curve = 1.0 / np.sqrt(1.0 + ((1.0 + cv_s ** 2) / (cv_s ** 2)) * cv_w2)
        sd_pp = 100.0 * np.sqrt(mu_maj * (1.0 - mu_maj) / (alphas + 1.0))

        # Verify crossing at r = 0.90 matches frozen CSV!
        r_at_target = np.interp(sd_target_pp, sd_pp[::-1], r_curve[::-1])
        assert abs(r_at_target - 0.90) < 0.002, (crop, sd_target_pp, r_at_target)

        ax_3d.plot(sd_pp, r_curve, color=col_d, lw=1.65, label=lbl, zorder=3)

        # Crossing marker at r = 0.90 + unobstructed label with white halo
        ax_3d.plot(sd_target_pp, 0.90, marker="D", ms=4.3, markerfacecolor="white", markeredgecolor=col_d, markeredgewidth=1.25, zorder=5)
        ax_x, ax_y, ha_s, va_s = ann_cfg
        ax_3d.text(
            ax_x, ax_y, f"{sd_target_pp:.1f}",
            fontsize=6.1, fontweight="bold", color=col_d, ha=ha_s, va=va_s,
            bbox=dict(boxstyle="square,pad=0.04", facecolor="white", edgecolor="none", alpha=0.88), zorder=6
        )

        # Observed cohort marker trapped at top!
        obs_sd_pp = 100.0 * float(fd[(fd["crop"] == crop) & (fd["sugar"].str.lower() == maj_sugar.lower())]["observed_SD_fraction"].iloc[0])
        obs_r = float(dr.loc[crop, "observed_r"])
        ax_3d.plot(obs_sd_pp, obs_r, marker="o", ms=5.0, color=col_d, markeredgecolor="white", markeredgewidth=0.65, zorder=6)

    # Clean callout in totally open lower-left above band pointing straight up to observed cluster (zero curve overlap!)
    ax_3d.annotate(
        "Observed cohorts\n2.3–10.2 pp\n(r ≥ 0.982)",
        xy=(3.8, 0.991), xytext=(1.2, 0.927),
        fontsize=5.95, fontweight="bold", color=CHARCOAL, ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor=LINE, lw=0.6),
        arrowprops=dict(arrowstyle="-|>,head_length=0.4,head_width=0.22", color=CHARCOAL, lw=0.75, connectionstyle="arc3,rad=-0.10"),
        zorder=7
    )

    ax_3d.set_xlabel("Major-sugar share SD (percentage points, pp)", fontsize=7.3, fontweight="bold", color=INK, labelpad=2.5)
    ax_3d.set_ylabel("Pearson r (SUM-mass vs weighted load)", fontsize=7.2, fontweight="bold", color=INK, labelpad=2.5)
    ax_3d.set_xticks([0, 5, 10, 15, 20, 25, 30])
    ax_3d.set_yticks([0.85, 0.90, 0.95, 1.00])
    ax_3d.grid(color=HAIR, lw=0.45, zorder=0)
    style_clean_axes(ax_3d)

    # Compact legend cleanly below Panel d X-axis (zero data occlusion!)
    leg_d_handles = [
        Line2D([0], [0], color=AMBER, lw=1.6, marker="o", ms=4.2, markeredgecolor="white", markeredgewidth=0.4, label="Apple"),
        Line2D([0], [0], color=PURPLE, lw=1.6, marker="o", ms=4.2, markeredgecolor="white", markeredgewidth=0.4, label="Blueberry"),
        Line2D([0], [0], color=BLUE, lw=1.6, marker="o", ms=4.2, markeredgecolor="white", markeredgewidth=0.4, label="Grape"),
        Line2D([0], [0], color=TEAL, lw=1.6, marker="o", ms=4.2, markeredgecolor="white", markeredgewidth=0.4, label="Tomato"),
    ]
    ax_3d.legend(
        handles=leg_d_handles, loc="upper center", bbox_to_anchor=(0.48, -0.175),
        ncol=4, frameon=False, fontsize=6.4, columnspacing=0.95, handletextpad=0.35, handlelength=1.35
    )

    svg_path = OUT_DIR / "PsyMix_Figure3.svg"
    pdf_path = OUT_DIR / "PsyMix_Figure3.pdf"
    fig.savefig(svg_path, format="svg")
    fig.savefig(pdf_path, format="pdf")
    plt.close(fig)
    print("Saved Figure 3:", pdf_path)


# =========================================================================
# FIGURE 4: 180.0 mm x 206.0 mm (Tasks T3 & T4 — Sweetener Mixtures)
# =========================================================================
def build_figure4():
    W, H = 180.0, 206.0
    fig = plt.figure(figsize=(W / 25.4, H / 25.4), dpi=600)

    ax_bg = fig.add_axes([0, 0, 1, 1], xlim=(0, W), ylim=(H, 0), zorder=0)
    ax_bg.set_axis_off()
    ax_bg.set_gid("Canvas_Background")
    ax_bg.add_patch(Rectangle((0, 0), W, H, facecolor="white", edgecolor="none", zorder=0))
    # Horizontal tier divider + vertical panel separators
    ax_bg.plot([4.0, W - 4.0], [111.5, 111.5], color=HAIR, lw=0.85, zorder=1)
    ax_bg.plot([68.0, 68.0], [4.5, 108.5], color=HAIR, lw=0.75, zorder=1)
    ax_bg.plot([117.5, 117.5], [114.5, 202.5], color=HAIR, lw=0.75, zorder=1)

    ax_hdr = fig.add_axes([0, 0, 1, 1], xlim=(0, W), ylim=(H, 0), zorder=10)
    ax_hdr.set_axis_off()
    ax_hdr.patch.set_alpha(0.0)
    ax_hdr.set_gid("Panel_Headers")

    # ---------------------------------------------------------------------
    # PANEL a (Top left, narrower: x = 4.0 .. 66.5 mm, ~0.82 ratio)
    # ---------------------------------------------------------------------
    ax_hdr.text(4.0, 6.6, "a", fontsize=11.0, fontweight="bold", color=CHARCOAL, ha="left", va="center")
    ax_hdr.text(9.0, 6.6, "Iso-sweet pair holdout (fold 1 of 5)", fontsize=8.3, fontweight="bold", color=CHARCOAL, ha="left", va="center")

    mon = pd.read_csv(r"C:\zhuang_taste_pre\phase29_direct_mixture_20261115\data\monomers.csv").set_index("code")
    folds = pd.read_csv(r"C:\zhuang_taste_pre\psymix_four_issue_audit_20261123\C1_frozen_outer_folds.csv")
    pair_folds = folds.drop_duplicates("group_id")[["group_id", "split_id"]]
    assert len(pair_folds) == 55
    test_pairs = set(pair_folds[pair_folds["split_id"] == 1]["group_id"])
    train_pairs = set(pair_folds[pair_folds["split_id"] != 1]["group_id"])
    assert len(test_pairs) == 11 and len(train_pairs) == 44

    # 11 core small-molecule sweeteners ordered clockwise by descending potency
    core_codes = ["ali", "neo", "sac", "reb", "apm", "ace", "ste", "cyc", "suc", "man", "sor"]
    code_disp = {
        "ali": "Alitame", "neo": "Neotame", "sac": "Saccharin", "reb": "Reb A",
        "apm": "Aspartame", "ace": "Ace-K", "ste": "Stevioside", "cyc": "Cyclamate",
        "suc": "Sucrose", "man": "Mannitol", "sor": "Sorbitol",
    }
    pot_se3 = {c: 3.0 / float(mon.loc[c, "wv_percent_SE3"]) for c in core_codes}
    log_pot = {c: np.log10(pot_se3[c]) for c in core_codes}
    lp_min, lp_max = min(log_pot.values()), max(log_pot.values())

    # Verify 4,317-8,679x potency span across SE3/SE5/SE7
    span_se3 = float(mon.loc[core_codes, "wv_percent_SE3"].max() / mon.loc[core_codes, "wv_percent_SE3"].min())
    span_se7 = float(mon.loc[core_codes, "wv_percent_SE7"].max() / mon.loc[core_codes, "wv_percent_SE7"].min())
    assert 8670 < span_se3 < 8690 and 4310 < span_se7 < 4325, (span_se3, span_se7)

    # Dedicated figure-mm canvas for Panel 4a network
    ax_4a = fig.add_axes([4.0 / W, 1.0 - 108.0 / H, 62.5 / W, 97.5 / H], xlim=(0, 62.5), ylim=(97.5, 0), zorder=3)
    ax_4a.set_axis_off()
    ax_4a.set_gid("Panel_4a_Network")

    cx, cy, R = 30.2, 37.5, 18.4
    angles = np.linspace(-np.pi / 2.0, 1.5 * np.pi, len(core_codes), endpoint=False)
    pos = {c: (cx + R * np.cos(a), cy + R * np.sin(a)) for c, a in zip(core_codes, angles)}

    # Draw 44 training edges first (ultra-thin translucent slate-blue, Rule 5!)
    for gid in sorted(train_pairs):
        a, b = gid.split("|")
        xa, ya = pos[a]
        xb, yb = pos[b]
        ax_4a.plot([xa, xb], [ya, yb], color=SLATE_BLUE, lw=0.60, alpha=0.25, zorder=2)

    # Draw 11 held-out test edges on top (solid warm orange lw=1.5, Rule 5!)
    for gid in sorted(test_pairs):
        a, b = gid.split("|")
        xa, ya = pos[a]
        xb, yb = pos[b]
        ax_4a.plot([xa, xb], [ya, yb], color=AMBER, lw=1.50, alpha=0.92, zorder=3)

    # Draw 11 core nodes scaled by log10 potency + STRICTLY HORIZONTAL labels!
    for c, ang in zip(core_codes, angles):
        x, y = pos[c]
        frac = (log_pot[c] - lp_min) / (lp_max - lp_min)
        r_mm = 1.45 + 1.95 * frac
        # Color gradient subtly: high-potency (>100x) BLUE, mid TEAL, bulk sugars/polyols SLATE_BLUE
        ncol = BLUE if pot_se3[c] >= 100.0 else (TEAL if pot_se3[c] >= 10.0 else "#5A7D9A")
        ax_4a.add_patch(Circle((x, y), r_mm, facecolor=ncol, edgecolor="white", linewidth=0.85, zorder=5))

        # Horizontal non-overlapping label placement
        ca, sa = np.cos(ang), np.sin(ang)
        if abs(ca) < 0.20:
            lx = x
            ly = y - r_mm - 2.3 if sa < 0 else y + r_mm + 2.4
            ha, va = "center", ("bottom" if sa < 0 else "top")
        elif ca > 0:
            lx = x + r_mm + 1.5
            ly = y + 0.4 * sa
            ha, va = "left", "center"
        else:
            lx = x - r_mm - 1.5
            ly = y + 0.4 * sa
            ha, va = "right", "center"
        ax_4a.text(lx, ly, code_disp[c], fontsize=6.25, fontweight="bold" if c in ("ali", "apm", "suc", "sor") else "normal",
                   color=INK, ha=ha, va=va, zorder=6)

    # Crisp opaque center badge cleanly masking background grey edges (P1/P2 Fix #5!)
    ax_4a.text(cx, cy, "55 core pairs\nSE 3 / 5 / 7%", fontsize=5.9, fontweight="bold", color=CHARCOAL,
               ha="center", va="center",
               bbox=dict(boxstyle="round,pad=0.24", facecolor="white", edgecolor=LINE, lw=0.55, alpha=0.96), zorder=6)

    # Delicate Stevia watercolor corner accent (top-right open corner, clear of labels!)
    im_stevia = load_cropped_rgba(SPEC_DIR / "stevia.png")
    place_image_mm(fig, W, H, im_stevia, 55.2, 11.2, 10.5, 11.5, gid="Specimen_4a_Stevia", alpha=0.92, zorder=4)

    # Rule 5: 3 outer extended-network complex sweeteners as dashed light-grey nodes (NO 36 extra edges!)
    add_rbox(ax_4a, 1.5, 67.2, 59.5, 13.0, fc=STRIPE_BG, ec=LINE, lw=0.6, r=1.5, ls="--", zorder=2)
    ext_items = [("Fructose", 10.5), ("Glucose", 26.5), ("Thaumatin", 44.5)]
    for ename, ex in ext_items:
        ax_4a.add_patch(Circle((ex, 71.8), 1.45, facecolor="white", edgecolor=GREY, linewidth=0.95, linestyle="--", zorder=4))
        ax_4a.text(ex + 2.3, 71.8, ename, fontsize=6.15, color=SUB, ha="left", va="center", zorder=4)
    ax_4a.text(31.25, 77.2, "+3 extended sweeteners (91-pair set; 36 extra edges omitted)",
               fontsize=5.85, color=MUTED, ha="center", va="center", zorder=4)

    # Crisp clean legend block at bottom of 4a (y = 82.8 .. 96.5 mm in ax_4a)
    ax_4a.plot([2.5, 8.2], [85.2, 85.2], color=SLATE_BLUE, lw=0.85, alpha=0.65)
    ax_4a.text(9.4, 85.2, "44 train pairs (grey-blue)", fontsize=6.15, color=SUB, ha="left", va="center")
    ax_4a.plot([34.5, 40.2], [85.2, 85.2], color=AMBER, lw=1.6)
    ax_4a.text(41.4, 85.2, "11 held-out pairs", fontsize=6.15, fontweight="bold", color=AMBER, ha="left", va="center")

    # Node potency scale key
    ax_4a.add_patch(Circle((4.5, 92.0), 1.45, facecolor="#5A7D9A", edgecolor="white", lw=0.6))
    ax_4a.add_patch(Circle((8.3, 92.0), 2.35, facecolor=TEAL, edgecolor="white", lw=0.6))
    ax_4a.add_patch(Circle((13.3, 92.0), 3.25, facecolor=BLUE, edgecolor="white", lw=0.6))
    ax_4a.text(18.2, 90.4, r"Node radius $\propto \log_{10}$ potency", fontsize=6.15, fontweight="bold", color=INK, ha="left", va="center")
    ax_4a.text(18.2, 94.1, "4,317–8,679× w/v span (SE 3–7%)", fontsize=5.95, color=MUTED, ha="left", va="center")

    # ---------------------------------------------------------------------
    # PANEL b (Top right, wider: x = 70.0 .. 176.0 mm, ~1.38 ratio)
    # Two aligned forest columns, NO '//' broken axis, clean right-edge overflow arrows!
    # ---------------------------------------------------------------------
    ax_hdr.text(70.0, 6.6, "b", fontsize=11.0, fontweight="bold", color=CHARCOAL, ha="left", va="center")
    ax_hdr.text(75.0, 6.6, "Dual-endpoint model inversion on 55 held-out pairs (T3)", fontsize=8.4, fontweight="bold", color=CHARCOAL, ha="left", va="center")

    f4b = pd.read_csv(r"C:\zhuang_taste_pre\psymix_manuscript_release\Figures_round1_20261124\source_data\Figure4B.csv")
    f4bo = pd.read_csv(r"C:\zhuang_taste_pre\psymix_manuscript_release\Figures_round1_20261124\source_data\Figure4B_other_contrasts.csv")

    # Verify exact fixed-input model gains & dual-endpoint inversion
    m_abs = f4bo[(f4bo["metric"] == "absolute_MAE") & (f4bo["candidate"] == "C1_ET_raw")].iloc[0]
    m_chg = f4bo[(f4bo["metric"] == "delta_MAE") & (f4bo["candidate"] == "C1_ET_raw")].iloc[0]
    assert abs(float(m_abs["diff"]) - (-0.2738)) < 0.001
    assert abs(float(m_chg["diff"]) - (-0.5536)) < 0.001

    model_order = [
        ("NULL-mean", "None"),
        ("RIDGE-id", "Identity"),
        ("RIDGE-anon", "Conc. (anon)"),
        ("ADD-nominal", "Additivity"),
        ("TREE-anon", "Conc. (anon)"),
        ("RIDGE-conc", "Named conc."),
        ("TREE-conc", "Named conc."),
        ("EQS", "Potency ref."),
        ("EQS×ID", "Ref. + slopes"),
    ]
    y_b = [0.0, 1.0, 2.0, 3.0, 4.0, 5.45, 6.45, 7.45, 8.45]

    x_bL, w_bL = 90.8, 39.5
    x_bR, w_bR = 135.8, 39.5
    y_bT, h_b = 17.5, 73.5

    ax_4bL = fig.add_axes([x_bL / W, 1.0 - (y_bT + h_b) / H, w_bL / W, h_b / H], zorder=3)
    ax_4bR = fig.add_axes([x_bR / W, 1.0 - (y_bT + h_b) / H, w_bR / W, h_b / H], zorder=3)
    ax_4bL.set_gid("Panel_4b_AbsMAE")
    ax_4bR.set_gid("Panel_4b_ChangeMAE")

    for ax_col, ep_name, ep_title, eqs_val, xlim_hi, x_cap, x_ticks, x_tlabels in [
        (ax_4bL, "absolute_MAE", "Absolute MAE (EQS = 0.930)", 0.930, 0.92, 0.85, [-0.3, 0.0, 0.3, 0.6], ["−0.3", "0.0", "+0.3", "+0.6"]),
        (ax_4bR, "delta_MAE", "Dose-change MAE (EQS = 0.610)", 0.610, 1.24, 1.15, [-0.3, 0.0, 0.3, 0.6, 0.9], ["−0.3", "0.0", "+0.3", "+0.6", "+0.9"]),
    ]:
        ax_col.set_ylim(9.15, -0.68)
        ax_col.set_xlim(-0.36, xlim_hi)
        ax_col.set_title(ep_title, fontsize=7.3, fontweight="bold", color=CHARCOAL, pad=4.0)

        # Highlight EQS reference row and TREE-conc inversion row softly
        ax_col.axhspan(7.02, 7.88, facecolor=STRIPE_BG, edgecolor="none", zorder=0)
        ax_col.axhspan(6.02, 6.88, facecolor=BLUE_LT if ep_name == "absolute_MAE" else AMBER_LT, alpha=0.55, edgecolor="none", zorder=0)
        ax_col.axhline(4.72, color=LINE, lw=0.75, ls="--", zorder=1)
        ax_col.axvline(0.0, color=CHARCOAL, lw=0.85, ls="--", zorder=2)
        ax_col.grid(axis="x", color=HAIR, lw=0.48, zorder=0)

        sub_ep = f4b[f4b["endpoint"] == ep_name].set_index("label")

        for yp, (mlbl, _) in zip(y_b, model_order):
            row = sub_ep.loc[mlbl]
            d = float(row["delta"])
            lo = float(row["lo"]) if pd.notna(row["lo"]) else None
            hi = float(row["hi"]) if pd.notna(row["hi"]) else None

            # Color semantics
            if mlbl == "EQS":
                col, mk, ms = CHARCOAL, "D", 4.4
            elif mlbl == "EQS×ID":
                col, mk, ms = TEAL, "o", 4.9
            elif mlbl == "TREE-conc":
                col = BLUE if ep_name == "absolute_MAE" else AMBER
                mk, ms = "o", 4.9
            elif mlbl == "RIDGE-conc":
                col, mk, ms = PURPLE, "s", 4.3
            elif mlbl in ("TREE-anon", "ADD-nominal"):
                col, mk, ms = SLATE_BLUE, "s", 4.1
            else:
                col, mk, ms = GREY, "s", 3.9

            if d > x_cap:
                # Rule 5: Clean right-edge overflow arrow (NO '//' broken axis!)
                arr_x0 = 0.48 if ep_name == "absolute_MAE" else 0.62
                ax_col.annotate(
                    "", xy=(xlim_hi - 0.015, yp), xytext=(arr_x0, yp),
                    arrowprops=dict(arrowstyle="-|>,head_length=0.45,head_width=0.25", color=GREY, lw=1.0),
                    zorder=4
                )
                ax_col.text(
                    arr_x0 - 0.02, yp, f"+{d:.2f}", fontsize=6.4, fontweight="bold", color=SUB,
                    ha="right", va="center",
                    bbox=dict(boxstyle="round,pad=0.14", facecolor=STRIPE_BG, edgecolor=LINE, lw=0.5), zorder=5
                )
            else:
                if lo is not None and hi is not None:
                    ax_col.plot([lo, hi], [yp, yp], color=col, lw=1.5, solid_capstyle="round", zorder=3)
                ax_col.plot(d, yp, marker=mk, ms=ms, color=col, markeredgecolor="white", markeredgewidth=0.45, zorder=5)
                # Direct concise annotation on key points (P1 Fix #4: consistent (ns) + inline inversion + zero collision!)
                if mlbl == "TREE-conc":
                    if ep_name == "absolute_MAE":
                        ax_col.text(0.04, yp, "−0.113 (ns)", fontsize=5.95, fontweight="bold", color=BLUE, ha="left", va="center",
                                    bbox=dict(boxstyle="square,pad=0.03", facecolor="#F0F7FC", edgecolor="none", alpha=0.92), zorder=6)
                    else:
                        ax_col.text(hi + 0.03, yp, "+0.083* (inversion)", fontsize=5.55, fontweight="bold", color=AMBER, ha="left", va="center",
                                    bbox=dict(boxstyle="square,pad=0.03", facecolor="#FEF7F0", edgecolor="none", alpha=0.92), zorder=6)
                elif mlbl == "EQS×ID":
                    if ep_name == "delta_MAE":
                        ax_col.text(0.04, yp, "−0.077* (0.533)", fontsize=6.1, fontweight="bold", color=TEAL, ha="left", va="center", zorder=6)
                    else:
                        ax_col.text(hi + 0.035, yp, "−0.051 (ns)", fontsize=5.95, color=TEAL, ha="left", va="center", zorder=6)
                elif mlbl == "EQS":
                    ax_col.text(0.045, yp, f"ref ({eqs_val:.3f})", fontsize=6.0, color=SUB, ha="left", va="center", zorder=6)

        # Rule 5 + P1 Fix #4: Symmetrical purple fixed-input model gain brackets clear of all error bars + symmetrical ← Better!
        if ep_name == "absolute_MAE":
            bx = 0.51
            ax_col.plot([0.33, bx, bx, 0.46], [5.45, 5.45, 6.45, 6.45], color=PURPLE, lw=0.85, zorder=4)
            ax_col.text(bx + 0.025, 5.95, "Model gain\n−0.274*", fontsize=5.85, fontweight="bold", color=PURPLE, ha="left", va="center", zorder=6)
            ax_col.text(-0.04, 8.98, "← Better", fontsize=5.65, color=TEAL, fontweight="bold", ha="right", va="center", zorder=6)
        else:
            bx = 0.92
            ax_col.plot([0.79, bx, bx, 0.88], [5.45, 5.45, 6.45, 6.45], color=PURPLE, lw=0.85, zorder=4)
            ax_col.text(bx + 0.025, 5.95, "Model\n−0.554*", fontsize=5.65, fontweight="bold", color=PURPLE, ha="left", va="center", zorder=6)
            ax_col.text(-0.04, 8.98, "← Better", fontsize=5.65, color=TEAL, fontweight="bold", ha="right", va="center", zorder=6)

        ax_col.set_xticks(x_ticks)
        ax_col.set_xticklabels(x_tlabels, fontsize=6.7)
        ax_col.set_xlabel("Δ MAE vs EQS (95% whole-pair CI)", fontsize=7.0, fontweight="bold", color=INK, labelpad=2.2)
        style_clean_axes(ax_col)
        ax_col.patch.set_alpha(0.0)

    # Shared Y-axis tick labels on left column only
    ax_4bL.set_yticks(y_b)
    ax_4bL.set_yticklabels([m[0] for m in model_order], fontsize=6.9, color=INK)
    for tl in ax_4bL.get_yticklabels():
        if tl.get_text() in ("EQS", "EQS×ID", "TREE-conc", "RIDGE-conc"):
            tl.set_fontweight("bold")
    ax_4bR.set_yticks(y_b)
    ax_4bR.set_yticklabels([])
    ax_4bR.tick_params(axis="y", length=0)

    # Clean footer note under Panel 4b (zero collision with X-axis labels or right margin!)
    ax_hdr.text(
        70.5, 105.0,
        "Right-edge arrows: out-of-range baselines (no broken axis). *95% CI excludes 0; purple bracket = fixed-input model gain.",
        fontsize=5.65, color=SUB, ha="left", va="center"
    )
    ax_hdr.text(
        70.5, 108.5,
        "Capacity ceiling test: deep set/dynamics adapters (ADPT-set/pool/dyn; Table S4) did not surpass EQS or TREE-conc.",
        fontsize=5.65, color=MUTED, ha="left", va="center"
    )

    # ---------------------------------------------------------------------
    # PANEL c (Bottom left+center: x = 4.0 .. 115.5 mm) — Task T4a (Table 1)
    # Left: 3% vs 7% residual reproducibility; Right: STRICTLY TWO stacked sub-axes!
    # ---------------------------------------------------------------------
    ax_hdr.text(4.0, 116.5, "c", fontsize=11.0, fontweight="bold", color=CHARCOAL, ha="left", va="center")
    ax_hdr.text(9.0, 116.5, "Pair-specific residual reproducibility & structural borrowing (T4a)", fontsize=8.3, fontweight="bold", color=CHARCOAL, ha="left", va="center")

    tr = pd.read_csv(r"C:\zhuang_taste_pre\sensory_pair_completion_mvp\schiffman\truth.csv")
    sim = pd.read_csv(r"C:\zhuang_taste_pre\sensory_pair_completion_mvp\schiffman\similarities.csv").set_index("ingredient")
    core11 = set(sim.index)
    tr11 = tr[tr["ingredient_a"].isin(core11) & tr["ingredient_b"].isin(core11)].copy()
    tr11["res"] = tr11["measured_score"] - tr11.groupby("level")["measured_score"].transform("mean")
    piv_res = tr11.pivot(index="edge", columns="level", values="res")
    r37 = pearsonr(piv_res[3], piv_res[7])[0]
    r35 = pearsonr(piv_res[3], piv_res[5])[0]
    r57 = pearsonr(piv_res[5], piv_res[7])[0]
    assert abs(r37 - 0.7447) < 0.001 and abs(r35 - 0.8752) < 0.001

    # 4c-1: Left scatter (x = 14.5 .. 57.5 mm, y = 126.5 .. 189.5 mm)
    ax_4c1 = fig.add_axes([14.5 / W, 1.0 - 189.5 / H, 43.0 / W, 63.0 / H], zorder=3)
    ax_4c1.set_gid("Panel_4c1_ReproScatter")

    x3 = piv_res[3].values
    y7 = piv_res[7].values
    ax_4c1.axhline(0.0, color=LINE, lw=0.65, ls=":", zorder=1)
    ax_4c1.axvline(0.0, color=LINE, lw=0.65, ls=":", zorder=1)
    ax_4c1.plot([-3.5, 2.5], [-3.5, 2.5], color=GREY, lw=0.75, ls="--", alpha=0.65, zorder=1)
    m_sl, b_in = np.polyfit(x3, y7, 1)
    xs_fit = np.linspace(x3.min(), x3.max(), 40)
    ax_4c1.plot(xs_fit, m_sl * xs_fit + b_in, color=BLUE, lw=1.2, zorder=2)

    # P1 Fix #3: Unified primary blue #0072B2 (continuous correlation, no arbitrary 2-color split!)
    ax_4c1.scatter(x3, y7, s=20, color=BLUE, alpha=0.80, edgecolors="white", linewidths=0.45, zorder=4)

    # Label only the 2 genuinely isolated extreme outer pairs; neo+reb shifted left 3-4 pt clear of right border!
    for edge_id, dx, dy, ha_e in [("ace|sac", 0.25, 0.0, "left"), ("neo|reb", 0.14, 0.34, "center")]:
        if edge_id in piv_res.index:
            px, py = float(piv_res.loc[edge_id, 3]), float(piv_res.loc[edge_id, 7])
            ax_4c1.text(px + dx, py + dy, edge_id.replace("|", "+"), fontsize=5.75, color=SUB, ha=ha_e, va="center",
                        bbox=dict(boxstyle="square,pad=0.03", facecolor="white", edgecolor="none", alpha=0.82), zorder=6)

    # Rule 6 mandatory callouts: exact 3% vs 7% r, all dose pairs range, and 86.5% vs 8.1% variance contrast!
    ax_4c1.text(
        0.04, 0.96,
        f"3% vs 7%: r = {r37:.3f}\nAll dose pairs: r = {r37:.3f}–{r35:.3f}",
        transform=ax_4c1.transAxes, fontsize=6.05, fontweight="bold", color=BLUE,
        ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor=LINE, lw=0.55, alpha=0.95), zorder=6
    )

    # Clean bottom-right variance comparison card (86.5% vs 8.1%)
    ax_4c1.text(
        0.96, 0.05,
        "Pair constant: 86.5% var.\nTREE-conc (cold): 8.1%",
        transform=ax_4c1.transAxes, fontsize=6.05, fontweight="bold", color=CHARCOAL,
        ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.22", facecolor=STRIPE_BG, edgecolor=CHARCOAL, lw=0.6, alpha=0.95), zorder=6
    )

    ax_4c1.set_xlim(-3.5, 2.55)
    ax_4c1.set_ylim(-5.8, 3.1)
    ax_4c1.set_xticks([-3, -2, -1, 0, 1, 2])
    ax_4c1.set_yticks([-5, -3, -1, 1])
    ax_4c1.set_xlabel("Level-centered residual at 3% SE", fontsize=7.0, fontweight="bold", color=INK, labelpad=2.2)
    ax_4c1.set_ylabel("Level-centered residual at 7% SE", fontsize=7.0, fontweight="bold", color=INK, labelpad=2.0)
    ax_4c1.set_title("55 core pairs (cross-dose consistency)", fontsize=7.1, fontweight="bold", color=CHARCOAL, pad=4.5)
    ax_4c1.grid(color=HAIR, lw=0.45, zorder=0)
    style_clean_axes(ax_4c1)

    # 4c-2 (Upper right sub-axis: x = 70.5 .. 114.5 mm, y = 126.5 .. 154.0 mm) — Global 2D similarity (ρ = 0.002)
    tr11_nn = tr11.dropna(subset=["observed"]).copy()
    tr11_nn["tanimoto"] = [sim.loc[a, b] for a, b in zip(tr11_nn["ingredient_a"], tr11_nn["ingredient_b"])]
    rho_tan, p_tan = spearmanr(tr11_nn["tanimoto"], tr11_nn["observed"])
    assert len(tr11_nn) == 136 and abs(rho_tan - 0.0020) < 0.001

    ax_4c2 = fig.add_axes([70.5 / W, 1.0 - 154.0 / H, 43.5 / W, 27.5 / H], zorder=3)
    ax_4c2.set_gid("Panel_4c2_GlobalTanimoto")

    ax_4c2.axhline(0.0, color=LINE, lw=0.7, ls="--", zorder=1)
    med_obs = float(tr11_nn["observed"].median())
    ax_4c2.axhline(med_obs, color=CRIMSON, lw=0.95, ls=":", alpha=0.85, zorder=2)
    # P1 Fix #3: Minimal clear label for horizontal crimson dotted line
    ax_4c2.text(0.76, med_obs + 0.18, f"median ({med_obs:.2f})", fontsize=5.55, color=CRIMSON, ha="right", va="bottom", zorder=4)
    ax_4c2.scatter(tr11_nn["tanimoto"], tr11_nn["observed"], s=11, color=SLATE_BLUE, alpha=0.78, edgecolors=CHARCOAL, linewidths=0.25, zorder=3)
    ax_4c2.text(0.733, 0.56, "man+sor", fontsize=5.4, color=MUTED, ha="center", va="bottom", zorder=4)

    ax_4c2.text(
        0.97, 0.94, f"Spearman ρ = {rho_tan:.3f}\n(p = {p_tan:.2f}, n = 136)",
        transform=ax_4c2.transAxes, fontsize=6.1, fontweight="bold", color=CRIMSON,
        ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.88), zorder=5
    )
    ax_4c2.set_xlim(-0.04, 0.78)
    ax_4c2.set_ylim(-2.4, 5.4)
    ax_4c2.set_xticks([0.0, 0.25, 0.50, 0.75])
    ax_4c2.set_yticks([-2, 0, 2, 4])
    ax_4c2.set_xlabel("Pair 2D Tanimoto similarity", fontsize=6.7, color=INK, labelpad=1.6)
    ax_4c2.set_ylabel("Method II resid.", fontsize=6.6, color=INK, labelpad=1.8)
    ax_4c2.set_title("Global 2D similarity vs interaction", fontsize=7.0, fontweight="bold", color=CHARCOAL, pad=2.5)
    ax_4c2.grid(color=HAIR, lw=0.45, zorder=0)
    style_clean_axes(ax_4c2, left_lw=0.6, bottom_lw=0.6)

    # 4c-3 (Lower right sub-axis: x = 70.5 .. 114.5 mm, y = 169.5 .. 189.5 mm) — Dedicated MAE axis (Rule 6!)
    ax_4c3 = fig.add_axes([70.5 / W, 1.0 - 189.5 / H, 43.5 / W, 20.0 / H], zorder=3)
    ax_4c3.set_gid("Panel_4c3_LocalBorrowing")

    mae_unw, mae_sim = 1.074, 0.921
    ax_4c3.set_xlim(0.78, 1.20)
    ax_4c3.set_ylim(-0.62, 1.32)
    ax_4c3.grid(axis="x", color=HAIR, lw=0.45, zorder=0)

    # Horizontal paired arrow 1.074 -> 0.921 on dedicated MAE axis
    ax_4c3.annotate(
        "", xy=(mae_sim + 0.015, 0.22), xytext=(mae_unw - 0.015, 0.22),
        arrowprops=dict(arrowstyle="-|>,head_length=0.5,head_width=0.28", color=TEAL, lw=1.7),
        zorder=4
    )
    ax_4c3.plot(mae_unw, 0.22, marker="o", ms=5.6, markerfacecolor="white", markeredgecolor=SUB, markeredgewidth=1.35, zorder=5)
    ax_4c3.plot(mae_sim, 0.22, marker="o", ms=6.0, color=TEAL, markeredgecolor="white", markeredgewidth=0.55, zorder=6)

    ax_4c3.text(mae_unw, -0.18, "Unweighted\n1.074", fontsize=6.0, color=SUB, ha="center", va="top")
    ax_4c3.text(mae_sim, -0.18, "Partner-sim.\n0.921", fontsize=6.0, fontweight="bold", color=TEAL, ha="center", va="top")
    ax_4c3.text(
        0.5 * (mae_unw + mae_sim), 0.78,
        "Δ = −0.154 [−0.283, −0.014]",
        fontsize=6.25, fontweight="bold", color=TEAL, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.18", facecolor=TEAL_LT, edgecolor=TEAL, lw=0.55), zorder=6
    )

    ax_4c3.set_yticks([])
    ax_4c3.set_xticks([0.80, 0.90, 1.00, 1.10, 1.20])
    ax_4c3.set_xlabel("Held-out pair MAE (sweetness units)", fontsize=6.8, fontweight="bold", color=INK, labelpad=2.0)
    ax_4c3.set_title("Local substituted-partner borrowing", fontsize=7.0, fontweight="bold", color=CHARCOAL, pad=2.5)
    style_clean_axes(ax_4c3, left_lw=0.0, bottom_lw=0.65)
    ax_4c3.spines["left"].set_visible(False)

    # ---------------------------------------------------------------------
    # PANEL d (Bottom right: x = 120.5 .. 176.0 mm) — Ternary Transfer (T4b, Table 1)
    # ---------------------------------------------------------------------
    ax_hdr.text(120.5, 116.5, "d", fontsize=11.0, fontweight="bold", color=CHARCOAL, ha="left", va="center")
    ax_hdr.text(125.5, 116.5, "Ternary transfer (T4b, n = 64)", fontsize=8.3, fontweight="bold", color=CHARCOAL, ha="left", va="center")

    # Top compact vector triangle schematic (x = 121.0 .. 175.8 mm, y = 122.0 .. 145.5 mm)
    add_rbox(ax_hdr, 121.0, 122.0, 54.8, 23.2, fc=STRIPE_BG, ec=LINE, lw=0.65, r=1.6, zorder=2)
    ax_hdr.text(148.4, 125.4, "Leave-3-target-edges-out protocol", fontsize=6.4, fontweight="bold", color=CHARCOAL, ha="center", va="center", zorder=3)

    # Mini triangle (A, B, C) on left of card with 3 red dashed withheld edges
    tri_pts = [(133.2, 129.6), (125.9, 141.0), (140.5, 141.0)]
    for i in range(3):
        p0, p1 = tri_pts[i], tri_pts[(i + 1) % 3]
        ax_hdr.plot([p0[0], p1[0]], [p0[1], p1[1]], color=CRIMSON, lw=1.1, ls="--", zorder=3)
    # Subtle permitted training stubs radiating from vertices
    stubs = [((133.2, 129.6), (138.2, 128.4)), ((125.9, 141.0), (122.4, 138.0)), ((140.5, 141.0), (143.8, 138.2))]
    for s0, s1 in stubs:
        ax_hdr.plot([s0[0], s1[0]], [s0[1], s1[1]], color=TEAL, lw=0.95, alpha=0.8, zorder=3)
    for (tx, ty), lbl_t in zip(tri_pts, ["A", "B", "C"]):
        ax_hdr.add_patch(Circle((tx, ty), 1.9, facecolor=BLUE, edgecolor="white", lw=0.6, zorder=4))
        ax_hdr.text(tx, ty, lbl_t, fontsize=5.6, fontweight="bold", color="white", ha="center", va="center", zorder=5)

    # Arrow + target triad summary text cleanly padded inside right card border!
    ax_hdr.annotate("", xy=(149.8, 136.0), xytext=(144.2, 136.0),
                    arrowprops=dict(arrowstyle="-|>,head_length=0.45,head_width=0.25", color=CHARCOAL, lw=1.0), zorder=4)
    ax_hdr.text(151.0, 131.6, "Train: other 1995 pairs", fontsize=5.65, color=TEAL, fontweight="bold", ha="left", va="center", zorder=4)
    ax_hdr.text(151.0, 135.8, "Withhold: A+B, B+C, A+C", fontsize=5.65, color=CRIMSON, fontweight="bold", ha="left", va="center", zorder=4)
    ax_hdr.text(151.0, 140.0, "Supply: 2000 pure refs", fontsize=5.65, color=SUB, ha="left", va="center", zorder=4)

    # Load frozen 64-recipe ternary predictions & compute exact recipe-bootstrap 95% CIs (seed=2901, 5000 draws)
    tp = pd.read_csv(r"C:\zhuang_taste_pre\psymix_four_issue_audit_20261123\ternary_predictions.csv")
    al = pd.read_csv(r"C:\zhuang_taste_pre\phase29_direct_mixture_20261115\evaluation\aligned_sealed.csv")
    piv_al = al.pivot(index="triad", columns="model", values="prediction").reset_index()
    tm = tp.merge(piv_al, on="triad", how="inner")
    assert len(tm) == 64

    rng = np.random.default_rng(2901)
    idx_b = rng.integers(0, len(tm), size=(5000, len(tm)))
    y_obs = tm["observed"].values

    def boot_mae(pred_col):
        e = np.abs(y_obs - tm[pred_col].values)
        b = e[idx_b].mean(axis=1)
        return float(e.mean()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975)), e

    m_self = boot_mae("SELF_mean")
    m_null = boot_mae("NULL_mean")
    m_tdir = boot_mae("M1")
    m_tgrp = boot_mae("M3_graph")
    diff_dg = m_tdir[3] - m_tgrp[3]
    bdg = diff_dg[idx_b].mean(axis=1)
    dg_pt, dg_lo, dg_hi = float(diff_dg.mean()), float(np.quantile(bdg, 0.025)), float(np.quantile(bdg, 0.975))

    assert abs(m_self[0] - 1.7688) < 0.001 and abs(m_null[0] - 1.0601) < 0.001
    assert abs(m_tdir[0] - 0.6663) < 0.001 and abs(m_tgrp[0] - 0.6881) < 0.001
    assert abs(dg_pt - (-0.0219)) < 0.001

    # Bottom discrete dot + 95% CI plot (x = 138.8 .. 175.0 mm, y = 150.5 .. 189.5 mm)
    ax_4d = fig.add_axes([138.8 / W, 1.0 - 189.5 / H, 36.2 / W, 39.0 / H], zorder=3)
    ax_4d.set_gid("Panel_4d_TernaryMAE")

    d_rows = [
        ("SELF-mean\n(pure-ref)", m_self, GREY, "s", True),
        ("NULL-mean\n(binary train)", m_null, SUB, "o", True),
        ("TER-graph", m_tgrp, TEAL, "D", False),
        ("TER-direct", m_tdir, BLUE, "o", False),
    ]
    y_d = [0, 1, 2, 3]
    ax_4d.set_ylim(3.65, -0.65)
    ax_4d.set_xlim(0.35, 2.28)
    ax_4d.grid(axis="x", color=HAIR, lw=0.48, zorder=0)
    ax_4d.axhspan(1.52, 3.55, facecolor=BLUE_LT, alpha=0.45, edgecolor="none", zorder=0)

    for yp, (lbl, (pt, lo, hi, _), col, mk, hollow) in zip(y_d, d_rows):
        ax_4d.plot([lo, hi], [yp, yp], color=col, lw=1.55, solid_capstyle="round", zorder=3)
        if hollow:
            ax_4d.plot(pt, yp, marker=mk, ms=5.0, markerfacecolor="white", markeredgecolor=col, markeredgewidth=1.25, zorder=5)
        else:
            ax_4d.plot(pt, yp, marker=mk, ms=5.2, color=col, markeredgecolor="white", markeredgewidth=0.5, zorder=5)
        # Clean value label directly beside CI
        if yp == 0:
            ax_4d.text(lo - 0.06, yp, f"{pt:.3f}", fontsize=6.4, fontweight="bold", color=SUB, ha="right", va="center")
        else:
            ax_4d.text(hi + 0.06, yp, f"{pt:.3f}", fontsize=6.4, fontweight="bold", color=col, ha="left", va="center")

    ax_4d.set_yticks(y_d)
    ax_4d.set_yticklabels([r[0] for r in d_rows], fontsize=6.55, color=INK)
    for tl in ax_4d.get_yticklabels():
        if "TER-" in tl.get_text():
            tl.set_fontweight("bold")
    ax_4d.set_xticks([0.5, 1.0, 1.5, 2.0])
    ax_4d.set_xlabel("Ternary MAE (95% recipe CI)", fontsize=6.9, fontweight="bold", color=INK, labelpad=2.0)
    style_clean_axes(ax_4d)
    ax_4d.patch.set_alpha(0.0)

    # Rule 6 mandatory callout below Panel 4d: TER-direct − TER-graph = −0.022 [−0.128, 0.081]
    add_rbox(ax_hdr, 120.8, 196.2, 55.2, 6.8, fc=BLUE_LT, ec=SLATE_BLUE, lw=0.65, r=1.4, zorder=3)
    ax_hdr.text(
        148.4, 199.6,
        f"TER-direct − TER-graph = −{abs(dg_pt):.3f} [−{abs(dg_lo):.3f}, +{dg_hi:.3f}]",
        fontsize=6.15, fontweight="bold", color=CHARCOAL, ha="center", va="center", zorder=4
    )

    svg_path = OUT_DIR / "PsyMix_Figure4.svg"
    pdf_path = OUT_DIR / "PsyMix_Figure4.pdf"
    fig.savefig(svg_path, format="svg")
    fig.savefig(pdf_path, format="pdf")
    plt.close(fig)
    print("Saved Figure 4:", pdf_path)


if __name__ == "__main__":
    build_figure3()
    build_figure4()
