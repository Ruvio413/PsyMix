# -*- coding: utf-8 -*-
"""
PsyMix Batch 3 Publication Figures — Figure 5 (a-d) & Figure 6 (a-e)
  - PsyMix Figure 5 (a-d): 180.0 mm x 202.0 mm (Task T5: Cross-crop calibration & @1 geometric hazards)
  - PsyMix Figure 6 (a-e): 180.0 mm x 228.0 mm (Tasks T6a & T6b: Real food perturbations & mechanistic limits)

Strict compliance with Batch 1-2 VISUAL_SPEC.md & User's 6 Batch 3 Hard Rules:
  1. Fig. 5a: Log-scale RMSE axis (2, 4, 8, 16, 32); single ○ NULL-mean@10 overlay per row;
     curved arcs / vertical stagger for non-monotonic foldback routes (Apple->Tomato & Tomato->Apple sour).
  2. Fig. 5c: Exact real yogurt sugar concentrations (CH1 38.02->144.62, YFL 41.84->148.29, AGY 167.57 g/L)
     on true 0-10 yogurt sweetness scale (never 0-18!), highlighting grape extrapolated zero crossing -b/a = 34.613 g/L.
  3. Fig. 5d: Single unified geometric ray plot H(c) vs Bitterness (0-15) mirroring 5c, contrasting Valencia V09
     near-floor anchor (I0=1.31, H0=0.0352, alpha=37.17 -> pred 14.30) vs Hamlin H10 (alpha=24.44) with overshoot
     bracket (+12.99 vs obs +4.65) and 2-juice MAE box (DOT-log@1=5.451, NULL-zero=3.565, SUM-mass@1=7.746).
  4. Fig. 6: Golden asymmetric grid (180 x 228 mm; top ~[1.55, 1.0], bottom ~[1.18, 0.74, 1.18]);
     6b lower draws ONE shared flavonol-3-O-rutinoside vector scaffold from RDKit with highlighted 3'-R substituent
     and live-verified RDKit Morgan(r=2, 2048-bit) Tanimoto = 0.875 (217x cliff, Scharbert et al., 2004).
  5. Fig. 6a: Vertical dodge (+/-0.16 row height) separating Unweighted / equal-threshold baseline (Table 3) from
     Potency / multi-marker model; explicit SUM-count micro-tag on Darjeeling astringent full; 10,000-draw CIs ONLY
     for n >= 6 (hollow markers with no CIs for n = 2-3); right-edge overflow arrow -> 2.17 (no '//' broken axis).
  6. Global continuity: Batch 1-2 palette, Arial + DejaVuSans mathtext, pdf.fonttype=42, lowercase bold a-d / a-e,
     zero truncated-zero bars, zero engineering codes (RIDGE-conc-fixed8 mapped cleanly to RIDGE-conc).
"""
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.lines import Line2D

# Add RDKit path for live Morgan Tanimoto verification
sys.path.insert(0, r"C:\zhuang_taste_pre\psymix_foodatlas_multimodel_20261121\.deps\uni_mordred")
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

# Strict VISUAL_SPEC.md §1.1 configuration (identical to accepted Batch 1 & Batch 2)
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "svg.hashsalt": "PsyMix-Batch3-Final",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 7.5,
    "axes.labelsize": 7.9,
    "axes.titlesize": 8.3,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.1,
    "legend.fontsize": 6.6,
    "axes.unicode_minus": True,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

OUT_DIR = Path(r"c:\gemi-antigravity\00_PsyMix_Submission_Master\02_Figures_Design_and_Drafts\final_figures\Batch3_Fig5_Fig6")
SPEC_DIR = Path(r"c:\gemi-antigravity\00_PsyMix_Submission_Master\02_Figures_Design_and_Drafts\specimens")
MOL_DIR = OUT_DIR / "molecules"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Unified Colorblind-Safe Palette (aligned 100% with Figures 1-4)
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
    ax.tick_params(colors=INK, width=0.6, length=2.4, pad=2.0)


def draw_rdkit_path_svg_mm(ax_mm, svg_path, meta_path, x_mm, y_mm, w_mm, h_mm, bond_lw=0.72, highlight_atom="atom-18", zorder=4):
    """
    Render RDKit MolDraw2DSVG (with M/L/Q/Z vector paths, including crisp glyph paths) natively inside
    ax_mm (figure-mm coords, y downward), highlighting the designated 3'-R substituent in crimson + halo!
    """
    root = ET.parse(svg_path).getroot()
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))

    # Parse all SVG <path> elements with M, L, Q, Z commands
    parsed = []
    all_x, all_y = [], []
    for el in root.iter():
        if not el.tag.endswith("path"):
            continue
        cls = el.get("class") or ""
        style = el.get("style") or ""
        fill_attr = el.get("fill")
        d = el.get("d") or ""
        if not d.strip():
            continue
        tokens = re.findall(r"[MLQZ]|[-+]?\d*\.?\d+", d)
        pts, codes = [], []
        i = 0
        start_pt = None
        while i < len(tokens):
            t = tokens[i]
            if t == "M":
                px, py = float(tokens[i + 1]), float(tokens[i + 2])
                pts.append((px, py)); codes.append(MplPath.MOVETO)
                start_pt = (px, py)
                all_x.append(px); all_y.append(py)
                i += 3
            elif t == "L":
                px, py = float(tokens[i + 1]), float(tokens[i + 2])
                pts.append((px, py)); codes.append(MplPath.LINETO)
                all_x.append(px); all_y.append(py)
                i += 3
            elif t == "Q":
                cx1, cy1 = float(tokens[i + 1]), float(tokens[i + 2])
                px, py = float(tokens[i + 3]), float(tokens[i + 4])
                pts.extend([(cx1, cy1), (px, py)])
                codes.extend([MplPath.CURVE3, MplPath.CURVE3])
                all_x.extend([cx1, px]); all_y.extend([cy1, py])
                i += 5
            elif t == "Z":
                if start_pt is not None:
                    pts.append(start_pt); codes.append(MplPath.CLOSEPOLY)
                i += 1
            else:
                i += 1
        if pts:
            is_filled = (fill_attr is not None and fill_attr != "none") or ("fill:#" in style and "fill:none" not in style)
            is_dashed = "stroke-dasharray" in style
            is_hi = highlight_atom in cls
            parsed.append((pts, codes, is_filled, is_dashed, is_hi))

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    span_x = max(max_x - min_x, 1e-3)
    span_y = max(max_y - min_y, 1e-3)
    scale = min(w_mm / span_x, h_mm / span_y)
    ox = x_mm + 0.5 * (w_mm - span_x * scale) - min_x * scale
    oy = y_mm + 0.5 * (h_mm - span_y * scale) - min_y * scale

    # Draw highlight circle centered directly on R substituent!
    rx_mm = ox + meta["r_draw_x"] * scale
    ry_mm = oy + meta["r_draw_y"] * scale
    ax_mm.add_patch(Circle((rx_mm, ry_mm), 2.55, facecolor=CRIMSON_LT, edgecolor=CRIMSON, linewidth=0.95, zorder=zorder))

    for pts, codes, is_filled, is_dashed, is_hi in parsed:
        tpts = [(ox + px * scale, oy + py * scale) for px, py in pts]
        col = CRIMSON if is_hi else INK
        path = MplPath(tpts, codes)
        if is_filled:
            ax_mm.add_patch(PathPatch(path, facecolor=col, edgecolor="none", zorder=zorder + (2 if is_hi else 1)))
        else:
            ax_mm.add_patch(PathPatch(
                path, facecolor="none", edgecolor=col,
                linewidth=bond_lw * (1.35 if is_hi else 1.0),
                linestyle="--" if is_dashed else "-",
                capstyle="round", joinstyle="round",
                zorder=zorder + (2 if is_hi else 1)
            ))
    return rx_mm, ry_mm


# =========================================================================
# FIGURE 5: 180.0 mm x 202.0 mm (Task T5 — Target-Scale Calibration Geometry)
# =========================================================================
def build_figure5():
    W, H = 180.0, 202.0
    fig = plt.figure(figsize=(W / 25.4, H / 25.4), dpi=600)

    ax_bg = fig.add_axes([0, 0, 1, 1], xlim=(0, W), ylim=(H, 0), zorder=0)
    ax_bg.set_axis_off()
    ax_bg.set_gid("Canvas_Background")
    ax_bg.add_patch(Rectangle((0, 0), W, H, facecolor="white", edgecolor="none", zorder=0))
    ax_bg.plot([4.0, W - 4.0], [105.5, 105.5], color=HAIR, lw=0.85, zorder=1)
    ax_bg.plot([97.5, 97.5], [4.5, 102.5], color=HAIR, lw=0.75, zorder=1)
    ax_bg.plot([91.0, 91.0], [108.5, 198.5], color=HAIR, lw=0.75, zorder=1)

    ax_hdr = fig.add_axes([0, 0, 1, 1], xlim=(0, W), ylim=(H, 0), zorder=10)
    ax_hdr.set_axis_off()
    ax_hdr.patch.set_alpha(0.0)
    ax_hdr.set_gid("Panel_Headers")

    # Load frozen T5 anchor baselines CSV
    t5 = pd.read_csv(r"C:\zhuang_taste_pre\psymix_A5_null_baseline_audit_20261124\PsyMix_T5_Anchor_Only_Baselines.csv")

    routes = [
        ("apple_blueberry_sweetness_intensity", "Apple → Blueb.", "sweet (n=219)", 0.0, False),
        ("apple_tomato_sweetness_intensity", "Apple → Tomato", "sweet (n=209)", 1.02, False),
        ("blueberry_apple_sweetness_intensity", "Blueb. → Apple", "sweet (n=40)", 2.04, False),
        ("tomato_apple_sweetness_intensity", "Tomato → Apple", "sweet (n=40)", 3.06, False),
        ("apple_tomato_sourness_intensity", "Apple → Tomato", "sour (n=209)", 4.62, True),
        ("tomato_apple_sourness_intensity", "Tomato → Apple", "sour (n=40)", 5.64, True),
    ]

    # ---------------------------------------------------------------------
    # PANEL 5a (Top left, wider: x = 4.0 .. 95.5 mm) — Log-scale RMSE trajectories!
    # ---------------------------------------------------------------------
    ax_hdr.text(4.0, 6.6, "a", fontsize=11.0, fontweight="bold", color=CHARCOAL, ha="left", va="center")
    ax_hdr.text(9.0, 6.6, "Cross-crop RMSE across @0 → @5 → @10 (T5)", fontsize=8.3, fontweight="bold", color=CHARCOAL, ha="left", va="center")

    x_5a_L, x_5a_W = 29.8, 53.2
    y_5_T, y_5_H = 16.5, 65.5
    ax_5a = fig.add_axes([x_5a_L / W, 1.0 - (y_5_T + y_5_H) / H, x_5a_W / W, y_5_H / H], zorder=3)
    ax_5a.set_gid("Panel_5a_CrossCropLogRMSE")

    y_min_5, y_max_5 = -0.62, 6.26
    ax_5a.set_ylim(y_max_5, y_min_5)
    ax_5a.set_xscale("log")
    ax_5a.set_xlim(2.4, 36.0)

    def y5_to_mm(yd):
        return y_5_T + (yd - y_min_5) / (y_max_5 - y_min_5) * y_5_H

    # Group stripes & headers
    y_sw0, y_sw1 = y5_to_mm(-0.50), y5_to_mm(3.56)
    y_so0, y_so1 = y5_to_mm(4.12), y5_to_mm(6.14)
    ax_bg.add_patch(Rectangle((4.0, y_sw0), 91.2, y_sw1 - y_sw0, facecolor="#F7FAFD", edgecolor="none", zorder=1))
    ax_bg.add_patch(Rectangle((4.0, y_so0), 91.2, y_so1 - y_so0, facecolor="#FAF8FC", edgecolor="none", zorder=1))
    ax_5a.axhline(3.84, color=LINE, lw=0.75, ls="--", zorder=2)

    ax_hdr.text(56.4, 13.4, "Target RMSE (log scale)", fontsize=6.7, fontweight="bold", color=CHARCOAL, ha="center", va="center")
    ax_hdr.text(94.2, 13.4, "@0 ρ", fontsize=6.7, fontweight="bold", color=BLUE, ha="right", va="center")
    ax_bg.plot([85.0, 94.8], [15.1, 15.1], color=LINE, lw=0.6, zorder=2)
    ax_5a.text(2.55, -0.46, "Sweetness (4 routes)", fontsize=6.0, fontweight="bold", color=BLUE, ha="left", va="top")
    ax_5a.text(2.55, 3.98, "Sourness (2 routes)", fontsize=6.0, fontweight="bold", color=PURPLE, ha="left", va="top")

    ax_5a.grid(axis="x", which="major", color=HAIR, lw=0.55, zorder=0)

    for route, disp, sub_lbl, yp, is_foldback in routes:
        r0 = t5[(t5["route"] == route) & (t5["anchors"] == 0) & (t5["contrast"] == "source_only")].iloc[0]
        r5 = t5[(t5["route"] == route) & (t5["anchors"] == 5) & (t5["contrast"] == "same_budget")].iloc[0]
        r10 = t5[(t5["route"] == route) & (t5["anchors"] == 10) & (t5["contrast"] == "same_budget")].iloc[0]
        rn10 = t5[(t5["route"] == route) & (t5["anchors"] == 10) & (t5["contrast"] == "anchor_only")].iloc[0]

        v0, v5, v10, vn10 = float(r0["rmse"]), float(r5["rmse"]), float(r10["rmse"]), float(rn10["rmse"])
        rho0 = float(r0["rho_sample_oof"])

        if route == "apple_blueberry_sweetness_intensity":
            assert abs(v0 - 22.602) < 0.002 and abs(v5 - 4.711) < 0.002 and abs(v10 - 4.098) < 0.002 and abs(rho0 - 0.466) < 0.002
        if route == "apple_tomato_sweetness_intensity":
            assert abs(v0 - 24.702) < 0.002 and abs(v10 - 3.543) < 0.002 and abs(vn10 - 4.888) < 0.002

        if is_foldback:
            # Rule 1 [P0]: Vertical dodge + clean curved/staggered trajectory for non-monotonic foldback routes!
            # @0 (upper) -> @5 (mid-right) -> @10 (lower) so 3.86 and 3.87 never collide!
            y_0, y_5, y_10 = yp - 0.16, yp, yp + 0.16
            ax_5a.annotate("", xy=(v5, y_5), xytext=(v0, y_0),
                           arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.18", color=AMBER, lw=0.95, connectionstyle="arc3,rad=-0.18"), zorder=3)
            ax_5a.annotate("", xy=(v10, y_10), xytext=(v5, y_5),
                           arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.18", color=TEAL, lw=0.95, connectionstyle="arc3,rad=-0.18"), zorder=3)
            ax_5a.plot(v0, y_0, marker="s", ms=4.5, markerfacecolor="white", markeredgecolor=AMBER, markeredgewidth=1.25, zorder=5)
            ax_5a.plot(v5, y_5, marker="o", ms=4.4, color=BLUE, markeredgecolor="white", markeredgewidth=0.45, zorder=5)
            ax_5a.plot(v10, y_10, marker="D", ms=4.4, color=TEAL, markeredgecolor="white", markeredgewidth=0.45, zorder=6)
            ax_5a.plot(vn10, yp, marker="o", ms=4.8, markerfacecolor="white", markeredgecolor=CHARCOAL, markeredgewidth=1.15, zorder=4)
        else:
            # Monotonic calibration drop @0 -> @5 -> @10
            ax_5a.plot([v10, v0], [yp, yp], color=SLATE_BLUE, lw=1.25, zorder=2)
            ax_5a.annotate("", xy=(v5 * 1.07, yp), xytext=(v0 * 0.93, yp),
                           arrowprops=dict(arrowstyle="-|>,head_length=0.38,head_width=0.20", color=SLATE_BLUE, lw=1.0), zorder=3)
            ax_5a.plot(v0, yp, marker="s", ms=4.6, markerfacecolor="white", markeredgecolor=AMBER, markeredgewidth=1.25, zorder=5)
            ax_5a.plot(v5, yp, marker="o", ms=4.6, color=BLUE, markeredgecolor="white", markeredgewidth=0.45, zorder=5)
            ax_5a.plot(v10, yp, marker="D", ms=4.6, color=TEAL, markeredgecolor="white", markeredgewidth=0.45, zorder=6)
            # Single ○ NULL-mean@10 overlay (Rule 1!)
            ax_5a.plot(vn10, yp + 0.19 if abs(np.log2(vn10 / v10)) < 0.14 else yp,
                       marker="o", ms=4.6, markerfacecolor="white", markeredgecolor=CHARCOAL, markeredgewidth=1.15, zorder=7)

        # Left two-line route label cleanly outside spine
        y_mm = y5_to_mm(yp)
        ax_hdr.text(28.6, y_mm - 1.4, disp, fontsize=6.65, fontweight="bold", color=INK, ha="right", va="center")
        ax_hdr.text(28.6, y_mm + 1.8, sub_lbl, fontsize=5.85, color=MUTED, ha="right", va="center")

        # Right aligned @0 Spearman rho column
        ax_hdr.text(94.2, y_mm, f"{rho0:.3f}", fontsize=6.65, fontweight="bold" if rho0 >= 0.44 else "normal",
                    color=BLUE if rho0 >= 0.44 else MUTED, ha="right", va="center")

    ax_5a.set_yticks([])
    ax_5a.set_xticks([2, 4, 8, 16, 32])
    ax_5a.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax_5a.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax_5a.set_xticklabels(["2", "4", "8", "16", "32"], fontsize=6.9)
    ax_5a.set_xlabel(r"Cross-crop RMSE ($\log_2$ spacing; native scale)", fontsize=7.1, fontweight="bold", color=INK, labelpad=2.5)
    style_clean_axes(ax_5a)
    ax_5a.patch.set_alpha(0.0)

    leg_5a = [
        Line2D([0], [0], marker="s", color="none", ms=4.5, markerfacecolor="white", markeredgecolor=AMBER, markeredgewidth=1.25, label="SUM-mass@0 (source)"),
        Line2D([0], [0], marker="o", color=BLUE, lw=0, ms=4.6, markeredgecolor="white", markeredgewidth=0.4, label="SUM-mass@5"),
        Line2D([0], [0], marker="D", color=TEAL, lw=0, ms=4.5, markeredgecolor="white", markeredgewidth=0.4, label="SUM-mass@10"),
        Line2D([0], [0], marker="o", color="none", ms=4.5, markerfacecolor="white", markeredgecolor=CHARCOAL, markeredgewidth=1.15, label="NULL-mean@10 (anchor only)"),
    ]
    ax_5a.legend(
        handles=leg_5a, loc="upper center", bbox_to_anchor=(0.48, -0.155),
        ncol=2, frameon=False, fontsize=6.15, columnspacing=0.95, handletextpad=0.5, handlelength=1.1, labelspacing=0.35
    )

    # ---------------------------------------------------------------------
    # PANEL 5b (Top right: x = 99.5 .. 176.0 mm) — Unified %ΔRMSE vs NULL-mean (A5)
    # ---------------------------------------------------------------------
    ax_hdr.text(99.5, 6.6, "b", fontsize=11.0, fontweight="bold", color=CHARCOAL, ha="left", va="center")
    ax_hdr.text(104.5, 6.6, "Net chemistry gain vs NULL-mean (A5)", fontsize=8.3, fontweight="bold", color=CHARCOAL, ha="left", va="center")

    x_5b_L, x_5b_W = 103.5, 45.0
    ax_5b = fig.add_axes([x_5b_L / W, 1.0 - (y_5_T + y_5_H) / H, x_5b_W / W, y_5_H / H], zorder=3)
    ax_5b.set_gid("Panel_5b_AnchorBaselineContrast")

    ax_5b.set_ylim(y_max_5, y_min_5)
    ax_5b.set_xlim(-38.0, 48.0)

    ax_bg.add_patch(Rectangle((99.5, y_sw0), 76.8, y_sw1 - y_sw0, facecolor="#F7FAFD", edgecolor="none", zorder=1))
    ax_bg.add_patch(Rectangle((99.5, y_so0), 76.8, y_so1 - y_so0, facecolor="#FAF8FC", edgecolor="none", zorder=1))
    ax_5b.axhline(3.84, color=LINE, lw=0.75, ls="--", zorder=2)
    ax_5b.axvline(0.0, color=CHARCOAL, lw=0.85, ls="--", zorder=2)
    ax_5b.grid(axis="x", color=HAIR, lw=0.5, zorder=0)

    ax_hdr.text(126.0, 13.4, "%ΔRMSE (SUM − NULL)", fontsize=6.6, fontweight="bold", color=CHARCOAL, ha="center", va="center")
    ax_hdr.text(176.0, 10.6, "Native ΔRMSE@10", fontsize=6.5, fontweight="bold", color=CHARCOAL, ha="right", va="center")
    ax_hdr.text(176.0, 13.4, "point [95% CI]", fontsize=6.15, color=MUTED, ha="right", va="center")
    ax_bg.plot([151.0, 176.2], [15.1, 15.1], color=LINE, lw=0.6, zorder=2)

    for route, disp, sub_lbl, yp, _ in routes:
        r5 = t5[(t5["route"] == route) & (t5["anchors"] == 5) & (t5["contrast"] == "same_budget")].iloc[0]
        r10 = t5[(t5["route"] == route) & (t5["anchors"] == 10) & (t5["contrast"] == "same_budget")].iloc[0]

        for row_k, dy_k, col_k, mk_k in [(r5, -0.17, BLUE, "o"), (r10, +0.17, TEAL, "D")]:
            ref_r = float(row_k["ref_rmse"])
            d_pt = float(row_k["delta_rmse"])
            d_lo = float(row_k["delta_rmse_lo"])
            d_hi = float(row_k["delta_rmse_hi"])
            pct = 100.0 * d_pt / ref_r
            plo = 100.0 * d_lo / ref_r
            phi = 100.0 * d_hi / ref_r

            # Highlight significant worse-than-NULL in AMBER
            c_use = AMBER if d_lo > 0.0 else col_k
            ax_5b.plot([plo, phi], [yp + dy_k, yp + dy_k], color=c_use, lw=1.35, solid_capstyle="round", zorder=3)
            ax_5b.plot(pct, yp + dy_k, marker=mk_k, ms=4.3, color=c_use, markeredgecolor="white", markeredgewidth=0.45, zorder=5)

        # Right two-line Native ΔRMSE@10 [95% CI]
        d10 = float(r10["delta_rmse"])
        lo10 = float(r10["delta_rmse_lo"])
        hi10 = float(r10["delta_rmse_hi"])
        if route == "apple_tomato_sweetness_intensity":
            assert abs(d10 - (-1.345)) < 0.002
        if route == "apple_blueberry_sweetness_intensity":
            assert abs(d10 - (-0.235)) < 0.002

        sgn = "+" if d10 > 0 else "−"
        slo = "+" if lo10 > 0 else "−"
        shi = "+" if hi10 > 0 else "−"
        col_txt = TEAL if hi10 < 0 else (AMBER if lo10 > 0 else SUB)
        y_mm = y5_to_mm(yp)
        ax_hdr.text(176.0, y_mm - 1.4, f"{sgn}{abs(d10):.3f}", fontsize=6.6, fontweight="bold" if col_txt != SUB else "normal", color=col_txt, ha="right", va="center")
        ax_hdr.text(176.0, y_mm + 1.8, f"[{slo}{abs(lo10):.2f}, {shi}{abs(hi10):.2f}]", fontsize=5.8, color=MUTED, ha="right", va="center")

    ax_5b.text(-3.5, 6.08, "← Beats NULL", fontsize=5.75, fontweight="bold", color=TEAL, ha="right", va="center")
    ax_5b.text(+3.5, 6.08, "Worse →", fontsize=5.75, fontweight="bold", color=AMBER, ha="left", va="center")
    ax_5b.set_yticks([])
    ax_5b.set_xticks([-30, -15, 0, 15, 30])
    ax_5b.set_xticklabels(["−30%", "−15%", "0%", "+15%", "+30%"], fontsize=6.7)
    ax_5b.set_xlabel("%ΔRMSE vs matched anchor-only NULL", fontsize=7.0, fontweight="bold", color=INK, labelpad=2.5)
    style_clean_axes(ax_5b)
    ax_5b.patch.set_alpha(0.0)

    leg_5b = [
        Line2D([0], [0], marker="o", color=BLUE, lw=1.2, ms=4.3, markeredgecolor="white", markeredgewidth=0.4, label="@5 anchors"),
        Line2D([0], [0], marker="D", color=TEAL, lw=1.2, ms=4.2, markeredgecolor="white", markeredgewidth=0.4, label="@10 anchors"),
        Line2D([0], [0], marker="o", color=AMBER, lw=1.2, ms=4.2, markeredgecolor="white", markeredgewidth=0.4, label="CI > 0 (worse)"),
    ]
    ax_5b.legend(
        handles=leg_5b, loc="upper center", bbox_to_anchor=(0.50, -0.155),
        ncol=3, frameon=False, fontsize=5.95, columnspacing=0.65, handletextpad=0.35, handlelength=1.1
    )
    ax_hdr.text(99.5, 102.2, "At @10, DOT-log vs DOT-lin 95% CIs span 0 across all 6 routes (T2 collinearity).", fontsize=5.8, color=MUTED, ha="left", va="center")

    # ---------------------------------------------------------------------
    # PANEL 5c (Bottom left: x = 4.0 .. 89.0 mm) — Grape -> Yogurt @1 geometry (TRUE 0-10 SCALE!)
    # ---------------------------------------------------------------------
    ax_hdr.text(4.0, 110.8, "c", fontsize=11.0, fontweight="bold", color=CHARCOAL, ha="left", va="center")
    ax_hdr.text(9.0, 110.8, "Retained zero crossing under @1 transfer (T5)", fontsize=8.2, fontweight="bold", color=CHARCOAL, ha="left", va="center")

    # Load exact frozen yogurt states directly from CSV!
    ydf = pd.read_csv(r"C:\zhuang_taste_pre\psymix_foodatlas_interpretation_20261121\yogurt\recipe_predictions_and_changes.csv")
    yg_st = ydf[(ydf["model"] == "GRAPE_TOTAL_MASS") & (ydf["panel"] == "four_measured") & (ydf["role"].isin(["state", "anchor"]))].set_index("state1")

    S_ch1, I_ch1 = float(yg_st.loc["CH1", "total_four_1_g_L"]), float(yg_st.loc["CH1", "observed_1"])
    S_ch1p, I_ch1p = float(yg_st.loc["CH1+", "total_four_1_g_L"]), float(yg_st.loc["CH1+", "observed_1"])
    S_yfl, I_yfl = float(yg_st.loc["YFL", "total_four_1_g_L"]), float(yg_st.loc["YFL", "observed_1"])
    S_yflp, I_yflp = float(yg_st.loc["YFL+", "total_four_1_g_L"]), float(yg_st.loc["YFL+", "observed_1"])
    S_agy, I_agy = float(yg_st.loc["AGY", "total_four_1_g_L"]), float(yg_st.loc["AGY", "observed_1"])

    # Verify exact frozen values & true 0-10 scale!
    assert abs(S_agy - 167.569) < 0.01 and abs(I_agy - 9.979) < 0.01
    assert abs(S_ch1 - 38.023) < 0.01 and abs(I_ch1 - 0.0) < 1e-6 and abs(I_ch1p - 8.160) < 0.01
    assert abs(S_yfl - 41.836) < 0.01 and abs(I_yflp - 8.805) < 0.01

    a_grp, b_grp = 0.03279676362, -1.135178698
    x_zero = -b_grp / a_grp  # 34.6125 g/L
    assert abs(x_zero - 34.613) < 0.002

    ax_5c = fig.add_axes([14.8 / W, 1.0 - 183.0 / H, 72.5 / W, 63.0 / H], zorder=3)
    ax_5c.set_gid("Panel_5c_YogurtZeroCrossing")

    ax_5c.set_xlim(0.0, 272.0)
    ax_5c.set_ylim(-0.6, 11.4)

    # Observed grape sugar range [109.845, 264.337] g L^-1 (unified g L^-1 units throughout 5c!)
    ax_5c.axvspan(109.845, 264.337, facecolor=BLUE_LT, alpha=0.50, zorder=0)
    ax_5c.text(261.0, 0.35, "Observed grape sugar\n" + r"109.8–264.3 $\mathrm{g\ L}^{-1}$", fontsize=5.6, color=BLUE, ha="right", va="bottom", zorder=3)

    # Scale ceiling hairline at 10 (true 0-10 yogurt scale!)
    ax_5c.axhline(10.0, color=LINE, lw=0.65, ls=":", zorder=1)
    ax_5c.axhline(0.0, color=LINE, lw=0.65, ls="-", zorder=1)

    # Extrapolated zero crossing x_zero = 34.613 g L^-1: crisp small bottom triangle clear of CH1 dot!
    ax_5c.axvline(x_zero, color=CRIMSON, lw=0.85, ls=":", zorder=2)
    ax_5c.plot(x_zero, -0.22, marker="^", ms=4.8, color=CRIMSON, markeredgecolor="white", markeredgewidth=0.4, zorder=6)

    # Calibrated rays terminating cleanly at AGY (167.569, 9.979) so nothing crosses the AGY label!
    xs_prop = np.linspace(0.0, S_agy, 100)
    ys_prop = I_agy * xs_prop / S_agy
    xs_aff = np.linspace(x_zero, S_agy, 100)
    ys_aff = I_agy * (xs_aff - x_zero) / (S_agy - x_zero)

    ax_5c.plot(xs_prop, ys_prop, color=GOLD, lw=1.65, ls="--", label="SUM-mass@1 (origin)", zorder=3)
    ax_5c.plot(xs_aff, ys_aff, color=BLUE, lw=1.85, ls="-", label="SUM-mass-source-affine@1", zorder=4)

    # Independent AGY anchor star + unobstructed right-aligned label
    ax_5c.plot(S_agy, I_agy, marker="*", ms=9.5, color=CRIMSON, markeredgecolor="white", markeredgewidth=0.5, zorder=7)
    ax_5c.text(S_agy + 7.5, I_agy - 0.05, "AGY anchor\n(167.6, 9.98)", fontsize=5.85, fontweight="bold", color=CRIMSON, ha="left", va="center", zorder=7)

    # Real observed yogurt states (slightly smaller CH1/YFL dots so cleanly separated from x_zero triangle!)
    ax_5c.scatter([S_ch1, S_yfl], [I_ch1, I_yfl], s=19, color=TEAL, edgecolors="white", linewidths=0.5, zorder=6)
    ax_5c.scatter([S_ch1p, S_yflp], [I_ch1p, I_yflp], s=25, color=TEAL, edgecolors="white", linewidths=0.5, zorder=6)

    # Clean vertical ΔÎ comparison brackets at CH1 -> CH1+ (S0=38.0 -> S1=144.6)
    y_prop_0 = I_agy * S_ch1 / S_agy
    y_prop_1 = I_agy * S_ch1p / S_agy
    y_aff_0 = I_agy * (S_ch1 - x_zero) / (S_agy - x_zero)
    y_aff_1 = I_agy * (S_ch1p - x_zero) / (S_agy - x_zero)
    assert abs((y_prop_1 - y_prop_0) - 6.348) < 0.005 and abs((y_aff_1 - y_aff_0) - 8.000) < 0.005

    # Vertical dashed drop lines at CH1 and CH1+ showing ΔS = +106.6 g L^-1
    ax_5c.plot([S_ch1, S_ch1], [0.0, y_prop_0], color=GOLD, lw=0.9, ls=":", zorder=2)
    ax_5c.plot([S_ch1p, S_ch1p], [0.0, I_ch1p], color=TEAL, lw=0.85, ls=":", alpha=0.7, zorder=2)

    # Labels for states & extrapolated crossing (unified g L^-1 units & 5.5 mm clear of gold dashed ray!)
    ax_5c.annotate(
        r"$-b/a = 34.6\ \mathrm{g\ L}^{-1}$" + "\n(extrap. zero)",
        xy=(x_zero + 1.5, 7.25), xytext=(43.0, 7.25),
        fontsize=5.35, fontweight="bold", color=CRIMSON, ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor=CRIMSON, lw=0.55, alpha=0.95),
        arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.18", color=CRIMSON, lw=0.75),
        zorder=8
    )
    ax_5c.text(S_yfl + 5.5, 0.30, "CH1 / YFL\n" + r"(38–42 $\mathrm{g\ L}^{-1}$, I=0)", fontsize=5.35, color=TEAL, fontweight="bold", ha="left", va="bottom", zorder=7)
    ax_5c.annotate(
        "CH1+ / YFL+\nobs: +8.16 / +8.81",
        xy=(S_ch1p - 1.5, I_ch1p), xytext=(55.0, 9.65),
        fontsize=5.55, fontweight="bold", color=TEAL, ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor=TEAL, lw=0.5, alpha=0.92),
        arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.18", color=TEAL, lw=0.7),
        zorder=8
    )

    # Compact algebraic readout pill in open right area (S = 178..268, Y = 3.2..7.6)
    ax_5c.text(
        265.0, 5.1,
        r"$\Delta\hat{I} = I_A(S_1 - S_0)/(S_A + b/a)$" + "\n"
        "Affine@1: +8.00 / +7.99\n"
        "SUM@1:   +6.35 / +6.34",
        fontsize=5.6, color=CHARCOAL, ha="right", va="center",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor=LINE, lw=0.6, alpha=0.95), zorder=7
    )

    ax_5c.set_xticks([0, 50, 100, 150, 200, 250])
    ax_5c.set_yticks([0, 2, 4, 6, 8, 10])
    ax_5c.set_xlabel(r"Total sugar concentration $S$ ($\mathrm{g\ L}^{-1}$)", fontsize=7.1, fontweight="bold", color=INK, labelpad=2.2)
    ax_5c.set_ylabel("Yogurt sweetness rating (0–10)", fontsize=7.1, fontweight="bold", color=INK, labelpad=2.2)
    ax_5c.grid(color=HAIR, lw=0.45, zorder=0)
    style_clean_axes(ax_5c)

    ax_5c.legend(
        loc="upper center", bbox_to_anchor=(0.48, -0.175),
        ncol=2, frameon=False, fontsize=6.05, columnspacing=0.85, handletextpad=0.45, handlelength=1.5
    )

    # Delicate watercolor accents in completely open top-left margin of 5c
    place_image_mm(fig, W, H, load_cropped_rgba(SPEC_DIR / "grape.png"), 15.8, 121.0, 8.2, 8.8, gid="Specimen_5c_Grape", alpha=0.9, zorder=8)
    place_image_mm(fig, W, H, load_cropped_rgba(SPEC_DIR / "yogurt.png"), 77.5, 134.5, 8.5, 8.5, gid="Specimen_5c_Yogurt", alpha=0.9, zorder=8)

    # ---------------------------------------------------------------------
    # PANEL 5d (Bottom right: x = 93.0 .. 176.0 mm) — Single Unified Geometric Ray Plot (Rule 3 [P0]!)
    # ---------------------------------------------------------------------
    ax_hdr.text(93.0, 110.8, "d", fontsize=11.0, fontweight="bold", color=CHARCOAL, ha="left", va="center")
    ax_hdr.text(98.0, 110.8, "Near-floor @1 error amplification (orange juice, A6)", fontsize=8.2, fontweight="bold", color=CHARCOAL, ha="left", va="center")

    dea = pd.read_csv(r"C:\zhuang_taste_pre\foodatlas_intake_20261121\source_followup\dea_recipe_composition_ratings.csv")
    K_bit = 2.336466
    h10_nat = dea[(dea["matrix"] == "H10") & (dea["condition"] == "native")].iloc[0]
    h10_pair = dea[(dea["matrix"] == "H10") & (dea["condition"] == "pair")].iloc[0]
    v09_nat = dea[(dea["matrix"] == "V09") & (dea["condition"] == "native")].iloc[0]
    v09_pair = dea[(dea["matrix"] == "V09") & (dea["condition"] == "pair")].iloc[0]

    H_h10_0, I_h10_0 = np.log1p(float(h10_nat["load"]) / K_bit), float(h10_nat["bitter"])
    H_h10_1, I_h10_1 = np.log1p(float(h10_pair["load"]) / K_bit), float(h10_pair["bitter"])
    H_v09_0, I_v09_0 = np.log1p(float(v09_nat["load"]) / K_bit), float(v09_nat["bitter"])
    H_v09_1, I_v09_1 = np.log1p(float(v09_pair["load"]) / K_bit), float(v09_pair["bitter"])

    alpha_h10 = I_h10_0 / H_h10_0
    alpha_v09 = I_v09_0 / H_v09_0
    I_h10_hat = alpha_h10 * H_h10_1
    I_v09_hat = alpha_v09 * H_v09_1

    assert abs(H_v09_0 - 0.03524) < 0.0002 and abs(alpha_v09 - 37.172) < 0.02
    assert abs((I_v09_hat - I_v09_0) - 12.985) < 0.02 and abs((I_v09_1 - I_v09_0) - 4.65) < 0.01

    ax_5d = fig.add_axes([103.5 / W, 1.0 - 183.0 / H, 71.5 / W, 63.0 / H], zorder=3)
    ax_5d.set_gid("Panel_5d_NearFloorRay")

    ax_5d.set_xlim(0.0, 0.435)
    ax_5d.set_ylim(0.0, 15.6)

    # Subtle near-floor anchor band at bottom (Y = 0 .. 2.0)
    ax_5d.axhspan(0.0, 2.0, facecolor=CRIMSON_LT, alpha=0.65, zorder=0)
    ax_5d.axhline(15.0, color=LINE, lw=0.65, ls=":", zorder=1)
    ax_5d.text(0.428, 0.28, "Near-floor anchor zone", fontsize=5.55, fontweight="bold", color=CRIMSON, ha="right", va="bottom", zorder=2)

    # Two proportional rays from (0, 0): V09 steep (alpha=37.17) vs H10 gentler (alpha=24.44)
    xh_v = np.linspace(0.0, H_v09_1, 80)
    xh_h = np.linspace(0.0, H_h10_1, 80)
    ax_5d.plot(xh_v, alpha_v09 * xh_v, color=AMBER, lw=1.9, ls="-", label="Valencia V09 ray (α = 37.17)", zorder=4)
    ax_5d.plot(xh_h, alpha_h10 * xh_h, color=BLUE, lw=1.6, ls="--", label="Hamlin H10 ray (α = 24.44)", zorder=3)

    # Native anchors (stars)
    ax_5d.plot(H_v09_0, I_v09_0, marker="*", ms=9.0, color=CRIMSON, markeredgecolor="white", markeredgewidth=0.45, zorder=7)
    ax_5d.plot(H_h10_0, I_h10_0, marker="*", ms=8.0, color=BLUE, markeredgecolor="white", markeredgewidth=0.45, zorder=7)

    # Predicted spiked points (open circles on rays) vs Observed spiked points (filled circles)
    ax_5d.plot(H_v09_1, I_v09_hat, marker="o", ms=5.5, markerfacecolor="white", markeredgecolor=AMBER, markeredgewidth=1.45, zorder=6)
    ax_5d.plot(H_v09_1, I_v09_1, marker="o", ms=5.5, color=AMBER, markeredgecolor="white", markeredgewidth=0.5, zorder=6)
    ax_5d.plot(H_h10_1, I_h10_hat, marker="s", ms=4.8, markerfacecolor="white", markeredgecolor=BLUE, markeredgewidth=1.35, zorder=6)
    ax_5d.plot(H_h10_1, I_h10_1, marker="s", ms=4.8, color=BLUE, markeredgecolor="white", markeredgewidth=0.5, zorder=6)

    # Bold vertical overshoot bracket on Valencia V09 (5.96 -> 14.30: pred +12.99 vs obs +4.65!)
    ax_5d.annotate(
        "", xy=(H_v09_1, I_v09_hat - 0.35), xytext=(H_v09_1, I_v09_1 + 0.35),
        arrowprops=dict(arrowstyle="<->,head_length=0.4,head_width=0.22", color=CRIMSON, lw=1.35), zorder=6
    )
    # Polish #1: narrow 3-line box centered cleanly between orange ray (x=0.29) and crimson arrow (x=0.385)!
    ax_5d.text(
        H_v09_1 - 0.014, 10.15,
        "V09 overshoot\npred +12.99\nvs obs +4.65",
        fontsize=5.25, fontweight="bold", color=CRIMSON, ha="right", va="center",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor=CRIMSON, lw=0.55, alpha=0.95), zorder=8
    )

    # Clean anchor & spiked observation callouts (exact H0=0.0352 matching §3.5 + H10 pred 8.09!)
    ax_5d.annotate(
        "V09 anchor\n" + r"($H_0$=0.0352, $I_0$=1.31)",
        xy=(H_v09_0 + 0.005, I_v09_0), xytext=(0.088, 1.15),
        fontsize=5.4, fontweight="bold", color=CRIMSON, ha="left", va="center",
        arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.18", color=CRIMSON, lw=0.7), zorder=7
    )
    ax_5d.annotate(
        "H10 anchor\n" + r"($I_0$=3.04)",
        xy=(H_h10_0 + 0.006, I_h10_0 - 0.08), xytext=(0.185, 2.75),
        fontsize=5.45, color=BLUE, fontweight="bold", ha="left", va="center",
        arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.18", color=BLUE, lw=0.7), zorder=7
    )
    ax_5d.text(H_v09_1 - 0.008, I_v09_hat + 0.45, "14.30", fontsize=5.8, fontweight="bold", color=AMBER, ha="right", va="bottom", zorder=7)
    ax_5d.text(H_h10_1 - 0.008, I_h10_hat + 0.42, "H10 pred (8.09)", fontsize=5.35, fontweight="bold", color=BLUE, ha="right", va="bottom", zorder=7)
    ax_5d.text(H_h10_1 - 0.012, I_h10_1, "H10 obs (5.52)", fontsize=5.35, fontweight="bold", color=BLUE, ha="right", va="center", zorder=7)
    ax_5d.text(H_v09_1 - 0.004, I_v09_1 - 0.82, "V09 obs (5.96)", fontsize=5.4, fontweight="bold", color=AMBER, ha="right", va="top", zorder=7)

    # Crisp inset 2-juice change MAE summary card in open top-left (H = 0.015..0.19, Y = 9.0..14.6)
    ax_5d.text(
        0.014, 14.7,
        "2-juice ΔMAE (n = 2):\n"
        "NULL-zero:   3.565\n"
        "DOT-log@1:   5.451\n"
        "SUM-mass@1: 7.746",
        fontsize=5.65, color=CHARCOAL, ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.22", facecolor=STRIPE_BG, edgecolor=LINE, lw=0.6, alpha=0.95), zorder=8
    )

    # Delicate orange watercolor corner accent tucked cleanly in lower-right open space above floor band!
    place_image_mm(fig, W, H, load_cropped_rgba(SPEC_DIR / "orange.png"), 163.2, 165.5, 9.8, 9.8, gid="Specimen_5d_Orange", alpha=0.9, zorder=7)

    ax_5d.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4])
    ax_5d.set_yticks([0, 3, 6, 9, 12, 15])
    ax_5d.set_xlabel(r"Unscaled log load $H(c) = \ln(1 + U/K)$", fontsize=7.1, fontweight="bold", color=INK, labelpad=2.2)
    ax_5d.set_ylabel("Bitterness rating (0–15 scale)", fontsize=7.1, fontweight="bold", color=INK, labelpad=2.2)
    ax_5d.grid(color=HAIR, lw=0.45, zorder=0)
    style_clean_axes(ax_5d)

    leg_5d = [
        Line2D([0], [0], color=AMBER, lw=1.8, ls="-", label="Valencia V09 (α = 37.17)"),
        Line2D([0], [0], color=BLUE, lw=1.5, ls="--", label="Hamlin H10 (α = 24.44)"),
        Line2D([0], [0], marker="o", color="none", ms=4.3, markerfacecolor="white", markeredgecolor=CHARCOAL, markeredgewidth=1.25, label="Hollow: DOT-log@1 pred"),
        Line2D([0], [0], marker="o", color=CHARCOAL, lw=0, ms=4.3, markeredgecolor="white", markeredgewidth=0.4, label="Filled: Observed"),
    ]
    ax_5d.legend(
        handles=leg_5d, loc="upper center", bbox_to_anchor=(0.48, -0.158),
        ncol=2, frameon=False, fontsize=5.75, columnspacing=0.75, handletextpad=0.42, handlelength=1.35, labelspacing=0.28
    )

    svg_path = OUT_DIR / "PsyMix_Figure5.svg"
    pdf_path = OUT_DIR / "PsyMix_Figure5.pdf"
    fig.savefig(svg_path, format="svg")
    fig.savefig(pdf_path, format="pdf")
    plt.close(fig)
    print("Saved Figure 5:", pdf_path)


# =========================================================================
# FIGURE 6: 180.0 mm x 228.0 mm (Tasks T6a & T6b — Golden Non-Uniform Grid!)
# =========================================================================
def build_figure6():
    W, H = 180.0, 228.0
    fig = plt.figure(figsize=(W / 25.4, H / 25.4), dpi=600)

    ax_bg = fig.add_axes([0, 0, 1, 1], xlim=(0, W), ylim=(H, 0), zorder=0)
    ax_bg.set_axis_off()
    ax_bg.set_gid("Canvas_Background")
    ax_bg.add_patch(Rectangle((0, 0), W, H, facecolor="white", edgecolor="none", zorder=0))

    # Golden grid dividers: Top ~[1.55, 1.0] (x=109.5), Bottom ~[1.18, 0.74, 1.18] (x=68.5, x=112.0), Tier y=126.5
    ax_bg.plot([4.0, W - 4.0], [126.5, 126.5], color=HAIR, lw=0.85, zorder=1)
    ax_bg.plot([109.5, 109.5], [4.5, 123.5], color=HAIR, lw=0.75, zorder=1)
    ax_bg.plot([68.5, 68.5], [129.5, 224.5], color=HAIR, lw=0.75, zorder=1)
    ax_bg.plot([112.0, 112.0], [129.5, 224.5], color=HAIR, lw=0.75, zorder=1)

    ax_hdr = fig.add_axes([0, 0, 1, 1], xlim=(0, W), ylim=(H, 0), zorder=10)
    ax_hdr.set_axis_off()
    ax_hdr.patch.set_alpha(0.0)
    ax_hdr.set_gid("Panel_Headers")

    # ---------------------------------------------------------------------
    # PANEL 6a (Top left: x = 4.0 .. 108.0 mm, ~1.55 ratio) — All 17 slices RER
    # ---------------------------------------------------------------------
    ax_hdr.text(4.0, 6.4, "a", fontsize=11.0, fontweight="bold", color=CHARCOAL, ha="left", va="center")
    ax_hdr.text(8.8, 6.4, "Relative error ratio across all 17 slices (T6a–T6b)", fontsize=8.2, fontweight="bold", color=CHARCOAL, ha="left", va="center")

    f6a = pd.read_csv(r"C:\zhuang_taste_pre\psymix_manuscript_release\Figures_round1_20261124\source_data\Figure6A.csv")
    # Clean engineering label RIDGE-conc-fixed8 -> RIDGE-conc!
    f6a["model"] = f6a["model"].replace({"RIDGE-conc-fixed8": "RIDGE-conc"})

    # Exact Table 3 order of all 17 archived slices: (study, scope, endpoint, label, base_model, cand_model, special_tag)
    slices_17 = [
        # Group 0: Omissions (T6a, 8 slices, n=6-8)
        ("Darjeeling2005", "full_threshold_coverage", "bitter", "Darj. bitter (full)", "SUM-molar@1", "DOT-log@1", None),
        ("Darjeeling2005", "structure_common", "bitter", "Darj. bitter (struct.)", "DOT-equal@1", "DOT-meas@1", None),
        ("Yu2014", "full_threshold_coverage", "bitter", "Green bitter (full)", "SUM-molar@1", "DOT-log@1", None),
        ("Yu2014", "structure_common", "bitter", "Green bitter (struct.)", "DOT-equal@1", "DOT-meas@1", None),
        ("Darjeeling2005", "full_threshold_coverage", "astringent", "Darj. astring. (full; SUM-count)", "SUM-count@1", "DOT-log@1", "SUM-count"),
        ("Darjeeling2005", "structure_common", "astringent", "Darj. astring. (struct.)", "DOT-equal@1", "DOT-meas@1", None),
        ("Yu2014", "full_threshold_coverage", "astringent", "Green astring. (full)", "SUM-molar@1", "DOT-log@1", None),
        ("Yu2014", "structure_common", "astringent", "Green astring. (struct.)", "DOT-equal@1", "DOT-meas@1", None),
        # Group 1: Addition / background (T6a, 3 slices, n=2)
        ("Dea", "two bitterants", "bitterness", "OJ + lim./nom. spike", "SUM-mass@1", "DOT-log@1", None),
        ("Dea", "sugar with fixed bitterants", "bitterness", "OJ + sucrose (fixed bit.)", "SUM-mass@1", "DOT-log@1", None),
        ("Dea", "acid with fixed bitterants", "bitterness", "OJ + acid (fixed bit.)", "SUM-mass@1", "DOT-log@1", None),
        # Group 2: Formulation (T6a, 1 slice, n=2)
        ("Ahmad", "sucrose + pectin", "sweetness", "Yogurt + suc. + pectin", "SUM-mass@1", "DOT-log@1", "affine"),
        # Group 3: Processing (T6b, 5 slices)
        ("McClure & Wiedemer", "roast versus raw", "bitter_natural_gLMS", "Cocoa bitter (natural)", "RIDGE-SUM-mass", "RIDGE-conc", None),
        ("McClure & Wiedemer", "roast versus raw", "bitter_noseclip_gLMS", "Cocoa bitter (nose-clip)", "RIDGE-SUM-mass", "RIDGE-conc", None),
        ("ancha", "external processing, one reference score", "bitter", "Ancha bitter (drying)", "SUM-molar@1", "DOT-log@1", None),
        ("ancha", "external processing, one reference score", "astringent", "Ancha astring. (drying)", "SUM-molar@1", "DOT-log@1", None),
        ("coffee", "external processing, one reference score", "bitter", "Coffee bitter (roast)*", "SUM-molar@1", "DOT-log@1", "prelock"),
    ]
    assert len(slices_17) == 17

    # Y positions with crisp gaps between the 4 perturbation groups
    y_6a = []
    cur = 0.0
    group_breaks = {8, 11, 12}
    for idx in range(17):
        if idx in group_breaks:
            cur += 0.55
        y_6a.append(cur)
        cur += 1.0

    x_6a_L, x_6a_W = 43.2, 57.0
    y_6a_T, y_6a_H = 14.5, 92.5
    ax_6a = fig.add_axes([x_6a_L / W, 1.0 - (y_6a_T + y_6a_H) / H, x_6a_W / W, y_6a_H / H], zorder=3)
    ax_6a.set_gid("Panel_6a_All17RER")

    y_min_6a, y_max_6a = -0.68, y_6a[-1] + 0.68
    ax_6a.set_ylim(y_max_6a, y_min_6a)
    ax_6a.set_xlim(0.0, 1.72)

    def y6a_to_mm(yd):
        return y_6a_T + (yd - y_min_6a) / (y_max_6a - y_min_6a) * y_6a_H

    # Pale crimson zone for RER > 1.0 (worse than NULL-zero)
    ax_6a.axvspan(1.0, 1.72, facecolor=CRIMSON_LT, alpha=0.45, zorder=0)
    ax_6a.axvline(1.0, color=CHARCOAL, lw=0.85, ls="--", zorder=2)
    ax_6a.grid(axis="x", color=HAIR, lw=0.48, zorder=0)

    # Group band dividers & subtle background stripes
    for g_idx, (r0, r1, g_col) in enumerate([(0, 7, "#F8FAFC"), (8, 10, "#FFFFFF"), (11, 11, "#F8FAFC"), (12, 16, "#FFFFFF")]):
        ya0, ya1 = y6a_to_mm(y_6a[r0] - 0.48), y6a_to_mm(y_6a[r1] + 0.48)
        if g_idx % 2 == 0:
            ax_bg.add_patch(Rectangle((4.0, ya0), 103.5, ya1 - ya0, facecolor=g_col, edgecolor="none", zorder=1))
    for brk in group_breaks:
        ax_6a.axhline(0.5 * (y_6a[brk - 1] + y_6a[brk]), color=LINE, lw=0.65, ls="--", zorder=1)

    ax_hdr.text(
        71.7, 11.8,
        r"RER = $\mathrm{MAE}_{\mathrm{model}}\ /\ \mathrm{MAE}_{\mathrm{NULL\mathrm{-}zero}}$ (<1 beats zero)",
        fontsize=6.4, fontweight="bold", color=CHARCOAL, ha="center", va="center"
    )
    ax_hdr.text(107.2, 11.8, "n", fontsize=6.5, fontweight="bold", color=SUB, ha="right", va="center")

    x_cap_6a = 1.66
    for idx, (st, sc, ep, lbl, bmod, cmod, stag) in enumerate(slices_17):
        yp = y_6a[idx]
        sub_s = f6a[(f6a["study"] == st) & (f6a["scope"] == sc) & (f6a["endpoint"] == ep)]
        rb = sub_s[sub_s["model"] == bmod].iloc[0]
        rc = sub_s[sub_s["model"] == cmod].iloc[0]
        n_s = int(rb["n"])
        has_ci = (n_s >= 6)

        # Rule 5 [P0]: Vertical dodge +/-0.16 row height so baseline & candidate NEVER overlap at RER = 1.00!
        for row_m, dy, col_m, mk_m in [(rb, -0.17, GOLD, "s"), (rc, +0.17, BLUE, "o")]:
            val = float(row_m["rer"])
            lo = float(row_m["lo"]) if (has_ci and pd.notna(row_m["lo"])) else None
            hi = float(row_m["hi"]) if (has_ci and pd.notna(row_m["hi"])) else None

            if val > x_cap_6a:
                # Right-edge overflow arrow (NO '//' broken axis!)
                ax_6a.annotate("", xy=(1.71, yp + dy), xytext=(1.42, yp + dy),
                               arrowprops=dict(arrowstyle="-|>,head_length=0.38,head_width=0.20", color=AMBER, lw=0.95), zorder=5)
                ax_6a.text(1.40, yp + dy, f"→ {val:.2f}", fontsize=5.35, fontweight="bold", color=AMBER, ha="right", va="center", zorder=6)
            else:
                if lo is not None and hi is not None:
                    if hi > 1.71:
                        ax_6a.plot([lo, 1.67], [yp + dy, yp + dy], color=col_m, lw=1.15, solid_capstyle="round", zorder=3)
                        ax_6a.annotate("", xy=(1.718, yp + dy), xytext=(1.63, yp + dy),
                                       arrowprops=dict(arrowstyle="-|>,head_length=0.32,head_width=0.16", color=col_m, lw=1.0), zorder=4)
                    else:
                        ax_6a.plot([lo, hi], [yp + dy, yp + dy], color=col_m, lw=1.15, solid_capstyle="round", zorder=3)
                if has_ci:
                    ax_6a.plot(val, yp + dy, marker=mk_m, ms=3.9, color=col_m, markeredgecolor="white", markeredgewidth=0.35, zorder=5)
                else:
                    # Distinct hollow marker for n=2-3 (no CI!)
                    ax_6a.plot(val, yp + dy, marker=mk_m, ms=3.9, markerfacecolor="white", markeredgecolor=col_m, markeredgewidth=1.15, zorder=5)

        # Special tags (Rule 5!): SUM-count cleanly inside left row label (zero CI collision!); Source-affine@1 shifted into open space!
        if stag == "affine":
            ra = sub_s[sub_s["model"] == "SUM-mass-source-affine@1"].iloc[0]
            ax_6a.plot(float(ra["rer"]), yp + 0.17, marker="D", ms=4.1, markerfacecolor="white", markeredgecolor=TEAL, markeredgewidth=1.15, zorder=6)
            ax_6a.text(float(ra["rer"]) + 0.055, yp + 0.17, "Affine", fontsize=5.0, fontweight="bold", color=TEAL, ha="left", va="center", zorder=6)

        y_mm = y6a_to_mm(yp)
        ax_hdr.text(42.2, y_mm, lbl, fontsize=5.5 if stag == "SUM-count" else 5.85, fontweight="bold" if stag == "prelock" else "normal",
                    color=CRIMSON if stag == "prelock" else INK, ha="right", va="center")
        ax_hdr.text(107.2, y_mm, f"{n_s}*" if stag == "prelock" else f"{n_s}", fontsize=5.85,
                    fontweight="bold" if has_ci else "normal", color=CHARCOAL if has_ci else MUTED, ha="right", va="center")

    ax_6a.set_yticks([])
    ax_6a.set_xticks([0.0, 0.5, 1.0, 1.5])
    ax_6a.set_xlabel("Relative error ratio (RER; 95% CI only for n ≥ 6)", fontsize=6.9, fontweight="bold", color=INK, labelpad=2.0)
    style_clean_axes(ax_6a)
    ax_6a.patch.set_alpha(0.0)

    # Rule 5 exact legend phrasing!
    leg_6a = [
        Line2D([0], [0], marker="s", color=GOLD, lw=1.1, ms=3.9, markeredgecolor="white", markeredgewidth=0.35, label="Unweighted / equal-threshold baseline (Table 3)"),
        Line2D([0], [0], marker="o", color=BLUE, lw=1.1, ms=3.9, markeredgecolor="white", markeredgewidth=0.35, label="Potency / multi-marker (DOT-log@1 / RIDGE-conc)"),
        Line2D([0], [0], marker="D", color="none", ms=3.9, markerfacecolor="white", markeredgecolor=TEAL, markeredgewidth=1.15, label="Source-affine@1 (Yogurt)"),
        Line2D([0], [0], marker="o", color="none", ms=3.9, markerfacecolor="white", markeredgecolor=SUB, markeredgewidth=1.15, label="Hollow: n = 2–3 (no CI); *pre-outcome lock"),
    ]
    ax_6a.legend(
        handles=leg_6a, loc="upper center", bbox_to_anchor=(0.35, -0.105),
        ncol=2, frameon=False, fontsize=5.55, columnspacing=0.65, handletextpad=0.4, handlelength=1.15, labelspacing=0.30
    )

    # ---------------------------------------------------------------------
    # PANEL 6b (Top right: x = 111.5 .. 176.5 mm, ~65 mm wide)
    # Upper: A3 Finite Derangements (Darjeeling + Green Tea)
    # Lower: Independent Framed Single-Core RDKit Flavonol-3-O-Rutinoside Cliff!
    # ---------------------------------------------------------------------
    ax_hdr.text(111.5, 6.4, "b", fontsize=11.0, fontweight="bold", color=CHARCOAL, ha="left", va="center")
    ax_hdr.text(116.2, 6.4, "Threshold reassignment (A3) & monomer cliff", fontsize=8.1, fontweight="bold", color=CHARCOAL, ha="left", va="center")

    # Load exact 199 Darjeeling draws & exact 9 Green Tea derangements
    p31 = pd.read_csv(r"C:\zhuang_taste_pre\phase31_cross_scale_perturbation\phase31_perturbation_results.csv")
    dj_shuf = p31[(p31["dataset"] == "Darjeeling2005") & (p31["endpoint"] == "bitter") & (p31["model"] == "E_shuffled")]
    dj_maes = dj_shuf.groupby("draw")["mae"].first().values
    assert len(dj_maes) == 199 and (dj_maes <= 0.647963 + 1e-9).sum() == 107

    gd = pd.read_csv(r"C:\zhuang_taste_pre\psymix_four_issue_audit_20261123\green_exact_derangements.csv")
    yu_maes = gd["MAE"].values
    assert len(yu_maes) == 9 and (yu_maes <= 0.370954 + 1e-9).sum() == 1

    # 6b-Upper Track 1: Darjeeling bitter (k=9, 199 draws)
    ax_6b1 = fig.add_axes([117.5 / W, 1.0 - 34.5 / H, 57.5 / W, 18.0 / H], zorder=3)
    ax_6b1.set_gid("Panel_6b_DarjeelingDerang")
    ax_6b1.hist(dj_maes, bins=16, color=SLATE_BLUE, edgecolor="white", linewidth=0.45, alpha=0.82, zorder=2)
    ax_6b1.axvline(0.648, color=BLUE, lw=1.6, ls="-", zorder=4)
    ax_6b1.axvline(0.938, color=GOLD, lw=1.1, ls="--", zorder=3)
    ax_6b1.text(0.656, 0.93, "Obs. 0.648\nrank 108/200\n(p = 0.540)", transform=ax_6b1.get_xaxis_transform(),
                fontsize=5.45, fontweight="bold", color=BLUE, ha="left", va="top",
                bbox=dict(boxstyle="round,pad=0.14", facecolor="white", edgecolor="none", alpha=0.86), zorder=6)
    ax_6b1.text(0.912, 0.55, "Equal\n0.938", transform=ax_6b1.get_xaxis_transform(), fontsize=5.2, color="#B87D00", ha="right", va="center", zorder=5)
    ax_6b1.set_xlim(0.44, 1.00)
    ax_6b1.set_xticks([0.5, 0.7, 0.9])
    ax_6b1.set_yticks([])
    ax_6b1.set_title("Darjeeling bitter (k = 9; 199 derangements)", fontsize=6.4, fontweight="bold", color=CHARCOAL, pad=2.0)
    ax_6b1.set_xlabel("Bitterness change MAE", fontsize=6.0, color=SUB, labelpad=1.2)
    style_clean_axes(ax_6b1, left_lw=0.0, bottom_lw=0.6)
    ax_6b1.spines["left"].set_visible(False)

    # 6b-Upper Track 2: Yu green tea bitter (k=4, exact 1 + D(4) = 10)
    ax_6b2 = fig.add_axes([117.5 / W, 1.0 - 59.5 / H, 57.5 / W, 14.5 / H], zorder=3)
    ax_6b2.set_gid("Panel_6b_GreenExactDerang")
    ax_6b2.axhline(0.0, color=LINE, lw=0.7, zorder=1)
    # Small y-jitter only where points are nearly tied so all 9 derangements are visible!
    yu_sorted = np.sort(yu_maes)
    y_jit = np.zeros_like(yu_sorted)
    for k_i in range(1, len(yu_sorted)):
        if yu_sorted[k_i] - yu_sorted[k_i - 1] < 0.004:
            y_jit[k_i] = 0.25 if y_jit[k_i - 1] <= 0 else -0.25
    ax_6b2.scatter(yu_sorted, y_jit, s=18, color=GREY, edgecolors="white", linewidths=0.4, zorder=3, label="9 derangements")
    ax_6b2.plot(0.3710, -0.35, marker="D", ms=5.0, color=BLUE, markeredgecolor="white", markeredgewidth=0.45, zorder=5)
    ax_6b2.annotate(
        "Obs. 0.371 (rank 2/10, p = 0.200; floor 0.10)",
        xy=(0.372, -0.32), xytext=(0.402, -0.58),
        fontsize=5.3, fontweight="bold", color=BLUE, ha="left", va="center",
        arrowprops=dict(arrowstyle="-|>,head_length=0.3,head_width=0.16", color=BLUE, lw=0.7), zorder=6
    )
    ax_6b2.set_xlim(0.355, 0.565)
    ax_6b2.set_ylim(-0.92, 0.65)
    ax_6b2.set_xticks([0.37, 0.42, 0.47, 0.52])
    ax_6b2.set_yticks([])
    ax_6b2.set_title("Green tea bitter (k = 4; exact 1 + D(4) = 10)", fontsize=6.4, fontweight="bold", color=CHARCOAL, pad=2.0)
    ax_6b2.set_xlabel("Bitterness change MAE", fontsize=6.0, color=SUB, labelpad=1.2)
    style_clean_axes(ax_6b2, left_lw=0.0, bottom_lw=0.6)
    ax_6b2.spines["left"].set_visible(False)

    # 6b-Lower: EXPLICITLY FRAMED Single-Core RDKit Flavonol-3-O-Rutinoside Monomer Cliff (Rule 4 [P0]!)
    # Live RDKit Morgan (r=2, 2048-bit) Tanimoto verification
    tc = pd.read_csv(r"C:\zhuang_taste_pre\psymix_four_issue_audit_20261123\threshold_cliffs.csv")
    m_rut = Chem.MolFromSmiles(tc.loc[0, "target_SMILES"])
    m_kae = Chem.MolFromSmiles(tc.loc[0, "donor_SMILES"])
    fp_rut = AllChem.GetMorganFingerprintAsBitVect(m_rut, 2, nBits=2048, useChirality=False)
    fp_kae = AllChem.GetMorganFingerprintAsBitVect(m_kae, 2, nBits=2048, useChirality=False)
    tan_live = DataStructs.TanimotoSimilarity(fp_rut, fp_kae)
    assert abs(tan_live - 0.875) < 1e-6

    add_rbox(ax_hdr, 112.0, 68.5, 64.2, 55.2, fc="#FAFBFD", ec=LINE, lw=0.8, r=1.8, ls="--", zorder=2)
    ax_hdr.text(144.1, 71.8, "Separate literature monomer astringency cliff", fontsize=6.35, fontweight="bold", color=CHARCOAL, ha="center", va="center", zorder=4)

    # Draw ONE razor-sharp RDKit vector scaffold with highlighted 3'-R substituent!
    rx_m, ry_m = draw_rdkit_path_svg_mm(
        ax_hdr,
        MOL_DIR / "flavonol_3O_rutinoside_R.svg",
        MOL_DIR / "flavonol_3O_rutinoside_R.json",
        115.5, 74.2, 57.5, 33.0, bond_lw=0.68, highlight_atom="atom-18", zorder=4
    )
    ax_hdr.text(rx_m + 3.2, ry_m - 0.4, "3'-R", fontsize=6.0, fontweight="bold", color=CRIMSON, ha="left", va="center", zorder=8)

    # Clean 2-row summary below the shared scaffold
    ax_hdr.text(114.5, 110.2, "R = OH  (Rutin): 0.00115 µM", fontsize=6.05, fontweight="bold", color=BLUE, ha="left", va="center", zorder=5)
    ax_hdr.text(114.5, 114.8, "R = H   (Kaempferol-3-O-rutinoside): 0.25 µM", fontsize=5.85, fontweight="bold", color=AMBER, ha="left", va="center", zorder=5)
    ax_hdr.text(
        144.1, 120.2,
        f"217× cliff · Morgan Tanimoto = {tan_live:.3f} (Scharbert et al., 2004)",
        fontsize=5.65, fontweight="bold", color=CRIMSON, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.16", facecolor=CRIMSON_LT, edgecolor="none"), zorder=5
    )

    # ---------------------------------------------------------------------
    # BOTTOM TIER (y = 129.0 .. 224.5 mm) — Golden Ratios [1.18, 0.74, 1.18]
    # ---------------------------------------------------------------------
    cp = pd.read_csv(r"C:\zhuang_taste_pre\psymix_final_tables_for_writing\T6_condition_predictions.csv")

    # =====================================================================
    # PANEL 6c (Bottom left: x = 4.0 .. 66.5 mm, width 62.5 mm) — OJ + Cocoa
    # =====================================================================
    ax_hdr.text(4.0, 131.5, "c", fontsize=11.0, fontweight="bold", color=CHARCOAL, ha="left", va="center")
    ax_hdr.text(8.5, 131.5, "Missing terms (OJ) & shared raw error (cocoa)", fontsize=7.65, fontweight="bold", color=CHARCOAL, ha="left", va="center")

    # 6c-Upper: Orange juice background shifts (+Sucrose vs +Citric acid vs 0.00 model flatline)
    ax_6c1 = fig.add_axes([19.5 / W, 1.0 - 172.0 / H, 46.0 / W, 31.0 / H], zorder=3)
    ax_6c1.set_gid("Panel_6c1_OJBackground")

    # Extract exact Dea shifts
    dea_bg = cp[(cp["dataset"] == "Dea") & (cp["model"] == "DOT-log@1")]
    oj_vals = {
        "H10_suc": float(dea_bg[dea_bg["condition"] == "H10: pair → sugar"]["observed_delta"].iloc[0]),
        "V09_suc": float(dea_bg[dea_bg["condition"] == "V09: pair → sugar"]["observed_delta"].iloc[0]),
        "H10_acid": float(dea_bg[dea_bg["condition"] == "H10: pair → acid"]["observed_delta"].iloc[0]),
        "V09_acid": float(dea_bg[dea_bg["condition"] == "V09: pair → acid"]["observed_delta"].iloc[0]),
    }
    assert abs(oj_vals["H10_suc"] - (-1.48)) < 0.01 and abs(oj_vals["V09_suc"] - (-1.81)) < 0.01
    assert abs(oj_vals["H10_acid"] - (-0.52)) < 0.01 and abs(oj_vals["V09_acid"] - (-0.73)) < 0.01

    ax_6c1.axvline(0.0, color=CRIMSON, lw=1.1, ls="--", zorder=2)
    ax_6c1.grid(axis="x", color=HAIR, lw=0.45, zorder=0)

    # Zero-anchored lollipops (NO truncated bars!)
    oj_plot = [
        (0.0 - 0.16, oj_vals["H10_suc"], BLUE, "o", "H10 −1.48*"),
        (0.0 + 0.16, oj_vals["V09_suc"], TEAL, "D", "V09 −1.81*"),
        (1.0 - 0.16, oj_vals["H10_acid"], GREY, "o", "H10 −0.52 (ns)"),
        (1.0 + 0.16, oj_vals["V09_acid"], MUTED, "D", "V09 −0.73 (ns)"),
    ]
    for yp_o, xv_o, col_o, mk_o, t_o in oj_plot:
        ax_6c1.plot([0.0, xv_o], [yp_o, yp_o], color=col_o, lw=1.45, zorder=3)
        ax_6c1.plot(xv_o, yp_o, marker=mk_o, ms=4.5, color=col_o, markeredgecolor="white", markeredgewidth=0.4, zorder=5)
        ax_6c1.text(xv_o - 0.06, yp_o, t_o, fontsize=5.35, fontweight="bold" if "*" in t_o else "normal", color=col_o, ha="right", va="center", zorder=6)

    ax_6c1.text(0.04, 0.5, "Bitterant\nmodels:\n" + r"$\Delta\hat{I}=0.00$", fontsize=5.15, fontweight="bold", color=CRIMSON, ha="left", va="center", zorder=6)
    ax_6c1.set_yticks([0.0, 1.0])
    ax_6c1.set_yticklabels(["+Sucrose\n(Tukey*)", "+Acid\n(a/ab ns)"], fontsize=6.0, color=INK)
    ax_6c1.invert_yaxis()
    ax_6c1.set_xlim(-2.85, 0.62)
    ax_6c1.set_xticks([-2.0, -1.0, 0.0])
    ax_6c1.set_xlabel("Observed Δbitterness (fixed bitterants)", fontsize=6.15, fontweight="bold", color=INK, labelpad=1.5)
    ax_6c1.set_title("Orange juice background (Dea et al., 2013)", fontsize=6.5, fontweight="bold", color=CHARCOAL, pad=2.0)
    style_clean_axes(ax_6c1)

    # 6c-Lower: Ghana cocoa roasting MAE (n=7, Natural vs Nose-clip, zero-based dot plot + EXTERNAL legend!)
    ax_6c2 = fig.add_axes([19.5 / W, 1.0 - 211.0 / H, 46.0 / W, 26.5 / H], zorder=3)
    ax_6c2.set_gid("Panel_6c2_GhanaCocoa")

    cocoa_nat = cp[(cp["dataset"] == "McClure & Wiedemer") & (cp["endpoint"] == "bitter_natural_gLMS")]
    cocoa_nc = cp[(cp["dataset"] == "McClure & Wiedemer") & (cp["endpoint"] == "bitter_noseclip_gLMS")]
    c_rows = [
        ("NULL-zero", float(cocoa_nat[cocoa_nat["model"] == "NULL-zero"]["absolute_error"].mean()),
         float(cocoa_nc[cocoa_nc["model"] == "NULL-zero"]["absolute_error"].mean())),
        ("RIDGE-SUM", float(cocoa_nat[cocoa_nat["model"] == "RIDGE-SUM-mass"]["absolute_error"].mean()),
         float(cocoa_nc[cocoa_nc["model"] == "RIDGE-SUM-mass"]["absolute_error"].mean())),
        ("RIDGE-conc", float(cocoa_nat[cocoa_nat["model"] == "RIDGE-conc-fixed8"]["absolute_error"].mean()),
         float(cocoa_nc[cocoa_nc["model"] == "RIDGE-conc-fixed8"]["absolute_error"].mean())),
    ]
    assert abs(c_rows[1][1] - 1.685) < 0.005 and abs(c_rows[2][1] - 2.377) < 0.005

    from decimal import Decimal, ROUND_HALF_UP
    def fmt2_t3(v):
        # Standard half-up rounding from Table 3 3-decimal value (e.g. 1.685 -> 1.69)
        return str(Decimal(f"{v:.3f}").quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

    ax_6c2.grid(axis="x", color=HAIR, lw=0.45, zorder=0)
    for ci_y, (clbl, v_nat, v_nc) in enumerate(c_rows):
        ax_6c2.plot([0.0, v_nat], [ci_y - 0.16, ci_y - 0.16], color=BLUE, lw=1.4, zorder=3)
        ax_6c2.plot(v_nat, ci_y - 0.16, marker="o", ms=4.4, color=BLUE, markeredgecolor="white", markeredgewidth=0.4, zorder=5)
        ax_6c2.text(v_nat + 0.06, ci_y - 0.16, fmt2_t3(v_nat), fontsize=5.35, fontweight="bold", color=BLUE, ha="left", va="center")

        ax_6c2.plot([0.0, v_nc], [ci_y + 0.16, ci_y + 0.16], color=PURPLE, lw=1.4, zorder=3)
        ax_6c2.plot(v_nc, ci_y + 0.16, marker="D", ms=4.1, color=PURPLE, markeredgecolor="white", markeredgewidth=0.4, zorder=5)
        ax_6c2.text(v_nc + 0.06, ci_y + 0.16, fmt2_t3(v_nc), fontsize=5.35, fontweight="bold", color=PURPLE, ha="left", va="center")

    ax_6c2.set_yticks([0, 1, 2])
    ax_6c2.set_yticklabels([r[0] for r in c_rows], fontsize=5.95, color=INK)
    ax_6c2.invert_yaxis()
    ax_6c2.set_xlim(0.0, 2.95)
    ax_6c2.set_xticks([0.0, 1.0, 2.0])
    ax_6c2.set_xlabel("Roast vs raw MAE (gLMS; n = 7)", fontsize=6.1, fontweight="bold", color=INK, labelpad=1.4)
    ax_6c2.set_title("Ghana cocoa (separate panels; shared raw)", fontsize=6.35, fontweight="bold", color=CHARCOAL, pad=2.0)
    style_clean_axes(ax_6c2)

    leg_c = [
        Line2D([0], [0], marker="o", color=BLUE, lw=0, ms=4.0, label="Natural panel"),
        Line2D([0], [0], marker="D", color=PURPLE, lw=0, ms=3.8, label="Nose-clip panel"),
    ]
    ax_6c2.legend(handles=leg_c, loc="upper center", bbox_to_anchor=(0.50, -0.34), ncol=2, frameon=False, fontsize=5.4, columnspacing=0.9, handletextpad=0.3)
    place_image_mm(fig, W, H, load_cropped_rgba(SPEC_DIR / "cocoa.png"), 4.6, 214.0, 8.0, 8.0, gid="Specimen_6c_Cocoa", alpha=0.88, zorder=7)

    # =====================================================================
    # PANEL 6d (Bottom center: x = 70.5 .. 110.0 mm, compact 0.74 ratio!) — Yogurt
    # =====================================================================
    ax_hdr.text(70.5, 131.5, "d", fontsize=11.0, fontweight="bold", color=CHARCOAL, ha="left", va="center")
    ax_hdr.text(74.8, 131.5, "Yogurt formulation (T6a)", fontsize=7.65, fontweight="bold", color=CHARCOAL, ha="left", va="center")

    ax_6d = fig.add_axes([78.2 / W, 1.0 - 203.0 / H, 31.0 / W, 62.0 / H], zorder=3)
    ax_6d.set_gid("Panel_6d_Yogurt")

    ah = cp[cp["dataset"] == "Ahmad"]
    def get_ah(mname):
        sub = ah[ah["model"] == mname]
        c1 = float(sub[sub["condition"].str.contains("CH1")]["predicted_delta"].iloc[0])
        y1 = float(sub[sub["condition"].str.contains("YFL")]["predicted_delta"].iloc[0])
        return c1, y1

    obs_ch1 = float(ah[ah["condition"].str.contains("CH1")]["observed_delta"].iloc[0])
    obs_yfl = float(ah[ah["condition"].str.contains("YFL")]["observed_delta"].iloc[0])
    d_models = [
        ("Obs.", obs_ch1, obs_yfl, CHARCOAL),
        ("Affine@1", 8.0003, 7.9894, TEAL),
        ("SUM@1", *get_ah("SUM-mass@1"), GOLD),
        ("DOT-log", *get_ah("DOT-log@1"), BLUE),
        ("NULL", 0.0, 0.0, GREY),
    ]
    assert abs(d_models[2][1] - 6.348) < 0.01 and abs(d_models[3][1] - 4.046) < 0.01

    ax_6d.grid(axis="y", color=HAIR, lw=0.45, zorder=0)
    ax_6d.axhline(0.0, color=LINE, lw=0.7, zorder=1)
    xs_d = np.arange(len(d_models))
    for xi, (mlbl, vc, vy, col_d) in enumerate(d_models):
        # Zero-based stems + distinct CH1+ / YFL+ markers (zero truncated bars!)
        ax_6d.plot([xi - 0.14, xi - 0.14], [0.0, vc], color=col_d, lw=1.4, alpha=0.85, zorder=3)
        ax_6d.plot([xi + 0.14, xi + 0.14], [0.0, vy], color=col_d, lw=1.4, alpha=0.85, zorder=3)
        ax_6d.plot(xi - 0.14, vc, marker="o", ms=4.3, color=col_d, markeredgecolor="white", markeredgewidth=0.4, zorder=5)
        ax_6d.plot(xi + 0.14, vy, marker="D", ms=4.0, color=col_d, markeredgecolor="white", markeredgewidth=0.4, zorder=5)

    ax_6d.text(1.0, 8.65, "MAE\n0.49", fontsize=5.25, fontweight="bold", color=TEAL, ha="center", va="bottom")
    ax_6d.text(2.0, 6.85, "2.14", fontsize=5.25, fontweight="bold", color="#B87D00", ha="center", va="bottom")
    ax_6d.text(3.0, 4.55, "4.49", fontsize=5.25, fontweight="bold", color=BLUE, ha="center", va="bottom")
    ax_6d.text(4.0, 0.55, "8.48", fontsize=5.25, fontweight="bold", color=MUTED, ha="center", va="bottom")

    ax_6d.set_ylim(-0.3, 10.2)
    ax_6d.set_xlim(-0.55, 4.55)
    ax_6d.set_xticks(xs_d)
    ax_6d.set_xticklabels([m[0] for m in d_models], fontsize=5.55, rotation=28, ha="right")
    ax_6d.set_yticks([0, 2, 4, 6, 8, 10])
    ax_6d.set_ylabel("ΔSweetness (0–10 scale)", fontsize=6.5, fontweight="bold", color=INK, labelpad=1.8)
    style_clean_axes(ax_6d)

    leg_d = [
        Line2D([0], [0], marker="o", color=CHARCOAL, lw=0, ms=3.8, label="CH1+"),
        Line2D([0], [0], marker="D", color=CHARCOAL, lw=0, ms=3.6, label="YFL+"),
    ]
    ax_6d.legend(handles=leg_d, loc="upper right", frameon=True, facecolor="white", edgecolor=HAIR, fontsize=5.2, handletextpad=0.25, borderpad=0.22)

    add_rbox(ax_hdr, 70.8, 213.5, 39.2, 10.2, fc=AMBER_LT, ec=AMBER, lw=0.6, r=1.4, zorder=4)
    ax_hdr.text(90.4, 218.6, "Sucrose + pectin co-varied\n(n = 2; viscosity confounded)", fontsize=5.4, fontweight="bold", color=AMBER, ha="center", va="center", zorder=5)

    # =====================================================================
    # PANEL 6e (Bottom right: x = 114.0 .. 176.5 mm, width 62.5 mm) — Ancha & Coffee
    # =====================================================================
    ax_hdr.text(114.0, 131.5, "e", fontsize=11.0, fontweight="bold", color=CHARCOAL, ha="left", va="center")
    ax_hdr.text(118.5, 131.5, "Processing: Ancha reversal & coffee lock (T6b)", fontsize=7.65, fontweight="bold", color=CHARCOAL, ha="left", va="center")

    # 6e-Upper: Ancha dark tea drying opposing astringency trajectory (CD, SD vs HD; EXTERNAL legend below!)
    ax_6e1 = fig.add_axes([127.5 / W, 1.0 - 167.5 / H, 48.0 / W, 27.5 / H], zorder=3)
    ax_6e1.set_gid("Panel_6e1_AnchaReversal")

    anc = cp[(cp["dataset"] == "ancha") & (cp["endpoint"] == "astringent")]
    def get_anc(mname, cond):
        return float(anc[(anc["model"] == mname) & (anc["condition"] == cond)]["predicted_delta"].iloc[0])
    obs_cd = float(anc[anc["condition"] == "CD"]["observed_delta"].iloc[0])
    obs_sd = float(anc[anc["condition"] == "SD"]["observed_delta"].iloc[0])
    sum_cd, sum_sd = get_anc("SUM-molar@1", "CD"), get_anc("SUM-molar@1", "SD")
    dot_cd, dot_sd = get_anc("DOT-log@1", "CD"), get_anc("DOT-log@1", "SD")
    assert abs(obs_sd - (-0.86)) < 0.01 and dot_sd > 0.02

    ax_6e1.axvline(0.0, color=CHARCOAL, lw=0.8, ls="--", zorder=2)
    ax_6e1.grid(axis="x", color=HAIR, lw=0.45, zorder=0)
    for row_i, (o_v, s_v, d_v) in enumerate([(obs_cd, sum_cd, dot_cd), (obs_sd, sum_sd, dot_sd)]):
        ax_6e1.plot([0.0, o_v], [row_i - 0.22, row_i - 0.22], color=CHARCOAL, lw=1.45, zorder=3)
        ax_6e1.plot(o_v, row_i - 0.22, marker="o", ms=4.5, color=CHARCOAL, markeredgecolor="white", markeredgewidth=0.4, zorder=5)
        ax_6e1.plot([0.0, s_v], [row_i, row_i], color=GOLD, lw=1.35, zorder=3)
        ax_6e1.plot(s_v, row_i, marker="s", ms=4.2, color=GOLD, markeredgecolor="white", markeredgewidth=0.4, zorder=5)
        ax_6e1.plot([0.0, d_v], [row_i + 0.22, row_i + 0.22], color=CRIMSON, lw=1.45, zorder=3)
        ax_6e1.plot(d_v, row_i + 0.22, marker="D", ms=4.3, color=CRIMSON, markeredgecolor="white", markeredgewidth=0.4, zorder=5)

    ax_6e1.annotate(
        "Sign reversal\n(+0.029 vs −0.86)",
        xy=(dot_sd + 0.02, 1.22), xytext=(0.06, 0.35),
        fontsize=5.2, fontweight="bold", color=CRIMSON, ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.14", facecolor=CRIMSON_LT, edgecolor=CRIMSON, lw=0.45),
        arrowprops=dict(arrowstyle="-|>,head_length=0.3,head_width=0.16", color=CRIMSON, lw=0.7), zorder=7
    )
    ax_6e1.set_yticks([0, 1])
    ax_6e1.set_yticklabels(["Charcoal\n(CD vs HD)", "Sun-dried\n(SD vs HD)"], fontsize=5.75, color=INK)
    ax_6e1.invert_yaxis()
    ax_6e1.set_xlim(-1.02, 0.58)
    ax_6e1.set_xticks([-0.8, -0.4, 0.0, 0.4])
    ax_6e1.set_xlabel("ΔAstringency (0–5 scale)", fontsize=6.05, fontweight="bold", color=INK, labelpad=1.2)
    ax_6e1.set_title("Ancha tea drying (MAE: SUM 0.19 vs DOT 0.68)", fontsize=6.15, fontweight="bold", color=CHARCOAL, pad=2.0)
    style_clean_axes(ax_6e1)

    leg_e1 = [
        Line2D([0], [0], marker="o", color=CHARCOAL, lw=0, ms=3.8, label="Obs."),
        Line2D([0], [0], marker="s", color=GOLD, lw=0, ms=3.6, label="SUM-molar"),
        Line2D([0], [0], marker="D", color=CRIMSON, lw=0, ms=3.6, label="DOT-log@1"),
    ]
    ax_6e1.legend(handles=leg_e1, loc="upper center", bbox_to_anchor=(0.48, -0.32), ncol=3, frameon=False, fontsize=5.15, columnspacing=0.6, handletextpad=0.25)

    # 6e-Lower: Arabica coffee roast (Internal Pre-Outcome SHA-256 Lock, 1/51 caffeine channel!)
    ax_6e2 = fig.add_axes([127.5 / W, 1.0 - 211.0 / H, 48.0 / W, 24.5 / H], zorder=3)
    ax_6e2.set_gid("Panel_6e2_CoffeePreLock")

    cof = cp[(cp["dataset"] == "coffee") & (cp["model"] == "DOT-log@1")].set_index("condition")
    c_conds = ["Medium", "Slow", "Baked"]
    ax_6e2.axvline(0.0, color=CHARCOAL, lw=0.8, ls="--", zorder=2)
    ax_6e2.grid(axis="x", color=HAIR, lw=0.45, zorder=0)

    for ci_k, cname in enumerate(c_conds):
        ov = float(cof.loc[cname, "observed_delta"])
        pv = float(cof.loc[cname, "predicted_delta"])
        ax_6e2.plot([0.0, ov], [ci_k - 0.15, ci_k - 0.15], color=CHARCOAL, lw=1.45, zorder=3)
        ax_6e2.plot(ov, ci_k - 0.15, marker="o", ms=4.4, color=CHARCOAL, markeredgecolor="white", markeredgewidth=0.4, zorder=5)
        ax_6e2.text(ov + 0.08, ci_k - 0.15, f"+{ov:.1f}", fontsize=5.3, fontweight="bold", color=CHARCOAL, ha="left", va="center")
        ax_6e2.plot([0.0, pv], [ci_k + 0.17, ci_k + 0.17], color=BLUE, lw=1.35, zorder=3)
        ax_6e2.plot(pv, ci_k + 0.17, marker="D", ms=4.0, color=BLUE, markeredgecolor="white", markeredgewidth=0.4, zorder=5)

    ax_6e2.text(
        3.48, -0.05,
        "1/51 matched\n(caffeine only)\nMAE 2.03 vs 2.00",
        fontsize=5.15, fontweight="bold", color=CRIMSON, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.15", facecolor=CRIMSON_LT, edgecolor=CRIMSON, lw=0.45), zorder=7
    )
    ax_6e2.set_yticks([0, 1, 2])
    ax_6e2.set_yticklabels(["Med.", "Slow", "Baked"], fontsize=5.9, color=INK)
    ax_6e2.invert_yaxis()
    ax_6e2.set_xlim(-0.45, 3.60)
    ax_6e2.set_xticks([0.0, 1.0, 2.0, 3.0])
    ax_6e2.set_xlabel("ΔBitterness vs Fast roast (15-pt)", fontsize=6.1, fontweight="bold", color=INK, labelpad=1.4)
    ax_6e2.set_title("Coffee roast (internal pre-outcome SHA-256 lock)", fontsize=6.05, fontweight="bold", color=CRIMSON, pad=2.0)
    style_clean_axes(ax_6e2)

    leg_e2 = [
        Line2D([0], [0], marker="o", color=CHARCOAL, lw=0, ms=3.8, label="Obs. Δbitterness"),
        Line2D([0], [0], marker="D", color=BLUE, lw=0, ms=3.6, label="DOT-log@1 (caffeine)"),
    ]
    ax_6e2.legend(handles=leg_e2, loc="upper center", bbox_to_anchor=(0.50, -0.37), ncol=2, frameon=False, fontsize=5.2, columnspacing=0.7, handletextpad=0.25)

    place_image_mm(fig, W, H, load_cropped_rgba(SPEC_DIR / "coffee.png"), 114.2, 214.0, 8.0, 8.0, gid="Specimen_6e_Coffee", alpha=0.88, zorder=7)

    svg_path = OUT_DIR / "PsyMix_Figure6.svg"
    pdf_path = OUT_DIR / "PsyMix_Figure6.pdf"
    fig.savefig(svg_path, format="svg")
    fig.savefig(pdf_path, format="pdf")
    plt.close(fig)
    print("Saved Figure 6:", pdf_path)


if __name__ == "__main__":
    build_figure5()
    build_figure6()
