# -*- coding: utf-8 -*-
"""
PsyMix Batch 1 Publication Figures — WP1 (Figure 1) & WP2 (Figure 2) Final Visual Spec Edition
  - PsyMix Figure 1 (a-c): 180.0 mm x 204.0 mm
  - PsyMix Figure 2 (a-c): 180.0 mm x 210.0 mm

Strict compliance with VISUAL_SPEC.md & refs/:
  - Global: pdf.fonttype=42, ps.fonttype=42, svg.fonttype='none', Arial + mathtext.fontset='dejavusans',
    zero rare box-drawing Unicode (no ┤/ㅓ), strict lowercase bold a/b/c, frozen data untouched.
  - WP1 (Figure 1):
      * 1a left: straight symmetrical hard-gate funnel (Botanicals 2-stage flat step -> flat stop at red gate,
        dashed rejection callout "× 0 cup doses", single thin dark-teal bar -> T1 sets); flat single-block muted
        slate-blue Food graphs (#E6EEF5/#9BB5CE) stopping flat at red gate with dashed callout "× FoodAtlas: 0
        paired panels"; un-sandwiched Curated paired studies below with 4 delicate converging royal-blue (#0072B2)
        curves crossing gate as a slender bar; independent USDA FDC collapse dashed callout card at very bottom.
      * 1a right: vertical square brackets (ref_1a_bracket.png) feeding two broken-top-line group frames
        (ref_1a_step_and_cards.png Panel B/C): "Qualitative sets (T1)" (T1a, T1b) and "Sample-matched quantitative
        domains (T2-T6)" (T2, T3-4, T5 grape->yogurt, T6 watercolor tea->ΔI) with unified delicate illustration style.
      * 1b: pastel-medallion S/P/Q/W/A icons, muted-grey Lock vs solid Vary pills, 706 donut (175/137/113/281 Mixed
        (multi-factor)), upright-subscript ΔR = R_cand - R_ref (<0 lowers error).
      * 1c: deep charcoal (#2C3E50) white-text grouped column banners (ref_1c_audit_matrix.jpg), alternating #F8FAFC
        row stripes, two-layer cells (top 42% crisp 3-state proportional bar + bottom 58% clean unobstructed numbers),
        centered single-row legend + T5 footnote.
  - WP2 (Figure 2):
      * 2a: split titles padded further right (x = 0.474) off black Y-axis spine; strict lowercase a/b/c.
      * 2b: wide 20.5 mm gutter + compact 1-column centered legends (> 30 mm white space between legends), clean
        discrete dot plot (no grey error-bar sticks), Prior = 0.992 aimed straight at grey hollow square.
      * 2c: Aspartame pure-vector RDKit structure matched to D-Sorbitol razor-sharpness; Aspartame dashed leader line
        shortened 5 pt cleanly above structure (zero grazing of NH2); Measured DoT shifted right; watercolor tea.png.
"""
import re
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Polygon, Circle, Ellipse, PathPatch
from matplotlib.path import Path as MplPath
import matplotlib.patheffects as pe

# Strict VISUAL_SPEC.md §1.1 configuration
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "svg.hashsalt": "PsyMix-VisualSpec-Final",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 7.8,
    "axes.labelsize": 8.2,
    "axes.titlesize": 8.6,
    "xtick.labelsize": 7.4,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.0,
    "axes.unicode_minus": True,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

OUT_DIR = Path(r"c:\gemi-antigravity\00_PsyMix_Submission_Master\02_Figures_Design_and_Drafts\final_figures\Batch1_Fig1_Fig2")
SPEC_DIR = Path(r"c:\gemi-antigravity\00_PsyMix_Submission_Master\02_Figures_Design_and_Drafts\specimens")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Unified Colorblind-Safe Palette (VISUAL_SPEC.md §1.2)
CHARCOAL = "#2C3E50"
INK = "#1E293B"
SUB = "#4A5568"
MUTED = "#718096"
LINE = "#CBD5E0"
HAIR = "#E6ECF2"

BLUE = "#0072B2"
BLUE_LT = "#E8F3FA"
TEAL = "#1B7F6B"
TEAL_BRIGHT = "#009E73"
TEAL_LT = "#D6F0E8"
TEAL_MID = "#A7DFCF"
SLATE_BG = "#E6EEF5"
SLATE_EC = "#9BB5CE"
SLATE_TXT = "#3E566E"
VIOLET = "#5E4FA2"
VIOLET_LT = "#F1EEFA"
AMBER = "#D55E00"
GOLD = "#E69F00"
AMBER_LT = "#FEF2E8"
CRIMSON = "#C0392B"
CRIMSON_LT = "#FDECEC"
SLATE = "#4D648D"
GREY = "#8C98A4"
NA_FILL = "#E5E7EB"
CALLOUT_BG = "#F3F4F6"
STRIPE_BG = "#F8FAFC"
SEPIA_INK = "#433B33"


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


def draw_rdkit_svg_vector(ax, svg_file, x_mm, y_mm, w_mm, h_mm, color=INK, bond_lw=0.90, min_atom_pt=4.1, font_mult=2.78, halo=True, zorder=5):
    """Render RDKit MolDraw2DSVG as razor-sharp native vector paths + ChemDraw-style merged atom text."""
    root = ET.parse(svg_file).getroot()
    raw_paths, raw_texts = [], []
    all_x, all_y = [], []

    for el in root.iter():
        tag = el.tag.split("}")[-1]
        if tag == "path":
            d = el.get("d", "")
            style = el.get("style", "")
            tokens = re.findall(r"[MLZ]|[-+]?\d*\.?\d+", d)
            pts, codes = [], []
            i = 0
            closed = False
            while i < len(tokens):
                t = tokens[i]
                if t == "M":
                    px, py = float(tokens[i + 1]), float(tokens[i + 2])
                    pts.append((px, py)); codes.append(MplPath.MOVETO)
                    all_x.append(px); all_y.append(py)
                    i += 3
                elif t == "L":
                    px, py = float(tokens[i + 1]), float(tokens[i + 2])
                    pts.append((px, py)); codes.append(MplPath.LINETO)
                    all_x.append(px); all_y.append(py)
                    i += 3
                elif t == "Z":
                    pts.append(pts[0]); codes.append(MplPath.CLOSEPOLY)
                    closed = True
                    i += 1
                else:
                    i += 1
            if pts:
                is_filled = ("fill:#000000" in style) or closed
                raw_paths.append((pts, codes, is_filled))
        elif tag == "text" and (el.text or "").strip():
            tx = float(el.get("x"))
            ty = float(el.get("y"))
            style = el.get("style", "")
            m_fs = re.search(r"font-size:(\d+(?:\.\d+)?)px", style)
            fs_px = float(m_fs.group(1)) if m_fs else 12.0
            raw_texts.append((tx, ty, el.text.strip(), fs_px))
            all_x.extend([tx, tx + fs_px * 0.7])
            all_y.extend([ty - fs_px * 0.8, ty])

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    span_x = max(max_x - min_x, 1e-3)
    span_y = max(max_y - min_y, 1e-3)
    s = min(w_mm / span_x, h_mm / span_y)
    ox = x_mm + (w_mm - span_x * s) / 2.0
    oy = y_mm + (h_mm - span_y * s) / 2.0

    for pts, codes, is_filled in raw_paths:
        tpts = [(ox + (px - min_x) * s, oy + (py - min_y) * s) for px, py in pts]
        patch = PathPatch(
            MplPath(tpts, codes),
            facecolor=color if is_filled else "none",
            edgecolor=color if not is_filled else "none",
            lw=bond_lw if not is_filled else 0.0,
            capstyle="round", joinstyle="round", zorder=zorder
        )
        ax.add_patch(patch)

    # Merge adjacent same-baseline same-size atom characters (O+H -> OH, H+O -> HO, N+H -> NH)
    raw_texts.sort(key=lambda item: (round(item[1] / 2.5), item[0]))
    merged_texts = []
    for tx, ty, txt, fs_px in raw_texts:
        if merged_texts:
            px, py, ptxt, pfs = merged_texts[-1]
            same_row = abs(ty - py) < 2.2
            same_size = abs(fs_px - pfs) < 1.5
            close_x = 0.0 <= (tx - (px + len(ptxt) * pfs * 0.62)) <= pfs * 0.48
            if same_row and same_size and close_x:
                merged_texts[-1] = (px, py, ptxt + txt, pfs)
                continue
        merged_texts.append((tx, ty, txt, fs_px))

    for tx, ty, txt, fs_px in merged_texts:
        mx = ox + (tx - min_x) * s
        my = oy + (ty - min_y) * s
        is_sub = txt.isdigit() and fs_px < 10.5
        pt_size = max(min_atom_pt * 0.72, fs_px * s * font_mult) if is_sub else max(min_atom_pt, fs_px * s * font_mult)
        t_obj = ax.text(mx, my, txt, fontsize=pt_size, fontweight="bold", color=color, ha="left", va="baseline", zorder=zorder + (0.40 if is_sub else 0.35))
        if halo and not is_sub:
            t_obj.set_path_effects([pe.withStroke(linewidth=1.35, foreground="white")])


def place_image_mm(fig, W, H, img, x_mm, y_mm, w_mm, h_mm, gid=None):
    iw, ih = img.size
    scale = min(w_mm / iw, h_mm / ih)
    rw, rh = iw * scale, ih * scale
    ox = x_mm + (w_mm - rw) / 2.0
    oy = y_mm + (h_mm - rh) / 2.0
    ax_i = fig.add_axes([ox / W, 1.0 - (oy + rh) / H, rw / W, rh / H])
    ax_i.imshow(img, interpolation="lanczos")
    ax_i.set_axis_off()
    if gid:
        ax_i.set_gid(gid)
    return ax_i


def add_rbox(ax, x, y, w, h, fc="white", ec=LINE, lw=0.65, ls="-", r=1.4, zorder=1):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=zorder
    )
    ax.add_patch(p)
    return p


def draw_broken_top_group_frame(ax, x, y, w, h, title, title_w_mm, ec, title_col, lw=0.75, zorder=2):
    """Draw FoldBench-style (ref_1a_step_and_cards.png Panel B/C) broken-top-border group box."""
    cx = x + w / 2.0
    gap_l = cx - title_w_mm / 2.0
    gap_r = cx + title_w_mm / 2.0
    # Left top stub -> left wall -> bottom wall -> right wall -> right top stub
    ax.plot([gap_l, x, x, x + w, x + w, gap_r], [y, y, y + h, y + h, y, y],
            color=ec, lw=lw, solid_capstyle="round", solid_joinstyle="round", zorder=zorder)
    ax.text(cx, y, title, fontsize=7.1, fontweight="bold", color=title_col, ha="center", va="center", zorder=zorder + 1)


def draw_square_bracket(ax, x_in, x_vert, x_tick, y_in, y_top, y_bot, color, lw=1.05, zorder=3):
    """Draw PathoROB-style (ref_1a_bracket.png) clean vertical square bracket separating stream & independent domains."""
    ax.plot([x_in, x_vert], [y_in, y_in], color=color, lw=lw, solid_capstyle="butt", zorder=zorder)
    ax.plot([x_tick, x_vert, x_vert, x_tick], [y_top, y_top, y_bot, y_bot],
            color=color, lw=lw, solid_capstyle="round", solid_joinstyle="round", zorder=zorder)


def draw_hatched_rect(ax, x, y, w, h, fc="#FEF0D9", ec=GOLD, lw=0.42, step=1.35, zorder=2):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder))
    u = -h + 0.2
    while u < w:
        x0 = max(0.0, u)
        x1 = min(w, u + h)
        if x1 > x0:
            y0 = y + h - (x0 - u)
            y1 = y + h - (x1 - u)
            ax.plot([x + x0, x + x1], [y0, y1], color=ec, lw=0.38, solid_capstyle="butt", zorder=zorder + 0.1)
        u += step


def s_ribbon(ax, x0, y0_top, y0_bot, x1, y1_top, y1_bot, fc, ec="none", alpha=0.90, lw=0.45, c_frac=0.52, zorder=2):
    xm0 = x0 + (x1 - x0) * c_frac
    xm1 = x0 + (x1 - x0) * (1.0 - (1.0 - c_frac) * 0.65)
    verts = [
        (x0, y0_top), (xm0, y0_top), (xm1, y1_top), (x1, y1_top),
        (x1, y1_bot), (xm1, y1_bot), (xm0, y0_bot), (x0, y0_bot), (x0, y0_top)
    ]
    codes = [
        MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.CLOSEPOLY
    ]
    patch = PathPatch(MplPath(verts, codes), facecolor=fc, edgecolor=ec, alpha=alpha, lw=lw, zorder=zorder)
    ax.add_patch(patch)


def draw_soft_beaker(ax, cx, cy, w, h, spheres, wash_top="#E1EFF8", wash_bot="#C2DCEF", zorder=4):
    """Delicate fine-sepia-ink & translucent watercolor-wash laboratory beaker harmonized with botanical specimens."""
    y_top = cy - h * 0.48
    y_bot = cy + h * 0.48
    y_liq = cy - h * 0.08
    w_top = w * 0.96
    w_bot = w * 0.80

    body_pts = [
        (cx - w_top * 0.5, y_top), (cx - w_bot * 0.5, y_bot - 0.5),
        (cx - w_bot * 0.42, y_bot), (cx + w_bot * 0.42, y_bot),
        (cx + w_bot * 0.5, y_bot - 0.5), (cx + w_top * 0.5, y_top)
    ]
    ax.add_patch(Polygon(body_pts, closed=True, facecolor="#F7FAFC", edgecolor="none", alpha=0.88, zorder=zorder))

    n_bands = 6
    c_top = mpl.colors.to_rgb(wash_top)
    c_bot = mpl.colors.to_rgb(wash_bot)
    for k in range(n_bands):
        f0, f1 = k / n_bands, (k + 1) / n_bands
        ya = y_liq + f0 * (y_bot - y_liq)
        yb = y_liq + f1 * (y_bot - y_liq)
        wa = w_top * (1 - 0.14 * ((ya - y_top) / h)) * 0.93
        wb = w_top * (1 - 0.14 * ((yb - y_top) / h)) * 0.93
        c_k = tuple(c_top[j] * (1 - f1) + c_bot[j] * f1 for j in range(3))
        poly = [(cx - wa * 0.5, ya), (cx + wa * 0.5, ya), (cx + wb * 0.5, yb), (cx - wb * 0.5, yb)]
        ax.add_patch(Polygon(poly, closed=True, facecolor=c_k, edgecolor="none", alpha=0.72, zorder=zorder + 0.1))

    ax.add_patch(Ellipse((cx, y_liq), w_top * 0.88, 1.15, facecolor=wash_top, edgecolor="#8FB2CE", lw=0.40, alpha=0.92, zorder=zorder + 0.25))

    for dx, dy, col in spheres:
        sx, sy = cx + dx, cy + dy
        ax.add_patch(Circle((sx, sy), 0.95, facecolor=col, edgecolor=SEPIA_INK, lw=0.38, alpha=0.88, zorder=zorder + 0.35))
        ax.add_patch(Circle((sx - 0.26, sy - 0.26), 0.28, facecolor="white", edgecolor="none", alpha=0.85, zorder=zorder + 0.42))

    ax.plot([cx - w_top * 0.33, cx - w_bot * 0.33], [y_top + 1.1, y_bot - 1.0], color="white", lw=0.80, alpha=0.78, solid_capstyle="round", zorder=zorder + 0.45)
    ax.plot([p[0] for p in body_pts], [p[1] for p in body_pts], color=SEPIA_INK, lw=0.52, solid_capstyle="round", solid_joinstyle="round", zorder=zorder + 0.50)
    ax.add_patch(Ellipse((cx, y_top), w_top, 1.20, facecolor="#FAFCFD", edgecolor=SEPIA_INK, lw=0.50, alpha=0.92, zorder=zorder + 0.55))


def draw_padlock(ax, cx, cy, locked=True, color="#7B8794", zorder=4):
    bw, bh = 2.8, 2.2
    add_rbox(ax, cx - bw / 2, cy - 0.05, bw, bh, fc=color if locked else "white", ec=color, lw=0.7, r=0.4, zorder=zorder)
    th = np.linspace(np.pi, 2 * np.pi, 20)
    sx = cx + 0.85 * np.cos(th)
    sy = (cy - 0.05) + 0.98 * np.sin(th)
    ax.plot(sx, sy, color=color, lw=0.75, zorder=zorder)
    if locked:
        ax.plot(cx, cy + 1.05, "o", ms=1.3, color="white", zorder=zorder + 0.2)


# =========================================================================
# VERIFY ALL FROZEN DATA (ZERO MODIFICATION)
# =========================================================================
f1_counts = pd.read_csv(r"C:\zhuang_taste_pre\psymix_manuscript_release\Figures_round1_20261124\source_data\Figure1_counts.csv")
assert int(f1_counts[f1_counts["check"] == "A1"][["clear", "limited", "na"]].to_numpy().sum()) == 44

herb_df = pd.read_csv(r"C:\zhuang_taste_pre\phase15_crossstudy_benchmark\results\herb_metrics.csv")
dmc_sweet = pd.read_csv(r"C:\zhuang_taste_pre\psymix_four_issue_audit_20261123\C1_DMC_published_outputs.csv")
monomers = pd.read_csv(r"C:\zhuang_taste_pre\phase29_direct_mixture_20261115\data\monomers.csv").set_index("code")
tea_prob = pd.read_csv(r"C:\zhuang_taste_pre\psymix_herbal_infusions_20261122\model_trace_v2\tea_licorice\tea_probability_predictions.csv")

darj = tea_prob[tea_prob["dataset"] == "Darjeeling2005"]
darj_summary = {(m, arm): float((g["predicted"] - g["observed_delta"]).abs().mean())
                for (m, arm), g in darj.groupby(["model", "arm"])}
assert round(darj_summary[("DmcFlavor author S3/S7 molecular table", "equal")], 3) == 0.771
assert round(darj_summary[("DmcFlavor author S3/S7 molecular table", "molecular_probability")], 3) == 0.925


# =========================================================================
# WP1 — FIGURE 1: 180.0 mm x 204.0 mm (VISUAL_SPEC.md Final)
# =========================================================================
def build_figure1():
    W, H = 180.0, 204.0
    fig = plt.figure(figsize=(W / 25.4, H / 25.4), dpi=600)

    ax_bg = fig.add_axes([0, 0, 1, 1], xlim=(0, W), ylim=(H, 0))
    ax_bg.set_axis_off()
    ax_bg.set_gid("Canvas_Background")
    ax_bg.add_patch(Rectangle((0, 0), W, H, facecolor="white", edgecolor="none", zorder=0))
    ax_bg.plot([4.0, W - 4.0], [84.5, 84.5], color=HAIR, lw=0.85, zorder=1)
    ax_bg.plot([75.8, 75.8], [88.5, H - 4.0], color=HAIR, lw=0.85, zorder=1)

    # ---------------------------------------------------------------------
    # Panel a: Flat Hard-Gate Funnel + Square Brackets + Broken-Top Group Cards
    # ---------------------------------------------------------------------
    ax_a = fig.add_axes([0, 0, 1, 1], xlim=(0, W), ylim=(H, 0))
    ax_a.set_axis_off()
    ax_a.patch.set_alpha(0.0)
    ax_a.set_gid("Panel_a_Funnel")

    ax_a.text(4.0, 6.8, "a", fontsize=11.0, fontweight="bold", color=CHARCOAL, ha="left", va="center")
    ax_a.text(9.5, 6.8, "Data-qualification funnel and benchmark domains", fontsize=9.2, fontweight="bold", color=CHARCOAL, ha="left", va="center")

    gate_x = 56.5

    # === ROW 1: Botanicals — Symmetrical Flat Two-Stage Step Funnel (ZERO curved droopy tail!) ===
    # Stage 1 (x=4.5..33.0, y=14.8..29.2, centered y=22.0)
    ax_a.add_patch(Rectangle((4.5, 14.8), 28.5, 14.4, facecolor=TEAL_LT, edgecolor=TEAL, lw=0.60, zorder=2))
    # Symmetrical straight horizontal trapezoid connector (x=33.0..36.4)
    trap1 = [(33.0, 14.8), (36.4, 17.2), (36.4, 26.8), (33.0, 29.2)]
    ax_a.add_patch(Polygon(trap1, closed=True, facecolor="#BEE8DC", edgecolor=TEAL, lw=0.55, zorder=2))
    # Stage 2 (x=36.4..56.0, y=17.2..26.8, flat flush stop against Sensory gate!)
    ax_a.add_patch(Rectangle((36.4, 17.2), 19.6, 9.6, facecolor=TEAL_MID, edgecolor=TEAL, lw=0.60, zorder=2))
    # Hard flush crimson stop along blocked section (y=19.6..26.8) strictly left of gate
    ax_a.plot([56.0, 56.0], [19.6, 26.8], color=CRIMSON, lw=1.35, zorder=3.2)

    # Single slender dark-teal bar passing horizontally across Sensory gate -> T1 sets bracket
    ax_a.add_patch(Rectangle((56.0, 17.2), 29.2, 2.4, facecolor=TEAL, edgecolor="none", zorder=2.8))

    ax_a.text(6.2, 18.2, "Botanicals", fontsize=7.4, fontweight="bold", color="#125546", ha="left", va="center", zorder=4)
    ax_a.text(6.2, 22.0, "2.53M records", fontsize=6.8, color=INK, ha="left", va="center", zorder=4)
    ax_a.text(6.2, 25.8, "434k structures", fontsize=6.7, color=SUB, ha="left", va="center", zorder=4)
    ax_a.text(46.2, 22.0, "169k plant\nstruct.", fontsize=6.7, fontweight="bold", color="#125546", ha="center", va="center", linespacing=1.12, zorder=4)
    ax_a.text(58.5, 15.2, "T1 sets (qualitative)", fontsize=6.8, fontweight="bold", color=TEAL, ha="left", va="center", zorder=4)

    # Clean dashed rejection callout box right opposite blocked wall (ref_1a_hard_gate.jpg style)
    add_rbox(ax_a, 58.6, 21.8, 24.4, 5.2, fc=CRIMSON_LT, ec="#D99B96", lw=0.55, ls="--", r=0.9, zorder=2.5)
    ax_a.text(70.8, 24.4, r"$\times$ 0 cup doses", fontsize=6.7, fontweight="bold", color=CRIMSON, ha="center", va="center", zorder=4)

    # === ROW 2: Food graphs — Single Flat Muted Grey-Blue Rectangle (NO mid-divider, NO fake waist!) ===
    ax_a.add_patch(Rectangle((4.5, 32.0), 51.5, 15.6, facecolor=SLATE_BG, edgecolor=SLATE_EC, lw=0.65, zorder=2))
    # Hard flush crimson stop along entire right edge (zero pass!)
    ax_a.plot([56.0, 56.0], [32.0, 47.6], color=CRIMSON, lw=1.40, zorder=3.2)

    ax_a.text(6.2, 35.1, "Food graphs", fontsize=7.4, fontweight="bold", color=SLATE_TXT, ha="left", va="center", zorder=4)
    ax_a.text(6.2, 38.6, "FooDB: 352k conc.", fontsize=6.7, color=INK, ha="left", va="center", zorder=4)
    ax_a.text(6.2, 42.1, "FoodAtlas v4.16: 218k", fontsize=6.7, color=INK, ha="left", va="center", zorder=4)
    ax_a.text(6.2, 45.6, "FlavorDB: 1,335", fontsize=6.7, color=SUB, ha="left", va="center", zorder=4)
    ax_a.text(47.2, 39.8, "Unpaired\nlinks", fontsize=6.7, fontweight="bold", color="#526A82", ha="center", va="center", linespacing=1.15, zorder=4)

    # Clean dashed rejection callout box right opposite blocked Food graphs wall
    add_rbox(ax_a, 58.6, 35.4, 24.4, 8.8, fc=CRIMSON_LT, ec="#D99B96", lw=0.55, ls="--", r=0.9, zorder=2.5)
    ax_a.text(70.8, 39.8, r"$\times$ FoodAtlas: 0" + "\npaired panels", fontsize=6.7, fontweight="bold", color=CRIMSON, ha="center", va="center", linespacing=1.20, zorder=4)

    # === ROW 3: Curated paired studies — 4 slender symmetrical royal-blue curves converging cleanly! ===
    ax_a.text(4.8, 51.4, "Curated paired studies (same-sample Q + A)", fontsize=6.75, fontweight="bold", color=BLUE, ha="left", va="center", zorder=4)
    trib_ys = [55.4, 57.1, 58.8, 60.5]
    y_merge_top = 57.0
    for idx_t, yt in enumerate(trib_ys):
        s_ribbon(
            ax_a,
            4.8, yt - 0.24, yt + 0.24,
            51.0, y_merge_top + idx_t * 0.50, y_merge_top + (idx_t + 1) * 0.50,
            fc="#4D9BD0" if idx_t % 2 == 0 else "#1B81C2", ec="none", alpha=0.90, lw=0, c_frac=0.52, zorder=2.3
        )
    # Merged slim royal-blue bar (2.0 mm, slimmer than T1 sets!) crossing gate -> T2-T6 bracket
    ax_a.add_patch(Rectangle((51.0, y_merge_top), 34.2, 2.0, facecolor=BLUE, edgecolor="none", zorder=2.8))
    ax_a.text(58.5, 54.4, "T2–T6 paired Q + A", fontsize=6.85, fontweight="bold", color=BLUE, ha="left", va="center", zorder=4)

    # === Vertical Sensory Gate Hard Wall ===
    ax_a.plot([gate_x, gate_x], [13.6, 63.6], color=CRIMSON, lw=1.30, ls="--", zorder=3.5)
    add_rbox(ax_a, gate_x - 9.5, 9.1, 19.0, 4.5, fc=CRIMSON_LT, ec=CRIMSON, lw=0.60, r=1.0, zorder=4.5)
    ax_a.text(gate_x, 11.35, "Sensory gate", fontsize=6.9, fontweight="bold", color=CRIMSON, ha="center", va="center", zorder=5)

    # === ROW 4 (Very Bottom): Independent USDA FDC Presence-Only Collapse Callout Card ===
    add_rbox(ax_a, 4.5, 67.6, 78.6, 11.8, fc=CALLOUT_BG, ec="#9CA3AF", lw=0.65, ls="--", r=1.4, zorder=2.2)
    ax_a.text(6.6, 70.8, "Why presence-only fails (USDA FDC):", fontsize=6.65, fontweight="bold", color=SUB, ha="left", va="center", zorder=4)
    ax_a.text(6.6, 75.8, "3,516 foods", fontsize=6.7, fontweight="bold", color=INK, ha="left", va="center", zorder=4)
    bx0, by0 = 20.8, 74.4
    for k, bit in enumerate([1, 0, 1, 1, 0, 0, 1, 0, 1]):
        ax_a.add_patch(Rectangle((bx0 + k * 1.12, by0), 0.78, 2.7, facecolor=SLATE if bit else "#D8DEE6", edgecolor="none", zorder=3.5))
    ax_a.text(bx0 + 10.8, 75.8, r"$\rightarrow$ 122 0/1 vectors (up to 1,385 identical)", fontsize=6.6, color=INK, ha="left", va="center", zorder=4)

    # === RIGHT SIDE: Square Brackets (ref_1a_bracket.png) + Broken-Top Group Frames (ref_1a_step_and_cards.png) ===
    # Top Group: Qualitative sets (T1) -> T1a & T1b
    draw_square_bracket(ax_a, 85.2, 87.2, 88.8, 18.4, 15.8, 33.4, color=TEAL, lw=1.05, zorder=3)
    draw_broken_top_group_frame(ax_a, 89.4, 12.8, 87.1, 23.4, "Qualitative sets (T1)", 32.5, ec=TEAL, title_col=TEAL, lw=0.78, zorder=2)

    # Bottom Group: Sample-matched quantitative domains (T2-T6) -> T2, T3-4, T5, T6 (2x2 clean grid)
    draw_square_bracket(ax_a, 85.2, 87.2, 88.8, 58.0, 44.8, 76.2, color=BLUE, lw=1.05, zorder=3)
    draw_broken_top_group_frame(ax_a, 89.4, 41.2, 87.1, 38.2, "Sample-matched quantitative domains (T2–T6)", 66.5, ec=BLUE, title_col=BLUE, lw=0.78, zorder=2)

    # Top Group Cards (2 spacious horizontal cards inside Qualitative sets frame)
    q_cards = [
        ("T1a", "Molecules", "17,202 mol", TEAL, 91.8),
        ("T1b", "Botanicals", "645 + 128 herbs", TEAL, 134.0),
    ]
    for code, title, sub, col, cx0 in q_cards:
        cy0, cw, ch = 16.0, 40.4, 17.8
        add_rbox(ax_a, cx0, cy0, cw, ch, fc="#FAFBFD", ec=LINE, lw=0.60, r=1.3, zorder=2.5)
        ax_a.add_patch(Rectangle((cx0, cy0), 1.8, ch, facecolor=col, edgecolor="none", zorder=3))
        ax_a.text(cx0 + 18.2, cy0 + 6.0, f"{code}  {title}", fontsize=7.1, fontweight="bold", color=INK, ha="left", va="center", zorder=4)
        ax_a.text(cx0 + 18.2, cy0 + 12.2, sub, fontsize=6.7, color=SUB, ha="left", va="center", zorder=4)
        if code == "T1a":
            draw_rdkit_svg_vector(ax_a, OUT_DIR / "mol_caffeine.svg", cx0 + 2.8, cy0 + 2.2, 14.2, 13.4, color=INK, bond_lw=0.80, min_atom_pt=3.8, halo=True, zorder=4)
        else:
            herb_im = load_cropped_rgba(SPEC_DIR / "herb.png")
            place_image_mm(fig, W, H, herb_im, cx0 + 3.0, cy0 + 1.5, 13.8, 14.8, gid="Icon_T1b_Herb")

    # Bottom Group Cards (2x2 spacious horizontal cards inside Sample-matched quantitative frame)
    quant_cards = [
        ("T2", "Fruit cohorts", "5 crops (n = 741)", BLUE, 91.8, 44.6),
        ("T3–4", "Mixtures", "55 pairs, 64 tern.", VIOLET, 134.0, 44.6),
        ("T5", "Scale transfer", "6 routes + yogurt", AMBER, 91.8, 61.8),
        ("T6", "Interventions", "7 foods (58 contrasts)", CRIMSON, 134.0, 61.8),
    ]
    for code, title, sub, col, cx0, cy0 in quant_cards:
        cw, ch = 40.4, 15.2
        add_rbox(ax_a, cx0, cy0, cw, ch, fc="#FAFBFD", ec=LINE, lw=0.60, r=1.3, zorder=2.5)
        ax_a.add_patch(Rectangle((cx0, cy0), 1.8, ch, facecolor=col, edgecolor="none", zorder=3))
        tx_lbl = cx0 + 18.2 if code != "T6" else cx0 + 17.6
        ax_a.text(tx_lbl, cy0 + 5.1, f"{code}  {title}", fontsize=6.95, fontweight="bold", color=INK, ha="left", va="center", zorder=4)
        ax_a.text(tx_lbl, cy0 + 10.6, sub, fontsize=6.35 if code == "T6" else 6.6, color=SUB, ha="left", va="center", zorder=4)

        if code == "T2":
            tom_im = load_cropped_rgba(SPEC_DIR / "tomato.png")
            blu_im = load_cropped_rgba(SPEC_DIR / "blueberry.png")
            place_image_mm(fig, W, H, tom_im, cx0 + 2.4, cy0 + 2.2, 7.6, 10.8, gid="Icon_T2_Tomato")
            place_image_mm(fig, W, H, blu_im, cx0 + 9.8, cy0 + 2.2, 7.6, 10.8, gid="Icon_T2_Blueberry")
        elif code == "T3–4":
            draw_soft_beaker(ax_a, cx0 + 6.4, cy0 + 7.7, 6.2, 9.6, [(-1.0, 1.0, BLUE), (1.0, 1.8, AMBER)], wash_top="#E1F0F9", wash_bot="#BDD9EE", zorder=4)
            draw_soft_beaker(ax_a, cx0 + 13.6, cy0 + 7.7, 6.2, 9.6, [(-1.1, 0.9, BLUE), (1.1, 0.8, AMBER), (0.0, 2.4, VIOLET)], wash_top="#EAE6F6", wash_bot="#C9C0E8", zorder=4)
        elif code == "T5":
            # Strictly grape -> yogurt per VISUAL_SPEC.md, Table 1, §3.5, Fig. 5c!
            grp_im = load_cropped_rgba(SPEC_DIR / "grape.png")
            yog_im = load_cropped_rgba(SPEC_DIR / "yogurt.png")
            place_image_mm(fig, W, H, grp_im, cx0 + 2.0, cy0 + 1.8, 6.8, 11.4, gid="Icon_T5_Grape")
            ax_a.annotate("", xy=(cx0 + 11.3, cy0 + 7.6), xytext=(cx0 + 8.6, cy0 + 7.6),
                          arrowprops=dict(arrowstyle="-|>,head_length=0.40,head_width=0.22", mutation_scale=6.0, color=AMBER, lw=0.9), zorder=5)
            place_image_mm(fig, W, H, yog_im, cx0 + 11.1, cy0 + 2.4, 6.6, 10.2, gid="Icon_T5_Yogurt")
        elif code == "T6":
            tea_im = load_cropped_rgba(SPEC_DIR / "tea.png")
            place_image_mm(fig, W, H, tea_im, cx0 + 2.0, cy0 + 2.0, 8.8, 11.0, gid="Icon_T6_Tea")
            ax_a.annotate("", xy=(cx0 + 13.6, cy0 + 7.8), xytext=(cx0 + 10.8, cy0 + 7.8),
                          arrowprops=dict(arrowstyle="-|>,head_length=0.38,head_width=0.20", mutation_scale=6.0, color=SLATE, lw=0.85), zorder=5)
            ax_a.text(cx0 + 15.6, cy0 + 7.8, r"$\Delta I$", fontsize=7.5, fontweight="bold", color=CRIMSON, ha="center", va="center", zorder=4)

    # ---------------------------------------------------------------------
    # Panel b: Pastel-Medallion Input Icons + Lock/Vary Matrix + 706 Donut
    # ---------------------------------------------------------------------
    ax_b = fig.add_axes([0, 0, 1, 1], xlim=(0, W), ylim=(H, 0))
    ax_b.set_axis_off()
    ax_b.patch.set_alpha(0.0)
    ax_b.set_gid("Panel_b_Contrasts")

    ax_b.text(4.0, 91.0, "b", fontsize=11.0, fontweight="bold", color=CHARCOAL, ha="left", va="center")
    ax_b.text(9.5, 91.0, "Orthogonal attribution design", fontsize=9.2, fontweight="bold", color=CHARCOAL, ha="left", va="center")

    # 5 Unified Soft-Medallion Input Cards (ref_1a_food_style.png / ref_1b_paired_grid.png polish)
    sw = 12.6
    gap = (68.5 - 5 * sw) / 4.0
    in_slots = [
        ("S", "Structure", CHARCOAL, "#EEF2F6"),
        ("P", "Presence", BLUE, BLUE_LT),
        ("Q", "Quantity", BLUE, BLUE_LT),
        ("W", "Potency", AMBER, AMBER_LT),
        ("A", "Anchors", "#B84A00", "#FDE8D7"),
    ]
    for i, (sym, lbl, col, bgc) in enumerate(in_slots):
        sx = 4.0 + i * (sw + gap)
        sy = 97.2
        add_rbox(ax_b, sx, sy, sw, 21.4, fc="white", ec=LINE, lw=0.60, r=1.4, zorder=2)
        ax_b.text(sx + sw / 2, sy + 3.5, sym, fontsize=8.3, fontweight="bold", color=col, ha="center", va="center", zorder=3)
        cx, cy = sx + sw / 2, sy + 11.4
        # Uniform soft pastel circular medallion behind each crisp icon
        ax_b.add_patch(Circle((cx, cy), 4.6, facecolor=bgc, edgecolor="none", zorder=2.5))
        if sym == "S":
            ang = np.linspace(0, 2 * np.pi, 7) + np.pi / 6
            ax_b.plot(cx + 2.8 * np.cos(ang), cy + 2.8 * np.sin(ang), color=CHARCOAL, lw=0.88, zorder=3.5)
        elif sym == "P":
            add_rbox(ax_b, cx - 3.3, cy - 1.5, 6.6, 3.0, fc="white", ec=BLUE, lw=0.72, r=1.4, zorder=3.5)
            ax_b.plot(cx + 1.5, cy, "o", ms=3.8, color=BLUE, zorder=4)
        elif sym == "Q":
            for bi, bh_i in enumerate([2.2, 3.8, 5.4]):
                ax_b.add_patch(Rectangle((cx - 3.1 + bi * 2.2, cy + 2.6 - bh_i), 1.6, bh_i, facecolor=BLUE, edgecolor="none", zorder=3.5))
        elif sym == "W":
            ux = np.linspace(-3.0, 3.0, 25)
            uy = cy + 2.3 - 4.6 * np.log1p((ux + 3.0) * 0.9) / np.log1p(6.0 * 0.9)
            ax_b.plot([cx - 3.2, cx - 3.2, cx + 3.3], [cy - 2.6, cy + 2.5, cy + 2.5], color=SUB, lw=0.55, zorder=3.5)
            ax_b.plot(cx + ux, uy, color=AMBER, lw=1.10, zorder=4)
        elif sym == "A":
            ax_b.plot([cx - 3.4, cx + 3.4], [cy, cy], color="#B84A00", lw=0.95, zorder=3.5)
            for tx_k in [-2.8, 0.0, 2.8]:
                ax_b.plot([cx + tx_k, cx + tx_k], [cy - 1.1, cy + 1.1], color="#B84A00", lw=0.68, zorder=3.5)
            ax_b.plot(cx + 1.0, cy, "v", ms=4.2, color="#B84A00", zorder=4)
        ax_b.text(sx + sw / 2, sy + 18.6, lbl, fontsize=6.8, color=INK, ha="center", va="center", zorder=3)

    # Orthogonal Lock-and-Vary Matrix (muted cool-grey lock cells vs bold Vary pills)
    ax_b.text(4.0, 124.8, "Contrast axis", fontsize=7.3, fontweight="bold", color=CHARCOAL, ha="left", va="center")
    col_x = [36.2, 51.0, 65.4]
    for cx_m, chdr in zip(col_x, ["Inputs S–W", "Learner M", "Anchors A"]):
        ax_b.text(cx_m, 124.8, chdr, fontsize=7.1, fontweight="bold", color=SUB, ha="center", va="center")
    ax_b.plot([4.0, 72.5], [127.8, 127.8], color=LINE, lw=0.65, zorder=2)

    mat_rows = [
        ("Information", "Fig. 3a", BLUE, ["VARY", "LOCK", "LOCK"]),
        ("Model", "Fig. 4b", VIOLET, ["LOCK", "VARY", "LOCK"]),
        ("Calibration", "Fig. 5a", AMBER, ["LOCK", "LOCK", "VARY"]),
    ]
    for ri, (rname, rfig, rcol, states) in enumerate(mat_rows):
        ry = 129.6 + ri * 11.8
        add_rbox(ax_b, 4.0, ry, 68.5, 10.2, fc=STRIPE_BG, ec=HAIR, lw=0.55, r=1.3, zorder=1)
        ax_b.add_patch(Rectangle((4.0, ry), 2.1, 10.2, facecolor=rcol, edgecolor="none", zorder=2))
        ax_b.text(7.5, ry + 3.6, rname, fontsize=7.7, fontweight="bold", color=rcol, ha="left", va="center", zorder=3)
        ax_b.text(7.5, ry + 7.6, rfig, fontsize=6.8, color=SUB, ha="left", va="center", zorder=3)
        for ci, st in enumerate(states):
            cx_m = col_x[ci]
            if st == "LOCK":
                add_rbox(ax_b, cx_m - 5.4, ry + 2.2, 10.8, 5.8, fc="#EDF1F5", ec="none", lw=0, r=2.4, zorder=2.5)
                draw_padlock(ax_b, cx_m, ry + 4.3, locked=True, color="#7B8794", zorder=3.5)
            else:
                add_rbox(ax_b, cx_m - 5.8, ry + 2.2, 11.6, 5.8, fc=rcol, ec=rcol, lw=0.5, r=2.6, zorder=3)
                ax_b.text(cx_m, ry + 5.1, "Vary", fontsize=7.0, fontweight="bold", color="white", ha="center", va="center", zorder=4)

    # Bottom 706 Registry Donut + Exact 4 Labels + Upright-Subscript Formula
    ax_donut = fig.add_axes([3.6 / W, 1.0 - 197.5 / H, 28.0 / W, 28.0 / H])
    ax_donut.set_gid("Panel_b_RegistryDonut")
    counts_reg = [175, 137, 113, 281]
    cols_reg = [BLUE, VIOLET, AMBER, "#CBD5E0"]
    ax_donut.pie(
        counts_reg, colors=cols_reg, startangle=90, counterclock=False,
        wedgeprops=dict(width=0.38, edgecolor="white", linewidth=1.0)
    )
    ax_donut.text(0, 0.08, "706", fontsize=8.4, fontweight="bold", color=CHARCOAL, ha="center", va="center")
    ax_donut.text(0, -0.22, "contrasts", fontsize=6.6, color=SUB, ha="center", va="center")
    ax_donut.set_aspect("equal")

    leg_items = [
        (BLUE, "175 Information"),
        (VIOLET, "137 Model"),
        (AMBER, "113 Calibration"),
        ("#94A3B8", "281 Mixed (multi-factor)"),
    ]
    for li, (lc, ltxt) in enumerate(leg_items):
        ly_i = 170.5 + li * 4.7
        ax_b.add_patch(Rectangle((33.2, ly_i - 1.2), 2.8, 2.4, facecolor=lc, edgecolor="none", zorder=3))
        ax_b.text(37.2, ly_i, ltxt, fontsize=6.95, color=INK, ha="left", va="center", zorder=3)

    add_rbox(ax_b, 32.0, 190.6, 41.4, 7.8, fc=BLUE_LT, ec="none", lw=0, r=1.2, zorder=2)
    ax_b.text(
        52.7, 194.5,
        r"$\Delta R = R_{\mathrm{cand}} - R_{\mathrm{ref}}$ (<0 lowers error)",
        fontsize=6.75, fontweight="bold", color=BLUE, ha="center", va="center", zorder=3
    )

    # ---------------------------------------------------------------------
    # Panel c: Charcoal-Header Two-Layer 44-Slice Audit Matrix (ref_1c_audit_matrix.jpg)
    # ---------------------------------------------------------------------
    ax_c = fig.add_axes([0, 0, 1, 1], xlim=(0, W), ylim=(H, 0))
    ax_c.set_axis_off()
    ax_c.patch.set_alpha(0.0)
    ax_c.set_gid("Panel_c_Identifiability")

    ax_c.text(79.2, 91.0, "c", fontsize=11.0, fontweight="bold", color=CHARCOAL, ha="left", va="center")
    ax_c.text(84.7, 91.0, "Attribution identifiability across 44 slices", fontsize=9.2, fontweight="bold", color=CHARCOAL, ha="left", va="center")
    ax_c.text(84.7, 96.2, r"Predictive accuracy $\neq$ attribution identifiability (criteria A1–A6)", fontsize=7.25, color=SUB, ha="left", va="center")

    x_task = 79.6
    x_dots = [110.2 + i * 2.95 for i in range(5)]
    dot_hdrs = ["S", "P", "Q", "W", "A"]
    x_aud_start = 126.2
    aud_w = 8.40
    aud_cols = [
        ("A1", "Holdout"),
        ("A2", "Collin."),
        ("A3", "Assign."),
        ("A4", "Cover."),
        ("A5", "Base."),
        ("A6", "Anchor"),
    ]

    # Deep Charcoal Grouped Column Header Banners (#2C3E50 fill, bold white text — ref_1c_audit_matrix.jpg!)
    hdr_y, hdr_h = 100.0, 5.2
    ax_c.add_patch(Rectangle((79.0, hdr_y), 29.4, hdr_h, facecolor=CHARCOAL, edgecolor="none", zorder=2))
    ax_c.text(93.7, hdr_y + hdr_h / 2.0, "Task & holdout unit", fontsize=7.1, fontweight="bold", color="white", ha="center", va="center", zorder=3)

    ax_c.add_patch(Rectangle((109.2, hdr_y), 16.0, hdr_h, facecolor=CHARCOAL, edgecolor="none", zorder=2))
    ax_c.text(117.2, hdr_y + hdr_h / 2.0, "Inputs", fontsize=7.1, fontweight="bold", color="white", ha="center", va="center", zorder=3)

    ax_c.add_patch(Rectangle((126.0, hdr_y), 50.6, hdr_h, facecolor=CHARCOAL, edgecolor="none", zorder=2))
    ax_c.text(151.3, hdr_y + hdr_h / 2.0, "Identifiability audit (A1–A6)", fontsize=7.1, fontweight="bold", color="white", ha="center", va="center", zorder=3)

    # Crisp Sub-Header Row underneath charcoal banners
    sub_y = 108.4
    ax_c.text(x_task, sub_y, "Domain · Holdout (slices)", fontsize=6.6, color=SUB, ha="left", va="center", zorder=3)
    for xp, dh in zip(x_dots, dot_hdrs):
        ax_c.text(xp, sub_y, dh, fontsize=6.9, fontweight="bold", color=CHARCOAL, ha="center", va="center", zorder=3)
    for j, (acode, asub) in enumerate(aud_cols):
        cx = x_aud_start + j * aud_w + aud_w / 2.0
        ax_c.text(cx, 107.4, acode, fontsize=7.0, fontweight="bold", color=CHARCOAL, ha="center", va="center", zorder=3)
        ax_c.text(cx, 110.8, asub, fontsize=6.3, color=SUB, ha="center", va="center", zorder=3)

    ax_c.plot([79.0, 176.6], [113.0, 113.0], color=CHARCOAL, lw=0.75, zorder=2)

    task_rows = [
        ("T1", "Molecules & flora (6)", "Split / Family", [1, 1, 0, 0, 0]),
        ("T2", "Fruit cohorts (10)", "Cultivar group", [0, 1, 1, 1, 1]),
        ("T3", "Binary sweeteners (1)", "Ingredient pair", [1, 1, 1, 1, 1]),
        ("T4", "Residuals & ternary (4)", "Pair / 3 edges", [1, 1, 1, 1, 1]),
        ("T5", "Cross-crop transfer (6)", "Crop + anchors", [0, 0, 1, 1, 1]),
        ("T6", "Food interventions (17)", "Shared-ref / pair", [0, 0, 1, 1, 1]),
    ]

    row_y0 = 114.2
    row_h = 11.8
    for r_idx, (tcode, tname, thold, active_inputs) in enumerate(task_rows):
        ry = row_y0 + r_idx * row_h
        if r_idx % 2 == 1:
            ax_c.add_patch(Rectangle((79.0, ry), 97.6, row_h, facecolor=STRIPE_BG, edgecolor="none", zorder=1))

        ax_c.text(x_task, ry + 4.0, f"{tcode} {tname.split('(')[0].strip()}", fontsize=7.1, fontweight="bold", color=INK, ha="left", va="center", zorder=3)
        n_str = "(" + tname.split("(")[1]
        ax_c.text(x_task, ry + 8.5, f"{thold} · {n_str}", fontsize=6.6, color=SUB, ha="left", va="center", zorder=3)

        for idx_d, act in enumerate(active_inputs):
            xx = x_dots[idx_d]
            if act:
                ax_c.plot(xx, ry + 6.0, marker="o", ms=4.0, color=BLUE, zorder=3)
            else:
                ax_c.plot(xx, ry + 6.0, marker="o", ms=3.8, markerfacecolor="white", markeredgecolor="#CBD5E0", markeredgewidth=0.72, zorder=3)

        sub_counts = f1_counts[f1_counts["task"] == tcode].set_index("check")
        for j, (acode, _) in enumerate(aud_cols):
            row_c = sub_counts.loc[acode]
            n_clr, n_lim, n_na = int(row_c["clear"]), int(row_c["limited"]), int(row_c["na"])
            tot = n_clr + n_lim + n_na
            bx = x_aud_start + j * aud_w + 0.65
            bw = aud_w - 1.30
            # Two-layer anti-clutter cell: Top 42% crisp 3-state proportional bar; Bottom 58% clean unobstructed numbers!
            by = ry + 1.5
            bh = 4.2
            ax_c.add_patch(Rectangle((bx, by), bw, bh, facecolor=NA_FILL, edgecolor="#D1D5DB", linewidth=0.45, zorder=2))
            cur_x = bx
            if n_clr > 0:
                w_c = bw * (n_clr / tot)
                ax_c.add_patch(Rectangle((cur_x, by), w_c, bh, facecolor=TEAL, edgecolor="white", linewidth=0.35, zorder=3))
                cur_x += w_c
            if n_lim > 0:
                w_l = bw * (n_lim / tot)
                draw_hatched_rect(ax_c, cur_x, by, w_l, bh, fc="#FEF0D9", ec=GOLD, lw=0.42, step=1.3, zorder=3)

            ax_c.text(
                bx + bw / 2.0, ry + 8.8, f"{n_clr}/{n_lim}/{n_na}",
                fontsize=6.8, fontweight="bold" if (n_clr > 0 or n_lim > 0) else "normal",
                color=INK if (n_clr > 0 or n_lim > 0) else MUTED,
                ha="center", va="center", zorder=4
            )

    ax_c.plot([79.0, 176.6], [row_y0 + 6 * row_h + 0.6, row_y0 + 6 * row_h + 0.6], color=CHARCOAL, lw=0.70, zorder=2)

    # Centered single-row legend + T5 footnote
    ly = 188.6
    ax_c.add_patch(Rectangle((85.0, ly), 5.0, 3.5, facecolor=TEAL, edgecolor=TEAL, lw=0.4, zorder=3))
    ax_c.text(91.2, ly + 1.75, "Clear", fontsize=7.2, fontweight="bold", color=INK, ha="left", va="center")

    draw_hatched_rect(ax_c, 105.0, ly, 5.0, 3.5, fc="#FEF0D9", ec=GOLD, lw=0.42, step=1.25, zorder=3)
    ax_c.text(111.2, ly + 1.75, "Design-limited", fontsize=7.2, fontweight="bold", color=INK, ha="left", va="center")

    ax_c.add_patch(Rectangle((139.0, ly), 5.0, 3.5, facecolor=NA_FILL, edgecolor="#CBD5E0", lw=0.5, zorder=3))
    ax_c.text(145.2, ly + 1.75, "N/A (clear/lim./N/A)", fontsize=7.0, color=SUB, ha="left", va="center")

    ax_c.text(
        79.2, 196.6,
        "Note: T5 evaluates scale calibration; A2/A3 are scored N/A (collinearity inherited from T2).",
        fontsize=6.55, color=SUB, ha="left", va="center"
    )

    svg_path = OUT_DIR / "PsyMix_Figure1.svg"
    pdf_path = OUT_DIR / "PsyMix_Figure1.pdf"
    fig.savefig(svg_path, format="svg")
    fig.savefig(pdf_path, format="pdf")
    plt.close(fig)
    print("Saved WP1 Figure 1:", pdf_path)


# =========================================================================
# WP2 — FIGURE 2: 180.0 mm x 210.0 mm (Final 3 Precision Tweaks)
# =========================================================================
def build_figure2():
    W, H = 180.0, 210.0
    fig = plt.figure(figsize=(W / 25.4, H / 25.4), dpi=600)

    ax_bg = fig.add_axes([0, 0, 1, 1], xlim=(0, W), ylim=(H, 0))
    ax_bg.set_axis_off()
    ax_bg.set_gid("Canvas_Background")
    ax_bg.add_patch(Rectangle((0, 0), W, H, facecolor="white", edgecolor="none", zorder=0))
    ax_bg.plot([4.0, W - 4.0], [63.5, 63.5], color=HAIR, lw=0.85, zorder=1)
    ax_bg.plot([4.0, W - 4.0], [146.5, 146.5], color=HAIR, lw=0.85, zorder=1)

    ax_hdr = fig.add_axes([0, 0, 1, 1], xlim=(0, W), ylim=(H, 0))
    ax_hdr.set_axis_off()
    ax_hdr.patch.set_alpha(0.0)
    ax_hdr.set_gid("Panel_Headers")

    # ---------------------------------------------------------------------
    # TIER 1 — Panel a: Molecular Taste Classification (y = 4 .. 61 mm)
    # ---------------------------------------------------------------------
    ax_hdr.text(4.0, 6.8, "a", fontsize=11.0, fontweight="bold", color=CHARCOAL, ha="left", va="center")
    ax_hdr.text(9.5, 6.8, "Molecular taste classification across canonical and unseen-scaffold splits (T1a)", fontsize=9.0, fontweight="bold", color=CHARCOAL, ha="left", va="center")

    ax_a1 = fig.add_axes([35.0 / W, 1.0 - 54.8 / H, 58.0 / W, 42.0 / H])
    ax_a1.set_gid("Panel_a_Forest")

    mol_rows = [
        ("Uni-Flavor (official)", 0.820285, None, None, 0.973402, "canon"),
        ("Tree ensemble", 0.783756, 0.724133, 0.861986, 0.961607, "canon"),
        ("ECFP4 + CatBoost", 0.779436, 0.725653, 0.862876, 0.963424, "canon"),
        ("ChemBERTa Transf.", 0.754334, 0.728100, 0.812629, 0.972122, "canon"),
        ("Logistic", 0.671914, 0.607687, 0.777472, 0.920749, "canon"),
        ("Tree ensemble", 0.638765, None, None, 0.920155, "scaf"),
        ("Logistic", 0.515000, None, None, 0.912136, "scaf"),
    ]
    y_pos = [0, 1, 2, 3, 4, 5.9, 6.9]

    ax_a1.axhspan(-0.55, 4.45, color="#F6FAFD", zorder=0)
    ax_a1.axhspan(5.35, 7.45, color="#F7F5FC", zorder=0)
    ax_a1.axhline(4.78, color=LINE, lw=0.7, ls="--", zorder=1)

    for yp, (lbl, ap, lo, hi, auroc, grp) in zip(y_pos, mol_rows):
        col = BLUE if grp == "canon" else VIOLET
        if lo is not None and hi is not None:
            ax_a1.plot([lo, hi], [yp, yp], color=col, lw=1.6, solid_capstyle="round", zorder=3)
        ax_a1.plot(ap, yp, marker="o" if grp == "canon" else "D", ms=5.2, color=col, markeredgecolor="white", markeredgewidth=0.5, zorder=4)
        ax_a1.text(0.912, yp, f"{ap:.3f}   {auroc:.3f}", fontsize=7.3, color=INK, ha="left", va="center", clip_on=False)

    # WP2 #3: Generous right shift (x = 0.476, +0.026 off black Y-axis spine!)
    ax_a1.text(0.476, -0.96, "Released canonical split (n = 3,441)", fontsize=7.3, fontweight="bold", color=BLUE, ha="left", va="center", clip_on=False)
    ax_a1.text(0.476, 5.14, "Unseen Murcko scaffolds (586 scaffolds)", fontsize=7.2, fontweight="bold", color=VIOLET, ha="left", va="center",
               bbox=dict(boxstyle="square,pad=0.06", facecolor="white", edgecolor="none", alpha=0.88), clip_on=False, zorder=5)
    ax_a1.text(0.912, -0.96, "AP     AUROC", fontsize=7.3, fontweight="bold", color=SUB, ha="left", va="center", clip_on=False)

    ax_a1.set_yticks(y_pos)
    ax_a1.set_yticklabels([r[0] for r in mol_rows], fontsize=7.4, color=INK)
    ax_a1.invert_yaxis()
    ax_a1.set_xlim(0.45, 0.90)
    ax_a1.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax_a1.set_xlabel("Six-label macro-AP (95% CI where estimable)", fontsize=7.7, fontweight="bold", color=INK, labelpad=3)
    ax_a1.spines["top"].set_visible(False)
    ax_a1.spines["right"].set_visible(False)
    ax_a1.spines["left"].set_linewidth(0.65)
    ax_a1.spines["bottom"].set_linewidth(0.65)
    ax_a1.tick_params(width=0.65, length=2.8, pad=2.5)
    ax_a1.grid(axis="x", color=HAIR, lw=0.5, zorder=0)

    # Right of Panel a: Murcko Core vs Side-Chain Diagram + Cross-Level Attenuation
    add_rbox(ax_hdr, 123.0, 11.6, 53.0, 48.0, fc="#FAFBFD", ec=LINE, lw=0.65, r=1.6, zorder=1)
    ax_hdr.text(149.5, 15.8, "Scaffold split & cross-level transfer", fontsize=7.6, fontweight="bold", color=CHARCOAL, ha="center", va="center", zorder=3)

    cx_s, cy_s = 136.5, 26.8
    ang6 = np.linspace(0, 2 * np.pi, 7)
    ax_hdr.plot(cx_s + 4.2 * np.cos(ang6), cy_s + 4.2 * np.sin(ang6), color=VIOLET, lw=1.5, zorder=3)
    ax_hdr.add_patch(Circle((cx_s, cy_s), 2.1, facecolor="none", edgecolor=VIOLET, lw=0.9, ls="--", zorder=3))
    ax_hdr.plot([cx_s + 4.2, cx_s + 7.5], [cy_s, cy_s - 2.2], color=GREY, lw=1.1, zorder=3)
    ax_hdr.text(cx_s + 8.0, cy_s - 2.4, "−R", fontsize=7.1, color=SUB, ha="left", va="center", zorder=3)
    ax_hdr.plot([cx_s - 2.1, cx_s - 5.2], [cy_s + 3.6, cy_s + 5.8], color=GREY, lw=1.1, zorder=3)
    ax_hdr.text(cx_s - 5.6, cy_s + 6.0, "R'−", fontsize=7.1, color=SUB, ha="right", va="center", zorder=3)
    ax_hdr.text(151.0, 24.6, "0 shared scaffolds", fontsize=7.3, fontweight="bold", color=VIOLET, ha="left", va="center", zorder=3)
    ax_hdr.text(151.0, 29.2, "Salty test n = 1", fontsize=7.1, color=CRIMSON, fontweight="bold", ha="left", va="center", zorder=3)

    ax_att = fig.add_axes([138.0 / W, 1.0 - 55.6 / H, 34.5 / W, 16.0 / H])
    ax_att.set_gid("Panel_a_AttenuationInset")
    ax_att.barh([0, 1], [0.110, 0.0013], height=0.52, color=[BLUE, TEAL], zorder=3)
    ax_att.errorbar([0.110, 0.0013], [0, 1], xerr=[[0.110 - 0.014, 0.0013 - (-0.008)], [0.173 - 0.110, 0.011 - 0.0013]],
                    fmt="none", ecolor=INK, elinewidth=1.0, capsize=2.2, zorder=4)
    ax_att.axvline(0.0, color=SLATE, lw=0.65, ls="--", zorder=2)
    ax_att.text(0.116, 0.36, "+0.110", fontsize=7.0, fontweight="bold", color=BLUE, ha="left", va="center")
    ax_att.text(0.018, 1.0, "+0.0013", fontsize=7.0, fontweight="bold", color=TEAL, ha="left", va="center")
    ax_att.set_yticks([0, 1])
    ax_att.set_yticklabels(["Mol.", "Flora"], fontsize=7.1, color=INK)
    ax_att.invert_yaxis()
    ax_att.set_xlim(-0.02, 0.21)
    ax_att.set_xticks([0.0, 0.1, 0.2])
    ax_att.set_title(r"Logistic $\rightarrow$ Tree gain ($\Delta\mathrm{AP}$)", fontsize=7.0, fontweight="bold", color=CHARCOAL, pad=3)
    ax_att.spines["top"].set_visible(False)
    ax_att.spines["right"].set_visible(False)
    ax_att.spines["left"].set_linewidth(0.55)
    ax_att.spines["bottom"].set_linewidth(0.55)
    ax_att.tick_params(labelsize=6.9, width=0.55, length=2.2, pad=1.5)

    # ---------------------------------------------------------------------
    # TIER 2 — Panel b: Botanical Multi-Flavor Sets (y = 67 .. 144 mm)
    # ---------------------------------------------------------------------
    ax_hdr.text(4.0, 69.8, "b", fontsize=11.0, fontweight="bold", color=CHARCOAL, ha="left", va="center")
    ax_hdr.text(9.5, 69.8, "Botanical constituent-set multi-flavor prediction (T1b, DmcFlavor-aligned)", fontsize=9.0, fontweight="bold", color=CHARCOAL, ha="left", va="center")

    ax_b0 = fig.add_axes([4.0 / W, 1.0 - 122.0 / H, 32.0 / W, 43.0 / H])
    ax_b0.set_gid("Panel_b_FlavorDonut")
    flv_counts = [384, 228, 201, 86, 64, 42, 7]
    flv_cols = [CRIMSON, VIOLET, AMBER, TEAL, "#A0AEC0", BLUE, SLATE]
    ax_b0.pie(
        flv_counts, colors=flv_cols, startangle=110, counterclock=False,
        wedgeprops=dict(width=0.36, edgecolor="white", linewidth=0.9)
    )
    ax_b0.set_aspect("equal")
    ax_b0.set_title("Zhuang flora (645)", fontsize=7.6, fontweight="bold", color=CHARCOAL, pad=2)

    herb_im = load_cropped_rgba(SPEC_DIR / "herb.png")
    place_image_mm(fig, W, H, herb_im, 13.5, 95.8, 13.0, 13.8, gid="Specimen_Herb_Center")

    flv_leg = [
        (CRIMSON, "Bitter (384)"),
        (VIOLET, "Pungent (228)"),
        (AMBER, "Sweet (201)"),
        (TEAL, "Astring. (86)"),
        ("#A0AEC0", "Bland (64)"),
        (BLUE, "Sour (42)"),
        (SLATE, "Salty (7)"),
    ]
    for fi, (fc_i, ftxt) in enumerate(flv_leg):
        fy = 83.2 + fi * 5.2
        ax_hdr.add_patch(Circle((37.8, fy), 1.2, facecolor=fc_i, edgecolor="none", zorder=3))
        ax_hdr.text(40.1, fy, ftxt, fontsize=6.9, fontweight="bold" if fi < 3 else "normal", color=INK if fi < 4 else SUB, ha="left", va="center", zorder=3)
    ax_hdr.text(4.8, 133.2, r"Achiral astringency AUROC: 0.555 $\rightarrow$ 0.473", fontsize=6.8, color=SUB, ha="left", va="center")

    # WP2 #2: Wide 21.5 mm gutter between ax_b1 (68.0..111.5) and ax_b2 (133.0..176.0) + compact 1-col legends (> 30 mm gap!)
    ax_b1 = fig.add_axes([68.0 / W, 1.0 - 120.0 / H, 43.5 / W, 39.5 / H])
    ax_b1.set_gid("Panel_b_Ranking")

    cp_sub = herb_df[(herb_df["dataset"] == "Pharmacopoeia2025") & (herb_df["task"] == "six")].set_index("model")
    rk_metrics = ["Top1", "Top2", "Top3", "MRR"]
    rk_labels = ["Top-1", "Top-2", "Top-3", "MRR"]
    x_rk = np.arange(len(rk_metrics))

    vals_att = [float(cp_sub.loc["A2a_Attention", m]) for m in rk_metrics]
    vals_dmc = [float(cp_sub.loc["DmcStyleTransformer_A2a", m]) for m in rk_metrics]
    vals_pri = [float(cp_sub.loc["TasteFrequency", m]) for m in rk_metrics]

    dx_b1 = 0.21
    ax_b1.plot(x_rk - dx_b1, vals_att, "o", ms=5.0, color=BLUE, markeredgecolor="white", markeredgewidth=0.45, label="Set-attention", zorder=5)
    ax_b1.plot(x_rk, vals_dmc, "D", ms=4.5, color=TEAL, markeredgecolor="white", markeredgewidth=0.45, label="Dmc-style SetTr", zorder=4)
    ax_b1.plot(x_rk + dx_b1, vals_pri, "s", ms=4.5, markerfacecolor="white", markeredgecolor=GREY, markeredgewidth=1.25, label="NULL-label-rate", zorder=4)

    ax_b1.annotate(
        "Prior = 0.992", xy=(2 + dx_b1, 1.012), xytext=(2 + dx_b1, 1.084),
        fontsize=6.9, fontweight="bold", color=CRIMSON, ha="center", va="center",
        arrowprops=dict(arrowstyle="-|>,head_length=0.45,head_width=0.25", mutation_scale=7, color=CRIMSON, lw=0.75),
        annotation_clip=False
    )
    ax_b1.set_title("External CP2025 ranking (n = 128)", fontsize=7.6, fontweight="bold", color=CHARCOAL, pad=6)
    ax_b1.set_xticks(x_rk)
    ax_b1.set_xticklabels(rk_labels, fontsize=7.2, color=INK)
    ax_b1.set_xlim(-0.58, 3.58)
    ax_b1.set_ylim(0.52, 1.12)
    ax_b1.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax_b1.set_ylabel("Hit rate / MRR", fontsize=7.6, fontweight="bold", color=INK, labelpad=3)
    ax_b1.spines["top"].set_visible(False)
    ax_b1.spines["right"].set_visible(False)
    ax_b1.spines["left"].set_linewidth(0.65)
    ax_b1.spines["bottom"].set_linewidth(0.65)
    ax_b1.tick_params(width=0.65, length=2.6, pad=2)
    ax_b1.grid(axis="y", color=HAIR, lw=0.5, zorder=0)
    ax_b1.legend(
        loc="upper center", bbox_to_anchor=(0.48, -0.19), ncol=1,
        frameon=False, fontsize=6.7, handletextpad=0.35, labelspacing=0.30
    )

    ax_b2 = fig.add_axes([133.0 / W, 1.0 - 120.0 / H, 43.0 / W, 39.5 / H])
    ax_b2.set_gid("Panel_b_Discrimination")

    def hval(ds, prot, tsk, mdl, col):
        q = herb_df[(herb_df["dataset"] == ds) & (herb_df["protocol"] == prot) & (herb_df["task"] == tsk) & (herb_df["model"] == mdl)]
        return float(q[col].iloc[0])

    cohorts = [
        ("Zhuang\nrandom", "Zhuang645", "random"),
        ("Zhuang\nfamily", "Zhuang645", "family"),
        ("CP2025\nexternal", "Pharmacopoeia2025", "external"),
    ]
    xc = np.arange(len(cohorts))
    au_core4 = [hval(ds, pr, "shared4", "A2a_Attention", "Macro_AUROC") for _, ds, pr in cohorts]
    au_six = [hval(ds, pr, "six", "A2a_Attention", "Macro_AUROC") for _, ds, pr in cohorts]
    au_pri = [hval(ds, pr, "six", "TasteFrequency", "Macro_AUROC") for _, ds, pr in cohorts]
    ap_core4 = [hval(ds, pr, "shared4", "A2a_Attention", "Macro_AP") for _, ds, pr in cohorts]

    ax_b2.axhline(0.50, color=LINE, lw=0.65, ls="--", zorder=1)
    for i in range(len(cohorts)):
        ax_b2.plot([i, i - 0.13], [au_pri[i], au_core4[i]], color="#D5DCE4", lw=0.95, zorder=2)
        ax_b2.plot([i, i + 0.13], [au_pri[i], au_six[i]], color="#D5DCE4", lw=0.95, zorder=2)

    ax_b2.plot(xc - 0.13, au_core4, "D", ms=5.2, color=AMBER, markeredgecolor="white", markeredgewidth=0.45, label="Set-attention (4 core)", zorder=4)
    ax_b2.plot(xc + 0.13, au_six, "o", ms=5.0, color=BLUE, markeredgecolor="white", markeredgewidth=0.45, label="Set-attention (6 attr.)", zorder=4)
    ax_b2.plot(xc, au_pri, "o", ms=4.8, markerfacecolor="white", markeredgecolor=GREY, markeredgewidth=1.25, label="NULL-label-rate (6 attr.)", zorder=4)

    ap_labels = [f"4-core AP: {ap_core4[0]:.2f}", f"{ap_core4[1]:.2f}", f"{ap_core4[2]:.2f}"]
    x_off_ap = [0.18, 0.0, -0.04]
    for i in range(len(cohorts)):
        ax_b2.text(i + x_off_ap[i], max(au_core4[i], au_six[i]) + 0.028, ap_labels[i],
                   fontsize=6.8, color=AMBER, fontweight="bold", ha="center", va="bottom", clip_on=False)

    ax_b2.set_title("Label-wise macro-AUROC", fontsize=7.6, fontweight="bold", color=CHARCOAL, pad=6)
    ax_b2.set_xticks(xc)
    ax_b2.set_xticklabels([c[0] for c in cohorts], fontsize=7.1, color=INK)
    ax_b2.set_xlim(-0.52, 2.52)
    ax_b2.set_ylim(0.28, 0.79)
    ax_b2.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7])
    ax_b2.spines["top"].set_visible(False)
    ax_b2.spines["right"].set_visible(False)
    ax_b2.spines["left"].set_linewidth(0.65)
    ax_b2.spines["bottom"].set_linewidth(0.65)
    ax_b2.tick_params(width=0.65, length=2.6, pad=2)
    ax_b2.grid(axis="y", color=HAIR, lw=0.5, zorder=0)
    ax_b2.legend(
        loc="upper center", bbox_to_anchor=(0.48, -0.25), ncol=1,
        frameon=False, fontsize=6.7, handletextpad=0.35, labelspacing=0.30
    )

    # ---------------------------------------------------------------------
    # TIER 3 — Panel c: 481x Iso-Sweet Span + Darjeeling Omissions (y = 149 .. 206 mm)
    # ---------------------------------------------------------------------
    ax_hdr.text(4.0, 151.6, "c", fontsize=11.0, fontweight="bold", color=CHARCOAL, ha="left", va="center")
    ax_hdr.text(9.5, 151.6, r"Qualitative probabilities saturate across a 481$\times$ dose span and fail to weight tea omissions", fontsize=8.8, fontweight="bold", color=CHARCOAL, ha="left", va="center")

    ax_c1 = fig.add_axes([15.5 / W, 1.0 - 200.2 / H, 74.5 / W, 39.2 / H])
    ax_c1.set_gid("Panel_c_SweetenerSaturation")

    pts_cfg = [
        ("apm", "Aspartame", "S7", 64, True),
        ("suc", "Sucrose", "S3", 2222, True),
        ("sor", "D-Sorbitol", "S3", 5413, True),
        ("reb", "Reb A", "S7", 10395, False),
        ("ste", "Stevioside", "S7", 11254, False),
        ("man", "D-Mannitol", "S7", 616, False),
    ]
    coords = {}
    for code, name, sh, rw, is_top in pts_cfg:
        conc = float(monomers.loc[code, "wv_percent_SE3"])
        r_dmc = dmc_sweet[(dmc_sweet["sheet"] == sh) & (dmc_sweet["row"] == rw)].iloc[0]
        ps = float(r_dmc["prob_Sweet"])
        coords[code] = (conc, ps)
        if is_top:
            ax_c1.plot(conc, ps, "o", ms=6.0, color=AMBER, markeredgecolor="white", markeredgewidth=0.6, zorder=5)
        else:
            dom_c = CRIMSON if code in ("reb", "ste") else SLATE
            ax_c1.plot(conc, ps, "s", ms=5.0, markerfacecolor=dom_c, markeredgecolor="white", markeredgewidth=0.5, zorder=5)

    c_apm, _ = coords["apm"]
    c_sor, _ = coords["sor"]

    ax_c1.text(0.0048, 0.84, "Aspartame\n0.990", fontsize=6.9, fontweight="bold", color=INK, ha="left", va="top")
    ax_c1.text(coords["suc"][0] * 0.70, 0.985, "Sucrose 0.992", fontsize=6.9, fontweight="bold", color=INK, ha="right", va="center")
    ax_c1.text(17.5, 0.84, "D-Sorbitol\n0.992", fontsize=6.9, fontweight="bold", color=INK, ha="right", va="top")

    # WP2 #1: Shortened Aspartame dashed leader line by ~5 pt so it terminates well above top whitespace (ZERO grazing of NH2!)
    ax_c1.plot([c_apm * 1.45, 0.052], [0.960, 0.805], color="#B8C4D0", lw=0.80, ls="--", zorder=2)
    ax_c1.plot([c_sor * 0.85, 2.35], [0.960, 0.775], color="#B8C4D0", lw=0.80, ls="--", zorder=2)

    ax_c1.annotate("Reb A (Bitter 0.95)", xy=coords["reb"], xytext=(0.015, 0.24),
                   fontsize=6.8, color=CRIMSON, fontweight="bold", ha="left", va="center",
                   arrowprops=dict(arrowstyle="-", color=CRIMSON, lw=0.6))
    ax_c1.annotate("Stevioside (Bitter 0.99)", xy=coords["ste"], xytext=(0.036, 0.08),
                   fontsize=6.8, color=CRIMSON, ha="left", va="center",
                   arrowprops=dict(arrowstyle="-", color=CRIMSON, lw=0.6))
    ax_c1.annotate("D-Mannitol\n(Salty 0.98)", xy=coords["man"], xytext=(2.5, 0.16),
                   fontsize=6.8, color=SLATE, fontweight="bold", ha="right", va="center",
                   arrowprops=dict(arrowstyle="-", color=SLATE, lw=0.6))

    ax_c1.annotate(
        "", xy=(c_sor, 1.06), xytext=(c_apm, 1.06),
        arrowprops=dict(arrowstyle="<->,head_length=0.5,head_width=0.28", mutation_scale=7, color=AMBER, lw=1.1),
        annotation_clip=False
    )
    ax_c1.text(np.sqrt(c_apm * c_sor), 1.10, r"481$\times$ iso-sweet mass span (0.013% vs 6.249% w/v)",
               fontsize=7.0, fontweight="bold", color=AMBER, ha="center", va="bottom", clip_on=False)

    ax_c1.set_xscale("log")
    ax_c1.set_xlim(0.004, 19.5)
    ax_c1.set_ylim(-0.06, 1.13)
    ax_c1.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
    ax_c1.set_xlabel("3% sucrose-equivalent concentration (% w/v, log scale)", fontsize=7.6, fontweight="bold", color=INK, labelpad=3)
    ax_c1.set_ylabel("DmcFlavor P(Sweet)", fontsize=7.6, fontweight="bold", color=INK, labelpad=3)
    ax_c1.spines["top"].set_visible(False)
    ax_c1.spines["right"].set_visible(False)
    ax_c1.spines["left"].set_linewidth(0.65)
    ax_c1.spines["bottom"].set_linewidth(0.65)
    ax_c1.tick_params(width=0.65, length=2.8, pad=2)
    ax_c1.grid(color=HAIR, lw=0.5, zorder=0)

    # Top vector overlay: 1-line marker key + matched razor-sharp Aspartame & D-Sorbitol RDKit vector structures
    ax_mol_top = fig.add_axes([0, 0, 1, 1], xlim=(0, W), ylim=(H, 0))
    ax_mol_top.set_axis_off()
    ax_mol_top.patch.set_alpha(0.0)
    ax_mol_top.set_gid("Panel_c_VectorMolecules")

    ax_mol_top.plot(16.8, 156.0, "o", ms=4.8, color=AMBER, zorder=8)
    ax_mol_top.text(18.6, 156.0, "Sweet-saturated", fontsize=6.7, color=SUB, ha="left", va="center", zorder=8)
    ax_mol_top.plot(40.2, 156.0, "s", ms=4.5, color=CRIMSON, zorder=8)
    ax_mol_top.text(41.9, 156.0, "Misclassified bitter", fontsize=6.7, color=SUB, ha="left", va="center", zorder=8)
    ax_mol_top.plot(66.8, 156.0, "s", ms=4.5, color=SLATE, zorder=8)
    ax_mol_top.text(68.5, 156.0, "Misclassified salty", fontsize=6.7, color=SUB, ha="left", va="center", zorder=8)

    # WP2 #1: Matched jet-black pure-vector clarity (bond_lw=0.92, crisp merged OH/HO/NH2) placed slightly lower for extra whitespace!
    draw_rdkit_svg_vector(ax_mol_top, OUT_DIR / "mol_aspartame.svg", 30.2, 173.8, 22.0, 12.6, color=INK, bond_lw=0.92, min_atom_pt=4.1, font_mult=2.80, halo=True, zorder=8)
    ax_mol_top.text(41.2, 187.4, "Aspartame", fontsize=6.5, color=SUB, ha="center", va="center", zorder=8)

    draw_rdkit_svg_vector(ax_mol_top, OUT_DIR / "mol_sorbitol.svg", 59.5, 173.8, 21.0, 12.2, color=INK, bond_lw=0.92, min_atom_pt=4.1, font_mult=2.80, halo=True, zorder=8)
    ax_mol_top.text(70.0, 187.0, "D-Sorbitol", fontsize=6.5, color=SUB, ha="center", va="center", zorder=8)

    # Right: Darjeeling 8 Bitterness Omissions + Watercolor Tea Specimen
    ax_c2 = fig.add_axes([120.0 / W, 1.0 - 200.2 / H, 54.0 / W, 37.5 / H])
    ax_c2.set_gid("Panel_c_DarjeelingOmissions")

    om_models = [
        ("DmcFlavor (n = 16)", "DmcFlavor author S3/S7 molecular table"),
        ("Uni-Flavor (n = 10)", "UniFlavor_author_code_project_heads"),
        ("Tree ensemble (n = 22)", "A2aET"),
        ("Logistic (n = 22)", "A2aLR"),
    ]
    yo = np.arange(len(om_models))

    ax_c2.axvline(0.647963, color=TEAL, lw=1.1, ls="--", zorder=2)
    ax_c2.text(
        0.668, 2.50, "Measured DoT\n0.648 (9 cmpd)",
        fontsize=6.8, fontweight="bold", color=TEAL, ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="none", alpha=0.94),
        zorder=4
    )

    for i, (mlbl, mkey) in enumerate(om_models):
        v_eq = darj_summary[(mkey, "equal")]
        v_pr = darj_summary[(mkey, "molecular_probability")]
        ax_c2.annotate(
            "", xy=(v_pr - 0.005, i), xytext=(v_eq + 0.005, i),
            arrowprops=dict(arrowstyle="-|>,head_length=0.45,head_width=0.25", mutation_scale=6.5, color=CRIMSON, lw=1.1),
            zorder=3
        )
        ax_c2.plot(v_eq, i, "o", ms=4.5, color=SLATE, markeredgecolor="white", markeredgewidth=0.35, zorder=4, label="Equal molar" if i == 0 else None)
        ax_c2.plot(v_pr, i, "o", ms=4.7, color=AMBER, markeredgecolor="white", markeredgewidth=0.35, zorder=5, label="Prob.-weighted" if i == 0 else None)
        ax_c2.text(
            v_pr + 0.015, i, rf"{v_eq:.3f} $\rightarrow$ {v_pr:.3f}",
            fontsize=6.9, fontweight="bold" if i == 0 else "normal",
            color=INK, ha="left", va="center", clip_on=False
        )

    ax_c2.set_yticks(yo)
    ax_c2.set_yticklabels([m[0] for m in om_models], fontsize=7.2, color=INK)
    ax_c2.invert_yaxis()
    ax_c2.set_xlim(0.60, 1.15)
    ax_c2.set_xticks([0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
    ax_c2.set_xlabel("Darjeeling bitterness omission MAE (0–3 scale)", fontsize=7.6, fontweight="bold", color=INK, labelpad=3)
    ax_c2.spines["top"].set_visible(False)
    ax_c2.spines["right"].set_visible(False)
    ax_c2.spines["left"].set_linewidth(0.65)
    ax_c2.spines["bottom"].set_linewidth(0.65)
    ax_c2.tick_params(width=0.65, length=2.8, pad=2.5)
    ax_c2.grid(axis="x", color=HAIR, lw=0.5, zorder=0)

    ax_c2.legend(
        loc="lower left", bbox_to_anchor=(-0.03, 1.04), ncol=2,
        frameon=False, fontsize=6.9, borderpad=0.1, handletextpad=0.25, columnspacing=0.8
    )
    tea_im = load_cropped_rgba(SPEC_DIR / "tea.png")
    place_image_mm(fig, W, H, tea_im, 165.8, 152.5, 10.6, 10.6, gid="Specimen_Tea_Fig2c")

    svg_path = OUT_DIR / "PsyMix_Figure2.svg"
    pdf_path = OUT_DIR / "PsyMix_Figure2.pdf"
    fig.savefig(svg_path, format="svg")
    fig.savefig(pdf_path, format="pdf")
    plt.close(fig)
    print("Saved WP2 Figure 2:", pdf_path)


if __name__ == "__main__":
    build_figure1()
    build_figure2()
    print("WP1 (FIGURE 1) & WP2 (FIGURE 2) FINAL VISUAL-SPEC MASTERS GENERATED.")
