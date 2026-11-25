# -*- coding: utf-8 -*-
"""
Export 1:1 Actual-Size JCIM TOC Preview + 1200 dpi Fig. S1 zooms + PyMuPDF Text-Layer QA Audit
"""
import os
import re
import sys

sys.path.insert(0, r'C:\zhuang_taste_pre\psymix_foodatlas_multimodel_20261121\.deps')
import fitz  # PyMuPDF
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'mathtext.fontset': 'dejavusans',
})

OUT_DIR = r'c:\gemi-antigravity\00_PsyMix_Submission_Master\02_Figures_Design_and_Drafts\final_figures\TOC_and_FigS1'


def build_toc_actual_size_preview():
    """
    Create a true-scale (1:1) ACS/JCIM Table-of-Contents proof sheet (8.27 x 4.40 in, A4 width)
    so the TOC graphic occupies EXACTLY 3.25 x 1.75 in (8.25 x 4.45 cm) on the page,
    with an inch/cm calibration ruler and realistic JCIM TOC article listing beside it.
    """
    toc_png = os.path.join(OUT_DIR, 'PsyMix_TOC_Graphic_600dpi.png')
    im = Image.open(toc_png)
    w_px, h_px = im.size
    assert (w_px, h_px) == (1950, 1050), f'Expected 1950x1050 px at 600 dpi, got {w_px}x{h_px}'

    page_w, page_h = 8.27, 4.35  # exact A4 width (210 mm) x 110.5 mm strip
    fig = plt.figure(figsize=(page_w, page_h), dpi=300, facecolor='#F8FAFC')
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0.0, page_w)
    ax.set_ylim(0.0, page_h)
    ax.set_aspect('equal')
    ax.axis('off')

    # White A4 page frame
    ax.add_patch(Rectangle((0.20, 0.20), page_w - 0.40, page_h - 0.40,
                           facecolor='#FFFFFF', edgecolor='#CBD5E1', linewidth=0.9, zorder=1))

    # Header bar simulating ACS JCIM journal TOC strip
    ax.add_patch(Rectangle((0.20, page_h - 0.72), page_w - 0.40, 0.52,
                           facecolor='#1E293B', edgecolor='none', zorder=2))
    ax.text(0.45, page_h - 0.46, 'ACS / JCIM Table of Contents — 1:1 Actual Print Scale Proof (A4 Width = 210 mm)',
            ha='left', va='center', fontsize=9.0, fontweight='bold', color='#FFFFFF', zorder=3)
    ax.text(page_w - 0.45, page_h - 0.46, 'Card: 3.25 × 1.75 in (8.25 × 4.45 cm)',
            ha='right', va='center', fontsize=8.0, color='#CBD5E1', zorder=3)

    # Place the TOC Graphic at EXACT physical dimensions: 3.25 in wide x 1.75 in high
    toc_x0, toc_y0 = 0.50, 1.35
    toc_w, toc_h = 3.25, 1.75
    ax_img = fig.add_axes([toc_x0 / page_w, toc_y0 / page_h, toc_w / page_w, toc_h / page_h], zorder=4)
    ax_img.imshow(im)
    ax_img.axis('off')

    # Crisp hairline dimension guides around the 1:1 TOC box
    ax.add_patch(Rectangle((toc_x0 - 0.015, toc_y0 - 0.015), toc_w + 0.03, toc_h + 0.03,
                           facecolor='none', edgecolor='#94A3B8', linewidth=0.55, linestyle='--', zorder=3))
    # Exact physical ruler directly below the TOC graphic (0 to 3.25 inches)
    ry = toc_y0 - 0.28
    ax.plot([toc_x0, toc_x0 + toc_w], [ry, ry], color='#2C3E50', lw=0.9, zorder=4)
    for tick_in in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.25]:
        tx = toc_x0 + tick_in
        tl = 0.09 if (tick_in % 1.0 == 0 or tick_in == 3.25) else 0.05
        ax.plot([tx, tx], [ry, ry - tl], color='#2C3E50', lw=0.75, zorder=4)
        if tick_in in [0.0, 1.0, 2.0, 3.0, 3.25]:
            lbl = '3.25 in' if tick_in == 3.25 else f'{tick_in:.0f}'
            ax.text(tx, ry - 0.19, lbl, ha='center', va='top', fontsize=6.8, color='#475569', zorder=4)

    # Vertical height callout on the left of the TOC box
    vx = toc_x0 - 0.14
    ax.plot([vx, vx], [toc_y0, toc_y0 + toc_h], color='#2C3E50', lw=0.8, zorder=4)
    ax.plot([vx - 0.04, vx + 0.04], [toc_y0, toc_y0], color='#2C3E50', lw=0.8, zorder=4)
    ax.plot([vx - 0.04, vx + 0.04], [toc_y0 + toc_h, toc_y0 + toc_h], color='#2C3E50', lw=0.8, zorder=4)
    ax.text(vx - 0.08, toc_y0 + toc_h / 2.0, '1.75 in (4.45 cm)', ha='center', va='center',
            rotation=90, fontsize=6.8, color='#475569', zorder=4)

    # Simulated JCIM TOC article title & abstract snippet to the right (authentic print context)
    tx0 = toc_x0 + toc_w + 0.38
    ax.text(tx0, toc_y0 + toc_h - 0.04, 'ARTICLES  |  Journal of Chemical Information and Modeling',
            ha='left', va='top', fontsize=7.0, fontweight='bold', color='#0072B2', zorder=4)
    ax.text(tx0, toc_y0 + toc_h - 0.25,
            'From Molecular Taste Labels to Human Sensory\nResponses: Contributions of Chemical Information,\nModels, and Sensory Calibration in Mixtures and Foods',
            ha='left', va='top', fontsize=9.2, fontweight='bold', color='#0F172A', linespacing=1.28, zorder=4)
    ax.text(tx0, toc_y0 + toc_h - 1.02,
            'PsyMix Benchmark Study  ·  2026  ·  6 Tasks (T1–T6), 153 Models, 706 Contrasts',
            ha='left', va='top', fontsize=7.2, color='#64748B', zorder=4)
    ax.text(tx0, toc_y0 + toc_h - 1.24,
            'Matched contrasts across molecular/botanical classification, fruit cohorts,\n'
            'controlled sweetener mixtures, cross-study transfer, and real food\n'
            'interventions separate information, model, and calibration gains.',
            ha='left', va='top', fontsize=7.5, color='#334155', linespacing=1.35, zorder=4)

    ax.text(0.50, 0.48,
            'Print verification note: When printed at 100% scale on A4 paper, the TOC graphic above measures exactly 3.25 × 1.75 in (8.25 × 4.45 cm).',
            ha='left', va='center', fontsize=7.0, style='italic', color='#64748B', zorder=4)

    out_prev = os.path.join(OUT_DIR, 'PsyMix_TOC_Graphic_ActualSize_Preview.png')
    fig.savefig(out_prev, dpi=300)
    plt.close(fig)
    print(f'[OK] Saved 1:1 Actual-Size Preview -> {out_prev}')


def render_figs1_zooms():
    pdf_path = os.path.join(OUT_DIR, 'PsyMix_FigureS1.pdf')
    doc = fitz.open(pdf_path)
    page = doc[0]
    rect = page.rect
    mat = fitz.Matrix(1200 / 72.0, 1200 / 72.0)

    # Top Panel a crop (top 64%)
    clip_a = fitz.Rect(0, 0, rect.width, rect.height * 0.64)
    pix_a = page.get_pixmap(matrix=mat, clip=clip_a, alpha=False)
    p_a = os.path.join(OUT_DIR, 'zoom_1200dpi_FigS1_top_a.png')
    pix_a.save(p_a)

    # Bottom Panel b crop (bottom 38%)
    clip_b = fitz.Rect(0, rect.height * 0.62, rect.width, rect.height)
    pix_b = page.get_pixmap(matrix=mat, clip=clip_b, alpha=False)
    p_b = os.path.join(OUT_DIR, 'zoom_1200dpi_FigS1_bot_b.png')
    pix_b.save(p_b)
    doc.close()
    print(f'[OK] Saved 1200 dpi zooms -> {p_a}, {p_b}')


def audit_pdf(pdf_path, label, expected_w_pt, expected_h_pt, required_symbols,
              forbidden_tokens, is_toc=False):
    print(f'\n==================== PDF AUDIT: {label} ====================')
    doc = fitz.open(pdf_path)
    assert len(doc) == 1, f'Expected 1 page, got {len(doc)}'
    page = doc[0]
    rect = page.rect
    w_pt, h_pt = rect.width, rect.height
    w_mm, h_mm = w_pt * 25.4 / 72.0, h_pt * 25.4 / 72.0
    print(f'Page size : {w_pt:.2f} x {h_pt:.2f} pt  ({w_mm:.2f} x {h_mm:.2f} mm, {w_pt/72:.3f} x {h_pt/72:.3f} in)')
    assert abs(w_pt - expected_w_pt) < 0.8 and abs(h_pt - expected_h_pt) < 0.8, \
        f'Page size mismatch! Expected {expected_w_pt:.2f}x{expected_h_pt:.2f} pt'

    # Font audit
    fonts = page.get_fonts(full=True)
    font_types = set(f[2] for f in fonts)
    font_names = sorted(set(f[3] for f in fonts))
    print(f'Fonts ({len(font_names)}): {font_names} | Types: {sorted(font_types)}')
    assert 'Type3' not in font_types, f'Forbidden Type3 bitmap font found: {fonts}'

    # Extract spans
    text_dict = page.get_text('dict')
    all_text = []
    min_fs = 999.0
    min_fs_txt = ''
    clipped = []
    for b in text_dict['blocks']:
        if b.get('type') != 0:
            continue
        for line in b.get('lines', []):
            for sp in line.get('spans', []):
                t = sp['text'].strip()
                if not t:
                    continue
                all_text.append(t)
                fs = sp['size']
                if fs < min_fs:
                    min_fs = fs
                    min_fs_txt = t
                x0, y0, x1, y1 = sp['bbox']
                if x0 < -0.5 or y0 < -0.5 or x1 > w_pt + 0.5 or y1 > h_pt + 0.5:
                    clipped.append((t, (round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1))))

    full_txt = ' '.join(all_text)
    print(f'Min font size : {min_fs:.2f} pt (token: {min_fs_txt!r})')
    print(f'Edge clipping : {len(clipped)} spans {clipped[:5] if clipped else "(0 — clean!)"}')

    # Check garbled / replacement glyphs
    bad_chars = [ch for ch in full_txt if ch in ('\ufffd', '□', 'ㅓ', '┤')]
    print(f'Garbled chars : {len(bad_chars)} {bad_chars}')
    assert len(bad_chars) == 0, f'Garbled glyphs found: {bad_chars}'
    assert len(clipped) == 0, f'Clipped text spans found: {clipped}'

    # Required symbols
    missing_syms = [s for s in required_symbols if s not in full_txt]
    present_syms = [s for s in required_symbols if s in full_txt]
    print(f'Symbols OK    : {present_syms} | Missing: {missing_syms}')
    assert len(missing_syms) == 0, f'Missing required symbols: {missing_syms}'

    # Forbidden engineering codes
    found_forb = [tok for tok in forbidden_tokens if tok in full_txt]
    print(f'Forbidden toks: {found_forb if found_forb else "NONE (clean!)"}')
    assert len(found_forb) == 0, f'Forbidden internal engineering tokens leaked: {found_forb}'

    if is_toc:
        # Word count audit (excluding single punctuation symbols like ·, ≠, ≥, +, ×)
        words = [w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-–\.]*", full_txt)]
        # Separate 5 single-letter badges S, P, Q, W, A
        prose_words = [w for w in words if w not in ('S', 'P', 'Q', 'W', 'A', 'r')]
        print(f'TOC full text : {full_txt}')
        print(f'TOC word count: {len(prose_words)} prose words (+ single-letter badges S/P/Q/W/A/r; total tokens={len(words)})')
        assert min_fs >= 6.45, f'TOC min font size {min_fs:.2f} pt < 6.5 pt!'
        assert len(prose_words) <= 24, f'TOC word count {len(prose_words)} > 24!'
        assert 'Fig' not in full_txt and 'Figure' not in full_txt, 'TOC must not contain Figure label!'
    else:
        # Check exact bold lowercase panel tags 'a' and 'b' at large font size (>= 10 pt)
        tags = []
        for b in text_dict['blocks']:
            if b.get('type') != 0: continue
            for line in b.get('lines', []):
                for sp in line.get('spans', []):
                    if sp['size'] >= 10.0 and sp['text'].strip() in ('a', 'b', 'c', 'd', 'A', 'B'):
                        tags.append(sp['text'].strip())
        print(f'Panel tags    : {tags}')
        assert tags == ['a', 'b'], f"Expected lowercase panel tags ['a', 'b'], got {tags}"

    doc.close()
    print(f'[PASS] {label} passed ALL automated PDF text-layer checks!')


if __name__ == '__main__':
    if '--figs1-only' not in sys.argv:
        build_toc_actual_size_preview()
    render_figs1_zooms()

    forbidden = ['A2a', 'fixed8', 'B1_', 'B2_', 'B3_', 'POOL_REG', 'DMC_SET_REG',
                 'PHYSSIM', 'Generic_BG', 'TasteFrequency', 'C1_Ridge', 'C1_ET', 'CLS-LM-FT']

    audit_pdf(
        os.path.join(OUT_DIR, 'PsyMix_TOC_Graphic.pdf'),
        label='PsyMix_TOC_Graphic.pdf',
        expected_w_pt=3.25 * 72.0,
        expected_h_pt=1.75 * 72.0,
        required_symbols=['≠', '≥', '·', '–'],
        forbidden_tokens=forbidden,
        is_toc=True,
    )

    audit_pdf(
        os.path.join(OUT_DIR, 'PsyMix_FigureS1.pdf'),
        label='PsyMix_FigureS1.pdf',
        expected_w_pt=180.0 * 72.0 / 25.4,
        expected_h_pt=175.0 * 72.0 / 25.4,
        required_symbols=['×', '→', '−', '–', 'Δ'],
        forbidden_tokens=forbidden,
        is_toc=False,
    )
