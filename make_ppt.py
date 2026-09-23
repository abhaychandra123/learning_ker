"""Generate progress presentation for the standoff propagator validation project."""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pathlib import Path

FIGURES = Path(__file__).parent / "figures"
OUT = Path(__file__).parent / "progress_presentation.pptx"

# Colours
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BLACK = RGBColor(0x00, 0x00, 0x00)
DARK_BLUE = RGBColor(0x1B, 0x3A, 0x5C)
MED_BLUE = RGBColor(0x2C, 0x5F, 0x8A)
LIGHT_BLUE = RGBColor(0xD6, 0xE8, 0xF7)
ACCENT = RGBColor(0xC0, 0x39, 0x2B)
GREY = RGBColor(0x55, 0x55, 0x55)
LIGHT_GREY = RGBColor(0xF2, 0xF2, 0xF2)

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)


def set_slide_bg(slide, color):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_notes(slide, text):
    notes = slide.notes_slide
    tf = notes.notes_text_frame
    tf.text = text


def add_textbox(slide, left, top, width, height, text, font_size=18,
                bold=False, color=BLACK, alignment=PP_ALIGN.LEFT,
                font_name="Calibri", line_spacing=1.2):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.font.name = font_name
    p.alignment = alignment
    p.space_after = Pt(0)
    p.line_spacing = Pt(font_size * line_spacing)
    return tf


def add_para(tf, text, font_size=18, bold=False, color=BLACK,
             alignment=PP_ALIGN.LEFT, font_name="Calibri",
             space_before=0, space_after=6, line_spacing=1.2,
             indent_level=0):
    p = tf.add_paragraph()
    p.text = text
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.font.name = font_name
    p.alignment = alignment
    p.space_before = Pt(space_before)
    p.space_after = Pt(space_after)
    p.line_spacing = Pt(font_size * line_spacing)
    p.level = indent_level
    return p


def add_header_bar(slide, text):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), SLIDE_W, Inches(0.9))
    shape.fill.solid()
    shape.fill.fore_color.rgb = DARK_BLUE
    shape.line.fill.background()
    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.6)
    tf.margin_top = Inches(0.15)
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.font.name = "Calibri"


def add_footer(slide, slide_num, total):
    add_textbox(slide, Inches(0.5), Inches(7.05), Inches(12), Inches(0.35),
                f"{slide_num} / {total}", font_size=10, color=GREY,
                alignment=PP_ALIGN.RIGHT)


def add_bullet_slide(slide, title, bullets, sub_bullets=None):
    add_header_bar(slide, title)
    tf = add_textbox(slide, Inches(0.7), Inches(1.2), Inches(11.8), Inches(5.8),
                     "", font_size=20, color=BLACK)
    tf.paragraphs[0].text = ""
    for i, b in enumerate(bullets):
        sz = 20
        clr = BLACK
        bld = False
        indent = 0
        if b.startswith(">>"):
            b = b[2:].strip()
            indent = 1
            sz = 17
            clr = GREY
        elif b.startswith("**") and b.endswith("**"):
            b = b[2:-2]
            bld = True
            clr = DARK_BLUE
        sp_before = 12 if i > 0 and indent == 0 else 2
        add_para(tf, f"• {b}" if indent == 0 else f"  - {b}",
                 font_size=sz, bold=bld, color=clr,
                 space_before=sp_before, indent_level=indent)


def add_figure_slide(slide, title, img_path, caption="", img_top=1.15,
                     img_max_w=12.0, img_max_h=5.4):
    add_header_bar(slide, title)
    from PIL import Image
    im = Image.open(img_path)
    w_px, h_px = im.size
    aspect = w_px / h_px
    if img_max_w / aspect <= img_max_h:
        w = img_max_w
        h = w / aspect
    else:
        h = img_max_h
        w = h * aspect
    left = (13.333 - w) / 2
    slide.shapes.add_picture(str(img_path), Inches(left), Inches(img_top),
                             Inches(w), Inches(h))
    if caption:
        add_textbox(slide, Inches(0.5), Inches(img_top + h + 0.08),
                    Inches(12.3), Inches(0.9),
                    caption, font_size=14, color=GREY,
                    alignment=PP_ALIGN.LEFT)


# =====================================================================
def build():
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    blank = prs.slide_layouts[6]  # blank layout
    TOTAL = 22

    sn = 0

    # ---- SLIDE 1: Title ----
    sn += 1
    s = prs.slides.add_slide(blank)
    set_slide_bg(s, DARK_BLUE)
    add_textbox(s, Inches(1), Inches(1.8), Inches(11.3), Inches(2.0),
                "Standoff Propagator Validation\nfor Micro-Coil Magnetic Fields",
                font_size=36, bold=True, color=WHITE, alignment=PP_ALIGN.LEFT,
                font_name="Calibri", line_spacing=1.3)
    add_textbox(s, Inches(1), Inches(4.0), Inches(11.3), Inches(0.6),
                "Learning the Blur Kernel from COMSOL Field Data",
                font_size=22, bold=False, color=LIGHT_BLUE,
                alignment=PP_ALIGN.LEFT)
    add_textbox(s, Inches(1), Inches(5.2), Inches(11.3), Inches(1.0),
                "Abhay Chandra\nProgress Presentation, September 2026",
                font_size=18, bold=False, color=RGBColor(0xAA, 0xCC, 0xEE),
                alignment=PP_ALIGN.LEFT, line_spacing=1.5)
    add_notes(s, """SPEAKER SCRIPT:
Good morning/afternoon. Today I will walk you through my progress on the standoff propagator validation project.

The goal of this work is to learn the blur kernel that describes how magnetic field maps change as we move away from a micro-coil surface. I have been working with COMSOL simulation data, and I have some important findings about both the physics and the data quality that I want to share with you.

Let me start with an outline of what we will cover.""")

    # ---- SLIDE 2: Outline ----
    sn += 1
    s = prs.slides.add_slide(blank)
    add_header_bar(s, "Outline")
    items = [
        "1.  Problem Statement: What are we trying to learn?",
        "2.  The Coil Model and Data",
        "3.  The Physics: Standoff Propagator",
        "4.  Why Fourier Space (not real-space convolution)?",
        "5.  Analysis Pipeline Built",
        "6.  Visual Inspection of the Data (7 figure sets)",
        "7.  Key Finding: Mesh Noise Limits Usable Bandwidth",
        "8.  Practical Implications and Next Steps",
    ]
    tf = add_textbox(s, Inches(1.5), Inches(1.4), Inches(10), Inches(5.5),
                     "", font_size=22, color=BLACK)
    tf.paragraphs[0].text = ""
    for item in items:
        add_para(tf, item, font_size=22, color=DARK_BLUE, space_before=14, bold=False)
    add_footer(s, sn, TOTAL)
    add_notes(s, """SPEAKER SCRIPT:
Here is the roadmap for today. I will start by framing the problem, then describe the coil and the data we are working with. After that I will explain the physics of the standoff propagator, which is the core equation behind this project. Then I will show you the analysis pipeline I have built and walk through all the figures one by one. The main finding comes from the spectral analysis, and I will finish with what this means practically and what the next steps are.

The presentation has a lot of figures because I want to show you every angle of the data before we get to the conclusion.""")

    # ---- SLIDE 3: Problem Statement ----
    sn += 1
    s = prs.slides.add_slide(blank)
    add_bullet_slide(s, "Problem Statement", [
        "**Goal: Learn the \"blur kernel\" that relates magnetic field planes at different heights**",
        "A micro-coil produces a magnetic field. As we move away from the coil surface, the field gets blurred (fine details decay faster than coarse ones)",
        "We have COMSOL simulations of the field at 11 different standoff heights above the coil",
        "Can we learn the kernel that transforms the field at one height to the field at another height?",
        "**This is not discovery. The kernel is known analytically. Learning it is a validation step.**",
        "The gap between the fitted kernel and the analytic one tells us how much extra smoothing the COMSOL mesh is injecting. That is a real result about the simulation's trustworthiness",
    ])
    add_footer(s, sn, TOTAL)
    add_notes(s, """SPEAKER SCRIPT:
So what are we actually trying to do here? We have a micro-coil, and COMSOL gives us the magnetic field at 11 different heights above it. As you go higher, the field gets blurred because fine spatial details decay faster than broad ones.

The key point I want to emphasise is that this is not a discovery problem. The blur kernel is known analytically. It comes straight from Laplace's equation. So when we learn it from the data, we are really doing a validation. We are asking: does the COMSOL data actually follow the physics it should follow?

And the answer turns out to be interesting. The gap between what we fit and what the theory says tells us exactly how much numerical smoothing the COMSOL mesh is adding on top of the real physics. That is a quantitative statement about the simulation quality.""")

    # ---- SLIDE 4: Coil Model ----
    sn += 1
    s = prs.slides.add_slide(blank)
    add_header_bar(s, "The Coil Model")
    specs = [
        ("Model", "Multilayer square spiral coil (COMSOL 5.5)"),
        ("Geometry", "2 um track width, 4 um pitch, 3 layers, 4 coils (~4.5 turns)"),
        ("Drive current", "600 uA"),
        ("Export grid", "0.1 um point resolution, giving 1400 x 1400 pixels per plane"),
        ("Field of view", "140 x 140 um (x: -42 to 98 um, y: -39 to 101 um)"),
        ("Coil footprint", "x in [-20, 22] um, y in [-22, 22] um (fully inside the FOV)"),
        ("Standoff heights", "z = 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 um (11 planes)"),
        ("Components", "Bx, By, Bz, |B| at each plane"),
    ]
    tf = add_textbox(s, Inches(0.7), Inches(1.2), Inches(11.8), Inches(5.8),
                     "", font_size=18, color=BLACK)
    tf.paragraphs[0].text = ""
    for label, val in specs:
        p = add_para(tf, f"{label}:  {val}", font_size=18, color=BLACK,
                     space_before=10)
        p.clear()
        run1 = p.add_run()
        run1.text = f"{label}:  "
        run1.font.size = Pt(18)
        run1.font.bold = True
        run1.font.color.rgb = DARK_BLUE
        run1.font.name = "Calibri"
        run2 = p.add_run()
        run2.text = val
        run2.font.size = Pt(18)
        run2.font.bold = False
        run2.font.color.rgb = BLACK
        run2.font.name = "Calibri"
    add_para(tf, "", font_size=10, space_before=16)
    add_para(tf, "Total raw data: ~2.5 GB of CSV, converted to 260 MB of .npy arrays (44 arrays + grid metadata)",
             font_size=16, color=GREY, space_before=8)
    add_footer(s, sn, TOTAL)
    add_notes(s, """SPEAKER SCRIPT:
This is the COMSOL model we are working with. It is a multilayer square spiral coil with 2 micrometre tracks on a 4 micrometre pitch, 3 layers, and 4 coils giving about 4.5 turns total. It is driven at 600 microamps.

The data was exported on a very fine grid: 0.1 micrometre resolution, which gives us nearly 2 million points per plane, or 1400 by 1400 pixels. We have 11 planes at different standoff heights from 0.5 to 10 micrometres.

At each height we get all four field components: Bx, By, Bz, and the magnitude |B|. So the total dataset is 44 arrays. The raw CSVs were about 2.5 gigabytes, but I converted them into numpy arrays which are about 260 megabytes and load instantly.

One thing to note: the coil sits entirely inside the field of view, but it is not centred. More than half of the FOV on the right side is empty, and two feed traces run out through the left boundary. This becomes relevant later.""")

    # ---- SLIDE 5: Propagator Physics ----
    sn += 1
    s = prs.slides.add_slide(blank)
    add_header_bar(s, "The Physics: Standoff Propagator")
    tf = add_textbox(s, Inches(0.7), Inches(1.2), Inches(11.8), Inches(5.8),
                     "", font_size=18, color=BLACK)
    tf.paragraphs[0].text = ""
    add_para(tf, "In source-free space above the coil, Laplace's equation governs the field.",
             font_size=20, color=BLACK, space_before=4)
    add_para(tf, "Any plane at height z1 determines every plane above it, exactly, with zero free parameters:",
             font_size=20, color=BLACK, space_before=12)
    add_para(tf, "B_tilde(k, z2)  =  B_tilde(k, z1) * exp(-2pi * k * dz)", font_size=26, bold=True,
             color=DARK_BLUE, space_before=20, alignment=PP_ALIGN.CENTER)
    add_para(tf, "where k = sqrt(fx^2 + fy^2) is the radial spatial frequency in cycles/um, and dz = z2 - z1",
             font_size=16, color=GREY, space_before=6, alignment=PP_ALIGN.CENTER)
    add_para(tf, "", font_size=6, space_before=10)
    add_para(tf, "What this means physically:", font_size=20, bold=True, color=DARK_BLUE, space_before=14)
    add_para(tf, "* Each spatial frequency k is attenuated by exp(-2pi*k*dz) as we move up by dz",
             font_size=18, color=BLACK, space_before=8)
    add_para(tf, "* High-k content (fine details) decays exponentially faster than low-k content (broad features)",
             font_size=18, color=BLACK, space_before=6)
    add_para(tf, "* This is why the field looks progressively \"blurred\" with increasing standoff",
             font_size=18, color=BLACK, space_before=6)
    add_para(tf, "* The same operator applies to all three components Bx, By, Bz independently",
             font_size=18, color=BLACK, space_before=6)
    add_footer(s, sn, TOTAL)
    add_notes(s, """SPEAKER SCRIPT:
This is the core equation. Above the coil there are no sources, so the magnetic field satisfies Laplace's equation. This means that if you know the field at any height z1, you can compute it exactly at any height z2 above it. There are zero free parameters.

The equation says: take the 2D Fourier transform of the field at z1, multiply each spatial frequency k by exp(-2pi*k*dz), and inverse transform. That gives you the field at z2.

What this means physically is simple. Each spatial frequency gets attenuated by an exponential factor. High spatial frequencies, meaning fine details, decay much faster than low spatial frequencies, meaning broad features. That is exactly what we see visually: the field looks more and more blurred as you go higher up.

The important point is that this is not a model or an approximation. It is exact, given Laplace's equation. So any deviation we see in the data from this relation is telling us something about the data quality, not about the physics.""")

    # ---- SLIDE 6: Why Fourier Space ----
    sn += 1
    s = prs.slides.add_slide(blank)
    add_header_bar(s, "Why Fourier Space, Not Real-Space Convolution?")
    tf = add_textbox(s, Inches(0.7), Inches(1.2), Inches(11.8), Inches(5.8),
                     "", font_size=18, color=BLACK)
    tf.paragraphs[0].text = ""
    add_para(tf, "The exact real-space kernel is the 2-D Poisson kernel:", font_size=20,
             color=BLACK, space_before=4)
    add_para(tf, "K(r) = dz / ( 2pi * (r^2 + dz^2)^(3/2) )", font_size=22, bold=True,
             color=DARK_BLUE, space_before=12, alignment=PP_ALIGN.CENTER)
    add_para(tf, "", font_size=6, space_before=6)
    add_para(tf, "Problem: This kernel has a very long tail.", font_size=20,
             bold=True, color=ACCENT, space_before=10)
    add_para(tf, "* At radius R = 5*dz, still 20% of the kernel mass is outside the window",
             font_size=18, color=BLACK, space_before=8)
    add_para(tf, "* At R = 20*dz, still 5% is outside",
             font_size=18, color=BLACK, space_before=4)
    add_para(tf, "* For dz = 9.5 um, you would need a ~3800 pixel kernel to capture 95%, wider than the image!",
             font_size=18, color=BLACK, space_before=4)
    add_para(tf, "", font_size=6, space_before=6)
    add_para(tf, "A compact real-space kernel (e.g. 31x31 or 64x64) would truncate the tails and force the fitting to distort the center to compensate, giving wrong results.",
             font_size=18, color=BLACK, space_before=6)
    add_para(tf, "", font_size=6, space_before=6)
    add_para(tf, "Solution: Work in Fourier space, where the propagator is a simple multiplication by exp(-2pi*k*dz). The kernel estimation becomes Wiener-regularised spectral division, which is exact and numerically stable.",
             font_size=18, bold=True, color=DARK_BLUE, space_before=8)
    add_footer(s, sn, TOTAL)
    add_notes(s, """SPEAKER SCRIPT:
This is an important design decision. You might ask: why not just learn a convolution kernel in real space? The reason is that the true kernel, the Poisson kernel, has an extremely long tail.

If you look at the numbers, at a radius of 5 times delta-z, 20 percent of the kernel mass is still outside your window. For a large standoff gap of 9.5 micrometres, you would need a kernel that is about 3800 pixels wide just to capture 95 percent. That is wider than our entire image.

So if you try to fit a compact kernel, say 31 by 31 or 64 by 64 pixels, you are throwing away most of the kernel and the least-squares fit will distort the center to try to compensate. The result would be physically wrong.

The solution is to work in Fourier space instead. There, the propagator is just a pointwise multiplication by exp(-2pi*k*dz). And the kernel estimation becomes a Wiener-regularised spectral division, which has a closed form. It is the same equation we use for validation, so the validation code and the fitting code are actually the same thing.""")

    # ---- SLIDE 7: Validation Checks ----
    sn += 1
    s = prs.slides.add_slide(blank)
    add_header_bar(s, "Three Validation Checks")
    tf = add_textbox(s, Inches(0.7), Inches(1.2), Inches(11.8), Inches(5.8),
                     "", font_size=18, color=BLACK)
    tf.paragraphs[0].text = ""
    checks = [
        ("P: Phase Check", "Phase of H(k) = B_tilde_2 / B_tilde_1 must be identically zero",
         "Are the 11 planes laterally registered? A sub-pixel shift would masquerade as extra smoothing and poison all amplitude numbers."),
        ("B: Propagation Error (NRMSE)", "Apply analytic propagator to plane at z1, compare to actual plane at z2",
         "The headline result. No division, no conditioning issues. This number decides what pairs are usable."),
        ("A: Amplitude Ratio vs. Analytic", "Radially-averaged |H(k)| vs exp(-2pi*k*dz), SNR-masked",
         "Explains the propagation error: at what spatial frequency k does the data break away from theory? This is the usable-bandwidth curve."),
    ]
    for label, what, why in checks:
        add_para(tf, label, font_size=22, bold=True, color=DARK_BLUE, space_before=16)
        add_para(tf, what, font_size=18, color=BLACK, space_before=4)
        add_para(tf, f"Why: {why}", font_size=16, color=GREY, space_before=2)
    add_footer(s, sn, TOTAL)
    add_notes(s, """SPEAKER SCRIPT:
Before we can learn the kernel, we need to validate the data. I defined three checks.

First, the Phase Check. If you take the ratio of two planes in Fourier space, the phase must be exactly zero, because the propagator is real and positive. If you see a nonzero phase, it means the planes are laterally shifted relative to each other, even by a fraction of a pixel. That shift would show up as false extra smoothing and would corrupt everything downstream. So this runs first.

Second, the Propagation Error. This is the headline number. Take the field at z1, apply the exact analytic propagator to predict the field at z2, and measure the normalised RMS error against the actual z2 plane. On perfectly converged data this should be well under one percent.

Third, the Amplitude Ratio. This is the diagnostic that explains the propagation error. We plot the measured transfer function versus the analytic one as a function of spatial frequency. Where they part company, that is the usable bandwidth for that pair. This is the most informative of the three checks.""")

    # ---- SLIDE 8: Pipeline ----
    sn += 1
    s = prs.slides.add_slide(blank)
    add_header_bar(s, "Analysis Pipeline")
    tf = add_textbox(s, Inches(0.7), Inches(1.2), Inches(11.8), Inches(5.8),
                     "", font_size=18, color=BLACK)
    tf.paragraphs[0].text = ""
    modules = [
        ("convert.py", "CSV to .npy cache",
         "Parses the 2.5 GB of COMSOL CSV exports (9 header lines + 1.96M rows each), infers the regular grid, validates uniformity across all 11 planes, and saves 44 compact .npy arrays (~7.8 MB each). Run once, about 3 minutes."),
        ("fields.py", "Loader, Grid, pair enumeration",
         "Single point of truth for paths, grid metadata, and array orientation. Provides load(z, comp) and pairs(reference_z) that yields {z_sharp, z_blur, dz, comp, sharp, blur} dicts, the seam that kernel estimation plugs into."),
        ("propagator.py", "Fourier machinery",
         "k-grid construction, analytic transfer exp(-2pi*k*dz), upward continuation (propagate), Hann windowing, amplitude spectrum, and radial averaging. Currently holds what visualisation needs; masking and kernel estimation land here next."),
        ("visualize.py", "All 10 figures, ~15 seconds",
         "Generates overview, montages (x4), zoom, linecuts, decay, edges, and spectra plots. Colour rules: diverging for signed components, sequential for |B|, ordered ramp for standoff."),
    ]
    for name, role, desc in modules:
        p = add_para(tf, "", font_size=18, color=BLACK, space_before=12)
        p.clear()
        run1 = p.add_run()
        run1.text = f"{name}"
        run1.font.size = Pt(20)
        run1.font.bold = True
        run1.font.color.rgb = DARK_BLUE
        run1.font.name = "Calibri"
        run2 = p.add_run()
        run2.text = f"  :  {role}"
        run2.font.size = Pt(18)
        run2.font.bold = False
        run2.font.color.rgb = BLACK
        run2.font.name = "Calibri"
        add_para(tf, desc, font_size=15, color=GREY, space_before=2)
    add_footer(s, sn, TOTAL)
    add_notes(s, """SPEAKER SCRIPT:
Here is the pipeline I built. There are four Python modules.

convert.py is a one-time script that reads the raw COMSOL CSV files, which are about 2.5 gigabytes total, and converts them into numpy arrays. Each CSV has about 2 million rows. The script infers the grid, checks that all 11 planes share the same grid, and writes out 44 arrays plus a grid metadata file. This runs once in about 3 minutes and after that everything loads instantly.

fields.py is the loading layer. It knows about paths, grid geometry, and array orientation. The key function is pairs(reference_z), which yields dictionaries with the sharp plane, the blurred plane, the delta-z, and the component name. This is the interface that kernel estimation will plug into.

propagator.py has the Fourier machinery: the k-grid, the analytic transfer function, upward continuation, windowing, and radial averaging.

visualize.py generates all the figures. It runs in about 15 seconds end to end and produces 10 figure sets. I will show you all of them now.""")

    # ---- SLIDE 9: Overview figure ----
    sn += 1
    s = prs.slides.add_slide(blank)
    add_figure_slide(s, "Field Components at Four Standoffs",
                     FIGURES / "overview.png",
                     "All four field components (Bx, By, Bz, |B|) at z = 0.5, 2, 5, 10 um. Colour limits are per-panel. Peak field spans ~200x across z. The spiral coil structure is clearly resolved at low standoff and progressively blurs with height.")
    add_footer(s, sn, TOTAL)
    add_notes(s, """SPEAKER SCRIPT:
This is the overview figure. Each row is a different standoff height: 0.5, 2, 5, and 10 micrometres. Each column is a field component: Bx, By, Bz, and the magnitude |B|.

The colour limits are per-panel, so you are comparing shapes, not amplitudes. What you can see immediately is that at z = 0.5, the individual windings of the spiral are clearly resolved, especially in Bz. As you go higher, the fine structure disappears and you are left with a smooth blob by z = 10.

Also notice that the coil is in the lower-left quadrant of the field of view. The upper-right half is essentially empty. And in the Bx column at z = 0.5, you can see some horizontal structures on the left side, those are the feed traces.

The peak field spans about 200 times across these four heights, from about 215 microtesla at z = 0.5 to about 5 microtesla at z = 10.""")

    # ---- SLIDE 10: Bz montage ----
    sn += 1
    s = prs.slides.add_slide(blank)
    add_figure_slide(s, "Bz Across All 11 Standoffs",
                     FIGURES / "montage_Bz.png",
                     "Bz component at every standoff from 0.5 to 10 um. Each panel has its own colour scale. The sharp spiral winding structure at z = 0.5 um progressively smooths out. By z = 10 um only the coarse envelope remains. This is the blur that the propagator describes.")
    add_footer(s, sn, TOTAL)
    add_notes(s, """SPEAKER SCRIPT:
Now let me show you all 11 standoffs for the Bz component, which is the one we focus on since it dominates for a planar coil viewed from above.

Each panel has its own colour scale because the amplitude drops so much. What matters is the shape. At z = 0.5 you can count the individual turns of the spiral. By z = 2 to 3 micrometres the windings start merging. By z = 5 you cannot distinguish individual turns at all. And by z = 10 it is just a smooth rounded square.

This is exactly the blur that the standoff propagator describes: high spatial frequencies die first, so fine details vanish before coarse ones.""")

    # ---- SLIDE 11: |B| montage ----
    sn += 1
    s = prs.slides.add_slide(blank)
    add_figure_slide(s, "|B| (Field Magnitude) Across All Standoffs",
                     FIGURES / "montage_Bnorm.png",
                     "|B| at every standoff. Peak drops from ~140 uT at z = 0.5 to ~5 uT at z = 10 um (a ~27x reduction). The winding pattern dissolves into a smooth blob. Bz dominates the magnitude at every height, as expected for a planar coil viewed from above.")
    add_footer(s, sn, TOTAL)
    add_notes(s, """SPEAKER SCRIPT:
This is the same montage but for the field magnitude |B|. The peak drops from about 140 microtesla at the surface to about 5 microtesla at z = 10, which is about a 27 times reduction.

You can see the same blurring effect. The spiral pattern dissolves into a smooth blob. Bz dominates the magnitude at every height, which makes sense because for a planar coil the out-of-plane component is always the strongest when viewed from directly above.""")

    # ---- SLIDE 12: Zoom Bz ----
    sn += 1
    s = prs.slides.add_slide(blank)
    add_figure_slide(s, "Zoomed Bz: Blur Isolated (Normalised per Panel)",
                     FIGURES / "zoom_Bz.png",
                     "Same 20 um crop of Bz, each panel normalised to its own peak so amplitude decay is removed. Only shape change (blur) remains. At z = 0.5 um, triangular mesh facets from COMSOL's finite-element mesh are clearly visible. By z = 3 um, fine structure is gone.")
    add_footer(s, sn, TOTAL)
    add_notes(s, """SPEAKER SCRIPT:
This figure is important. I am showing a 20 micrometre crop of Bz, and each panel is normalised to its own peak. That means the amplitude decay has been divided out and you are seeing only the shape change, which is the blur.

But look at the z = 0.5 panel carefully. You can see triangular facets. These are the actual finite-element mesh triangles from COMSOL. The field was solved on this mesh and then interpolated onto our fine export grid, but the mesh itself is coarser than 0.1 micrometres, so you can see the individual triangles.

By z = 1 or z = 2, the facets are less visible because the physical blurring smooths them out. And by z = 3 or 4, all the fine structure is gone. This is the first visual hint that the data has a numerical noise floor from the mesh.""")

    # ---- SLIDE 13: Linecuts ----
    sn += 1
    s = prs.slides.add_slide(blank)
    add_figure_slide(s, "Line Cuts Through the Coil, Every Standoff",
                     FIGURES / "linecuts.png",
                     "Top: Bz along one row through the spiral (y = 20 um), absolute values. Sharp spikes at conductor edges (x = +/-20 um) collapse with height. Bottom: same data normalised to each curve's peak, pure shape. Note the stair-step noise at low z (dark curves): these are piecewise-linear mesh interpolation artifacts, not physics.")
    add_footer(s, sn, TOTAL)
    add_notes(s, """SPEAKER SCRIPT:
This is the single most convincing figure for showing the mesh noise. I took one horizontal row through the coil at y = 20 micrometres and plotted Bz for all 11 standoff heights, colour-coded from dark purple at z = 0.5 to yellow at z = 10.

The top panel shows absolute values. You see sharp spikes at the conductor edges around x = plus and minus 20 micrometres. These collapse with height, which is the decay.

The bottom panel is more revealing. Each curve is normalised to its own peak, so you are seeing pure shape. Look at the dark curves, the low standoff ones. They have a jagged, stair-step pattern. This is not physics. The real magnetic field half a micrometre above 2 micrometre tracks would be a smooth, regular ripple at the winding pitch. These random steps are the signature of a piecewise-linear interpolant on a mesh coarser than our sample grid.

If someone asks whether this could be a real field feature, the answer is no. The stair-steps are not periodic at the winding pitch and they are random in amplitude. This is mesh noise.""")

    # ---- SLIDE 14: Decay ----
    sn += 1
    s = prs.slides.add_slide(blank)
    add_figure_slide(s, "Amplitude Decay with Standoff",
                     FIGURES / "decay.png",
                     "Left: peak |field|. Right: RMS. Both log-y. Over z = 0.5 to 10 um the field drops ~43x at peak (215 to 4.9 uT) and ~27x in RMS. The curves bend because high-k content dies first. Peak panel wobbles (single-pixel mesh sensitivity); RMS is perfectly smooth. The disagreement is the mesh noise, measured.")
    add_footer(s, sn, TOTAL)
    add_notes(s, """SPEAKER SCRIPT:
Here I plot the amplitude decay with standoff. Left panel is peak field, right panel is RMS, both on a log scale.

Two things to notice. First, the curves bend. On a log scale, a pure exponential would be a straight line. These are steep at low z and flatten out at high z. That is because the field is a sum of many spatial frequencies, and the high-k ones die off quickly, leaving only the slow-decaying low-k content at higher standoffs. The curve flattening IS the blur, expressed as a scalar.

Second, and more subtle: the peak panel has small wobbles, but the RMS panel is perfectly smooth. Look at Bz peak around z = 8 and z = 9, they are almost flat when they should drop by about 20 percent. That is because peak is a single pixel and it latches onto whichever mesh facet happens to spike. RMS averages nearly 2 million pixels, so the noise cancels. The disagreement between these two panels is the mesh noise, measured quantitatively.""")

    # ---- SLIDE 15: Edges ----
    sn += 1
    s = prs.slides.add_slide(blank)
    add_figure_slide(s, "Boundary Effects: Feed Traces at the Domain Edge",
                     FIGURES / "edges.png",
                     "Left: Bz at z = 0.5 hard-clipped to +/-20 uT (peak is 215) to reveal weak structure, the spiral and two feed traces running left. Middle: zoom on feed traces, showing triangular mesh facets. Right: Bz along the leftmost column at all standoffs. The FFT wraps this hot edge into the empty right side, a real artifact to handle.")
    add_footer(s, sn, TOTAL)
    add_notes(s, """SPEAKER SCRIPT:
This figure is about a boundary effect. The left panel shows Bz at z = 0.5, but I hard-clipped the colour scale to plus or minus 20 microtesla even though the peak is 215. This saturates the coil to solid colour and reveals the weak structure around it.

You can now see the square spiral clearly, and importantly, two feed traces running out to the left. These carry current and produce a field of about 90 microtesla right at the domain boundary.

The middle panel zooms in on those traces. You can see the triangular mesh facets very clearly here.

The right panel shows Bz along the leftmost column of the image at all standoff heights. There are two sharp bipolar spikes at z = 0.5 that collapse with height.

Why does this matter? The FFT treats the image as periodic. On the right edge the field is nearly zero, but on the left edge it is 90 microtesla. Wrapping those together creates a discontinuity that leaks across all spatial frequencies. We handle this with windowing, and the spectra figure I will show next uses a Hann window for this reason.""")

    # ---- SLIDE 16: Spectra ----
    sn += 1
    s = prs.slides.add_slide(blank)
    add_figure_slide(s, "Spectral Analysis: The Central Finding",
                     FIGURES / "spectra.png",
                     "Left: radially-averaged Bz spectrum at each standoff. The high-k tail is entirely numerical noise (physical content at k = 1 cyc/um, z = 10 would be suppressed by ~10^-27). Right: measured transfer ratio (solid) vs analytic exp(-2pi*k*dz) (dashed). The departure point is the usable bandwidth, and it shrinks with dz.",
                     img_max_h=4.8)
    add_footer(s, sn, TOTAL)
    add_notes(s, """SPEAKER SCRIPT:
This is the most important figure. It is the spectral analysis and it contains the central finding of this work so far.

The left panel shows the radially-averaged spectrum of Bz at each standoff. There is a peak near k = 0.05 cycles per micrometre, which corresponds to the overall coil size, then it drops off, and then there is a long shallow tail extending to k = 2.

That tail is the giveaway. Physical content at k = 1 cycle per micrometre at z = 10 micrometres would be suppressed by exp(-2pi*1*10), which is about 10 to the power of negative 27. It would be completely gone. Yet the z = 10 curve sits at about 10 to the minus 5 microtesla out there. So the entire high-k tail is purely numerical noise.

Now the right panel, which is the actual finding. The solid lines are the measured transfer ratio: the spectrum at height z divided by the spectrum at the reference height z = 0.5. The dashed lines are the exact analytic prediction, exp(-2pi*k*dz), colour-matched to each pair.

At low k, the solid sits right on the dashed. The physics is correct there. Then the solid peels off and flattens into a plateau while the dashed plunges down. The point where they separate is the usable bandwidth for that pair.

For a small gap of 0.5 micrometres, the separation happens around k = 0.375. For a large gap of 9.5 micrometres, it happens around k = 0.11. Past that point you are measuring the ratio of two noise floors, which is why the solid curves flatten instead of decaying.""")

    # ---- SLIDE 17: Usable Bandwidth Table ----
    sn += 1
    s = prs.slides.add_slide(blank)
    add_header_bar(s, "Usable Bandwidth vs. Standoff Gap")
    tf = add_textbox(s, Inches(0.7), Inches(1.3), Inches(11.8), Inches(1.0),
                     "Where the measured transfer departs from the analytic curve:",
                     font_size=20, color=BLACK)

    rows_data = [
        ("dz (um)", "k_break (cyc/um)", "Finest resolvable feature"),
        ("0.5", "0.375", "2.7 um"),
        ("1.0", "0.30 - 0.31", "3.2 um"),
        ("1.5", "0.283", "3.5 um"),
        ("2.0", "0.18", "5.6 um"),
        ("3.0", "0.15", "6.6 um"),
        ("5.0", "0.11", "9.0 um"),
    ]
    table_shape = s.shapes.add_table(len(rows_data), 3,
                                      Inches(2.0), Inches(2.2),
                                      Inches(9.0), Inches(2.8))
    table = table_shape.table
    table.columns[0].width = Inches(2.5)
    table.columns[1].width = Inches(3.5)
    table.columns[2].width = Inches(3.0)

    for i, row_data in enumerate(rows_data):
        for j, cell_text in enumerate(row_data):
            cell = table.cell(i, j)
            cell.text = cell_text
            for paragraph in cell.text_frame.paragraphs:
                paragraph.font.size = Pt(16)
                paragraph.font.name = "Calibri"
                paragraph.alignment = PP_ALIGN.CENTER
                if i == 0:
                    paragraph.font.bold = True
                    paragraph.font.color.rgb = WHITE
                else:
                    paragraph.font.color.rgb = BLACK
            if i == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = DARK_BLUE
            elif i % 2 == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = LIGHT_GREY

    tf2 = add_textbox(s, Inches(0.7), Inches(5.2), Inches(11.8), Inches(1.8),
                      "", font_size=18, color=BLACK)
    tf2.paragraphs[0].text = ""
    add_para(tf2, "Critical comparison:", font_size=20, bold=True, color=ACCENT, space_before=4)
    add_para(tf2, "The coil's winding fundamental is at 0.25 cyc/um (= 1 / 4 um pitch).", font_size=18, color=BLACK, space_before=4)
    add_para(tf2, "For dz >= 2 um, this fundamental is at or below the noise floor. The data stops carrying usable information about the coil structure at the very length scale the study cares about.",
             font_size=18, bold=True, color=ACCENT, space_before=4)
    add_footer(s, sn, TOTAL)
    add_notes(s, """SPEAKER SCRIPT:
Let me put numbers on that spectral finding. This table shows the breakpoint spatial frequency, k_break, for each standoff gap, and what physical feature size that corresponds to.

At dz = 0.5 micrometres, we can resolve features down to about 2.7 micrometres. At dz = 2, only down to 5.6 micrometres. At dz = 5, only 9 micrometres.

Now here is the critical comparison. Our coil has 2 micrometre tracks on a 4 micrometre pitch. The fundamental spatial frequency of the winding pattern is 1 over 4, which is 0.25 cycles per micrometre.

Look at the table. For dz of 2 micrometres, k_break is 0.18, which is already below 0.25. That means for standoff gaps of 2 micrometres or more, the coil's own winding pattern is at or below the noise floor. The data does not carry usable information about the coil structure at the very spatial frequency we care about.

This is the central quantitative result: most of our pairs are noise-dominated at the length scale that matters.""")

    # ---- SLIDE 18: What Causes This ----
    sn += 1
    s = prs.slides.add_slide(blank)
    add_header_bar(s, "What Causes the Noise Floor?")
    tf = add_textbox(s, Inches(0.7), Inches(1.2), Inches(11.8), Inches(5.8),
                     "", font_size=18, color=BLACK)
    tf.paragraphs[0].text = ""
    add_para(tf, "It is mesh interpolation noise from COMSOL's finite-element export, not a flaw in our analysis.",
             font_size=20, bold=True, color=DARK_BLUE, space_before=4)
    add_para(tf, "", font_size=6, space_before=6)
    add_para(tf, "Four independent lines of evidence:", font_size=20, bold=True, color=BLACK, space_before=8)
    add_para(tf, "1. Invariant to windowing: the breakpoint is the same whether we use the full plane, a Tukey taper, or an interior crop, so it is not FFT edge leakage",
             font_size=18, color=BLACK, space_before=10)
    add_para(tf, "2. Directly visible: zoom_Bz shows triangular mesh facets across the windings; linecuts show stair-steps of tens of uT, the shape of a piecewise-linear interpolant",
             font_size=18, color=BLACK, space_before=8)
    add_para(tf, "3. Propagation error: analytic propagation leaves 13-17% NRMSE even at dz = 1 um, where an exact relation on converged data should give less than 1%",
             font_size=18, color=BLACK, space_before=8)
    add_para(tf, "4. Scaling with dz: k_break falls with dz exactly as a fixed additive noise floor predicts. Physical content drops as exp(-2pi*k*dz), the floor does not, so the crossing moves to lower k",
             font_size=18, color=BLACK, space_before=8)
    add_para(tf, "", font_size=6, space_before=6)
    add_para(tf, "Root cause: The COMSOL mesh in the air region above the coil is coarser than the 0.1 um export grid. The field is solved on the coarse mesh (likely quadratic elements for A, giving piecewise-linear B = curl(A)), then interpolated onto the fine export grid.",
             font_size=17, color=GREY, space_before=6)
    add_footer(s, sn, TOTAL)
    add_notes(s, """SPEAKER SCRIPT:
I want to make sure you are convinced this is a real mesh artifact and not a bug in my analysis. There are four independent lines of evidence.

First, I tested three different windowing strategies: using the full plane with no window, a Tukey taper, and an interior crop that avoids the edges entirely. The breakpoint frequency is the same in all three cases, to within a few percent. So it is not caused by FFT edge leakage.

Second, the artifact is directly visible. The zoomed Bz figure shows triangular mesh facets. The linecuts show stair-steps of tens of microtesla. These are the exact shapes you would expect from a piecewise-linear interpolant of a mesh coarser than our sample grid.

Third, the propagation error is way too high. When I propagate one plane to another using the exact analytic formula, I get 13 to 17 percent normalised RMS error even at dz = 1 micrometre. For an exact mathematical relation on converged data, this should be well under 1 percent.

Fourth, the way k_break scales with dz is exactly what a fixed additive noise floor predicts. The physical content drops as exp(-2pi*k*dz), but the numerical noise floor stays constant. So the crossing point between the two moves to lower k as dz increases. That is exactly what we observe.

The root cause is most likely that COMSOL is using quadratic elements for the vector potential A, which means B = curl(A) is only piecewise-linear. Then this piecewise-linear field is interpolated onto our 0.1 micrometre export grid.""")

    # ---- SLIDE 19: Practical Implications ----
    sn += 1
    s = prs.slides.add_slide(blank)
    add_header_bar(s, "Practical Implications for Kernel Learning")
    tf = add_textbox(s, Inches(0.7), Inches(1.2), Inches(11.8), Inches(5.8),
                     "", font_size=18, color=BLACK)
    tf.paragraphs[0].text = ""
    add_para(tf, "This is a data-quality ceiling, not a method problem. The formulation is sound.",
             font_size=20, bold=True, color=DARK_BLUE, space_before=4)
    add_para(tf, "", font_size=6, space_before=6)
    add_para(tf, "1.  Usable pairs are limited to dz <= 1.5 um",
             font_size=20, bold=True, color=BLACK, space_before=10)
    add_para(tf, "Anchored at z = 0.5 um, that means targets at z = 1 and z = 2 um. We have 2 to 3 genuinely informative pairs, not 10.",
             font_size=18, color=GREY, space_before=4)
    add_para(tf, "2.  Regularisation parameter lambda is not a free knob",
             font_size=20, bold=True, color=BLACK, space_before=14)
    add_para(tf, "It must roll off at k_break, and k_break depends on dz. Each pair needs its own lambda.",
             font_size=18, color=GREY, space_before=4)
    add_para(tf, "3.  Each pair needs a separate kernel",
             font_size=20, bold=True, color=BLACK, space_before=14)
    add_para(tf, "The propagator is not linear in dz: exp(-k*dz) is different for each dz. We cannot learn one universal kernel. We learn one kernel per standoff gap.",
             font_size=18, color=GREY, space_before=4)
    add_para(tf, "4.  Stacking Bx/By/Bz becomes more valuable, not less",
             font_size=20, bold=True, color=BLACK, space_before=14)
    add_para(tf, "With only 2 to 3 usable pairs, independent constraints are scarce. Using all three field components gives 3x the data per dz.",
             font_size=18, color=GREY, space_before=4)
    add_footer(s, sn, TOTAL)
    add_notes(s, """SPEAKER SCRIPT:
What does all this mean for the kernel learning problem?

First and most importantly, this is a data-quality ceiling, not a method problem. Our formulation is sound. The issue is that the data does not have enough information at the spatial frequencies we need.

There are four practical implications.

One, usable pairs are limited to dz of about 1.5 micrometres or less. If we anchor at z = 0.5, that means we can use targets at z = 1 and z = 2. So instead of 10 informative pairs, we really have 2 to 3.

Two, the regularisation parameter lambda is not something we can tune freely. It has to cut off the fit at k_break, because above that frequency we would be fitting noise. And k_break is different for each pair, so each pair needs its own lambda.

Three, we cannot learn one universal kernel and apply it to all pairs. The propagator depends on dz through the exponential. Each standoff gap has its own unique kernel.

Four, the silver lining. Since we have so few usable pairs, stacking all three field components becomes more valuable. The propagator is the same for Bx, By, and Bz, so using all three gives us three times the data per standoff gap.""")

    # ---- SLIDE 20: Next Steps ----
    sn += 1
    s = prs.slides.add_slide(blank)
    add_header_bar(s, "Next Steps: Requests to the COMSOL Team")
    tf = add_textbox(s, Inches(0.7), Inches(1.2), Inches(11.8), Inches(5.8),
                     "", font_size=18, color=BLACK)
    tf.paragraphs[0].text = ""
    add_para(tf, "In order of leverage:", font_size=20, bold=True, color=DARK_BLUE, space_before=4)
    add_para(tf, "", font_size=6, space_before=4)
    steps = [
        ("1. Refine the air-region mesh (root fix)",
         "Mesh in the air slab above the coil (z = 0 to 12 um) should have max element size <= 0.5 um. A swept/mapped mesh in z would be cost-effective."),
        ("2. Raise discretisation order (quick win)",
         "Quadratic to cubic elements for A. Since B = curl(A), quadratic A gives only piecewise-linear B, which is the faceting we see. Cubic may help a lot without much mesh cost."),
        ("3. One convergence pair (the single most useful deliverable)",
         "Same z = 1 and z = 2 um planes, exported at two mesh refinements (current + refined). If k_break moves up, the diagnosis is confirmed. This costs only two exports."),
        ("4. Finer z spacing near the surface",
         "z = 0.5, 0.75, 1.0, 1.25, 1.5 um would let us compare reference planes at matched dz below 1 um."),
        ("5. Re-centre the export window",
         "The spiral is off-centre and two feed traces exit through x_min at ~90 uT. Re-centring (or extending left) costs nothing and removes a real FFT edge artifact."),
    ]
    for title, desc in steps:
        add_para(tf, title, font_size=19, bold=True, color=BLACK, space_before=12)
        add_para(tf, desc, font_size=16, color=GREY, space_before=2)
    add_footer(s, sn, TOTAL)
    add_notes(s, """SPEAKER SCRIPT:
Here are the next steps, ordered by how much leverage they give us.

The root fix is to refine the mesh in the air region above the coil. The air slab from z = 0 to about 12 micrometres over the coil footprint needs to have a maximum element size at or below 0.5 micrometres. A swept or mapped mesh in z over that region would be cheaper than refining everything.

A potentially quick win is raising the discretisation order. If COMSOL is using quadratic elements for the vector potential A, then B = curl(A) is only piecewise-linear. Going to cubic elements for A would give us piecewise-quadratic B, which could dramatically reduce the faceting without adding many more elements.

But the single most useful thing they could send us is a convergence pair. Just give us the same z = 1 and z = 2 micrometre planes at two mesh refinements, the current one and a finer one. If k_break moves up with refinement, that proves the diagnosis. If it does not move, then my analysis is wrong and we have learned that for the price of two exports. Either way, it is the cheapest experiment with the most information.

On the lower priority side, finer z spacing near the surface would be helpful for comparing reference planes at small matched dz. And re-centring the export window would remove the edge artifact for free.""")

    # ---- SLIDE 21: Summary ----
    sn += 1
    s = prs.slides.add_slide(blank)
    set_slide_bg(s, DARK_BLUE)
    add_textbox(s, Inches(1), Inches(0.8), Inches(11.3), Inches(0.8),
                "Summary", font_size=32, bold=True, color=WHITE)

    tf = add_textbox(s, Inches(1), Inches(1.8), Inches(11.3), Inches(5.0),
                     "", font_size=20, color=WHITE)
    tf.paragraphs[0].text = ""
    summaries = [
        "Built a complete analysis pipeline: CSV conversion, field loading, Fourier propagator, visualisation (10 figure sets)",
        "Validated the COMSOL data against the exact analytic standoff propagator",
        "Discovered a mesh-interpolation noise floor that limits the usable spatial-frequency bandwidth",
        "For dz >= 2 um, the coil's winding fundamental (0.25 cyc/um) falls below the noise floor, the very scale we care about",
        "Usable pairs for kernel learning are limited to dz <= 1.5 um (2 to 3 pairs, not 10)",
        "This is a data-quality ceiling, not a method problem. The fix is on the COMSOL side (mesh refinement)",
        "Requested convergence pair from the COMSOL team to confirm the diagnosis",
    ]
    for sm in summaries:
        add_para(tf, f"*  {sm}", font_size=18, color=WHITE, space_before=10, line_spacing=1.25)

    add_textbox(s, Inches(1), Inches(6.6), Inches(11.3), Inches(0.5),
                "Thank You. Questions?", font_size=24, bold=True,
                color=LIGHT_BLUE, alignment=PP_ALIGN.CENTER)
    add_notes(s, """SPEAKER SCRIPT:
To summarise. I built a full analysis pipeline from raw COMSOL CSVs all the way to spectral diagnostics. I validated the data against the exact analytic propagator and discovered that there is a mesh-interpolation noise floor that limits how much useful spatial-frequency information the data contains.

The key number is that for standoff gaps of 2 micrometres or more, the coil winding fundamental at 0.25 cycles per micrometre is already at or below this noise floor. So out of the 10 pairs we thought we had, only 2 to 3 are genuinely informative.

This is not a problem with our method. The formulation is correct. It is a data quality issue that needs to be fixed on the COMSOL side by refining the mesh in the air region above the coil.

I have already sent a request to the COMSOL team asking for a convergence pair, which is the cheapest way to confirm this diagnosis.

Thank you. I am happy to take questions.""")

    # ---- SLIDE 22: Q&A Preparation ----
    sn += 1
    s = prs.slides.add_slide(blank)
    add_header_bar(s, "Anticipated Questions and Answers")
    tf = add_textbox(s, Inches(0.7), Inches(1.2), Inches(11.8), Inches(5.8),
                     "", font_size=18, color=BLACK)
    tf.paragraphs[0].text = ""
    qas = [
        ("Q: Could this noise be from your analysis (e.g. FFT artifacts) rather than the mesh?",
         "A: No. The breakpoint is invariant to windowing (full/Tukey/interior crop agree within a few percent), the facets are directly visible in the images, and the NRMSE is 13-17% where it should be <1%. All four lines of evidence point to the mesh."),
        ("Q: Why not just use a smaller real-space kernel and accept some truncation error?",
         "A: The Poisson kernel tail is too heavy. At R = 5*dz, 20% of the mass is still outside. Truncating forces the fit to distort the center. Working in Fourier space avoids this entirely with a closed-form solution."),
        ("Q: Can you still learn something useful from the large-dz pairs?",
         "A: Yes, but only at low spatial frequencies where the physics is already simple and well-understood. The kernel would be well-determined where we already know the answer, and noise-fitted where we do not. It is the high-k band at the winding pitch that matters, and that is exactly where those pairs fail."),
        ("Q: Why can you not learn one kernel for all standoff gaps?",
         "A: Because the propagator exp(-k*dz) is not linear in dz. The kernel for dz = 2 is fundamentally different from the kernel for dz = 5. Each gap has its own kernel, and each must be learned independently."),
        ("Q: What if the COMSOL team cannot refine the mesh?",
         "A: We can still proceed with the 2-3 usable pairs at small dz, stacking Bx/By/Bz for 3x data. The results will be valid but limited to small standoff gaps. We should also explore whether higher-order element export is available in COMSOL."),
    ]
    for q, a in qas:
        add_para(tf, q, font_size=16, bold=True, color=DARK_BLUE, space_before=10)
        add_para(tf, a, font_size=14, color=GREY, space_before=2)
    add_footer(s, sn, TOTAL)
    add_notes(s, """SPEAKER SCRIPT (this is a backup slide, not meant to be presented):

This slide lists anticipated questions. Keep it hidden during the presentation and refer to it during Q&A if needed.

Additional questions that might come up:

Q: How did you choose the reference plane z = 0.5 um?
A: It is the closest plane to the source, so it has the highest spatial-frequency content. Ideally we would anchor at z = 0 (the coil surface), but that is not available. We also checked whether z = 1.0 would be a better anchor using a ladder test approach, but it does not have a measurable advantage at equal dz.

Q: What is the Wiener regularisation parameter lambda?
A: It controls the trade-off between noise amplification and signal suppression in the spectral division. It should be set so that the filter rolls off around k_break, the frequency where data departs from theory. Below k_break we trust the data; above it we suppress. Each pair has its own lambda because each has its own k_break.

Q: How long did this analysis take to build?
A: The pipeline runs in under 5 minutes including conversion. Development was about 2 weeks, with most of the time spent understanding the data and designing the spectral diagnostics.

Q: Why is the field of view asymmetric?
A: The COMSOL export window was set up with the coil in the lower-left quadrant. More than half of the FOV on the right side is empty. This is not a problem per se, but re-centring the window would remove the edge artifact from the feed traces and give us more usable area.

Q: Could you use a different simulation tool instead of COMSOL?
A: The physics is the same regardless of the tool. The issue is mesh resolution in the air region, which any FEM solver would face. The fix is the same: refine the mesh or use higher-order elements.

Q: Is 13-17% NRMSE really that bad?
A: Yes, for an exact mathematical relation. The propagator has zero free parameters. On noise-free data with a converged mesh, the error should be limited only by floating-point precision, well under 0.1%. 13-17% means the mesh noise is comparable to the signal at the spatial frequencies that matter.""")

    # Save
    prs.save(str(OUT))
    print(f"Saved: {OUT}")
    print(f"Total slides: {sn}")


if __name__ == "__main__":
    build()
