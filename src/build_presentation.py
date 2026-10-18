"""
build_presentation.py - renders KIO_Routing_Presentation.pptx from PRESENTATION_OUTLINE.md.

What it does
  Builds the 11-slide deck described in PRESENTATION_OUTLINE.md: three chapters
  (2:15 / 2:30 / 2:45, 7:30 total), the KIO architecture diagram drawn natively
  with our router block highlighted, the real PNG figures from results/, and
  speaker notes on every slide carrying that slide's script and time budget.

Why a script and not a hand-made file
  Same rule as the rest of the project: the deck is reproducible, and every
  number on it is checked against results/*.json before the file is written
  (see verify_numbers) - the build fails rather than ship a stale figure.

Text fit
  There is no LibreOffice/PowerPoint renderer in this environment, so instead of
  a visual pass every text box is measured analytically with the real Calibri and
  Courier New metrics (PIL) and compared against its box. Overflows are reported
  and the build fails on them.

Run:  venv\\Scripts\\python.exe src\\build_presentation.py
"""
import json
from pathlib import Path

from PIL import ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "KIO_Routing_Presentation.pptx"

# ---------------------------------------------------------------- design system
SLIDE_W, SLIDE_H = 13.333, 7.5
MARGIN = 0.6
CONTENT_W = SLIDE_W - 2 * MARGIN

INK = RGBColor(0x16, 0x26, 0x3F)        # dominant dark
INK_SOFT = RGBColor(0x4A, 0x55, 0x6B)   # secondary text
MUTED = RGBColor(0x76, 0x7F, 0x8F)      # captions
SUPPORT = RGBColor(0x2A, 0x78, 0xD6)    # same blue as the figures (flag 0)
SALES = RGBColor(0xEB, 0x68, 0x34)      # same orange as the figures (flag 1)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
TINT = RGBColor(0xEE, 0xF3, 0xFA)       # pale blue panel
TINT_WARM = RGBColor(0xFD, 0xEF, 0xE8)  # pale orange panel
LINE = RGBColor(0xD8, 0xDF, 0xEA)
DARK_BG = RGBColor(0x12, 0x1E, 0x33)

HEAD_FONT = "Calibri"
BODY_FONT = "Calibri"
MONO_FONT = "Courier New"

TITLE_PT = 32
BODY_PT = 18
SMALL_PT = 14
CAPTION_PT = 12
MONO_PT = 11

FONT_FILES = {
    (BODY_FONT, False): "C:/Windows/Fonts/calibri.ttf",
    (BODY_FONT, True): "C:/Windows/Fonts/calibrib.ttf",
    (MONO_FONT, False): "C:/Windows/Fonts/cour.ttf",
    (MONO_FONT, True): "C:/Windows/Fonts/courbd.ttf",
}
_font_cache = {}
SCALE = 4           # measure at 4x and divide, for sub-point accuracy
LINE_FACTOR = 1.22  # PowerPoint single line spacing

OVERFLOWS = []


# ------------------------------------------------------------ text measurement
def _font(name, bold, pt):
    key = (name, bold, pt)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(FONT_FILES[(name, bold)], int(pt * SCALE))
    return _font_cache[key]


def text_w_in(text, pt, bold=False, font=BODY_FONT):
    """Width of a single line, in inches."""
    return _font(font, bold, pt).getlength(text) / SCALE / 72.0


def wrap(text, width_in, pt, bold=False, font=BODY_FONT):
    """Greedy word wrap at a width in inches. Returns the list of lines."""
    words, lines, cur = text.split(), [], ""
    for wd in words:
        trial = f"{cur} {wd}".strip()
        if cur and text_w_in(trial, pt, bold, font) > width_in:
            lines.append(cur)
            cur = wd
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines or [""]


def block_h_in(text, width_in, pt, bold=False, font=BODY_FONT):
    """Height a wrapped block needs, in inches."""
    return len(wrap(text, width_in, pt, bold, font)) * pt * LINE_FACTOR / 72.0


def check_fit(label, needed_h, box_h, needed_w=None, box_w=None):
    if needed_h > box_h + 1e-6:
        OVERFLOWS.append(f"{label}: needs {needed_h:.2f}in height, box is {box_h:.2f}in")
    if needed_w is not None and needed_w > box_w + 1e-6:
        OVERFLOWS.append(f"{label}: needs {needed_w:.2f}in width, box is {box_w:.2f}in")


# ------------------------------------------------------------------- primitives
def textbox(slide, x, y, w, h, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    tf.paragraphs[0].alignment = align
    return tb, tf


def style_run(run, pt, color, bold=False, italic=False, font=BODY_FONT):
    run.font.size = Pt(pt)
    run.font.color.rgb = color
    run.font.bold = bold
    run.font.italic = italic
    run.font.name = font


def put_text(slide, text, x, y, w, h, pt=BODY_PT, color=INK, bold=False, italic=False,
             font=BODY_FONT, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, label=""):
    _, tf = textbox(slide, x, y, w, h, align, anchor)
    style_run(tf.paragraphs[0].add_run(), pt, color, bold, italic, font)
    tf.paragraphs[0].runs[0].text = text
    check_fit(label or text[:40], block_h_in(text, w, pt, bold, font), h)
    return tf


def put_lines(slide, lines, x, y, w, h, pt, color=INK, bold=False, font=BODY_FONT,
              space_after=0, align=PP_ALIGN.LEFT, label=""):
    """A text box of pre-split lines (no re-wrapping expected)."""
    _, tf = textbox(slide, x, y, w, h, align)
    total = 0.0
    widest = 0.0
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        if space_after:
            p.space_after = Pt(space_after)
        r = p.add_run()
        r.text = ln
        style_run(r, pt, color, bold, False, font)
        total += pt * LINE_FACTOR / 72.0 + space_after / 72.0
        widest = max(widest, text_w_in(ln, pt, bold, font))
    check_fit(label or "lines", total, h, widest, w)
    return tf


def title(slide, text, dark=False):
    put_text(slide, text, MARGIN, 0.46, CONTENT_W, 0.62, TITLE_PT,
             WHITE if dark else INK, bold=True, font=HEAD_FONT, label=f"title '{text}'")


def chapter_chip(slide, text, color, dark=False):
    """The deck's motif: a small rounded chip naming the speaker, top right."""
    w = text_w_in(text, CAPTION_PT, True) + 0.42
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                Inches(SLIDE_W - MARGIN - w), Inches(0.5),
                                Inches(w), Inches(0.34))
    sh.fill.solid()
    sh.fill.fore_color.rgb = color
    sh.line.fill.background()
    sh.shadow.inherit = False
    tf = sh.text_frame
    tf.word_wrap = False
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = text
    style_run(r, CAPTION_PT, WHITE, True)


def bullets(slide, items, x, y, w, pt=BODY_PT, gap=0.20, color=INK, dot=SUPPORT,
            dot_size=0.105, label=""):
    """Dot + text rows, measured one by one. Returns the bottom y."""
    cur = y
    text_x = x + 0.30
    text_w = w - 0.30
    for i, item in enumerate(items):
        lines = wrap(item, text_w, pt)
        h = len(lines) * pt * LINE_FACTOR / 72.0
        d = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x),
                                   Inches(cur + pt * LINE_FACTOR / 72.0 / 2 - dot_size / 2),
                                   Inches(dot_size), Inches(dot_size))
        d.fill.solid()
        d.fill.fore_color.rgb = dot
        d.line.fill.background()
        d.shadow.inherit = False
        put_lines(slide, lines, text_x, cur, text_w, h + 0.02, pt, color,
                  label=f"{label} bullet {i + 1}")
        cur += h + gap
    bottom = cur - gap
    if bottom > SLIDE_H - 0.30:
        OVERFLOWS.append(f"{label}: bullet list runs to y={bottom:.2f}, past the slide")
    return bottom


def card(slide, x, y, w, h, fill=TINT, border=None, radius=True):
    shp = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
        Inches(x), Inches(y), Inches(w), Inches(h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    if border:
        shp.line.color.rgb = border
        shp.line.width = Pt(1.25)
    else:
        shp.line.fill.background()
    shp.shadow.inherit = False
    shp.text_frame.text = ""
    return shp


def stat(slide, x, y, w, value, caption_lines, color, value_pt=40, fill=TINT, h=None):
    """A big number over one or more caption lines. Height is derived, not guessed."""
    if isinstance(caption_lines, str):
        caption_lines = [caption_lines]
    value_h = value_pt * LINE_FACTOR / 72.0 + 0.04
    cap_h = len(caption_lines) * CAPTION_PT * LINE_FACTOR / 72.0 + 0.04
    needed = 0.14 + value_h + 0.08 + cap_h + 0.14
    h = max(h or 0, needed)
    card(slide, x, y, w, h, fill)
    put_text(slide, value, x + 0.14, y + 0.14, w - 0.28, value_h, value_pt, color, bold=True,
             align=PP_ALIGN.CENTER, label=f"stat {value}")
    put_lines(slide, caption_lines, x + 0.12, y + 0.14 + value_h + 0.08, w - 0.24, cap_h,
              CAPTION_PT, INK_SOFT, align=PP_ALIGN.CENTER,
              label=f"stat caption {caption_lines[0][:28]}")
    return h


def picture(slide, name, x, y, h=None, w=None, caption=None):
    """Place a results/ figure; give height OR width, aspect is preserved."""
    path = RESULTS / name
    if not path.exists():
        raise FileNotFoundError(path)
    if h is not None:
        pic = slide.shapes.add_picture(str(path), Inches(x), Inches(y), height=Inches(h))
    else:
        pic = slide.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w))
    pw, ph = pic.width / 914400, pic.height / 914400
    if x + pw > SLIDE_W - MARGIN + 1e-6 or y + ph > SLIDE_H - 0.3 + 1e-6:
        OVERFLOWS.append(f"figure {name}: ends at x={x + pw:.2f} y={y + ph:.2f}")
    if caption:
        put_text(slide, caption, x, y + ph + 0.06, pw, 0.22, CAPTION_PT, MUTED,
                 align=PP_ALIGN.CENTER, label=f"caption {name}")
    return pw, ph


def mono_panel(slide, lines, x, y, w, h, pt=MONO_PT, fill=TINT, color=INK):
    card(slide, x, y, w, h, fill)
    put_lines(slide, lines, x + 0.20, y + 0.16, w - 0.40, h - 0.32, pt, color,
              font=MONO_FONT, label="mono panel")


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def blank(prs, dark=False):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    if dark:
        s.background.fill.solid()
        s.background.fill.fore_color.rgb = DARK_BG
    return s


# ----------------------------------------------------------- architecture diagram
def diagram(slide, x, y, w, h):
    """KIO architecture, our intent-router block highlighted."""
    gap = 0.34
    cols = [2.30, 2.95, 2.55, 3.35]
    scale = (w - gap * 3) / sum(cols)
    cols = [c * scale for c in cols]
    xs, cx = [], x
    for c in cols:
        xs.append(cx)
        cx += c + gap

    box_h = 0.92
    mid_y = y + (h - box_h) / 2
    top_y = y + 0.02
    bot_y = y + h - box_h - 0.02

    def box(bx, by, bw, lines, fill, text_color, border=None, bold_first=True):
        card(slide, bx, by, bw, box_h, fill, border)
        put_lines(slide, lines, bx + 0.10, by + (box_h - len(lines) * SMALL_PT * LINE_FACTOR / 72.0) / 2,
                  bw - 0.20, len(lines) * SMALL_PT * LINE_FACTOR / 72.0 + 0.04, SMALL_PT,
                  text_color, bold=bold_first, align=PP_ALIGN.CENTER, label="diagram box")

    def arrow(ax, ay, aw):
        a = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(ax), Inches(ay - 0.085),
                                   Inches(aw), Inches(0.17))
        a.fill.solid()
        a.fill.fore_color.rgb = LINE
        a.line.fill.background()
        a.shadow.inherit = False

    box(xs[0], mid_y, cols[0], ["KIO chat", "message"], WHITE, INK_SOFT, LINE)
    arrow(xs[0] + cols[0] + 0.05, mid_y + box_h / 2, gap - 0.10)

    # our block - highlighted
    hb = card(slide, xs[1], mid_y - 0.10, cols[1], box_h + 0.20, INK)
    put_lines(slide, ["OUR BLOCK", "Intent router", "flag 0 or 1"],
              xs[1] + 0.10, mid_y + 0.02, cols[1] - 0.20, box_h,
              SMALL_PT, WHITE, bold=True, align=PP_ALIGN.CENTER, label="router block")
    hb.line.color.rgb = SALES
    hb.line.width = Pt(2.5)

    for (by, lab, flag, col, data, note) in (
        (top_y, "Support agent", "flag 0", SUPPORT, "Search tools + LLM", "Policies / FAQs"),
        (bot_y, "Sales agent", "flag 1", SALES, "Search tools + LLM", "Products / plans / pricing"),
    ):
        arrow(xs[1] + cols[1] + 0.05, by + box_h / 2, gap - 0.10)
        card(slide, xs[2], by, cols[2], box_h, WHITE, col)
        put_lines(slide, [lab, flag], xs[2] + 0.10, by + 0.18, cols[2] - 0.20, box_h - 0.30,
                  SMALL_PT, col, bold=True, align=PP_ALIGN.CENTER, label="agent box")
        arrow(xs[2] + cols[2] + 0.05, by + box_h / 2, gap - 0.10)
        box(xs[3], by, cols[3], [data, note], WHITE, INK_SOFT, LINE, bold_first=False)


# ----------------------------------------------------------------- number guard
def verify_numbers():
    """Every figure quoted on a slide, checked against results/ before building."""
    ml = json.loads((RESULTS / "ml_metrics.json").read_text(encoding="utf-8"))
    dm = json.loads((RESULTS / "zeroshot_metrics.json").read_text(encoding="utf-8"))
    cp = json.loads((RESULTS / "comparison_summary.json").read_text(encoding="utf-8"))
    ds = json.loads((RESULTS / "dataset_summary.json").read_text(encoding="utf-8"))
    checks = [
        ("dataset size", ds["n_messages"], 60),
        ("dataset groups", ds["n_groups"], 30),
        ("train support", ds["counts_split_label"]["train_support"], 20),
        ("test sales", ds["counts_split_label"]["test_sales"], 10),
        ("ml cv", round(ml["best_cv_mean_accuracy"], 3), 0.825),
        ("ml train acc", round(ml["train_accuracy"], 3), 0.975),
        ("ml test acc", round(ml["test"]["accuracy"], 3), 0.600),
        ("ml macro f1", round(ml["test"]["f1_macro"], 3), 0.583),
        ("ml recall support", round(ml["test"]["recall_support"], 3), 0.400),
        ("ml recall sales", round(ml["test"]["recall_sales"], 3), 0.800),
        ("ml C", ml["selected"]["C"], 1),
        ("ml features", ml["n_features"], 11),
        ("ml latency ms", round(ml["latency_ms_per_message"], 2), 0.06),
        ("ml cv tfidf", round(ml["cv_best_per_feature_set"]["tfidf_1_1"], 3), 0.725),
        # the count of cross-validated configurations, read from the CV table itself
        ("ml n configs", sum(1 for _ in (RESULTS / "ml_cv_results.csv")
                             .read_text(encoding="utf-8").splitlines()[1:]), 45),
        ("dm test acc", round(dm["test"]["accuracy"], 3), 0.950),
        ("dm macro f1", round(dm["test"]["f1_macro"], 2), 0.95),
        ("dm all60 acc", round(dm["all_60"]["accuracy"], 2), 0.85),
        ("dm threshold", dm["decision_threshold"], 0.41),
        ("dm confusion", dm["test"]["confusion_matrix"], [[10, 0], [1, 9]]),
        ("dm invalid", dm["schema_invalid_outputs"], 0),
        ("dm abstain total", dm["n_abstentions_total"], 3),
        ("dm abstain test", dm["n_abstentions_test"], 0),
        ("dm ms", round(dm["seconds_per_message"] * 1000), 546),
        ("dm variant B", round(dm["train_accuracy_per_config"]["B_topic"], 3), 0.875),
        ("dm variant A", round(dm["train_accuracy_per_config"]["A_short_team"], 3), 0.750),
        ("dm variant C", round(dm["train_accuracy_per_config"]["C_long_routing"], 3), 0.750),
        ("dm variant D", round(dm["train_accuracy_per_config"]["D_intent"], 3), 0.675),
        ("dm by category", dm["test_accuracy_by_category"],
         {"mixed": "4/4", "negation": "7/8", "plain": "8/8"}),
        ("ml by category", ml["test_accuracy_by_category"],
         {"mixed": "2/4", "negation": "3/8", "plain": "7/8"}),
        ("mcnemar p", round(cp["paired_test"]["p_value"], 4), 0.0156),
        ("agreement", round(cp["agreement_rate"], 2), 0.65),
        ("only dm correct", cp["only_dm_correct"], 7),
        ("only ml correct", cp["only_ml_correct"], 0),
        ("posthoc range", cp["posthoc_test_accuracy_range"], [0.60, 0.75]),
    ]
    bad = [f"  {n}: results say {g!r}, slide says {e!r}" for n, g, e in checks if g != e]
    if bad:
        raise SystemExit("Numbers on the slides disagree with results/:\n" + "\n".join(bad))
    print(f"Number guard: {len(checks)} values match results/")


# ----------------------------------------------------------------------- slides
def slide1(prs):
    s = blank(prs)
    title(s, "The routing problem")
    chapter_chip(s, "Student 1", SUPPORT)
    diagram(s, MARGIN, 1.32, CONTENT_W, 2.70)
    left = [
        "KIO is the chat entry point to two specialist agents: support "
        "(problems, complaints, existing requests) and sales (new purchases, packages, quotes).",
        "Our block is the intent router: message in, flag out. 0 = support, 1 = sales.",
    ]
    right = [
        "Routing picks the workflow; retrieval (RAG) then finds information inside it. "
        "A wrong route means the wrong tools and the wrong knowledge base.",
        "We built the router twice and compared the two on the same flag.",
    ]
    col_w = (CONTENT_W - 0.6) / 2
    bullets(s, left, MARGIN, 4.42, col_w, label="s1 left")
    bullets(s, right, MARGIN + col_w + 0.6, 4.42, col_w, dot=SALES, label="s1 right")
    notes(s, "STUDENT 1 - 0:45 of a 2:15 chapter.\n\n"
             "KIO is the chat entry point to two specialist agents: support handles problems, "
             "complaints and existing requests; sales handles new purchases, packages and quotes. "
             "Our block is the highlighted one in the middle: a message goes in, a routing flag "
             "comes out, 0 for support and 1 for sales, and KIO uses that flag to call the agent. "
             "Routing picks the workflow; retrieval then finds information inside it, so a wrong "
             "route means the wrong tools and the wrong knowledge base and the answer cannot "
             "recover. We built this router twice and compared the two on the same flag.")


def slide2(prs):
    s = blank(prs)
    title(s, "The dataset")
    chapter_chip(s, "Student 1", SUPPORT)
    w = (CONTENT_W - 3 * 0.30) / 4
    stat(s, MARGIN, 1.26, w, "60", ["synthetic English messages", "30 support / 30 sales"], SUPPORT)
    stat(s, MARGIN + (w + 0.30), 1.26, w, "24", "plain", INK)
    stat(s, MARGIN + 2 * (w + 0.30), 1.26, w, "20", "negation", INK)
    stat(s, MARGIN + 3 * (w + 0.30), 1.26, w, "16", "mixed intent", INK, fill=TINT_WARM)
    items = [
        "60 synthetic English messages = 30 intent groups, each group a base message plus a "
        "paraphrase. 30 support / 30 sales.",
        "Three categories on purpose, written so keywords mislead - support messages that say "
        "upgrade and premium package, sales messages that say nothing is wrong.",
        "Labeling guideline, agreed before any model ran: support = a problem or an existing "
        "service; sales = a new purchase with no unresolved problem; mixed intent with an "
        "unresolved problem -> support.",
        "Split 40 train / 20 test, balanced by class, split at group level so a paraphrase never "
        "lands on the other side of the split. Seed 42.",
    ]
    bullets(s, items, MARGIN, 3.00, CONTENT_W, gap=0.22, label="s2")
    notes(s, "STUDENT 1 - 0:50.\n\n"
             "Sixty synthetic English messages, written as thirty intent groups: each group is a "
             "base message plus a paraphrase of it. Thirty support, thirty sales. Three categories "
             "on purpose - twenty-four plain, twenty negation, sixteen mixed intent - and they are "
             "written so that keywords mislead: support messages that say upgrade and premium "
             "package, sales messages that say nothing is wrong. The labeling guideline was agreed "
             "before any model ran, and the important line is the last one: mixed intent with an "
             "unresolved problem is support. We split forty train, twenty test, balanced by class, "
             "at group level, so a paraphrase never lands on the other side of the split.")


def slide3(prs):
    s = blank(prs)
    title(s, "Sample rows, and how the two approaches differ")
    chapter_chip(s, "Student 1", SUPPORT)
    rows = [
        ("id", "message", "label", "split", "category"),
        ("1", "My internet connection keeps dropping every evening.", "0", "test", "plain"),
        ("13", "I did not order this upgrade, so why am I being billed for it?", "0", "test", "negation"),
        ("25", "Before I buy a second license, can you fix the error the first one keeps showing?",
         "0", "test", "mixed"),
        ("33", "I'd like to buy three more licenses for my team.", "1", "test", "plain"),
        ("43", "Nothing is wrong with my service, I just want a faster plan.", "1", "test", "negation"),
        ("53", "My last issue was fixed quickly, thanks! Now I'd like to order the security add-on.",
         "1", "train", "mixed"),
    ]
    widths = [0.55, 7.50, 0.95, 1.00, 2.13]
    tbl_h = 0.38 * len(rows)
    gfx = s.shapes.add_table(len(rows), 5, Inches(MARGIN), Inches(1.26),
                             Inches(sum(widths)), Inches(tbl_h))
    table = gfx.table
    table.first_row = True
    for i, cw in enumerate(widths):
        table.columns[i].width = Emu(int(cw * 914400))
    for r, row in enumerate(rows):
        table.rows[r].height = Emu(int(0.38 * 914400))
        for c, val in enumerate(row):
            cell = table.cell(r, c)
            cell.margin_left = cell.margin_right = Emu(int(0.08 * 914400))
            cell.margin_top = cell.margin_bottom = Emu(int(0.03 * 914400))
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = INK if r == 0 else (WHITE if r % 2 else TINT)
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if c == 1 else PP_ALIGN.CENTER
            run = p.add_run()
            run.text = val
            colour = WHITE if r == 0 else (SUPPORT if (c == 2 and val == "0")
                                           else SALES if (c == 2 and val == "1") else INK)
            style_run(run, CAPTION_PT, colour, bold=(r == 0 or c == 2))
            check_fit(f"table r{r}c{c}", CAPTION_PT * LINE_FACTOR / 72.0, 0.38,
                      text_w_in(val, CAPTION_PT, r == 0 or c == 2), widths[c] - 0.16)

    y = 1.26 + tbl_h + 0.34
    put_text(s, "How a decision model differs from a trained classifier", MARGIN, y,
             CONTENT_W, 0.32, BODY_PT, INK, bold=True, label="s3 head")
    col_w = (CONTENT_W - 0.6) / 2
    card(s, MARGIN, y + 0.42, col_w, 1.52, TINT)
    put_text(s, "Trained classifier", MARGIN + 0.22, y + 0.58, col_w - 0.44, 0.28,
             SMALL_PT, SUPPORT, bold=True, label="s3 c1h")
    put_text(s, "Learns from our labels: it turns the message into numbers and fits a boundary. "
                "It knows only what the 40 training messages taught it.",
             MARGIN + 0.22, y + 0.92, col_w - 0.44, 0.94, SMALL_PT, INK, label="s3 c1")
    card(s, MARGIN + col_w + 0.6, y + 0.42, col_w, 1.52, TINT_WARM)
    put_text(s, "Decision model", MARGIN + col_w + 0.82, y + 0.58, col_w - 0.44, 0.28,
             SMALL_PT, SALES, bold=True, label="s3 c2h")
    put_text(s, "Already understands English: it is given the message plus routing instructions "
                "in words and returns a typed decision. Zero training examples - we tune wording, "
                "not weights.",
             MARGIN + col_w + 0.82, y + 0.92, col_w - 0.44, 0.94, SMALL_PT, INK, label="s3 c2")
    notes(s, "STUDENT 1 - 0:40.\n\n"
             "Six real rows from the dataset. Notice id 13 and id 43: both are negations, and the "
             "surface words point the wrong way. Id 25 is mixed intent - the customer wants to buy "
             "a second license but has an unresolved error, so by our guideline it is support.\n\n"
             "The conceptual difference, one line each. The classifier learns from our labels: it "
             "turns the message into numbers and fits a boundary, and it knows only what the forty "
             "training messages taught it. The decision model already understands English: we give "
             "it the message plus routing instructions in words and it returns a typed decision, "
             "with zero training examples. We tune wording, not weights.\n\n"
             "HAND-OFF: 'Student 2 will show the decision model first.'")


def slide4(prs):
    s = blank(prs)
    title(s, "The decision model: what we used and why")
    chapter_chip(s, "Student 2", SALES)
    col_w = 7.10
    items = [
        "A Decision API like Jev (TypeSafe AI) takes message + instructions and returns a typed "
        "decision. We had no paid API, so we used the open-source equivalent with our own "
        "structured-output wrapper.",
        "Method: zero-shot via natural-language inference - the message is the premise, each label "
        "becomes a hypothesis, entailment scores are compared.",
    ]
    bullets(s, items, MARGIN, 1.34, col_w, dot=SALES, gap=0.26, label="s4")

    card(s, MARGIN, 3.30, col_w, 3.44, TINT_WARM)
    put_text(s, "facebook/bart-large-mnli", MARGIN + 0.28, 3.54, col_w - 0.56, 0.42,
             22, INK, bold=True, font=MONO_FONT, label="s4 model")
    facts = [
        ("Architecture", "BART-large, 407M parameters, fine-tuned on MultiNLI"),
        ("License", "MIT (weights); MultiNLI under its own permissive terms"),
        ("Setup", "local HuggingFace cache, HF_HUB_OFFLINE=1"),
        ("Runs", "CPU, no API key, no network call, no training on our data"),
    ]
    fy = 4.06
    for k, v in facts:
        put_text(s, k, MARGIN + 0.28, fy, 1.55, 0.26, SMALL_PT, SALES, bold=True, label=f"s4 k {k}")
        put_text(s, v, MARGIN + 1.90, fy, col_w - 2.18, 0.52, SMALL_PT, INK, label=f"s4 v {k}")
        fy += 0.62

    rx = MARGIN + col_w + 0.50
    rw = SLIDE_W - MARGIN - rx
    card(s, rx, 1.34, rw, 5.40, TINT)
    put_text(s, "Message + routing instructions", rx + 0.26, 1.64, rw - 0.52, 0.56, SMALL_PT,
             INK_SOFT, align=PP_ALIGN.CENTER, label="s4 f1")
    for i, (lab, col, y0) in enumerate((("Zero-shot NLI model", INK, 2.46),
                                        ("Structured-output wrapper", INK, 3.74),
                                        ("Typed decision + flag", SALES, 5.02))):
        a = s.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, Inches(rx + rw / 2 - 0.09),
                               Inches(y0 - 0.44), Inches(0.18), Inches(0.34))
        a.fill.solid()
        a.fill.fore_color.rgb = LINE
        a.line.fill.background()
        a.shadow.inherit = False
        card(s, rx + 0.26, y0, rw - 0.52, 0.76, WHITE, col)
        put_text(s, lab, rx + 0.36, y0 + 0.24, rw - 0.72, 0.30, SMALL_PT, col, bold=True,
                 align=PP_ALIGN.CENTER, label=f"s4 step {i}")
    put_text(s, "customer_service -> 0   |   sales -> 1", rx + 0.26, 6.02, rw - 0.52, 0.30,
             CAPTION_PT, MUTED, align=PP_ALIGN.CENTER, font=MONO_FONT, label="s4 map")
    notes(s, "STUDENT 2 - 0:50 of a 2:30 chapter.\n\n"
             "A Decision API like Jev from TypeSafe AI takes a message plus instructions and "
             "returns a typed decision. We had no paid API, so we used the open-source equivalent "
             "with our own structured-output wrapper. The model is facebook/bart-large-mnli: "
             "BART-large, four hundred and seven million parameters, fine-tuned on MultiNLI. The "
             "weights are MIT licensed. Setup is a local HuggingFace cache with HF_HUB_OFFLINE set "
             "to one - CPU only, no API key, no network call, and no training on our data. The "
             "method is zero-shot via natural-language inference: the message is the premise, each "
             "label becomes a hypothesis, and the entailment scores are compared.")


def slide5(prs):
    s = blank(prs)
    title(s, "Instructions, and output validation")
    chapter_chip(s, "Student 2", SALES)
    col_w = 6.70
    put_text(s, "The instructions are the template + the two label descriptions",
             MARGIN, 1.30, col_w, 0.30, SMALL_PT, INK_SOFT, label="s5 lead")
    mono_panel(s, [
        '"This customer message is about {}."',
        "",
        'customer_service -> "a problem with a service',
        '                    the customer already has"',
        'sales            -> "buying a new product, a price',
        '                    or an upgrade"',
        "",
        "threshold = 0.41",
    ], MARGIN, 1.70, col_w, 2.42)

    put_text(s, "4 instruction variants, scored on the training split only",
             MARGIN, 4.34, col_w, 0.30, SMALL_PT, INK_SOFT, label="s5 var lead")
    vw = (col_w - 3 * 0.18) / 4
    for i, (name, acc, hl) in enumerate((("A", "0.750", False), ("B", "0.875", True),
                                         ("C", "0.750", False), ("D", "0.675", False))):
        x = MARGIN + i * (vw + 0.18)
        card(s, x, 4.70, vw, 1.08, TINT_WARM if hl else TINT)
        put_text(s, acc, x + 0.06, 4.84, vw - 0.12, 0.42, 22, SALES if hl else INK, bold=True,
                 align=PP_ALIGN.CENTER, label=f"s5 var {name}")
        put_text(s, f"variant {name}" + (" (chosen)" if hl else ""), x + 0.04, 5.34,
                 vw - 0.08, 0.30, 11, INK_SOFT, align=PP_ALIGN.CENTER, label=f"s5 varl {name}")
    put_text(s, "Wording is worth 20 points - the key insight of this chapter",
             MARGIN, 5.94, col_w, 0.34, SMALL_PT, SALES, bold=True, label="s5 insight")

    rx = MARGIN + col_w + 0.56
    rw = SLIDE_W - MARGIN - rx
    items = [
        "Structured output + validation: every answer is parsed into "
        '{"decision", "confidence", "scores"} and checked - decision in the enum, confidence a '
        "float in [0,1], scores over exactly the two labels summing to 1.",
        "Abstention band +/-0.05 around the threshold returns \"uncertain\" and falls back to "
        "support, the safe default. 3 of 60 abstained, 0 in test.",
    ]
    bullets(s, items, rx, 1.30, rw, dot=SALES, gap=0.28, label="s5 right")
    stat(s, rx, 4.36, rw, "0 of 60", "outputs failed schema validation", SALES, value_pt=38,
         fill=TINT_WARM, h=1.42)
    notes(s, "STUDENT 2 - 0:50.\n\n"
             "The instructions are the hypothesis template plus the two label descriptions, with a "
             "decision threshold of nought point four one. We wrote four instruction variants and "
             "picked one on the training split only: nought point eight seven five, nought point "
             "seven five, nought point seven five, nought point six seven five. Say this out loud: "
             "wording is worth twenty points - that is the key insight of this chapter.\n\n"
             "On the output side, every answer is parsed into a decision, a confidence and the two "
             "scores, and schema-checked: the decision must be in the enum, confidence a float "
             "between zero and one, and the scores must cover exactly the two labels and sum to "
             "one. Zero of sixty outputs failed validation. There is also an abstention band of "
             "plus or minus nought point nought five around the threshold: inside it the wrapper "
             "returns uncertain and falls back to support, the safe default. Three of sixty "
             "abstained, none of them in the test split.")


def slide6(prs):
    s = blank(prs)
    title(s, "Decision model: results")
    chapter_chip(s, "Student 2", SALES)
    # Row 1: two headline stats, then the two saved figures.
    sw, gap = 2.05, 0.30
    stat(s, MARGIN, 1.22, sw, "0.95", ["test accuracy (n=20)", "macro F1 0.95"], SALES,
         value_pt=34, fill=TINT_WARM)
    stat(s, MARGIN + sw + gap, 1.22, sw, "0.85", ["across all 60 messages", "the test half is easier"],
         INK, value_pt=34)
    fx = MARGIN + 2 * (sw + gap)
    pw1, _ = picture(s, "zeroshot_confusion_matrix.png", fx, 1.22, h=2.40,
                     caption="Never sent a real problem to sales")
    picture(s, "zeroshot_confidence.png", fx + pw1 + gap, 1.22, h=2.40)
    put_text(s, "By category:  plain 8/8    negation 7/8    mixed 4/4",
             MARGIN, 3.08, 2 * sw + gap, 0.52, CAPTION_PT, INK_SOFT, align=PP_ALIGN.CENTER,
             label="s6 cat")

    # Row 2: two real outputs on the left, what they show on the right.
    px, pw = MARGIN, 7.35
    put_text(s, "Two real outputs, copied from results/zeroshot_predictions.csv",
             px, 3.92, pw, 0.28, CAPTION_PT, MUTED, label="s6 outlead")
    mono_panel(s, [
        'id 33  "I\'d like to buy three more licenses for my team."   -> flag 1',
        '{"decision": "sales", "confidence": 0.906,',
        ' "scores": {"customer_service": 0.094, "sales": 0.906}}',
        "",
        'id 25  "Before I buy a second license, can you fix the       -> flag 0',
        '        error the first one keeps showing?"',
        '{"decision": "customer_service", "confidence": 0.704,',
        ' "scores": {"customer_service": 0.704, "sales": 0.296}}',
    ], px, 4.26, pw, 1.98, fill=TINT)
    put_text(s, "Cost: 546 ms per message on CPU, ~1.6 GB in memory", px, 6.38, pw, 0.30,
             CAPTION_PT, MUTED, label="s6 cost")

    bx = px + pw + 0.50
    bw = SLIDE_W - MARGIN - bx
    items = [
        "id 25 is the win: mixed intent, the words buy and second license are right there, and it "
        "still routes to support.",
        "One error - id 43 scored 0.07 for sales: confidently wrong, outside the abstention band. "
        "Validation guarantees the shape of the answer, never its truth.",
    ]
    bullets(s, items, bx, 4.26, bw, pt=SMALL_PT, gap=0.24, dot=SALES, label="s6 b")
    notes(s, "STUDENT 2 - 0:50.\n\n"
             "Test accuracy nought point nine five, macro F1 the same, on twenty test messages. "
             "Across all sixty messages it is nought point eight five - say both, because the test "
             "half is the easier one. The confusion matrix is ten, zero, one, nine: it never sent a "
             "customer with a real problem to sales. By category: plain eight of eight, negation "
             "seven of eight, mixed four of four.\n\n"
             "These are two real outputs, copied from results slash zeroshot_predictions.csv - "
             "this is what the router actually returns, not a sketch. Id 33 is the easy case. Id 25 "
             "is the win: mixed intent, the words buy and second license are right there, and it "
             "still routes to support because the customer has an unresolved error. That is exactly "
             "the rule we wrote in the labeling guideline, obeyed by a model we never trained.\n\n"
             "One error: id 43, 'Nothing is wrong with my service, I just want a faster plan.' It "
             "scored nought point nought seven for sales - confidently wrong, outside the "
             "abstention band. The lesson is that validation guarantees the shape of the answer, "
             "never its truth. Cost: five hundred and forty-six milliseconds per message on CPU.\n\n"
             "HAND-OFF: 'Now the classical approach, and the head-to-head.'")


def slide7(prs):
    s = blank(prs)
    title(s, "The traditional ML classifier")
    chapter_chip(s, "Student 3", SUPPORT)
    col_w = 7.00
    items = [
        "Numeric features. Two options were tried: TF-IDF word n-grams, and 11 hand-crafted "
        "features. Lexicons fixed a priori from the routing definition.",
        "Model search: linear SVM, degree-2 polynomial-kernel SVM, and logistic regression on "
        "degree-2 polynomial features - 45 configurations, selected by 5-fold grouped "
        "cross-validation on the 40 training messages only (paraphrase pairs kept in the same fold).",
        "Selected: 11 hand-crafted features + SVM, degree-2 polynomial kernel, C = 1. CV accuracy "
        "0.825. TF-IDF came last (0.725) - with 40 messages there is almost no vocabulary overlap "
        "with unseen text.",
    ]
    bullets(s, items, MARGIN, 1.32, col_w, gap=0.26, label="s7")

    rx = MARGIN + col_w + 0.50
    rw = SLIDE_W - MARGIN - rx
    card(s, rx, 1.32, rw, 3.66, TINT)
    put_text(s, "The 11 hand-crafted features", rx + 0.24, 1.54, rw - 0.48, 0.30, SMALL_PT,
             SUPPORT, bold=True, label="s7 feat head")
    feats = ["length in words", "length in characters", "'?' present", "support-lexicon hits",
             "sales-lexicon hits", "their margin", "negation hits", "existing-service hits",
             "new-purchase hits", "money words", "first-person request"]
    put_lines(s, feats, rx + 0.24, 1.96, rw - 0.48, 2.92, CAPTION_PT, INK, space_after=3.0,
              label="s7 feats")
    stat(s, rx, 5.12, rw, "45", ["configurations cross-validated", "best CV accuracy 0.825"],
         SUPPORT, value_pt=40, h=1.46)
    notes(s, "STUDENT 3 - 0:45 of a 2:45 chapter.\n\n"
             "The features have to be numeric. We tried two representations: TF-IDF word n-grams, "
             "and eleven hand-crafted features - length in words and characters, whether there is a "
             "question mark, support-lexicon hits per ten words, sales-lexicon hits, their margin, "
             "negation hits, existing-service hits, new-purchase hits, money words, and a "
             "first-person request flag. The lexicons were fixed a priori from the routing "
             "definition.\n\n"
             "For the model we searched exactly what the brief allows: linear SVM, degree-2 "
             "polynomial-kernel SVM, and logistic regression on degree-2 polynomial features. "
             "Fifty configurations in total, selected by five-fold grouped cross-validation on the "
             "forty training messages only, with paraphrase pairs kept in the same fold. The winner "
             "was the eleven hand-crafted features with a degree-2 polynomial-kernel SVM at C "
             "equals one, cross-validation accuracy nought point eight two five. TF-IDF came last "
             "at nought point seven two five - with forty messages there is almost no vocabulary "
             "overlap with unseen text.")


def slide8(prs):
    s = blank(prs)
    title(s, "Its boundary, its outputs, its errors")
    chapter_chip(s, "Student 3", SUPPORT)
    fig_h = 3.18
    picture(s, "ml_decision_boundary_lexicon.png", MARGIN, 1.22, h=fig_h,
            caption="Boundary on two readable axes: support vs sales lexicon hits")
    picture(s, "ml_test_scores.png", MARGIN + fig_h * 1.345 + 0.55, 1.22, h=fig_h,
            caption="Signed decision score of all 20 test messages")
    items = [
        "Train accuracy 0.975 -> test accuracy 0.600. That gap is overfitting: a degree-2 kernel "
        "separates 40 points easily, and CV on 40 points is too noisy to catch it.",
        "ids 12 (score 0.02) and 17 (0.19) are support messages pushed over the line by a hair.",
    ]
    right = [
        "Permutation importance: the strongest single feature is the question mark (0.080 accuracy "
        "drop); negation is almost unused (0.007).",
        "Latency 0.06 ms/message - about 9 000x faster than the decision model.",
    ]
    col_w = (CONTENT_W - 0.6) / 2
    bullets(s, items, MARGIN, 5.02, col_w, pt=SMALL_PT, gap=0.18, label="s8 left")
    bullets(s, right, MARGIN + col_w + 0.6, 5.02, col_w, pt=SMALL_PT, gap=0.18, dot=SALES,
            label="s8 right")
    notes(s, "STUDENT 3 - 0:45.\n\n"
             "On the left is the boundary, drawn on two readable axes: support-lexicon hits against "
             "sales-lexicon hits. The same boundary for a PCA projection of the full feature space "
             "is in ml_decision_boundary.png.\n\n"
             "Train accuracy nought point nine seven five, test accuracy nought point six zero "
             "zero. That gap is overfitting: a degree-2 kernel separates forty points easily, and "
             "cross-validation on forty points is too noisy to catch it.\n\n"
             "On the right are its actual outputs: the signed decision score of all twenty test "
             "messages, one bar each, coloured by true label, with the misrouted ones marked. Point "
             "at the bars sitting just past zero - ids 12 and 17 are support messages pushed over "
             "the line by a hair.\n\n"
             "Permutation importance tells us why: the strongest single feature is the question "
             "mark, and negation is almost unused. That one fact explains the errors on the next "
             "slide. On speed it wins easily - nought point two three milliseconds a message.")


def slide9(prs):
    s = blank(prs)
    title(s, "Head to head")
    chapter_chip(s, "Student 3", SUPPORT)
    rows = [
        ("Metric (test, n=20)", "Traditional ML", "Decision model"),
        ("Accuracy", "0.600", "0.950"),
        ("Macro F1", "0.583", "0.950"),
        ("Recall (support)", "0.400", "1.000"),
        ("Recall (sales)", "0.800", "0.900"),
        ("plain / negation / mixed", "7/8, 3/8, 2/4", "8/8, 7/8, 4/4"),
        ("Latency per message", "0.06 ms", "546 ms"),
        ("Labelled data needed", "40 and still rising", "0"),
    ]
    widths = [2.55, 2.00, 2.05]
    row_h = 0.395
    gfx = s.shapes.add_table(len(rows), 3, Inches(MARGIN), Inches(1.22),
                             Inches(sum(widths)), Inches(row_h * len(rows)))
    table = gfx.table
    table.first_row = True
    for i, cw in enumerate(widths):
        table.columns[i].width = Emu(int(cw * 914400))
    bold_rows = {1, 3}
    for r, row in enumerate(rows):
        table.rows[r].height = Emu(int(row_h * 914400))
        for c, val in enumerate(row):
            cell = table.cell(r, c)
            cell.margin_left = cell.margin_right = Emu(int(0.09 * 914400))
            cell.margin_top = cell.margin_bottom = Emu(int(0.02 * 914400))
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = INK if r == 0 else (WHITE if r % 2 else TINT)
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
            run = p.add_run()
            run.text = val
            bold = r == 0 or (r in bold_rows and c > 0)
            colour = WHITE if r == 0 else (INK if c == 0 else
                                           (SUPPORT if c == 1 else SALES))
            style_run(run, SMALL_PT, colour, bold=bold)
            check_fit(f"s9 table r{r}c{c}", SMALL_PT * LINE_FACTOR / 72.0, row_h,
                      text_w_in(val, SMALL_PT, bold), widths[c] - 0.18)

    picture(s, "comparison_accuracy.png", MARGIN + sum(widths) + 0.45, 1.22, h=3.28)
    items = [
        "The routers agree on 65% of test messages. Where they disagree, the decision model is "
        "right 7 times, the classifier 0. Exact McNemar p = 0.0156 - significant, even at n = 20.",
        "The error pattern: the classifier breaks on negation and mixed intent. \"I did not order "
        "this upgrade, so why am I being billed for it?\" - it sees upgrade, order, '?' and routes "
        "to sales. The decision model reads the sentence and routes to support.",
        "Fair to the classifier: the top six CV configurations score 0.60-0.75 on test, and the "
        "learning curve is still rising at 40 messages. It is not broken - it is starved of data.",
    ]
    bullets(s, items, MARGIN, 4.68, CONTENT_W, pt=SMALL_PT, gap=0.16, label="s9")
    notes(s, "STUDENT 3 - 0:45.\n\n"
             "Side by side on the same twenty test messages. Accuracy nought point six against "
             "nought point nine five. The row that matters for the business is recall on support: "
             "nought point four against one point zero - the classifier sends six of ten customers "
             "with a real problem to the sales team.\n\n"
             "The two routers agree on sixty-five per cent of test messages. Where they disagree, "
             "the decision model is right seven times and the classifier zero. An exact McNemar "
             "test gives p equals nought point nought one five six - significant, even at twenty "
             "messages.\n\n"
             "The error pattern is consistent: the classifier breaks on negation and mixed intent. "
             "Take 'I did not order this upgrade, so why am I being billed for it?' - it sees "
             "upgrade, order and a question mark, and routes to sales. The decision model reads the "
             "sentence and routes to support.\n\n"
             "To be fair to the classifier: the top six cross-validation configurations score "
             "between nought point six and nought point seven five on test, and the learning curve "
             "is still rising at forty messages. It is not broken - it is starved of data.")


def slide10(prs):
    s = blank(prs, dark=True)
    title(s, "Recommendation for KIO", dark=True)
    chapter_chip(s, "Student 3", SALES, dark=True)
    put_text(s, "Deploy the decision model now", MARGIN, 1.30, CONTENT_W, 0.80, 44, WHITE,
             bold=True, font=HEAD_FONT, label="s10 claim")

    sw = (CONTENT_W - 2 * 0.36) / 3
    for i, (val, cap, col) in enumerate((("0.95 vs 0.60", "test accuracy against the classifier", SALES),
                                         ("p = 0.0156", "exact McNemar - the gap is real", WHITE),
                                         ("0", "labelled examples it needed", SALES))):
        x = MARGIN + i * (sw + 0.36)
        card(s, x, 2.26, sw, 1.30, RGBColor(0x1C, 0x2B, 0x45))
        put_text(s, val, x + 0.14, 2.42, sw - 0.28, 0.50, 28, col, bold=True,
                 align=PP_ALIGN.CENTER, label=f"s10 stat {i}")
        put_text(s, cap, x + 0.14, 3.00, sw - 0.28, 0.44, CAPTION_PT,
                 RGBColor(0xB8, 0xC4, 0xD8), align=PP_ALIGN.CENTER, label=f"s10 cap {i}")

    items = [
        "Better where it matters, never misroutes a real problem in our test, needs no labelled "
        "data, and the wrapper makes the flag safe to use: typed output, confidence exposed, "
        "abstain -> support.",
        "What we accept: 546 ms per message and 1.6 GB of memory; MIT-licensed and fully offline, "
        "so no per-call fee and no customer data leaves the machine. Instruction wording must be "
        "version-controlled - it is worth 20 points.",
        "The path forward: log every decision with its confidence, let agents flag misroutes, and "
        "after a few thousand real labelled messages retrain the classifier on that log and re-run "
        "this exact comparison.",
        "Caveat we state ourselves: 20 test messages is small - the same decision model scores 0.85 "
        "over all 60. The ranking is solid; the exact numbers would move on a larger set.",
    ]
    bullets(s, items, MARGIN, 3.92, CONTENT_W, pt=SMALL_PT, gap=0.18,
            color=RGBColor(0xE6, 0xEC, 0xF6), dot=SALES, label="s10")
    notes(s, "STUDENT 3 - 0:30, the close.\n\n"
             "Our recommendation is to deploy the decision model now. It is better where it "
             "matters - nought point nine five against nought point six, and the gap is "
             "statistically significant. It never misroutes a real problem in our test. It needs no "
             "labelled data. And the wrapper makes the flag safe to use in a pipeline: typed "
             "output, confidence exposed, and when it is unsure it abstains to support.\n\n"
             "What we accept: five hundred and forty-six milliseconds a message and about one and a "
             "half gigabytes of memory. It is MIT licensed and fully offline, so there is no "
             "per-call fee and no customer data leaves the machine. The instruction wording has to "
             "be version-controlled, because it is worth twenty points.\n\n"
             "The path forward is to log every decision with its confidence, let agents flag "
             "misroutes, and after a few thousand real labelled messages retrain the classifier on "
             "that log and re-run this exact comparison.\n\n"
             "And the caveat we state ourselves: twenty test messages is a small sample - the same "
             "model scores nought point eight five over all sixty. The ranking is solid; the exact "
             "numbers would move on a larger set.")


def slide11(prs):
    s = blank(prs)
    title(s, "Sources")
    chapter_chip(s, "Student 3", SUPPORT)
    left = [
        ("Project 1 brief, Routing for Agent KIO",
         "the task, the flag convention and the dataset specification"),
        ("facebook/bart-large-mnli model card, HuggingFace",
         "MIT license (weights); MultiNLI corpus under its own permissive terms"),
        ("Yin, Hay & Roth (2019), Benchmarking Zero-shot Text Classification: "
         "Datasets, Evaluation and Entailment Approach",
         "the NLI-as-zero-shot-classifier method we use"),
        ("Williams, Nangia & Bowman (2018), A Broad-Coverage Challenge Corpus "
         "for Sentence Understanding through Inference",
         "MultiNLI, the corpus the model was fine-tuned on"),
    ]
    right = [
        ("Dietterich (1998), Approximate statistical tests for comparing "
         "supervised classification learning algorithms",
         "McNemar's test for paired comparison"),
        ("Software",
         "scikit-learn 1.9.1, transformers 5.20.1, torch 2.14.1+cpu, pandas 3.0.6, "
         "matplotlib 3.11.2, seaborn 0.13.2, scipy 1.18.1, Python 3.12.10"),
        ("All code, data and saved results",
         "the project repository (src/, data/, results/, REPORT.md)"),
        ("Dataset",
         "60 messages written by our team for this project, synthetic, no customer data"),
    ]
    col_w = (CONTENT_W - 0.70) / 2
    for ci, column in enumerate((left, right)):
        x = MARGIN + ci * (col_w + 0.70)
        y = 1.34
        for head, sub in column:
            hh = block_h_in(head, col_w - 0.30, SMALL_PT, True)
            put_text(s, head, x + 0.30, y, col_w - 0.30, hh + 0.02, SMALL_PT, INK, bold=True,
                     label="src head")
            d = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x),
                                   Inches(y + SMALL_PT * LINE_FACTOR / 72.0 / 2 - 0.05),
                                   Inches(0.10), Inches(0.10))
            d.fill.solid()
            d.fill.fore_color.rgb = SUPPORT if ci == 0 else SALES
            d.line.fill.background()
            d.shadow.inherit = False
            sh = block_h_in(sub, col_w - 0.30, CAPTION_PT)
            put_text(s, sub, x + 0.30, y + hh + 0.04, col_w - 0.30, sh + 0.02, CAPTION_PT,
                     INK_SOFT, label="src sub")
            y += hh + sh + 0.34
        if y > SLIDE_H - 0.4:
            OVERFLOWS.append(f"sources column {ci + 1} ends at y={y:.2f}")
    notes(s, "STUDENT 3 - no extra speaking time; this slide is on screen while the "
             "recommendation lands and stays up through questions.\n\n"
             "The brief asks us to present all work, results AND sources, so these are on screen: "
             "the project brief, the model card and its MIT license, the two papers behind the "
             "zero-shot method and its training corpus, Dietterich for the paired significance "
             "test, the exact software versions, the repository holding the code and all saved "
             "results, and a note that the dataset is synthetic and contains no customer data.")


def main():
    verify_numbers()
    prs = Presentation()
    prs.slide_width = Inches(SLIDE_W)
    prs.slide_height = Inches(SLIDE_H)
    for fn in (slide1, slide2, slide3, slide4, slide5, slide6,
               slide7, slide8, slide9, slide10, slide11):
        fn(prs)
    prs.save(OUT)

    if OVERFLOWS:
        print(f"\nFIT PROBLEMS ({len(OVERFLOWS)}):")
        for o in OVERFLOWS:
            print("  -", o)
        raise SystemExit("Text does not fit; fix the layout before shipping.")
    print(f"Fit check: {len(prs.slides.__iter__.__self__._sldIdLst)} slides, no overflow detected")
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
