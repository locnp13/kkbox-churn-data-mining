"""Small python-docx helpers used by build_report.py."""
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt


def new_document():
    d = Document()
    st = d.styles["Normal"]
    st.font.name = "Times New Roman"
    st.font.size = Pt(12)
    return d


def para(d, text, bold=False, italic=False, size=None):
    p = d.add_paragraph()
    r = p.add_run(text)
    r.bold, r.italic = bold, italic
    if size:
        r.font.size = Pt(size)
    return p


def bullets(d, items):
    for it in items:
        d.add_paragraph(it, style="List Bullet")


def figure(d, path: Path, caption: str, width=6.0):
    if not Path(path).exists():
        para(d, f"[Thiếu hình: {Path(path).name}]", italic=True)
        return
    d.add_picture(str(path), width=Inches(width))
    d.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    c = para(d, caption, italic=True, size=10)
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER


def table(d, df: pd.DataFrame, caption: str = None, fmt: dict = None, max_rows=20):
    """Render a DataFrame. ``fmt`` maps column -> format string such as '{:.1%}'."""
    fmt = fmt or {}
    df = df.head(max_rows)
    if caption:
        para(d, caption, bold=True, size=10)
    t = d.add_table(rows=1, cols=len(df.columns))
    t.style = "Light Grid Accent 1"
    for c, h in zip(t.rows[0].cells, df.columns):
        c.text = str(h)
    for _, r in df.iterrows():
        cells = t.add_row().cells
        for c, col in zip(cells, df.columns):
            v = r[col]
            if pd.isna(v):
                c.text = ""
            elif col in fmt:
                c.text = fmt[col].format(v)
            elif isinstance(v, float):
                c.text = f"{v:,.3f}"
            elif isinstance(v, int):
                c.text = f"{v:,}"
            else:
                c.text = str(v)
    for row in t.rows:
        for c in row.cells:
            for p in c.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(9)
    d.add_paragraph()
    return t
