"""Convert the manuscript .docx files to markdown, so the text is diffable.

Uses only the standard library: a .docx is a zip whose word/document.xml holds
the content. Deliberately not pandoc or python-docx, so this runs anywhere
without an install.

The conversion is lossy by design -- it keeps headings, paragraphs, emphasis,
tables and figure placeholders, and drops styling. The .docx remains the
document of record; the markdown is for reading and diffing revisions.

Usage:
    python docs/docx_to_markdown.py           # convert every .docx in docs/
"""

import re
import shutil
import zipfile
from pathlib import Path
from xml.etree import ElementTree

W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
DOCS = Path(__file__).parent

# .docx name -> (markdown stem, figure prefix). Explicit rather than derived:
# the source filenames carry spaces and revision tags we don't want in the repo.
TARGETS = {
    'Estimating cervical cancer burden by HIV in Zambia_revised_v2.docx':
        ('manuscript', 'main'),
    'Supplementary material_estimating cervical_Sci_Rep_revision.docx':
        ('supplementary_material', 'supp'),
}


def _run_text(run):
    """Text of a single run, with markdown emphasis and superscript markers."""
    text = ''.join(node.text or '' for node in run.iter(f'{W}t'))
    if not text:
        return ''
    props = run.find(f'{W}rPr')
    if props is not None:
        vert = props.find(f'{W}vertAlign')
        if vert is not None and vert.get(f'{W}val') == 'superscript':
            return f'^{text}^'
        # Only mark emphasis on non-blank text, so we never emit stray '**'.
        if text.strip():
            if props.find(f'{W}b') is not None:
                text = f'**{text}**'
            if props.find(f'{W}i') is not None:
                text = f'*{text}*'
    return text


def _paragraph(par, images, fig_prefix):
    """One paragraph as a markdown line (or '' for an empty one)."""
    text = ''.join(_run_text(r) for r in par.findall(f'{W}r')).strip()
    # Collapse the '**a** **b**' that Word's run-splitting produces.
    text = re.sub(r'\*\*(\s*)\*\*', r'\1', text)

    # An inline image: emit a figure reference in document order.
    if par.find(f'.//{W}drawing') is not None or par.find(f'.//{W}pict') is not None:
        images[0] += 1
        ref = f'![Figure {images[0]}](figures_original/{fig_prefix}_image{images[0]}.png)'
        return f'{ref}\n' if not text else f'{ref}\n\n{text}'

    style = par.find(f'{W}pPr/{W}pStyle')
    if style is not None and text:
        val = style.get(f'{W}val', '')
        level = re.search(r'(\d)', val)
        if val.lower().startswith('heading') and level:
            return '#' * (int(level.group(1)) + 1) + f' {text}'
    return text


def _table(tbl):
    """One table as a markdown table, using its first row as the header."""
    rows = []
    for tr in tbl.findall(f'{W}tr'):
        cells = []
        for tc in tr.findall(f'{W}tc'):
            parts = [''.join(_run_text(r) for r in p.findall(f'{W}r'))
                     for p in tc.findall(f'{W}p')]
            cells.append(' '.join(' '.join(parts).split()).replace('|', r'\|'))
        if cells:
            rows.append(cells)
    if not rows:
        return ''
    width = max(len(r) for r in rows)
    rows = [r + [''] * (width - len(r)) for r in rows]
    out = ['| ' + ' | '.join(rows[0]) + ' |',
           '|' + '---|' * width]
    out += ['| ' + ' | '.join(r) + ' |' for r in rows[1:]]
    return '\n'.join(out)


def convert(docx_path, stem, fig_prefix):
    """Write <stem>.md next to the .docx, and extract its images."""
    zf = zipfile.ZipFile(docx_path)
    body = ElementTree.fromstring(zf.read('word/document.xml')).find(f'{W}body')

    figdir = DOCS / 'figures_original'
    figdir.mkdir(exist_ok=True)
    media = sorted(n for n in zf.namelist() if n.startswith('word/media/'))
    for name in media:
        target = figdir / f'{fig_prefix}_{Path(name).name}'
        with zf.open(name) as src, open(target, 'wb') as dst:
            shutil.copyfileobj(src, dst)

    images = [0]  # list so _paragraph can increment it
    blocks = []
    for child in body:
        if child.tag == f'{W}p':
            blocks.append(_paragraph(child, images, fig_prefix))
        elif child.tag == f'{W}tbl':
            blocks.append(_table(child))

    # Collapse runs of blank blocks into a single blank line.
    lines, blank = [], False
    for block in blocks:
        if block:
            lines.append(block)
            blank = False
        elif not blank:
            lines.append('')
            blank = True

    header = (f'<!-- Generated from "{docx_path.name}" by docs/docx_to_markdown.py.\n'
              f'     The .docx is the document of record; edit that, then regenerate. -->\n')
    out = DOCS / f'{stem}.md'
    out.write_text(header + '\n'.join(lines).strip() + '\n')
    return out, len(media)


if __name__ == '__main__':
    for name, (stem, prefix) in TARGETS.items():
        path = DOCS / name
        if not path.exists():
            print(f'skipped (not found): {name}')
            continue
        out, n_img = convert(path, stem, prefix)
        words = len(out.read_text().split())
        print(f'{out.relative_to(DOCS.parent)}: {words:,} words, {n_img} figures')
