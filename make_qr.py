#!/usr/bin/env python3
"""Generate the QR code of this page's GitHub Pages URL and write it into index.html.

Run from the folder that holds index.html:

    python3 -I make_qr.py AnnaHolubarova mlr-ex08

Nothing here talks to the network: segno draws the code, so no QR image
service is involved and the finished page asks nothing of a third party.
The script writes qr.svg and replaces the block between the QR_SVG markers
in index.html, so the QR can always be regenerated and the diff reviewed.
"""
import re
import sys

import segno

MARKER = "<!--QR_SVG-->"
START, END = "<!--QR_SVG-->", "<!--/QR_SVG-->"


def pages_url(owner: str, repo: str) -> str:
    return f"https://{owner.lower()}.github.io/{repo.lower()}/"


def svg_text(url: str) -> str:
    """The QR as bare inline SVG markup (no XML prologue, no xmlns needed in HTML).

    A viewBox is added because the page sizes the code with CSS: an <svg> with
    only width/height attributes does not scale to a smaller box, it clips —
    which silently cut about a quarter off the code on its first try.
    """
    qr = segno.make(url, error="M")           # 'M' = 15 % recovery, plenty here
    qr.save("qr.svg", kind="svg", scale=4, border=4)   # standalone copy, with xmlns
    data = qr.svg_inline(scale=4, border=4)
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    if not data.lstrip().startswith("<svg"):
        data = data[data.index("<svg"):]
    tag_end = data.index(">")
    tag = data[:tag_end]
    if "viewbox" not in tag.lower():
        w = re.search(r'width="(\d+)', tag)
        h = re.search(r'height="(\d+)', tag)
        if w and h:
            tag += f' viewBox="0 0 {w.group(1)} {h.group(1)}"' \
                   ' preserveAspectRatio="xMidYMid meet"'
            data = tag + data[tag_end:]
    return data.strip()


def main() -> int:
    owner = sys.argv[1] if len(sys.argv) > 1 else "AnnaHolubarova"
    repo = sys.argv[2] if len(sys.argv) > 2 else "mlr-ex08"
    url = pages_url(owner, repo)
    svg = svg_text(url)

    html = open("index.html", encoding="utf-8").read()
    # the placeholder comment and the opening marker are the same text, so the
    # filled case has to be recognised first or the code would be inserted twice
    if START in html and END in html:            # regenerate in place
        html = re.sub(re.escape(START) + r".*?" + re.escape(END),
                      f'{START}\n{svg}\n{END}', html, flags=re.S)
    elif MARKER in html:                         # first run: fill the placeholder
        html = html.replace(MARKER, f'{START}\n{svg}\n{END}')
    else:
        sys.exit("index.html has no QR_SVG marker to fill")
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html)

    print(f"wrote qr.svg and index.html for {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
