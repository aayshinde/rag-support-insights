"""Build the static demo page: inline results/demo_data.json into docs/demo_template.html.

Writes docs/index.html, a single self-contained file (no backend) that GitHub Pages or any static host can serve.

Usage:  python -m src.build_demo
"""
from __future__ import annotations

from src import config

DOCS = config.ROOT / "docs"
HEAD = (
    '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
    '<meta name="description" content="A support-ticket triage system built on retrieval-augmented generation that escalates to a human when its evidence is weak.">\n'
)


def main() -> None:
    template = (DOCS / "demo_template.html").read_text()
    data = (config.RESULTS_DIR / "demo_data.json").read_text().replace("</", "<\\/")
    body = template.replace("__DATA__", data)
    title_end = body.index("</title>") + len("</title>")
    html = HEAD + body[:title_end] + "\n" + body[title_end:].replace("<style>", "<style>\nbody{margin:0}", 1)
    html = html.replace("<div class=\"wrap\">", "</head>\n<body>\n<div class=\"wrap\">", 1) + "\n</body>\n</html>\n"
    (DOCS / "index.html").write_text(html)
    (config.RESULTS_DIR / "demo_artifact.html").write_text(body)
    print(f"wrote {DOCS / 'index.html'} ({len(html) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
