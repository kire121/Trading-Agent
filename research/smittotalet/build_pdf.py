"""Build a single self-contained PDF of REPORT.md + README.md + the full
source code (config through tests), mirroring the Dammluckan.pdf /
Efterskalvsklockan.pdf convention used by the sibling research branches.

Usage: python -m research.smittotalet.build_pdf
"""
import html
import json
import os

import markdown
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import JsonLexer, PythonLexer
from weasyprint import HTML

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

SOURCE_FILES = [
    "config.py", "eodhd_client.py", "fetch_data.py", "data.py",
    "events.py", "signal.py", "scheduling.py", "tsmom.py", "costs.py",
    "backtest.py", "twins.py", "battery.py", "episodes.py", "nulls.py",
    "grid.py", "metrics.py", "run_research.py",
    "tests/test_config.py", "tests/test_events.py", "tests/test_signal.py",
    "tests/test_tsmom.py", "tests/test_costs.py", "tests/test_twins.py",
    "tests/test_episodes.py", "tests/test_nulls.py", "tests/test_metrics.py",
]

PYGMENTS_STYLE = "default"
CSS = """
@page { size: A4; margin: 2cm 1.8cm; @bottom-center { content: counter(page); font-size: 9px; color: #888; } }
body { font-family: "Helvetica Neue", Arial, sans-serif; font-size: 10.5px; line-height: 1.5; color: #1a1a1a; }
h1 { font-size: 22px; border-bottom: 2px solid #222; padding-bottom: 6px; margin-top: 0; }
h2 { font-size: 15px; margin-top: 22px; border-bottom: 1px solid #ccc; padding-bottom: 3px; }
h3 { font-size: 12.5px; margin-top: 16px; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 9.5px; }
th, td { border: 1px solid #999; padding: 4px 7px; text-align: left; }
th { background: #eee; }
code { font-family: "DejaVu Sans Mono", monospace; background: #f2f2f2; padding: 1px 3px; font-size: 9px; }
pre { background: #f2f2f2; padding: 6px; border-radius: 3px; font-size: 8.5px; overflow-x: auto;
      white-space: pre-wrap; word-wrap: break-word; }
.titlepage { text-align: center; margin-top: 30%; page-break-after: always; }
.titlepage h1 { border: none; font-size: 32px; }
.titlepage .verdict { font-size: 16px; margin-top: 30px; padding: 10px 20px; display: inline-block;
                       border: 2px solid #a00; color: #a00; font-weight: bold; }
.titlepage .meta { color: #555; margin-top: 40px; font-size: 11px; }
.toc { page-break-after: always; }
.toc ol { font-size: 11px; line-height: 2; }
.sourcefile { page-break-before: always; }
.sourcefile h2 { font-family: "DejaVu Sans Mono", monospace; }
""" + HtmlFormatter(style=PYGMENTS_STYLE).get_style_defs(".highlight")


def _read(path):
    with open(os.path.join(_THIS_DIR, path), encoding="utf-8") as f:
        return f.read()


def _md_to_html(md_text):
    return markdown.markdown(md_text, extensions=["tables", "fenced_code", "toc"])


def _highlight_python(code, filename):
    return f'<div class="sourcefile"><h2>{html.escape(filename)}</h2>' \
           + highlight(code, PythonLexer(), HtmlFormatter(style=PYGMENTS_STYLE)) + "</div>"


def _highlight_json(code, filename):
    return f'<div class="sourcefile"><h2>{html.escape(filename)}</h2>' \
           + highlight(code, JsonLexer(), HtmlFormatter(style=PYGMENTS_STYLE)) + "</div>"


def build(output_path=None):
    output_path = output_path or os.path.join(_THIS_DIR, "Smittotalet.pdf")

    verdict = "FORKASTAD"
    results_path = os.path.join(_THIS_DIR, "output", "is_results_summary.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            v = json.load(f).get("verdict", "")
            if v:
                verdict = v

    parts = []
    parts.append(f"""
    <div class="titlepage">
      <h1>Smittotalet</h1>
      <div class="meta">Epidemiskt reproduktionstal (R_t) som portfoljgasreglage pa en TSMOM-basbok</div>
      <div class="verdict">{html.escape(verdict)}</div>
      <div class="meta">Fullstandig kod, dokumentation och resultat<br/>Genererad av Claude Code</div>
    </div>
    """)

    toc_items = ["README", "REPORT (resultat och verdikt)", "Kallkod"] + SOURCE_FILES
    parts.append('<div class="toc"><h1>Innehall</h1><ol>' +
                 "".join(f"<li>{html.escape(t)}</li>" for t in toc_items) + "</ol></div>")

    parts.append('<div>' + _md_to_html(_read("README.md")) + "</div>")
    parts.append('<div style="page-break-before: always;">' + _md_to_html(_read("REPORT.md")) + "</div>")

    for rel in SOURCE_FILES:
        code = _read(rel)
        parts.append(_highlight_python(code, rel))

    if os.path.exists(results_path):
        with open(results_path) as f:
            pretty = json.dumps(json.load(f), indent=2, ensure_ascii=False)
        parts.append(_highlight_json(pretty, "output/is_results_summary.json"))

    html_doc = f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{''.join(parts)}</body></html>"
    HTML(string=html_doc, base_url=_THIS_DIR).write_pdf(output_path)
    return output_path


if __name__ == "__main__":
    path = build()
    print(f"Wrote {path}")
