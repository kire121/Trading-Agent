"""Builds a single self-contained PDF of REPORT.md + README.md + full source
code, mirroring the Dammluckan.pdf/Efterskalvsklockan.pdf/Smittotalet.pdf
convention (research/smittotalet/build_pdf.py, branch claude/smittotalet-
portfolio-overlay-0bl1sh, commit a67df1b -- ported near-verbatim, only
SOURCE_FILES/title/subtitle changed for Metusalem).

The PDF embeds the actual committed results.json/config_frozen.sha256 so
the config-hash it references and the numbers it shows are read from the
same files the audit trail uses -- never hand-typed here.

Usage: python -m research.metusalem.build_pdf
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
_RESULTS_DIR = os.path.join(_THIS_DIR, "..", "..", "results", "metusalem")

SOURCE_FILES = [
    "config.py", "survival_trend.py", "basbok.py", "scheduling.py", "costs.py",
    "data.py", "signal_construction.py", "oracle.py", "battery.py", "gates.py",
    "synth.py", "run_steg0c.py", "run_research.py", "deliver.py", "finalize_delivery.py",
    "tests/test_survival_trend.py",
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


def _read_results(name):
    with open(os.path.join(_RESULTS_DIR, name), encoding="utf-8") as f:
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
    output_path = output_path or os.path.join(_THIS_DIR, "Metusalem.pdf")

    results = json.loads(_read_results("results.json"))
    config_hash = _read_results("config_frozen.sha256").strip()
    verdict = "DÖD" if not results.get("all_steps_passed") else "LEVER"

    parts = []
    parts.append(f"""
    <div class="titlepage">
      <h1>Metusalem</h1>
      <div class="meta">Trendålderns hasard som tvärsnittstilt</div>
      <div class="verdict">{html.escape(verdict)}</div>
      <div class="meta">Config-hash: {html.escape(config_hash)}<br/>
      Genererad av Claude Code</div>
    </div>
    """)

    toc_items = ["README", "REPORT (resultat och verdikt)", "results.json", "assertions.jsonl",
                 "config_frozen.yaml", "AVVIKELSER.md", "Källkod"] + SOURCE_FILES
    parts.append('<div class="toc"><h1>Innehåll</h1><ol>' +
                 "".join(f"<li>{html.escape(t)}</li>" for t in toc_items) + "</ol></div>")

    parts.append('<div>' + _md_to_html(_read("README.md")) + "</div>")
    parts.append('<div style="page-break-before: always;">' + _md_to_html(_read("REPORT.md")) + "</div>")

    parts.append(_highlight_json(json.dumps(results, indent=2, ensure_ascii=False, sort_keys=True),
                                  "results/metusalem/results.json"))

    assertions_lines = _read_results("assertions.jsonl").strip().splitlines()
    assertions_pretty = "\n".join(json.dumps(json.loads(line), ensure_ascii=False) for line in assertions_lines)
    parts.append(f'<div class="sourcefile"><h2>results/metusalem/assertions.jsonl</h2>'
                 f'<pre>{html.escape(assertions_pretty)}</pre></div>')

    parts.append(f'<div class="sourcefile"><h2>results/metusalem/config_frozen.yaml</h2>'
                 f'<pre>{html.escape(_read_results("config_frozen.yaml"))}</pre></div>')

    parts.append('<div class="sourcefile"><h2>AVVIKELSER.md</h2>' +
                 _md_to_html(_read_results("AVVIKELSER.md")) + "</div>")

    for rel in SOURCE_FILES:
        code = _read(rel)
        parts.append(_highlight_python(code, rel))

    html_doc = f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{''.join(parts)}</body></html>"
    HTML(string=html_doc, base_url=_THIS_DIR).write_pdf(output_path)
    return output_path


if __name__ == "__main__":
    path = build()
    print(f"Wrote {path}")
