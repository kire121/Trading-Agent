"""Build a single self-contained PDF from REPORT.md, referencing the
config-hash. Session rule 6: "PDF:en får inte innehålla siffror som saknas
i results.json" -- REPORT.md (build_report.py) sources every number
directly from results.json, so rendering it verbatim satisfies this by
construction. AVVIKELSER.md's prose is intentionally NOT reproduced here
(it contains illustrative/diagnostic numbers from test construction that do
not live in results.json, e.g. correlation figures from synthetic-test
debugging) -- the PDF points readers to that file instead of re-stating it,
to keep every number in the PDF strictly traceable to results.json.

Mirrors the Dammluckan.pdf / Efterskalvsklockan.pdf / Smittotalet.pdf
convention (markdown -> weasyprint) used by sibling research branches;
provenance: research/smittotalet/build_pdf.py, branch
claude/smittotalet-portfolio-overlay-0bl1sh, commit a67df1b (structure/CSS
pattern reused, content generation is new since Timglaset's report schema
is unrelated).
"""
import json
from pathlib import Path

import markdown
from weasyprint import HTML

from . import config

_THIS_DIR = Path(__file__).resolve().parent

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
.titlepage { text-align: center; margin-top: 25%; page-break-after: always; }
.titlepage h1 { border: none; font-size: 30px; }
.titlepage .verdict { font-size: 15px; margin-top: 24px; padding: 10px 20px; display: inline-block;
                       border: 2px solid #a00; color: #a00; font-weight: bold; }
.titlepage .hash { font-family: "DejaVu Sans Mono", monospace; font-size: 10px; color: #555;
                    margin-top: 30px; word-break: break-all; }
.titlepage .meta { color: #555; margin-top: 16px; font-size: 11px; }
.deliverables { page-break-before: always; }
.deliverables code { display: block; margin: 4px 0; }
"""


def _md_to_html(md_text):
    return markdown.markdown(md_text, extensions=["tables", "fenced_code"])


def build(results_path=None, report_md_path=None, output_path=None):
    results_dir = Path(config.RESULTS_DIR).resolve()
    results_path = results_path or (results_dir / "results.json")
    report_md_path = report_md_path or (_THIS_DIR / "REPORT.md")
    output_path = output_path or (_THIS_DIR / "Timglaset.pdf")

    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)
    with open(report_md_path, encoding="utf-8") as f:
        report_md = f.read()

    config_hash = results["config_hash"]
    stopped_at = results["stopped_at"]
    verdict = "SAMTLIGA KÖRDA STEG PASSERADE" if stopped_at is None else f"DÖD -- {stopped_at.upper()}"

    title_html = f"""
    <div class="titlepage">
      <h1>TIMGLASET</h1>
      <div class="meta">Volymsubordinerad trendklocka -- autogenererad slutrapport</div>
      <div class="verdict">{verdict}</div>
      <div class="hash">config_hash:<br>{config_hash}</div>
      <div class="meta">Genererad från results/timglaset/results.json</div>
    </div>
    """

    body_html = _md_to_html(report_md)

    deliverables_html = f"""
    <div class="deliverables">
      <h2>Leveransfiler (exakta sökvägar)</h2>
      <code>{results_dir / 'results.json'}</code>
      <code>{results_dir / 'assertions.jsonl'}</code>
      <code>{results_dir / 'config_frozen.yaml'}</code>
      <code>{results_dir / 'config_frozen.sha256'}</code>
      <code>{results_dir / 'AVVIKELSER.md'}</code>
      <p>AVVIKELSER.md innehåller den fullständiga loggen över tolkningsval och
      metodologiska fynd (inte återgiven här, eftersom den innehåller
      illustrativa/diagnostiska tal från testkonstruktionen som inte finns i
      results.json -- se sessionsregel 6).</p>
    </div>
    """

    html = f"<html><head><meta charset='utf-8'></head><body>{title_html}{body_html}{deliverables_html}</body></html>"
    from weasyprint import CSS as WeasyCSS
    HTML(string=html, base_url=str(_THIS_DIR)).write_pdf(str(output_path), stylesheets=[WeasyCSS(string=CSS)])
    print(f"[build_pdf] Wrote {output_path}")
    return output_path


if __name__ == "__main__":
    build()
