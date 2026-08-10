"""
Efterskalvsklockan -- build a single self-contained PDF of the report + full source.

Renders REPORT.md and README.md as formatted HTML (with real tables), the
key result artifacts, and every Python source file with Pygments syntax
highlighting, then converts the whole thing to PDF via WeasyPrint. Layout
ported verbatim from research/dammluckan/build_pdf.py (the one established
PDF-build precedent in this repo's research-branch series).

    python -m research.omori.build_pdf [-o OUTPUT.pdf]
"""
import argparse
import html as html_mod
import os

import markdown
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer

HERE = os.path.dirname(os.path.abspath(__file__))

SOURCE_ORDER = [
    "config.py",
    "fetch_data.py",
    "data.py",
    "events.py",
    "signal.py",
    "priors.py",
    "sizing.py",
    "costs.py",
    "backtest.py",
    "battery.py",
    "nulls.py",
    "twins.py",
    "calibrate.py",
    "grid.py",
    "metrics.py",
    "run_research.py",
    "run_oos.py",
    "tests/test_signal.py",
    "tests/test_events.py",
    "tests/test_costs.py",
    "tests/test_sizing.py",
    "tests/test_metrics.py",
    "tests/test_backtest.py",
    "tests/test_battery.py",
    "tests/test_nulls.py",
    "tests/test_twins.py",
    "tests/test_priors.py",
]

CSS = """
@page {
  size: A4;
  margin: 18mm 16mm 20mm 16mm;
  @bottom-center {
    content: counter(page);
    font-family: 'DejaVu Sans', sans-serif;
    font-size: 8pt;
    color: #888;
  }
}
@page :first { margin: 0; @bottom-center { content: none; } }

body {
  font-family: 'DejaVu Sans', 'Helvetica', sans-serif;
  font-size: 9.2pt;
  line-height: 1.5;
  color: #1a1a1a;
}

/* ---- cover ---- */
.cover {
  page-break-after: always;
  height: 297mm;
  padding: 55mm 22mm 0 22mm;
  box-sizing: border-box;
  background: #12253d;
  color: #fff;
}
.cover .kicker {
  font-size: 9pt; letter-spacing: .22em; text-transform: uppercase;
  color: #7fa8d4; margin-bottom: 10mm;
}
.cover h1 { font-size: 30pt; line-height: 1.15; margin: 0 0 5mm 0; font-weight: 700; }
.cover .sub { font-size: 12.5pt; color: #b9cde4; margin-bottom: 20mm; line-height: 1.45; }
.cover .verdict {
  display: inline-block; border: 1.6pt solid #ff6b6b; color: #ff8f8f;
  padding: 3.5mm 7mm; font-size: 13pt; font-weight: 700; letter-spacing: .05em;
  margin-bottom: 18mm;
}
.cover .meta {
  font-size: 9pt; color: #8fb0d0; line-height: 1.9;
  border-top: .5pt solid #2e4a6b; padding-top: 6mm;
}
.cover .meta b { color: #d5e4f2; font-weight: 600; }

/* ---- headings ---- */
h1 {
  font-size: 17pt; margin: 0 0 4mm 0; padding-bottom: 2.5mm;
  border-bottom: 1.6pt solid #12253d; page-break-after: avoid;
}
h2 {
  font-size: 12.5pt; margin: 8mm 0 3mm 0; color: #12253d;
  page-break-after: avoid; border-bottom: .5pt solid #d4dde6; padding-bottom: 1.5mm;
}
h3 { font-size: 10.5pt; margin: 5mm 0 2mm 0; color: #24425f; page-break-after: avoid; }
p { margin: 0 0 2.6mm 0; text-align: justify; }
strong { font-weight: 700; color: #000; }
ul, ol { margin: 0 0 3mm 0; padding-left: 5.5mm; }
li { margin-bottom: 1.4mm; }

/* ---- tables ---- */
table {
  border-collapse: collapse; width: 100%; margin: 3mm 0 5mm 0;
  font-size: 8pt; page-break-inside: avoid;
}
th {
  background: #12253d; color: #fff; text-align: left; font-weight: 600;
  padding: 2mm 2.2mm; border: .4pt solid #12253d;
}
td { padding: 1.7mm 2.2mm; border: .4pt solid #cfd8e2; vertical-align: top; }
tbody tr:nth-child(even) { background: #f4f7fa; }

/* ---- inline code ---- */
code {
  font-family: 'DejaVu Sans Mono', monospace; font-size: 8pt;
  background: #eef2f6; padding: .3mm 1mm; border-radius: 1.5px; color: #0b3d62;
}

/* ---- section divider pages ---- */
.section-break {
  page-break-before: always;
  margin-top: 12mm; padding: 6mm 0 4mm 0;
  border-top: 2.5pt solid #12253d; border-bottom: .5pt solid #12253d;
}
.section-break h1 { border: none; margin: 0; padding: 0; font-size: 20pt; }
.section-break .lead { font-size: 9.5pt; color: #55697e; margin-top: 2.5mm; }

/* ---- source code ---- */
.srcfile { page-break-before: always; }
.srcfile-head {
  background: #12253d; color: #fff; padding: 2.2mm 3.5mm;
  font-family: 'DejaVu Sans Mono', monospace; font-size: 9.5pt; font-weight: 700;
  page-break-after: avoid;
}
.srcfile-head .lines { float: right; font-weight: 400; color: #9dc0e0; font-size: 8pt; }
.highlight {
  border: .4pt solid #cfd8e2; border-top: none;
  padding: 2.5mm 3mm; background: #fbfcfd; margin: 0 0 4mm 0;
}
.highlight pre {
  font-family: 'DejaVu Sans Mono', monospace; font-size: 6.6pt; line-height: 1.35;
  margin: 0; white-space: pre-wrap; word-wrap: break-word;
}
pre.plain {
  font-family: 'DejaVu Sans Mono', monospace; font-size: 6.9pt; line-height: 1.35;
  border: .4pt solid #cfd8e2; background: #fbfcfd; padding: 2.5mm 3mm;
  white-space: pre-wrap; word-wrap: break-word; margin: 0 0 4mm 0;
}

/* ---- toc ---- */
.toc { page-break-after: always; }
.toc ol { padding-left: 6mm; }
.toc li { margin-bottom: 1.8mm; font-size: 9.5pt; }
.toc .fn { font-family: 'DejaVu Sans Mono', monospace; font-size: 8.6pt; color: #12253d; }
.toc .desc { color: #55697e; font-size: 8.4pt; }
"""

FILE_BLURBS = {
    "config.py": "Alla brief-pinnade och deklarerade konstanter: universum, trösklar, grid, kostnadshinkar.",
    "fetch_data.py": "EODHD-hämtning med JSON-cache; skriver OHLCV-paneler till data/.",
    "data.py": "Panel-dataklass och kausala rullande statistik (MAD-z, sigma, ADV, trailing-korrelation).",
    "events.py": "Händelsedetektion (volym-z + |r0|) och samma-dags-klustring (Dammluckans samtidighetslärdom).",
    "signal.py": "Omori-fit (Huber, c-profilerad), empirisk-Bayes-krympning, tau_exit, den signerade signalen Z.",
    "priors.py": "Instrument- och globalprior p_bar_i/p_bar_global, skattade en gång på IS-panelen.",
    "sizing.py": "Vol-target-bas x Omori-tilt, portföljbruttotak.",
    "costs.py": "ADV-hinkad transaktionskostnadsmodell (porterad från Formdriften).",
    "backtest.py": "Händelsedriven, dag-för-dag, kausal motor: entry, daglig re-fit, hård stopp, bruttotak.",
    "battery.py": "Redundansscreening: p_hat/halveringstid mot faktorbatteriet -- körs före allt annat.",
    "nulls.py": "Estimatornull (within-event blockshuffle) och blockpermutations-IC-tester.",
    "twins.py": "T1 (fast horisont), T2 (p_tilde blockshufflad inom instrument), T3 (slumpad entry).",
    "calibrate.py": "IS-enbart kalibrering av kappa (leave-one-instrument-out) och p*.",
    "grid.py": "27-cells parametergrid, teckenstabilitet, tre-erors-konsistens, DSR-underlag.",
    "metrics.py": "Sharpe, max drawdown, PSR/DSR (Bailey & Lopez de Prado).",
    "run_research.py": "Orkestrerar hela IS-pipelinen, cachead per steg.",
    "run_oos.py": "Den enda, låsta OOS-avläsningen på lands+råvarupanelen.",
    "tests/test_signal.py": "Kända-svar-återhämtning, full-krympning, identifierbarhetsgrind, tau_exit-självkonsistens.",
    "tests/test_events.py": "Kausalitet (ingen framtidsdata i rullande baslinjer), klustringslogik.",
    "tests/test_costs.py": "ADV-hinkgränser och handberäknad kostnadsfraktion.",
    "tests/test_sizing.py": "Tilt-monotoni, riktningstecken, bruttotaksskalning.",
    "tests/test_metrics.py": "Sharpe, PSR/DSR-kalibrering, drawdown.",
    "tests/test_backtest.py": "Entry-timing, öppen-instrument-dedup, hård stopp, bruttotak (syntetisk panel).",
    "tests/test_battery.py": "Redundansscreening-kill-switch, absorptionskvot.",
    "tests/test_nulls.py": "Blockshuffle-nolla, blockpermutations-IC.",
    "tests/test_twins.py": "T1/T2/T3 på syntetisk panel.",
    "tests/test_priors.py": "Terminalfits, prior-fallback till global.",
}


def md_to_html(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    return markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])


def source_html(rel_path, formatter):
    abs_path = os.path.join(HERE, rel_path)
    with open(abs_path, encoding="utf-8") as f:
        code = f.read()
    n_lines = code.count("\n") + 1
    body = highlight(code, PythonLexer(), formatter)
    blurb = FILE_BLURBS.get(rel_path, "")
    return (
        f'<div class="srcfile">'
        f'<div class="srcfile-head">{html_mod.escape(rel_path)}'
        f'<span class="lines">{n_lines} rader</span></div>'
        f"{body}"
        f"</div>"
    ), n_lines, blurb


def csv_table_html(path, max_rows=None):
    import csv
    with open(path, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        return ""
    head, body = rows[0], rows[1:]
    if max_rows:
        body = body[:max_rows]

    def fmt(v):
        try:
            fv = float(v)
            return f"{fv:.4g}" if abs(fv) < 1e6 else v
        except (TypeError, ValueError):
            return v

    th = "".join(f"<th>{html_mod.escape(c)}</th>" for c in head)
    trs = "".join(
        "<tr>" + "".join(f"<td>{html_mod.escape(str(fmt(c)))}</td>" for c in r) + "</tr>"
        for r in body
    )
    return f"<table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>"


def json_pre_html(path):
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as f:
        return f'<pre class="plain">{html_mod.escape(f.read())}</pre>'


def build(output_path):
    formatter = HtmlFormatter(style="friendly", nowrap=False, cssclass="highlight")
    pygments_css = formatter.get_style_defs(".highlight")

    report_html = md_to_html(os.path.join(HERE, "REPORT.md"))
    readme_html = md_to_html(os.path.join(HERE, "README.md"))

    out_dir = os.path.join(HERE, "output")
    grid_html = csv_table_html(os.path.join(out_dir, "grid_table.csv"))
    is_summary_html = json_pre_html(os.path.join(out_dir, "is_results_summary.json"))
    oos_summary_html = json_pre_html(os.path.join(out_dir, "oos_results_summary.json"))
    oos_twins_html = json_pre_html(os.path.join(out_dir, "oos_twins.json"))
    diversification_html = json_pre_html(os.path.join(out_dir, "diversification.json"))

    sources, toc_items, total_lines = [], [], 0
    for rel in SOURCE_ORDER:
        block, n_lines, blurb = source_html(rel, formatter)
        sources.append(block)
        total_lines += n_lines
        toc_items.append(
            f'<li><span class="fn">{html_mod.escape(rel)}</span> '
            f'<span class="desc">— {html_mod.escape(blurb)} ({n_lines} rader)</span></li>'
        )

    cover = f"""
    <div class="cover">
      <div class="kicker">Systematisk strategiforskning · Trading-Agent</div>
      <h1>Efterskalvsklockan</h1>
      <div class="sub">Omori-relaxation som adaptiv exitklocka<br>
        En förregistrerad falsifieringsstudie på volymchock-händelser, 2003–2026</div>
      <div class="verdict">VERDIKT: FÖRKASTAD</div>
      <div class="meta">
        <b>Universum</b> &nbsp; 40 US-ETF:er (IS) + 16 lands-ETF:er + 12 råvaru-ETF:er (OOS, låst)<br>
        <b>Data</b> &nbsp; EODHD, dagliga OHLCV, 2003-01-02 – 2026-08-07<br>
        <b>Kod</b> &nbsp; {len(SOURCE_ORDER)} Python-filer, {total_lines} rader, 80 tester<br>
        <b>Branch</b> &nbsp; claude/omori-exit-strategy-du0qvg
      </div>
    </div>
    """

    toc = f"""
    <div class="toc">
      <h1>Innehåll</h1>
      <h2>Del I — Resultat</h2>
      <ol>
        <li>Forskningsrapport (REPORT.md) — hypotes, grindar, nollbatteri, OOS, verdikt</li>
        <li>Arkitektur och deklarerade avvikelser (README.md)</li>
        <li>Resultatartefakter — 27-cells grid, IS/OOS-sammanfattningar, diversifiering</li>
      </ol>
      <h2>Del II — Källkod ({len(SOURCE_ORDER)} filer, {total_lines} rader)</h2>
      <ol>{''.join(toc_items)}</ol>
    </div>
    """

    doc = f"""<!DOCTYPE html>
<html lang="sv"><head><meta charset="utf-8"><title>Efterskalvsklockan — forskningsrapport och källkod</title>
<style>{CSS}
{pygments_css}
</style></head><body>
{cover}
{toc}
<div class="section-break"><h1>Del I — Resultat</h1>
  <div class="lead">Forskningsrapporten i sin helhet, följd av arkitekturnoteringar och råa resultatartefakter.</div>
</div>
{report_html}
<div class="section-break"><h1>Arkitektur</h1>
  <div class="lead">Modulöversikt, återanvänt material från systergrenar och deklarerade avvikelser från brief:en.</div>
</div>
{readme_html}
<div class="section-break"><h1>Resultatartefakter</h1>
  <div class="lead">Den fullständiga 27-cells griden samt IS-, OOS- och diversifierings-JSON:erna som rapportens
  siffror citerar.</div>
</div>
<h2>27-cells grid (z × theta × tau_cap)</h2>
{grid_html}
<h2>is_results_summary.json</h2>
{is_summary_html}
<h2>oos_results_summary.json</h2>
{oos_summary_html}
<h2>oos_twins.json</h2>
{oos_twins_html}
<h2>diversification.json</h2>
{diversification_html}
<div class="section-break"><h1>Del II — Källkod</h1>
  <div class="lead">Samtliga {len(SOURCE_ORDER)} Python-filer ({total_lines} rader) i läsordning: konfiguration och
  data först, därefter signal, motor, statistiskt maskineri, orkestrering och tester.</div>
</div>
{''.join(sources)}
</body></html>"""

    from weasyprint import HTML
    HTML(string=doc, base_url=HERE).write_pdf(output_path)
    return output_path, total_lines


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default=os.path.join(HERE, "Efterskalvsklockan.pdf"))
    args = ap.parse_args()
    path, n = build(args.output)
    size_mb = os.path.getsize(path) / 1e6
    print(f"Wrote {path} ({size_mb:.2f} MB, {n} lines of source)")
