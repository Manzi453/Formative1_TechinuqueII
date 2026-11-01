#!/usr/bin/env python3
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import markdown
import pandas as pd
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MD, PDF, TMP = HERE / "Formative2_HMM_Report.md", HERE / "Formative2_HMM_Report.pdf", HERE / "_build.html"
CHROME = os.environ.get("CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")

CSS = """
@page { size: A4; margin: 13mm 14mm 14mm 14mm; @bottom-center { content: counter(page) " / " counter(pages); font: 8pt Helvetica, Arial, sans-serif; color: #666; } }
body { font: 9.3pt/1.32 Helvetica, Arial, sans-serif; color: #1b1b1b; margin: 0; }
h1 { font-size: 16pt; margin: 0 0 3px; } h2 { font-size: 11.5pt; margin: 11px 0 3px; border-bottom: 1px solid #bbb; padding-bottom: 1px; }
h3 { font-size: 9.8pt; margin: 7px 0 2px; } p { margin: 3px 0 4px; text-align: justify; } li { margin: 1px 0; } ul { margin: 2px 0 4px 16px; padding: 0; }
code { font: 8.2pt Menlo, Consolas, monospace; background: #f1f1f1; padding: 0 2px; border-radius: 2px; }
table { border-collapse: collapse; margin: 4px auto 2px; font-size: 8.6pt; } th, td { border: 1px solid #999; padding: 2px 7px; text-align: center; }
th { background: #eee; } p.cap, figcaption { font-size: 8pt; color: #333; text-align: left; margin: 2px 6px 6px; }
figure { margin: 5px 0 4px; text-align: center; break-inside: avoid; } figure img { max-width: 100%; }
.row { display: flex; gap: 8px; justify-content: center; align-items: flex-start; break-inside: avoid; } .row figure { flex: 1 1 0; margin: 4px 0; } .row img { width: 78%; }
.ph { background: #fff3a0; border: 1px dashed #c70; padding: 0 3px; font-weight: bold; color: #a00; }
"""


def check_numbers(text):
    """Fail loudly if a quoted number does not match the code outputs."""
    m = json.loads((ROOT / "results/metrics.json").read_text())
    act = pd.read_csv(ROOT / "results/test_per_activity.csv")
    man = pd.read_csv(ROOT / "Data_clean/manifest.csv")
    cmp_ = pd.read_csv(ROOT / "results/model_comparison.csv")
    A = pd.read_csv(ROOT / "results/transition_matrix.csv", index_col=0)
    expect = {
        "rate mean": f"{m['measured_hz_mean']:.3f} Hz", "rate min": f"{m['measured_hz_min']:.3f}", "rate max": f"{m['measured_hz_max']:.3f}",
        "rate std": f"{m['measured_hz_std']:.3f} Hz",
        "train windows": f"{m['n_train_windows']}", "test windows": f"{m['n_test_windows']}",
        "overall": f"{m['overall_accuracy_per_clip'] * 100:.1f} %", "correct": f"{m['correct_windows']} of {m['test_windows']}",
        "wilson": f"{m['overall_accuracy_wilson95'][0] * 100:.1f} % to {m['overall_accuracy_wilson95'][1] * 100:.1f} %",
        "emission only": f"{m['accuracy_emission_only_per_clip'] * 100:.1f} %",
        "ll first": f"{m['ll_first']:.2f}", "ll final": f"{m['ll_final']:.2f}", "bw iterations": f"{m['bw_iterations']} iterations",
        "last dLL": f"{m['ll_min_step'] * 1e5:.1f}e-5", "at floor": f"{m['variances_at_floor'] * 100:.0f} %",
        "median var": f"{m['median_unfloored_variance']:.3f}", "var floor": f"({m['var_floor']} in Z-score units)",
        "diag min": f"{A.values.diagonal().min():.2f} for still", "diag max": f"{A.values.diagonal().max():.3f} for jumping",
    }
    for _, r in act.iterrows():
        row = f"| {r['Activity']} | {int(r['Number of Samples'])} | {r['Sensitivity']:.3f} | {r['Specificity']:.3f} | {r['Accuracy (one-vs-rest)']:.3f} |"
        expect[f"table2 {r['Activity']}"] = row
    expect["table2 overall"] = f"**{m['overall_accuracy_per_clip']:.3f} ({m['correct_windows']} / {m['test_windows']})**"
    sec = man.groupby("activity").duration_s.sum()
    cnt = man.pivot_table(index="activity", columns="split", values="new_name", aggfunc="count")
    for a in ["standing", "walking", "jumping", "still"]:
        expect[f"table1 {a}"] = f"| {a} | {cnt.loc[a, 'train']} | {cnt.loc[a, 'test']} | {sec[a]:.1f} |"
    expect["table1 total"] = f"| **Total** | **{cnt['train'].sum()}** | **{cnt['test'].sum()}** | **{sec.sum():.1f}** |"
    expect["emission-only count"] = f"{round(cmp_.iloc[2, 1] * m['test_windows'])} of {m['test_windows']}"
    missing = {k: v for k, v in expect.items() if v not in text}
    if missing:
        sys.exit("NUMBER MISMATCH between report and results:\n" + "\n".join(f"  {k}: expected '{v}'" for k, v in missing.items()))
    print(f"consistency check passed ({len(expect)} quoted values match results/ and Data_clean/)")


def main():
    text = MD.read_text(encoding="utf-8")
    check_numbers(text)
    body = markdown.markdown(text, extensions=["tables", "sane_lists"])
    body = re.sub(r"\[VIDEO LINK: (\w+)\]", r'<span class="ph">[VIDEO LINK: \1]</span>', body)
    TMP.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>Formative 2 HMM Report</title><style>{CSS}</style></head><body>{body}</body></html>", encoding="utf-8")
    try:
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--allow-file-access-from-files",
                        f"--print-to-pdf={PDF}", TMP.as_uri()], check=True, capture_output=True, timeout=180)
    finally:
        TMP.unlink(missing_ok=True)
    n = len(PdfReader(str(PDF)).pages)
    print(f"wrote {PDF.relative_to(ROOT)}: {n} pages")
    if not 4 <= n <= 5:
        sys.exit(f"PAGE COUNT {n} is outside 4-5")


if __name__ == "__main__":
    main()
