#!/usr/bin/env python3
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import markdown
import numpy as np
import pandas as pd
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "report"
MD, PDF, TMP = REPORT / "Formative2_HMM_Report.md", REPORT / "Formative2_HMM_Report.pdf", REPORT / "_build.html"
CHROME = os.environ.get("CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
PLACEHOLDER = re.compile(r"\[(?:VIDEO LINK: \w+|WRITE IN YOUR OWN WORDS|CONFIRM[^\]]*)\]")
ACTS = ["standing", "walking", "jumping", "still"]
VIDEO_BASE = "https://github.com/Manzi453/Formative1_TechinuqueII/blob/main/Data/videos/"
NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

CSS = """
@page { size: A4; margin: 14mm 16mm 15mm 16mm; @bottom-center { content: "Page " counter(page) " of " counter(pages); font: 8.5pt Calibri, Carlito, Helvetica, Arial, sans-serif; color: #333; } }
body { font: 9.6pt/1.34 Calibri, Carlito, "Helvetica Neue", Helvetica, Arial, sans-serif; color: #111; margin: 0; }
.hdr { text-align: center; border-bottom: 1px solid #888; padding-bottom: 5px; margin-bottom: 2px; } .hdr h1 { font-size: 13.5pt; margin: 0 0 2px; } .hdr p { margin: 1px 0; text-align: center; font-size: 10pt; }
h2 { font-size: 11pt; margin: 9px 0 2px; } p { margin: 3px 0 5px; text-align: justify; } li { margin: 2px 0 4px; text-align: justify; } ul { margin: 2px 0 4px 18px; padding: 0; }
a { color: #111; text-decoration: underline; }
code { font: 8.4pt Menlo, Consolas, monospace; background: #f1f1f1; padding: 0 2px; border-radius: 2px; overflow-wrap: anywhere; }
table { border-collapse: collapse; margin: 5px auto 1px; font-size: 8.8pt; max-width: 100%; } th, td { border: 1px solid #c3c9d3; padding: 2px 9px; text-align: left; vertical-align: top; }
th { background: #dde3ec; } p.cap, figcaption { font-style: italic; font-size: 8.6pt; color: #222; text-align: center; margin: 2px 10px 7px; }
figure { margin: 5px 0 4px; text-align: center; break-inside: avoid; } figure img { max-width: 100%; }
.row { display: flex; gap: 8px; justify-content: center; align-items: flex-start; break-inside: avoid; } .row figure { flex: 1 1 0; margin: 4px 0; } .row img { width: 78%; }
.ph { background: #fff3a0; border: 1px dashed #c70; padding: 0 3px; font-weight: bold; color: #a00; }
blockquote { background: #fffbe6; border-left: 3px solid #c70; margin: 4px 0; padding: 1px 9px; }
"""


def pct(x, nd=1):
    return f"{x * 100:.{nd}f} %"


def check_numbers(text):
    """Fail loudly if a quoted number does not match the code outputs."""
    m = json.loads((ROOT / "results/metrics.json").read_text())
    f = json.loads((ROOT / "results/findings.json").read_text())
    act = pd.read_csv(ROOT / "results/test_per_activity.csv").set_index("Activity")
    cmp_ = pd.read_csv(ROOT / "results/model_comparison.csv")
    A = pd.read_csv(ROOT / "results/transition_matrix.csv", index_col=0)
    cv = pd.read_csv(ROOT / "results/variance_floor_cv.csv")
    pi_cv = pd.read_csv(ROOT / "results/pi_update_cv.csv")
    sens = pd.read_csv(ROOT / "results/variance_floor_sensitivity_posthoc.csv")
    errs = pd.read_csv(ROOT / "results/test_errors.csv")
    cat = pd.read_csv(ROOT / "results/test_confusion_matrix_concatenated.csv", index_col=0)
    man = pd.read_csv(ROOT / "Data_clean/manifest.csv")
    exc = pd.read_csv(ROOT / "Data_clean/excluded.csv")
    sweep = pd.read_csv(ROOT / "Data_clean/trim_sweep.csv").set_index("trim_s")
    log = (ROOT / "Data_clean/build_log.txt").read_text()

    # windows per activity/split recomputed from the cleaned files
    win, hop = m["window_samples"], m["hop_samples"]
    man["windows"] = (man["n_samples"] - win) // hop + 1
    sec = man.groupby("activity").duration_s.sum()
    cnt = man.pivot_table(index="activity", columns="split", values="new_name", aggfunc="count")
    wn = man.pivot_table(index="activity", columns="split", values="windows", aggfunc="sum")
    assert int(wn["train"].sum()) == m["n_train_windows"] and int(wn["test"].sum()) == m["n_test_windows"], \
        "window recount differs from metrics.json"

    raw_tot = json.loads(re.search(r"raw recorded seconds per activity: (\{.*\})", log).group(1).replace("'", '"'))
    raw_min, raw_max = (float(x) for x in re.search(r"raw durations: min ([\d.]+) s, max ([\d.]+) s", log).groups())
    seg = exc[exc.kind == "segment"]
    seg_s = int(seg.reason.str.extract(r"removed (\d+) s")[0].astype(int).sum())
    seg_worst = max(float(x) for r in seg.reason for x in re.findall(r"([\d.]+)x", r))
    mis = exc[exc.reason.str.contains("mislabeled")].reason.iloc[0]
    mis_std = re.search(r"accel-magnitude std ([\d.]+)", mis).group(1)
    mis_med = re.search(r"standing ([\d.]+)", mis).group(1)
    short = exc[(exc.kind == "recording") & exc.reason.str.contains("shorter than")].reason.iloc[0]
    short_s = re.search(r"\(([\d.]+) s;", short).group(1)
    miss = 90 - sec["still"]
    still01 = man[man.new_name == "still_01.csv"].duration_s.iloc[0]
    n_cat_err = int(cat.values.sum() - np.trace(cat.values))
    still_err = errs[errs.true == "still"]

    E = {}
    E["rate"] = (f"{m['measured_hz_mean']:.3f} Hz on average (range {m['measured_hz_min']:.3f} to {m['measured_hz_max']:.3f} Hz, "
                 f"std {m['measured_hz_std']:.3f} Hz)")
    E["raw clips"] = f"{raw_min:.1f} to {raw_max:.1f} s each (" + ", ".join(f"{raw_tot[a]:.1f} s {a}" for a in ACTS) + ")"
    E["raw count"] = "52 clips, 13 per activity" if "'standing': 13, 'walking': 13, 'jumping': 13, 'still': 13" in log else "RAW COUNT CHANGED"
    E["sweep"] = (f"(1.0 s leaves standing at {sweep.loc[1.0, 'standing']:.1f} s and still at {sweep.loc[1.0, 'still']:.1f} s; "
                  f"0.75 s leaves standing at {sweep.loc[0.75, 'standing']:.1f} s; 0.5 s leaves still at {sweep.loc[0.5, 'still']:.1f} s)")
    E["trim used"] = f"so I used {m['trim_s']:g} s"
    E["edge rule"] = f"removed {seg_s} s in total from {seg.new_name.nunique()} clips"
    E["edge worst"] = f"({seg_worst:.1f} times the median)"
    E["mislabelled"] = f"accelerometer-magnitude std {mis_std} against a standing median of {mis_med}"
    E["short clip"] = f"is {short_s} s long after the edge rule"
    for a in ACTS:
        b = "**" if a == "still" else ""
        E[f"table1 {a}"] = f"| {a} | {cnt.loc[a, 'train']} | {cnt.loc[a, 'test']} | {b}{sec[a]:.1f}{b} | {wn.loc[a, 'train']} / {wn.loc[a, 'test']} |"
    E["table1 total"] = (f"| **Total** | **{cnt['train'].sum()}** | **{cnt['test'].sum()}** | **{sec.sum():.1f}** | "
                         f"**{wn['train'].sum()} / {wn['test'].sum()}** |")
    E["table1 caption"] = f"Cleaned dataset ({len(man)} clips)"
    E["still short"] = f"<b>still is {miss:.1f} s short</b> ({sec['still']:.1f} s)"
    E["windows"] = f"giving {m['n_train_windows']} training and {m['n_test_windows']} test windows"
    E["window samples"] = f"({m['window_samples']} samples at {m['measured_hz_mean']:.3f} Hz)"
    assert abs(cv.iloc[0, 1] - cv.iloc[2, 1]) < 1e-12, "floors 0.01..0.1 no longer tie"
    E["floor cv"] = (f"the accuracy was {pct(cv.iloc[0, 1])} for 0.01 to 0.1, {pct(cv.iloc[3, 1])} for 0.25 and "
                     f"{pct(cv.iloc[4, 1])} for 0.5, so the floor is {m['var_floor']:g}")
    assert abs(pi_cv.iloc[0, 1] - pi_cv.iloc[1, 1]) < 1e-12 and not m["update_pi"], "pi decision changed"
    E["pi cv"] = f"it was not ({pct(pi_cv.iloc[0, 1])} both ways)"
    E["training clips"] = f"all {m['n_train_clips']} training clips concatenated"
    E["bw"] = f"converged after {m['bw_iterations']} iterations** (|&Delta;LL| = {m['ll_min_step'] * 1e5:.1f}e-5"
    E["ll"] = f"from {m['ll_first']:.2f} to {m['ll_final']:.2f}"
    assert abs(cmp_.iloc[0, 1] - cmp_.iloc[1, 1]) < 1e-12, "test predictions differ with/without Baum-Welch"
    E["bw same"] = f"({m['correct_windows']} of {m['test_windows']} correct in both cases)"
    E["floor pinned"] = f"{pct(m['variances_at_floor'])} of the state variances sit exactly at the floor"
    E["median var"] = f"is only {m['median_unfloored_variance']:.3f})"
    for a in ACTS:
        r = act.loc[a]
        E[f"table2 {a}"] = f"| {a} | {int(r['Number of Samples'])} | {r['Sensitivity']:.3f} | {r['Specificity']:.3f} | {r['Accuracy (one-vs-rest)']:.3f} |"
    E["table2 overall"] = (f"| **Overall accuracy** | **{m['test_windows']}** | | | "
                           f"**{m['overall_accuracy_per_clip']:.3f} ({m['correct_windows']} / {m['test_windows']})** |")
    E["wilson"] = f"{pct(m['overall_accuracy_wilson95'][0])} to {pct(m['overall_accuracy_wilson95'][1])}"
    E["overall"] = f"{m['correct_windows']} of {m['test_windows']} windows ({pct(m['overall_accuracy_per_clip'])}) are correct"
    E["train acc"] = f"scores {pct(m['train_accuracy_resubstitution'])} on its own training windows"
    E["lofo acc"] = f"and {pct(m['train_accuracy_lofo'])} in leave-one-clip-out"
    E["emission only"] = f"also gets {round(m['accuracy_emission_only_per_clip'] * m['test_windows'])} of {m['test_windows']} per clip"
    E["concat"] = (f"gives {round(m['overall_accuracy_concatenated'] * m['test_windows'])} of {m['test_windows']} "
                   f"({pct(m['overall_accuracy_concatenated'])}) with Viterbi against "
                   f"{round(cmp_.iloc[2, 2] * m['test_windows'])} of {m['test_windows']} ({pct(cmp_.iloc[2, 2])}) without transitions")
    assert round(m["overall_accuracy_concatenated"] * m["test_windows"]) - round(cmp_.iloc[2, 2] * m["test_windows"]) == 1, \
        "'a gain of one window' is no longer true"
    assert act.loc["standing", "Sensitivity"] == 1.0 and act.loc["jumping", "Sensitivity"] == 1.0
    E["never missed"] = "(17 of 17 each)" if int(act.loc["standing", "Number of Samples"]) == int(act.loc["jumping", "Number of Samples"]) == 17 else "COUNTS CHANGED"
    E["top4"] = ", ".join(f"`{t}`" for t in m["top_features"])
    n_still = int(act.loc["still", "Number of Samples"])
    E["still"] = f"sensitivity {act.loc['still', 'Sensitivity']:.3f}, {round(act.loc['still', 'Sensitivity'] * n_still)} of {n_still}"
    assert set(still_err.decoded) == {"standing"} and len(still_err) == 4
    E["still errors"] = "all 4 missed windows were decoded as standing, " + " and ".join(
        f"{(still_err["clip"] == c).sum()} in `{c[:-4]}`" for c in ["still_01.csv", "still_12.csv"])
    wk = errs[errs.true == "walking"]
    assert set(wk["clip"]) == {"walking_13.csv"} and set(wk.decoded) == {"jumping"} and len(wk) == 2
    E["errors first2"] = f"**{m['n_test_errors_in_first_two_windows']} of {m['n_test_errors']} are in the first two windows"
    last = errs[errs["clip"] == "still_01.csv"]
    E["still_01 tail"] = f"windows starting at {last.window_start_s.iloc[0]:.1f} s and {last.window_start_s.iloc[1]:.1f} s of an {still01:.1f} s clip"
    E["fig6"] = f"({m['test_windows']} windows, shuffled clip order); crosses mark the {NUMBER_WORDS[n_cat_err]} errors"
    d = A.values.diagonal()
    off = A.values[~np.eye(4, dtype=bool)]
    E["A"] = (f"({d.min():.3f} for still to {d.max():.3f} for jumping, an expected dwell of {min(f['dwell_windows'].values()):.1f} to "
              f"{max(f['dwell_windows'].values()):.1f} windows) with small, near-uniform off-diagonal entries ({off.min():.3f} to {off.max():.3f})")
    assert d.argmin() == ACTS.index("still") and d.argmax() == ACTS.index("jumping") and f["A_self_transitions_dominate"]
    acc_after = sens["test accuracy (after BW)"]
    assert abs(acc_after.min() - acc_after[sens.var_floor == m["var_floor"]].iloc[0]) < 1e-12, "chosen floor is no longer the lowest"
    E["floor test"] = f"test accuracies between {pct(acc_after.min())} and {acc_after.max() * 100:.0f} % across the five candidate floors"
    # facts quoted in Sections 1 and 5
    pa = f["per_activity"]
    sd = f["median_accel_std"]
    dg = f["A_diagonal"]
    assert pa["standing"]["specificity"] == min(v["specificity"] for v in pa.values()), "standing no longer has the lowest specificity"
    E["s1 result"] = f"decoded {pct(m['overall_accuracy_per_clip'])} ({m['correct_windows']} of {m['test_windows']}) of the windows correctly"
    E["s5 jumping"] = (f"sensitivity {pa['jumping']['sensitivity']:.3f} and specificity {pa['jumping']['specificity']:.3f}, with an accelerometer-magnitude "
                       f"std (median per clip {sd['jumping']:.3f}) far above walking ({sd['walking']:.3f}), standing ({sd['standing']:.3f}) and still ({sd['still']:.3f})")
    E["s5 standing"] = f"({pa['standing']['n']} of {pa['standing']['n']}) but has the lowest specificity ({pa['standing']['specificity']:.3f})"
    E["s5 still"] = (f"(sensitivity {pa['still']['sensitivity']:.3f}, {round(pa['still']['sensitivity'] * pa['still']['n'])} of {pa['still']['n']} windows) "
                     f"and all {len(still_err)} errors were decoded as standing: the median std of still and standing differs only by a factor of "
                     f"{f['still_to_standing_std_ratio']:.1f}")
    E["s5 walking"] = f"Walking reached {pa['walking']['sensitivity']:.3f} ({round(pa['walking']['sensitivity'] * pa['walking']['n'])} of {pa['walking']['n']})"
    rest = errs[errs.window >= 2]
    assert set(rest["clip"]) == {"still_01.csv"} and (rest.window >= rest.windows_in_clip - 2).all(), "late errors are no longer the last windows of still_01"
    k2, n_err = m["n_test_errors_in_first_two_windows"], m["n_test_errors"]
    E["s5 positions"] = (f"{k2} of the {n_err} test errors are in the first two windows of a clip and the other {n_err - k2} are the last two "
                         f"windows of `still_01`")
    E["s5 A"] = ", ".join(f"{dg[a]:.3f} for {a}" for a in ACTS) + f"; off-diagonal entries {off.min():.3f} to {off.max():.3f}; expected dwell " \
        f"{min(f['dwell_windows'].values()):.1f} to {max(f['dwell_windows'].values()):.1f} windows of 1 s"
    E["s5 nyquist"] = f"the Nyquist frequency is {f['nyquist_hz']:.2f} Hz"
    fm = f["median_dominant_freq_hz"]
    assert round(fm["walking"]["y"], 1) == round(fm["walking"]["z"], 1) == 1.5 and round(fm["jumping"]["x"], 1) == 2.0, "cadence statements changed"
    fr = fm["still"]
    E["s5 still freq"] = f"median dominant frequency of still is {fr['x']:.1f} to {fr['z']:.1f} Hz per axis"
    E["s5 walk"] = f"only {pct(f['walking_share_dom_1_to_2p5_hz'])} of walking windows"
    E["s5 ll"] = f"raised the training log-likelihood by {m['ll_final'] - m['ll_first']:.2f} in {m['bw_iterations']} iterations"
    E["s5 floor share"] = f"{pct(m['variances_at_floor'])} of the state variances sit at the variance floor"
    E["s5 train vs test"] = f"training windows score {pct(m['train_accuracy_resubstitution'])} against {pct(m['overall_accuracy_per_clip'])} on test"
    E["s5 short"] = f"still has {sec['still']:.1f} s, {miss:.1f} s below the 90 s target"
    E["s5 conclusion"] = f"with {pct(m['overall_accuracy_per_clip'])} window accuracy ({m['correct_windows']} of {m['test_windows']})"
    # video links: real URLs of the files in this repository, which must exist locally and be tracked by git
    for a in ["standing", "walking", "jumping", "still"]:
        path = ROOT / "Data" / "videos" / f"{a}.mp4"
        assert path.exists(), f"missing video file {path}"
        tracked = True                     # cannot be checked without a git repository
        if (ROOT / ".git").exists():
            try:
                tracked = subprocess.run(["git", "ls-files", "--error-unmatch", f"Data/videos/{a}.mp4"], cwd=ROOT, capture_output=True).returncode == 0
            except FileNotFoundError:
                pass
        assert tracked, f"Data/videos/{a}.mp4 is not tracked by git, so the link would not open"
        E[f"video {a}"] = f"[{a.capitalize()}]({VIDEO_BASE}{a}.mp4)"
    E["header github"] = 'GitHub: <a href="https://github.com/Manzi453/Formative1_TechinuqueII">'
    E["table2 sampling"] = (f"| iPhone 13 | 100 Hz | {m['measured_hz_mean']:.3f} Hz ({m['measured_hz_min']:.3f} to {m['measured_hz_max']:.3f} Hz) | "
                            f"{m['measured_hz_mean']:.3f} Hz, no resampling |")
    E["features"] = f"I compute {m['n_features'] // 2} features, **{m['n_features']} in total** (Table 3)"
    E["table4 caption"] = "Table 4. Results on the unseen test clips"
    assert "three orders" not in text, "outdated magnitude claim is back"
    missing = {k: v for k, v in E.items() if v not in text}
    if missing:
        sys.exit("NUMBER MISMATCH between report and results:\n" + "\n".join(f"  {k}: expected '{v}'" for k, v in missing.items()))
    print(f"consistency check passed ({len(E)} quoted values match results/ and Data_clean/)")


def main():
    final = "--final" in sys.argv[1:]
    text = MD.read_text(encoding="utf-8")
    check_numbers(text)
    found = PLACEHOLDER.findall(text)
    if found:
        print(f"{len(found)} placeholder(s) still in the report:")
        for p in found:
            print("  ", p)
        if final:
            sys.exit("FINAL BUILD REFUSED: replace every placeholder first")
    body = markdown.markdown(text, extensions=["tables", "sane_lists"])
    body = PLACEHOLDER.sub(lambda mt: f'<span class="ph">{mt.group(0)}</span>', body)
    TMP.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>Formative 2 HMM Report</title>"
                   f"<style>{CSS}</style></head><body>{body}</body></html>", encoding="utf-8")
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
