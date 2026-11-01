#!/usr/bin/env python3
import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "Data"
OUT = ROOT / "Data_clean"

ACTIVITIES = ["standing", "walking", "jumping", "still"]
TRIM_CANDIDATES = (1.0, 0.75, 0.5)   # seconds removed from both ends; largest passing value is used
CAP_S = 10.0          # maximum clip length kept
MIN_S = 5.0           # minimum clip length kept
TOL_NS = 5_000_000    # 5 ms tolerance for accelerometer/gyroscope alignment
MIN_TOTAL_S = 90.0    # required usable seconds per activity (train + test)
SEG = 100             # samples per 1 s segment at the nominal 100 Hz
EDGE_RATIO = 2.0      # edge segment is a transient if its std exceeds this multiple of the clip median
EDGE_MAX_DROP = 3     # at most this many 1 s segments are removed from each end

# Recordings whose signal contradicts the label (walking-like signal in a "standing" file).
MISLABELLED = {
    "standing_02-2026-10-28_17-23-27": "mislabeled: signal looks like walking, not standing",
}


class Tee:
    """Mirror stdout into a log file."""

    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")
        self.o = sys.stdout

    def write(self, s):
        self.o.write(s)
        self.f.write(s)

    def flush(self):
        self.o.flush()
        self.f.flush()


def rel(p):
    return Path(p).resolve().relative_to(ROOT).as_posix()


def read_zip(path):
    """Return (acc, gyr, meta) DataFrames read straight from the zip (nothing is extracted)."""
    with zipfile.ZipFile(path) as z:
        names = {Path(n).name: n for n in z.namelist()}
        rd = lambda key: pd.read_csv(io.BytesIO(z.read(names[key])))
        return rd("Accelerometer.csv"), rd("Gyroscope.csv"), rd("Metadata.csv")


def rate_stats(df):
    t = df["time"].to_numpy(dtype="int64") / 1e9
    dt = np.diff(t)
    return dict(
        n=len(df),
        dur=float(t[-1] - t[0]),
        hz=float((len(t) - 1) / (t[-1] - t[0])),
        max_gap_ms=float(dt.max() * 1000),
        nonmono=int((dt <= 0).sum()),
        dups=int(df["time"].duplicated().sum()),
        nans=int(df[["x", "y", "z"]].isna().sum().sum()),
    )


def merge_streams(acc, gyr):
    """Align gyro to accelerometer rows by nearest timestamp; merge strictly by column NAME."""
    a = acc[["time", "x", "y", "z"]].rename(columns={"x": "ax", "y": "ay", "z": "az"}).sort_values("time")
    g = gyr[["time", "x", "y", "z"]].rename(columns={"x": "gx", "y": "gy", "z": "gz"}).sort_values("time")
    g = g.assign(_gyro_time=g["time"])
    m = pd.merge_asof(a, g, on="time", direction="nearest", tolerance=TOL_NS)
    unmatched = m[["gx", "gy", "gz"]].isna().any(axis=1)
    info = dict(
        acc_rows=len(a),
        gyr_rows=len(g),
        dropped_acc=int(unmatched.sum()),
    )
    m = m[~unmatched].copy()
    info["gyro_reused"] = int(m["_gyro_time"].duplicated().sum())
    info["gyro_unused"] = int(len(g) - m["_gyro_time"].nunique())
    return m.drop(columns="_gyro_time").reset_index(drop=True), info


def mag(df, cols):
    return np.sqrt((df[cols] ** 2).sum(axis=1)).to_numpy()


def drop_edge_transients(m):
    """Rule 3. Returns (frame, seconds dropped at the start, seconds dropped at the end, ratios)."""
    a = mag(m, ["ax", "ay", "az"])
    n = len(a)
    nseg = n // SEG
    if nseg < 3:
        return m, 0.0, 0.0, []
    med = float(np.median([a[i * SEG:(i + 1) * SEG].std() for i in range(nseg)]))
    if med <= 0:
        return m, 0.0, 0.0, []
    lo, hi, ratios = 0, n, []
    for _ in range(EDGE_MAX_DROP):
        r = a[lo:lo + SEG].std() / med
        if r <= EDGE_RATIO or hi - lo <= 3 * SEG:
            break
        ratios.append(("start", float(r)))
        lo += SEG
    for _ in range(EDGE_MAX_DROP):
        r = a[hi - SEG:hi].std() / med
        if r <= EDGE_RATIO or hi - lo <= 3 * SEG:
            break
        ratios.append(("end", float(r)))
        hi -= SEG
    t = m["time"].to_numpy() / 1e9
    return m.iloc[lo:hi].reset_index(drop=True), float(t[lo] - t[0]), float(t[n - 1] - t[hi - 1]), ratios


def clean_clip(m, trim):
    """Rules 2-4 for one merged recording. Returns (clip or None, meta dict)."""
    t = (m["time"] - m["time"].iloc[0]).to_numpy() / 1e9
    seg = m[(t >= trim) & (t <= t[-1] - trim)].reset_index(drop=True)
    meta = dict(dropped_start_s=0.0, dropped_end_s=0.0, ratios=[], capped=False)
    if len(seg) < 2 * SEG:
        return None, meta
    seg, d0, d1, ratios = drop_edge_transients(seg)
    meta.update(dropped_start_s=d0, dropped_end_s=d1, ratios=ratios)
    ts = (seg["time"] - seg["time"].iloc[0]).to_numpy() / 1e9
    if ts[-1] > CAP_S:
        centre = ts[-1] / 2
        seg = seg[(ts >= centre - CAP_S / 2) & (ts <= centre + CAP_S / 2)]
        meta["capped"] = True
    out = seg.reset_index(drop=True)
    out.insert(0, "timestamp", (out["time"] - out["time"].iloc[0]) / 1e9)
    return out.drop(columns="time"), meta


def clean_all(records, trim):
    """Apply the rules to every recording at one trim length. Returns (kept, excluded)."""
    kept, excluded = [], []
    for r in records:
        stem = r["stem"]
        clip, meta = clean_clip(r["merged"], trim)
        dur = float(clip["timestamp"].iloc[-1]) if clip is not None else 0.0
        if stem in MISLABELLED:
            excluded.append(dict(original_zip=rel(r["zip"]), activity=r["act"], kind="recording", new_name="",
                                 reason=MISLABELLED[stem], _clip=clip))
            continue
        if clip is None or dur < MIN_S:
            excluded.append(dict(original_zip=rel(r["zip"]), activity=r["act"], kind="recording", new_name="",
                                 reason=f"shorter than {MIN_S:.0f} s after trimming {trim} s per end and the edge-transient "
                                        f"rule ({dur:.2f} s; raw {r['sa']['dur']:.2f} s)",
                                 _clip=clip))
            continue
        notes = []
        info = r["info"]
        if info["dropped_acc"] or info["gyro_unused"] or info["gyro_reused"]:
            notes.append(f"aligned by merge_asof: acc rows without gyro match {info['dropped_acc']}, "
                         f"gyro rows unused {info['gyro_unused']}, gyro rows reused {info['gyro_reused']}")
        if meta["capped"]:
            notes.append("capped to central 10 s")
        kept.append(dict(rec=r, clip=clip, notes=notes, dur=dur, meta=meta))
    return kept, excluded


def totals_of(kept):
    return {a: float(sum(k["dur"] for k in kept if k["rec"]["act"] == a)) for a in ACTIVITIES}


def main():
    OUT.mkdir(exist_ok=True)
    (OUT / "train").mkdir(exist_ok=True)
    (OUT / "test").mkdir(exist_ok=True)
    sys.stdout = Tee(OUT / "build_log.txt")

    # ------------------------------------------------------------------ 1. raw data checks
    zips = []
    for act in ACTIVITIES:
        for z in sorted((RAW / act).glob("*.zip")):
            zips.append((act, z))
    print(f"== PHASE 1: raw data checks ({len(zips)} zips in {rel(RAW)}/, read-only) ==")
    records = []
    col_orders = set()
    for act, z in zips:
        acc, gyr, meta = read_zip(z)
        col_orders.add(tuple(c for c in acc.columns if c in "xyz"))
        sa, sg = rate_stats(acc), rate_stats(gyr)
        rec_time = re.search(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})$", z.stem).group(1)
        merged, info = merge_streams(acc, gyr)
        records.append(dict(
            act=act, zip=z, stem=z.stem, rec_time=rec_time,
            hand_named=not z.stem[0].isdigit(),
            acc=acc, gyr=gyr, meta=meta, sa=sa, sg=sg, merged=merged, info=info,
        ))
    per_act = {a: sum(r["act"] == a for r in records) for a in ACTIVITIES}
    print("files per activity:", per_act, "| total:", len(records))
    print("accelerometer x/y/z column order in files:", sorted(col_orders), "(merge is done by NAME)")
    hz = np.array([r["sa"]["hz"] for r in records])
    print(f"measured accelerometer rate over all zips: min {hz.min():.3f}  max {hz.max():.3f}  "
          f"mean {hz.mean():.3f}  std {hz.std():.4f} Hz")
    print("NaNs in acc/gyro:", sum(r["sa"]["nans"] + r["sg"]["nans"] for r in records),
          "| non-monotonic steps:", sum(r["sa"]["nonmono"] + r["sg"]["nonmono"] for r in records),
          "| duplicate timestamps:", sum(r["sa"]["dups"] + r["sg"]["dups"] for r in records),
          f"| max inter-sample gap: {max(r['sa']['max_gap_ms'] for r in records):.2f} ms")
    mism = [(r["stem"], r["sa"]["n"], r["sg"]["n"]) for r in records if r["sa"]["n"] != r["sg"]["n"]]
    print("acc/gyro row-count mismatches (stem, acc rows, gyro rows):", mism)
    durs = np.array([r["sa"]["dur"] for r in records])
    print(f"raw durations: min {durs.min():.2f} s, max {durs.max():.2f} s, files > 10 s: {(durs > 10).sum()}")
    devs = {(r["meta"]["device name"].iloc[0], r["meta"]["device id"].iloc[0]) for r in records}
    rates = {str(r["meta"]["sampleRateMs"].iloc[0]) for r in records}
    apps = {str(r["meta"]["appVersion"].iloc[0]) + "/" + str(r["meta"]["platform"].iloc[0]) for r in records}
    print("devices:", devs, "| metadata sampleRateMs:", rates, "| appVersion/platform:", apps)
    print("annotation files empty:", all(
        zipfile.ZipFile(r["zip"]).getinfo("Annotation.csv").file_size == 0 for r in records))
    hand = [r["stem"] for r in records if r["hand_named"]]
    print(f"hand-named recordings ({len(hand)}):", *hand, sep="\n  ")
    raw_tot = {a: sum(r["sa"]["dur"] for r in records if r["act"] == a) for a in ACTIVITIES}
    print("raw recorded seconds per activity:", {a: round(v, 1) for a, v in raw_tot.items()})
    print()

    # ------------------------------------------------------------------ 2. trim sweep
    print(f"== PHASE 2: trim sweep (rule: largest trim for which every activity reaches {MIN_TOTAL_S:.0f} s; "
          f"edge-transient rule > {EDGE_RATIO:g}x median, clips {MIN_S:.0f}-{CAP_S:.0f} s) ==")
    runs, sweep_rows = {}, []
    for trim in TRIM_CANDIDATES:
        kept_t, excl_t = clean_all(records, trim)
        tot = totals_of(kept_t)
        ok = all(v >= MIN_TOTAL_S for v in tot.values())
        runs[trim] = (kept_t, excl_t, tot, ok)
        sweep_rows.append(dict(trim_s=trim, **{a: round(tot[a], 1) for a in ACTIVITIES},
                               clips=len(kept_t), all_activities_reach_90s=ok))
        print(f"trim {trim:4.2f} s: " + "  ".join(f"{a} {tot[a]:6.1f}" for a in ACTIVITIES)
              + f"  clips {len(kept_t)}  -> {'PASS' if ok else 'FAIL'}")
    passing = [t for t in TRIM_CANDIDATES if runs[t][3]]
    trim_sel = max(passing) if passing else min(TRIM_CANDIDATES)
    print(f"selected trim: {trim_sel} s" + ("" if passing else "  (NO candidate reaches the target; smallest candidate used)"))
    pd.DataFrame(sweep_rows).assign(selected=lambda d: d.trim_s == trim_sel).to_csv(OUT / "trim_sweep.csv", index=False)
    kept, excluded, tot_sel, _ = runs[trim_sel]
    print()

    # ------------------------------------------------------------------ 3. split + rename
    print(f"== PHASE 3: clean set at trim {trim_sel} s ==")
    for k in kept:
        k["split"] = "test" if k["rec"]["hand_named"] else "train"
    kept.sort(key=lambda k: (k["rec"]["act"], k["rec"]["rec_time"]))
    counters = {a: 0 for a in ACTIVITIES}
    for k in kept:
        counters[k["rec"]["act"]] += 1
        k["new_name"] = f"{k['rec']['act']}_{counters[k['rec']['act']]:02d}.csv"

    # ------------------------------------------------------------------ QC (heuristic)
    for k in kept:
        c = k["clip"]
        k["a_std"] = float(mag(c, ["ax", "ay", "az"]).std())
        k["g_std"] = float(mag(c, ["gx", "gy", "gz"]).std())
        k["hz"] = float((len(c) - 1) / c["timestamp"].iloc[-1])
        a = mag(c, ["ax", "ay", "az"])
        sec = np.array([a[i * SEG:(i + 1) * SEG].std() for i in range(len(a) // SEG)])
        k["edge_ratio"] = float(max(sec[0], sec[-1]) / np.median(sec)) if len(sec) >= 3 else np.nan
    med = {a: (np.median([np.log(k["a_std"]) for k in kept if k["rec"]["act"] == a]),
               np.median([np.log(k["g_std"]) for k in kept if k["rec"]["act"] == a])) for a in ACTIVITIES}
    for k in kept:
        pt = (np.log(k["a_std"]), np.log(k["g_std"]))
        d = {a: np.hypot(pt[0] - m[0], pt[1] - m[1]) for a, m in med.items()}
        k["nearest"] = min(d, key=d.get)
        flags = []
        if k["nearest"] != k["rec"]["act"]:
            flags.append(f"QC: std-profile closer to '{k['nearest']}'")
        if k["edge_ratio"] > EDGE_RATIO:
            flags.append(f"QC: residual edge transient (edge/median 1-s std = {k['edge_ratio']:.1f})")
        k["flags"] = flags
        m0 = k["meta"]
        edge = []
        if m0["dropped_start_s"] > 0:
            edge.append(f"first {m0['dropped_start_s']:.2f} s")
        if m0["dropped_end_s"] > 0:
            edge.append(f"last {m0['dropped_end_s']:.2f} s")
        k["edge_note"] = ("edge-transient rule removed " + " and ".join(edge)) if edge else ""
        k["notes"] = (["hand-named session recording (held out as unseen test file)"] if k["split"] == "test" else []) \
            + ([k["edge_note"]] if k["edge_note"] else []) + k["notes"] + flags

    # evidence for the mislabelled-recording exclusion
    for e in excluded:
        if "mislabeled" in e["reason"] and e["_clip"] is not None:
            c = e["_clip"]
            ea, eg = float(mag(c, ["ax", "ay", "az"]).std()), float(mag(c, ["gx", "gy", "gz"]).std())
            e["reason"] += (f" (accel-magnitude std {ea:.3f}, gyro std {eg:.3f}; class medians of kept clips: "
                            + ", ".join(f"{a} {np.exp(med[a][0]):.3f}" for a in ACTIVITIES) + ")")

    # removed edge segments are logged next to the excluded recordings
    for k in kept:
        for side in ("start", "end"):
            rs = [r for s, r in k["meta"]["ratios"] if s == side]
            if rs:
                excluded.append(dict(original_zip=rel(k["rec"]["zip"]), activity=k["rec"]["act"], kind="segment",
                                     new_name=k["new_name"],
                                     reason=f"removed {len(rs)} s at the {side}: 1-s accel-magnitude std was "
                                            + ", ".join(f"{r:.1f}x" for r in rs)
                                            + f" the clip median (rule: > {EDGE_RATIO:g}x)", _clip=None))

    # ------------------------------------------------------------------ write files
    for sub in ("train", "test"):
        for f in (OUT / sub).glob("*.csv"):
            f.unlink()
    manifest = []
    for k in kept:
        c = k["clip"]
        c.to_csv(OUT / k["split"] / k["new_name"], index=False, float_format="%.6f")
        manifest.append(dict(
            new_name=k["new_name"], original_zip=rel(k["rec"]["zip"]), activity=k["rec"]["act"],
            split=k["split"], duration_s=round(k["dur"], 3), n_samples=len(c),
            measured_hz=round(k["hz"], 3), trim_s=trim_sel,
            edge_dropped_start_s=round(k["meta"]["dropped_start_s"], 3),
            edge_dropped_end_s=round(k["meta"]["dropped_end_s"], 3),
            qc_std_profile_nearest=k["nearest"], qc_edge_ratio=round(k["edge_ratio"], 2),
            notes="; ".join(k["notes"]),
        ))
    man = pd.DataFrame(manifest).sort_values(["activity", "new_name"])
    man.to_csv(OUT / "manifest.csv", index=False)
    pd.DataFrame([{k: v for k, v in e.items() if not k.startswith("_")} for e in excluded],
                 columns=["original_zip", "activity", "kind", "new_name", "reason"]).sort_values(
        ["kind", "original_zip"]).to_csv(OUT / "excluded.csv", index=False)

    # ------------------------------------------------------------------ report
    n_rec = sum(e["kind"] == "recording" for e in excluded)
    n_seg = sum(e["kind"] == "segment" for e in excluded)
    print(f"kept {len(kept)} clips; excluded {n_rec} recordings and removed {n_seg} edge segments "
          f"(see Data_clean/excluded.csv):")
    for e in excluded:
        print(f"  - [{e['kind']}] {e['original_zip']}" + (f" -> {e['new_name']}" if e["new_name"] else "") + f": {e['reason']}")
    print()
    print("== PER-FILE QC (measured Hz; accel/gyro magnitude std; nearest activity by log-std medians) ==")
    q = pd.DataFrame([dict(file=k["new_name"], split=k["split"], dur_s=round(k["dur"], 2),
                           n=len(k["clip"]), hz=round(k["hz"], 2), accel_std=round(k["a_std"], 3),
                           gyro_std=round(k["g_std"], 3), nearest=k["nearest"],
                           edge_ratio=round(k["edge_ratio"], 1)) for k in kept])
    pd.set_option("display.width", 200, "display.max_rows", 200)
    print(q.to_string(index=False))
    print()
    print("class medians (accel std, gyro std) of kept clips:",
          {a: (round(float(np.exp(m[0])), 3), round(float(np.exp(m[1])), 3)) for a, m in med.items()})
    flagged = [k for k in kept if k["flags"]]
    print(f"QC-flagged clips ({len(flagged)}), kept (the rules above are the only filters):")
    for k in flagged:
        print(f"  - {k['split']}/{k['new_name']} <- {k['rec']['stem']}: {'; '.join(k['flags'])}")
    print()
    print("== TOTALS (train + test combined, after cleaning) ==")
    shortfalls = {}
    for a in ACTIVITIES:
        ks = [k for k in kept if k["rec"]["act"] == a]
        tot = sum(k["dur"] for k in ks)
        ntr = sum(k["split"] == "train" for k in ks)
        nte = len(ks) - ntr
        print(f"{a:9s} files {len(ks):2d} (train {ntr:2d} / test {nte}) total {tot:6.1f} s  "
              f"{'OK' if tot >= MIN_TOTAL_S else 'BELOW 90 s'}")
        if tot < MIN_TOTAL_S:
            shortfalls[a] = MIN_TOTAL_S - tot
    print(f"total files: {len(kept)} (train {sum(k['split'] == 'train' for k in kept)}, "
          f"test {sum(k['split'] == 'test' for k in kept)})")
    for a in ACTIVITIES:
        assert sum(k["split"] == "test" and k["rec"]["act"] == a for k in kept) >= 2, f"<2 test files for {a}"
    assert not any(k["rec"]["stem"] in MISLABELLED for k in kept)
    train_src = {k["rec"]["stem"] for k in kept if k["split"] == "train"}
    test_src = {k["rec"]["stem"] for k in kept if k["split"] == "test"}
    assert train_src.isdisjoint(test_src), "a recording is in both train and test"
    if shortfalls:
        print(f"\nSHORTFALL vs the {MIN_TOTAL_S:.0f} s per-activity target (trim {trim_sel} s, edge-transient rule, "
              f"{MIN_S:.0f}-{CAP_S:.0f} s clips):")
        for a, s in shortfalls.items():
            need = CAP_S + 2 * trim_sel
            print(f"  {a:9s} missing {s:5.1f} s -> record {int(np.ceil(s / CAP_S))} more recording(s) of at least "
                  f"{need:.1f} s (each yields 10 s after trimming; add margin for edge-transient seconds)")
    print("\nwritten:", rel(OUT / "manifest.csv"), rel(OUT / "excluded.csv"), rel(OUT / "trim_sweep.csv"),
          rel(OUT / "build_log.txt"))
    sys.stdout = sys.stdout.o


if __name__ == "__main__":
    main()
