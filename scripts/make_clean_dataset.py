#!/usr/bin/env python3
"""Build Data_clean/ from the raw Sensor Logger zips in Data/ (Data/ is only ever read).

Stages
  1. Audit the raw zips (counts, column order, sampling rate, NaNs, timestamp health,
     row-count mismatches, durations).
  2. Per zip: align Accelerometer.csv and Gyroscope.csv by nearest timestamp
     (merge_asof, 5 ms tolerance), trim 1.5 s from both ends, cap at 10 s (centre crop),
     keep only clips >= 5 s.
  3. Split into Data_clean/train and Data_clean/test (test = the hand-named recordings
     from the separate sessions, see README), rename to <activity>_<NN>.csv.
  4. Write manifest.csv, excluded.csv, build_log.txt and print per-file QC.

Run from anywhere:  python scripts/make_clean_dataset.py
"""
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
TRIM_S = 1.5          # removed from both ends of every recording
CAP_S = 10.0          # maximum clip length kept
MIN_S = 5.0           # minimum clip length kept after trimming
TOL_NS = 5_000_000    # 5 ms tolerance for accelerometer/gyroscope alignment
MIN_TOTAL_S = 90.0    # rubric: >= 1 min 30 s per activity

# Found mislabeled in the audit: the signal is walking-like, not standing.
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


def trim_and_cap(m):
    """Trim TRIM_S from each end, then centre-crop to CAP_S. Returns (clip, reasons)."""
    t = (m["time"] - m["time"].iloc[0]).to_numpy() / 1e9
    seg = m[(t >= TRIM_S) & (t <= t[-1] - TRIM_S)]
    if len(seg) < 2:
        return None, "nothing left after trimming"
    ts = (seg["time"] - seg["time"].iloc[0]).to_numpy() / 1e9
    capped = False
    if ts[-1] > CAP_S:
        centre = ts[-1] / 2
        seg = seg[(ts >= centre - CAP_S / 2) & (ts <= centre + CAP_S / 2)]
        capped = True
    out = seg.reset_index(drop=True)
    out.insert(0, "timestamp", (out["time"] - out["time"].iloc[0]) / 1e9)
    out = out.drop(columns="time")
    return out, "capped to central 10 s" if capped else ""


def mag(df, cols):
    return np.sqrt((df[cols] ** 2).sum(axis=1)).to_numpy()


def main():
    OUT.mkdir(exist_ok=True)
    (OUT / "train").mkdir(exist_ok=True)
    (OUT / "test").mkdir(exist_ok=True)
    sys.stdout = Tee(OUT / "build_log.txt")

    # ------------------------------------------------------------------ 1. audit
    zips = []
    for act in ACTIVITIES:
        for z in sorted((RAW / act).glob("*.zip")):
            zips.append((act, z))
    print(f"== PHASE 1: raw audit ({len(zips)} zips in {rel(RAW)}/, read-only) ==")
    records = []
    col_orders = set()
    for act, z in zips:
        acc, gyr, meta = read_zip(z)
        col_orders.add(tuple(c for c in acc.columns if c in "xyz"))
        sa, sg = rate_stats(acc), rate_stats(gyr)
        rec_time = re.search(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})$", z.stem).group(1)
        records.append(dict(
            act=act, zip=z, stem=z.stem, rec_time=rec_time,
            hand_named=not z.stem[0].isdigit(),
            acc=acc, gyr=gyr, meta=meta, sa=sa, sg=sg,
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
    print()

    # ------------------------------------------------------------------ 2. clean
    print("== PHASE 2: clean (align -> trim 1.5 s each end -> cap 10 s -> keep >= 5 s) ==")
    kept, excluded = [], []
    for r in records:
        stem = r["stem"]
        merged, info = merge_streams(r["acc"], r["gyr"])
        clip, cap_note = trim_and_cap(merged)
        notes = []
        if info["dropped_acc"] or info["gyro_unused"] or info["gyro_reused"]:
            notes.append(f"aligned by merge_asof: acc rows without gyro match {info['dropped_acc']}, "
                         f"gyro rows unused {info['gyro_unused']}, gyro rows reused {info['gyro_reused']}")
        if cap_note:
            notes.append(cap_note)
        if stem in MISLABELLED:
            excluded.append(dict(original_zip=rel(r["zip"]), activity=r["act"], reason=MISLABELLED[stem],
                                 _clip=clip))
            continue
        dur = float(clip["timestamp"].iloc[-1]) if clip is not None else 0.0
        if clip is None or dur < MIN_S:
            excluded.append(dict(original_zip=rel(r["zip"]), activity=r["act"],
                                 reason=f"too short after trimming {TRIM_S} s from each end "
                                        f"({dur:.2f} s < {MIN_S:.0f} s; raw {r['sa']['dur']:.2f} s)",
                                 _clip=clip))
            continue
        kept.append(dict(rec=r, clip=clip, notes=notes, dur=dur))

    # ------------------------------------------------------------------ 3. split + rename
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
        # edge check: residual transient = first/last second far more active than the clip median
        a = mag(c, ["ax", "ay", "az"])
        sec = np.array([a[i * 100:(i + 1) * 100].std() for i in range(len(a) // 100)])
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
        if k["edge_ratio"] >= 3:
            flags.append(f"QC: residual edge transient (edge/median 1-s std = {k['edge_ratio']:.1f})")
        k["flags"] = flags
        k["notes"] = ([f"hand-named session recording (held out as unseen test file)"] if k["split"] == "test" else []) \
            + k["notes"] + flags

    # standing_02 evidence for the exclusion log
    for e in excluded:
        if "mislabeled" in e["reason"] and e["_clip"] is not None:
            c = e["_clip"]
            ea, eg = float(mag(c, ["ax", "ay", "az"]).std()), float(mag(c, ["gx", "gy", "gz"]).std())
            e["reason"] += (f" (trimmed accel-magnitude std {ea:.3f}, gyro std {eg:.3f}; class medians of kept files: "
                            + ", ".join(f"{a} {np.exp(med[a][0]):.3f}" for a in ACTIVITIES) + ")")

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
            measured_hz=round(k["hz"], 3), notes="; ".join(k["notes"]),
        ))
    man = pd.DataFrame(manifest).sort_values(["activity", "new_name"])
    man.to_csv(OUT / "manifest.csv", index=False)
    pd.DataFrame([{k: v for k, v in e.items() if not k.startswith("_")} for e in excluded]).to_csv(
        OUT / "excluded.csv", index=False)

    # ------------------------------------------------------------------ report
    print(f"kept {len(kept)} clips, excluded {len(excluded)} (see Data_clean/excluded.csv):")
    for e in excluded:
        print(f"  - {e['original_zip']}: {e['reason']}")
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
    print(f"QC-flagged clips ({len(flagged)}):")
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
    raw_tot = {a: sum(r["sa"]["dur"] for r in records if r["act"] == a) for a in ACTIVITIES}
    print("raw recorded seconds per activity (before cleaning):", {a: round(v, 1) for a, v in raw_tot.items()})
    for a in ACTIVITIES:
        assert sum(k["split"] == "test" and k["rec"]["act"] == a for k in kept) >= 2, f"<2 test files for {a}"
    assert not any(k["rec"]["stem"] in MISLABELLED for k in kept)
    if shortfalls:
        print("\nSHORTFALL vs the 90 s per-activity target (after trimming 1.5 s per end, 5 s minimum clip):")
        for a, s in shortfalls.items():
            print(f"  {a:9s} missing {s:5.1f} s -> record {int(np.ceil(s / 7.0))} more 10 s clip(s) "
                  f"(each yields 7 s after trimming) or {int(np.ceil(s / 10.0))} 13 s clip(s) (each yields 10 s)")
        # informational only: what a shorter trim would have kept
        for trim in (1.0, 0.5):
            tot = {}
            for a in ACTIVITIES:
                tot[a] = sum(min(r["sa"]["dur"] - 2 * trim, CAP_S) for r in records
                             if r["act"] == a and r["stem"] not in MISLABELLED and r["sa"]["dur"] - 2 * trim >= MIN_S)
            print(f"  (info only, not applied) with {trim} s trim the clean totals would be:",
                  {a: round(v, 1) for a, v in tot.items()})
    print("\nwritten:", rel(OUT / "manifest.csv"), rel(OUT / "excluded.csv"), rel(OUT / "build_log.txt"))
    sys.stdout = sys.stdout.o


if __name__ == "__main__":
    main()
