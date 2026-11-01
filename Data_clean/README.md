# Data_clean: the cleaned CSV dataset

One CSV per clip, built from the raw Sensor Logger zips in `Data/` by `scripts/make_clean_dataset.py` (`Data/` is only read, never modified).

## Files

| Path | Content |
|---|---|
| `train/<activity>_<NN>.csv`, `test/<activity>_<NN>.csv` | the clips (50 in total: 40 train, 10 test) |
| `manifest.csv` | one row per clip: new name, source zip, activity, split, duration, samples, measured Hz, trim, removed edge seconds, QC notes |
| `excluded.csv` | every excluded recording and every removed edge segment, with the reason (`kind` = `recording` or `segment`) |
| `trim_sweep.csv` | usable seconds per activity for each candidate trim length and which one was selected |
| `build_log.txt` | raw-data checks, trim sweep, per-file QC table, totals and any shortfall |

## Columns

| Column | Meaning | Unit |
|---|---|---|
| `timestamp` | time since the first sample of the clip | seconds |
| `ax`, `ay`, `az` | calibrated accelerometer (`Accelerometer.csv`), matched to the raw x, y, z columns by name | m/s&sup2; (inferred, **UNVERIFIED**) |
| `gx`, `gy`, `gz` | calibrated gyroscope (`Gyroscope.csv`), aligned to the accelerometer rows by nearest timestamp (5 ms) | rad/s (inferred, **UNVERIFIED**) |

Units are not stated in the Sensor Logger files; they are inferred from the signal scale. The raw accelerometer export is ordered z, y, x, which is why columns are matched by name.

## Sampling rate

Nominal 100 Hz (`sampleRateMs = 10` in `Metadata.csv`). Measured per clip from the timestamps: 100.105-100.124 Hz (mean 100.118, std 0.005), 9.98-9.99 ms between samples. The rate is uniform, so the clips are not resampled.

## Naming scheme and split

* File name: `<activity>_<NN>.csv`, with activity in `standing, walking, jumping, still`; numbers run in recording-time order within an activity across train and test.
* **Split rule:** a recording whose zip name was typed by hand (it does not start with a digit, e.g. `still-2026-10-28_16-40-36.zip`) is a **test** clip; recordings with the app's default timestamp name are **train**.
* Clip length is 5-10 s. Cleaning rules (applied to every recording): trim both ends by the largest of 1.0, 0.75 and 0.5 s for which every activity reaches 90 s (0.5 s here, because no candidate reaches 90 s for all four); drop up to three 1 s edge segments per end while their accelerometer-magnitude std exceeds twice the clip median; central crop above 10 s.

## Adding recordings

Still is 1.8 s short of 90 s. Record at least one more still clip of 11 s or longer (10 s after the 0.5 s trims) with the app's default name (it becomes a train clip), copy the zip to `Data/still/`, then run `python scripts/make_clean_dataset.py` and the notebook. The trim sweep is re-evaluated automatically and may pick a larger trim once every activity has margin. The notebook, `results/` and the report numbers all change, so re-run the notebook and `python scripts/build_report.py` afterwards and update the report text that quotes numbers.
