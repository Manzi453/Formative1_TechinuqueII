# Formative 2: Modeling Human Activity States Using Hidden Markov Models

**CSV data:** [`Data_clean/`](Data_clean/) (`train/` and `test/`, columns `timestamp, ax, ay, az, gx, gy, gz`; see [`Data_clean/README.md`](Data_clean/README.md)).
**Submitted report:** [`report/Formative2_HMM_Report.pdf`](report/Formative2_HMM_Report.pdf). **Notebook:** [`HMM_Activity_Recognition.ipynb`](HMM_Activity_Recognition.ipynb).
**Videos (one per activity):** [Standing](https://github.com/Manzi453/Formative1_TechinuqueII/blob/main/Data/videos/standing.mp4) · [Walking](https://github.com/Manzi453/Formative1_TechinuqueII/blob/main/Data/videos/walking.mp4) · [Jumping](https://github.com/Manzi453/Formative1_TechinuqueII/blob/main/Data/videos/jumping.mp4) · [Still](https://github.com/Manzi453/Formative1_TechinuqueII/blob/main/Data/videos/still.mp4)

**Author:** Ivan Manzi Ya Musana | **Type:** individual assignment (one student, one iPhone 13, one device id)

Four activity states (**standing, walking, jumping, still**) are recognized from smartphone accelerometer and gyroscope
data with a Hidden Markov Model written from scratch in numpy (log-space forward-backward, Baum-Welch with a
log-likelihood convergence check, Viterbi). Items that could not be verified are marked **UNVERIFIED**.

## Repository structure

```
README.md
requirements.txt
Data/                          original recordings, never modified: Data/<activity>/*.zip (13 per activity, 6 CSVs each) and Data/videos/*.mp4
Data_clean/                    cleaned CSV dataset used by the notebook
  train/  test/                <activity>_<NN>.csv, columns: timestamp, ax, ay, az, gx, gy, gz
  manifest.csv                 one row per clip: source recording, split, duration, samples, measured Hz, QC notes
  excluded.csv                 excluded recordings and removed edge segments, with the reason
  trim_sweep.csv               usable seconds per activity for each candidate trim length
  build_log.txt                raw-data checks, per-file QC table and totals
  README.md                    columns, units, sampling rate, naming scheme
scripts/
  make_clean_dataset.py        raw zips -> Data_clean/ (reads Data/ only)
  build_report.py              report markdown -> PDF, with a check of every quoted number
figures/                       PNG figures written by the notebook (fig01 ... fig08)
results/                       CSV/JSON results written by the notebook (metrics.json, test tables, transition matrix, ...)
report/                        Formative2_HMM_Report.md and .pdf
HMM_Activity_Recognition.ipynb notebook with saved outputs
```

## How to run (in this order)

1. Python 3.14.7 was used; any recent Python 3 with the pinned packages should work: `pip install -r requirements.txt`.
2. *(optional, `Data_clean/` is already included)* rebuild the cleaned data from the raw zips: `python scripts/make_clean_dataset.py`.
3. Run the notebook from the repository root: open `HMM_Activity_Recognition.ipynb` and *Restart & Run All*, or
   `python -m nbconvert --to notebook --execute --inplace HMM_Activity_Recognition.ipynb` (about 15 s). All paths are relative and the random seed is 42;
   two consecutive runs gave byte-identical files in `results/`. It writes `figures/` and `results/`.
4. *(optional)* rebuild the PDF report: `python scripts/build_report.py` (needs Chrome or Chromium; set `CHROME` to its path). It checks that the numbers quoted in the
   report match `results/` and `Data_clean/`, lists placeholders that are still in the text, and requires 4 to 5 pages. `--final` refuses to build while any placeholder remains.

## Dataset

* **Device / app:** iPhone 13 (one device id in all 52 recordings, iOS 18.2.1). The export format matches Sensor Logger (appVersion 1.68),
  but the app name is not stored in the files: **UNVERIFIED**.
* **Sampling rate:** nominal 100 Hz (`sampleRateMs = 10`); **measured** 100.118 Hz on average over the cleaned clips (100.105-100.124 Hz, std 0.005 Hz) with
  9.98-9.99 ms between samples, so no resampling was required. Accelerometer and gyroscope are aligned by nearest timestamp (5 ms).
* **Signals used:** calibrated `Accelerometer.csv` and `Gyroscope.csv` (the uncalibrated accelerometer is on a different scale). Units are not stated in the
  files and are inferred as m/s&sup2; and rad/s: **UNVERIFIED**.
* **Raw:** 52 recordings, 13 per activity, 6.7-17.0 s each (118.3 s standing, 130.1 s walking, 121.7 s jumping, 115.2 s still).
* **Cleaning (the same rules for every recording):** trim both ends by the largest of 1.0, 0.75 and 0.5 s for which every activity reaches 90 s (none does, so
  0.5 s is used); drop a 1 s edge segment (at most 3 per end) while its accelerometer-magnitude std exceeds 2x the clip median; keep 5-10 s clips (central crop above 10 s);
  rename to `<activity>_<NN>.csv`. Two recordings are excluded (`Data_clean/excluded.csv`): `standing_02-2026-10-28_17-23-27` (mislabeled, walking-like signal)
  and `still01-2026-10-28_17-23-58` (4.76 s after the edge rule). Details and the trim sweep are in `Data_clean/build_log.txt` and `Data_clean/trim_sweep.csv`.
* **Cleaned totals (train + test), 50 clips (40 train, 10 test):**

| Activity | Clips (train / test) | Seconds | Windows (train / test) |
|---|---|---|---|
| standing | 10 / 2 | 92.3 | 58 / 17 |
| walking | 10 / 3 | 108.6 | 68 / 24 |
| jumping | 10 / 3 | 107.8 | 73 / 17 |
| still | 10 / 2 | **88.2** | 58 / 13 |

**Still is 1.8 s below the 90 s target.** One more still recording of at least 11 s (10 s after trimming) closes the gap; see "Adding recordings" in `Data_clean/README.md`.

### Train / test split

The **10 test clips** are the recordings I named by hand in separate sessions; every other recording is train. The split never looks at the signal. Test clips are never used for
normalization statistics, model fitting, Baum-Welch or any model choice (variance floor, pi), and the notebook asserts that no test name or source recording appears in train.

| Test clip | Activity | Source zip (in `Data/<activity>/`) | Seconds | Samples |
|---|---|---|---|---|
| jumping_01.csv | jumping | jumping_-2026-10-28_16-41-22.zip | 6.17 | 619 |
| jumping_12.csv | jumping | jumping_02-2026-10-28_17-22-14.zip | 7.10 | 712 |
| jumping_13.csv | jumping | jumping_03-2026-10-28_17-22-24.zip | 7.53 | 755 |
| standing_01.csv | standing | standing_-2026-10-28_16-39-57.zip | 9.32 | 934 |
| standing_12.csv | standing | standing_01-2026-10-28_17-23-40.zip | 10.00 | 1002 |
| still_01.csv | still | still-2026-10-28_16-40-36.zip | 8.57 | 859 |
| still_12.csv | still | still02-2026-10-28_17-24-09.zip | 7.56 | 758 |
| walking_01.csv | walking | walking_01-2026-10-28_16-39-06.zip | 9.99 | 1001 |
| walking_12.csv | walking | walking-2026-10-28_17-29-40.zip | 7.08 | 710 |
| walking_13.csv | walking | walking-2026-10-28_17-29-53.zip | 10.00 | 1002 |

The 40 training clips are every other file in `Data_clean/train/` (10 per activity: `standing_02`-`standing_11`, `walking_02`-`walking_11`, `jumping_02`-`jumping_11`, `still_02`-`still_11`).
`still_05` and `still_09` have a standard-deviation profile closer to standing; the rules above are the only filters, so they were kept.

## Main results (unseen test clips, 71 windows)

Overall accuracy **91.5 % (65 / 71)**, 95 % Wilson interval 82.8-96.1 % (optimistic, windows overlap). Per activity (sensitivity / specificity):
standing 1.00 / 0.93, walking 0.92 / 1.00, jumping 1.00 / 0.96, still 0.69 / 1.00. Training windows score 98.8 % (the model is initialized from them) and
leave-one-clip-out on the training clips 98.4 %. Honest caveats: Baum-Welch converged in 4 iterations but left the test predictions unchanged; 79.5 % of the state
variances sit at the variance floor; the floor chosen on training data (0.5) gave the lowest test accuracy of the five candidates (91.5 % against up to 100 %), and I kept it
because choosing a floor from test results would be tuning on the test set; 4 of the 6 test errors are in the first two windows of a clip; the test set is small and single-person.
Details in `HMM_Activity_Recognition.ipynb` (sections 7-9) and `report/Formative2_HMM_Report.pdf`.

## Video evidence

One video per activity, in `Data/videos/` (public repository, no login needed):

* Standing: <https://github.com/Manzi453/Formative1_TechinuqueII/blob/main/Data/videos/standing.mp4>
* Walking: <https://github.com/Manzi453/Formative1_TechinuqueII/blob/main/Data/videos/walking.mp4>
* Jumping: <https://github.com/Manzi453/Formative1_TechinuqueII/blob/main/Data/videos/jumping.mp4>
* Still: <https://github.com/Manzi453/Formative1_TechinuqueII/blob/main/Data/videos/still.mp4>
