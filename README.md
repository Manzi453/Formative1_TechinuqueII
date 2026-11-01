# Formative 2: Modeling Human Activity States Using Hidden Markov Models

**Author:** Ivan Manzi Ya Musana | **Type:** individual assignment (one student, one iPhone 13, one device id)

Four activity states (**standing, walking, jumping, still**) are recognized from smartphone accelerometer and gyroscope
data with a Hidden Markov Model written from scratch in numpy (log-space forward-backward, Baum-Welch with a
log-likelihood convergence check, Viterbi).

> **Status / honesty note.** Every number in the notebook and report is produced by the code in this repo. Items that
> could not be verified are marked **UNVERIFIED**. After cleaning, every activity has **less than 90 s** of usable data
> (see "Dataset"). The four video links below are **placeholders** until the videos are uploaded.

## Repository structure

```
Data/                         ORIGINAL recordings (never modified): Data/<activity>/*.zip (13 per activity, 6 CSVs each) and Data/videos/*.mp4
Data_clean/                   cleaned dataset used by the notebook
  train/  test/               <activity>_<NN>.csv, columns: timestamp, ax, ay, az, gx, gy, gz
  manifest.csv                new_name, original_zip, activity, split, duration_s, n_samples, measured_hz, notes
  excluded.csv                every excluded recording and the reason
  build_log.txt               audit of the raw data + per-file QC table + totals
scripts/make_clean_dataset.py raw zips -> Data_clean/ (reads Data/ only)
HMM_Activity_Recognition.ipynb notebook with saved outputs (setup, plots, sampling rate, windowing, features, HMM, evaluation, discussion)
figures/                      PNG figures produced by the notebook (fig01 ... fig08)
results/                      CSV/JSON results written by the notebook (metrics.json, test tables, transition matrix, ...)
report/                       Formative2_HMM_Report.md / .pdf (5 pages) and build_report.py
videos_trimmed/               optional 5.5 s COPIES of the videos (trimming does not fix framing, see VIDEO_TODO.md)
VIDEO_TODO.md                 what must be re-recorded / uploaded by hand
requirements.txt              pinned versions that were used
```

## How to run

1. Python 3.14.7 was used; any recent Python 3 with the pinned packages should work: `pip install -r requirements.txt`.
2. *(optional, Data_clean/ is already included)* rebuild the cleaned data from the raw zips: `python scripts/make_clean_dataset.py`.
3. Run the notebook from the repository root: open `HMM_Activity_Recognition.ipynb` and *Restart & Run All*, or
   `python -m nbconvert --to notebook --execute --inplace HMM_Activity_Recognition.ipynb`. All paths are relative; random seed 42; the
   results are deterministic (two full re-runs gave a byte-identical `results/metrics.json`).
4. *(optional)* rebuild the PDF report: `python report/build_report.py` (needs Chrome/Chromium; it also checks that the numbers
   quoted in the report match `results/`).

## Dataset

* **Device / app:** iPhone 13 (one device id in all 52 recordings). The export format matches Sensor Logger (appVersion 1.68, iOS),
  but the app name is not stored in the files: **UNVERIFIED**, please confirm.
* **Sampling rate:** nominal 100 Hz (`sampleRateMs = 10`); **measured** 100.118 Hz on average (100.105-100.124 Hz, std 0.005 Hz) with
  9.98-9.99 ms between samples, so no resampling was required. Accelerometer and gyroscope are aligned by nearest timestamp (5 ms).
* **Signals used:** calibrated `Accelerometer.csv` + `Gyroscope.csv` (the uncalibrated accelerometer is on a different scale). Units are
  not stated in the files and are inferred as m/s&sup2; and rad/s: **UNVERIFIED**.
* **Raw:** 52 recordings, 13 per activity, 6.7-17.0 s each.
* **Cleaning:** 1.5 s trimmed from both ends, capped at 10 s (central crop), clips under 5 s dropped, renamed `<activity>_<NN>.csv`.
  Nine recordings are excluded (`Data_clean/excluded.csv`): `standing_02-2026-10-28_17-23-27` (mislabeled, walking-like signal) and eight clips shorter than 5 s after trimming.
* **Cleaned totals (train + test):** standing 64.0 s, walking 86.7 s, jumping 73.7 s, still 58.5 s, so **all four are below the 90 s target**
  (raw recordings: 115-130 s per activity). Reaching 90 s would need about 4 standing, 1 walking, 3 jumping and 5 still extra 10 s clips.
* **43 clips: 33 train, 10 test.**

### Train / test split

The **10 test clips** are the hand-named recordings from two separate sessions; they are never used for normalization statistics, model
fitting, Baum-Welch or hyper-parameter choice (the notebook asserts that no test name or source zip appears in train).

| Test clip | Activity | Source zip (in `Data/<activity>/`) | Seconds | Samples |
|---|---|---|---|---|
| jumping_10.csv | jumping | jumping_02-2026-10-28_17-22-14.zip | 5.10 | 512 |
| jumping_11.csv | jumping | jumping_03-2026-10-28_17-22-24.zip | 5.53 | 555 |
| standing_01.csv | standing | standing_-2026-10-28_16-39-57.zip | 7.32 | 734 |
| standing_10.csv | standing | standing_01-2026-10-28_17-23-40.zip | 8.24 | 826 |
| still_01.csv | still | still-2026-10-28_16-40-36.zip | 9.99 | 1001 |
| still_08.csv | still | still01-2026-10-28_17-23-58.zip | 5.76 | 578 |
| still_09.csv | still | still02-2026-10-28_17-24-09.zip | 5.56 | 558 |
| walking_01.csv | walking | walking_01-2026-10-28_16-39-06.zip | 9.99 | 1001 |
| walking_12.csv | walking | walking-2026-10-28_17-29-40.zip | 5.08 | 510 |
| walking_13.csv | walking | walking-2026-10-28_17-29-53.zip | 8.06 | 808 |

The **33 training clips** are every other file in `Data_clean/train/`: standing_02-09 (8), walking_02-11 (10), jumping_01-09 (9), still_02-07 (6).
`still_01` and `still_08` are flagged by the quality check (standing-like std profile, residual start-up transient) and were kept.

## Main results (unseen test clips, 58 windows)

Overall accuracy **91.4 % (53 / 58)**, 95 % Wilson interval 81.4-96.3 % (optimistic, windows overlap). Per activity (sensitivity / specificity):
standing 1.00 / 0.93, walking 0.90 / 1.00, jumping 1.00 / 0.96, still 0.82 / 1.00. Honest caveats: Baum-Welch converged in 3 iterations but
changed the supervised estimate very little (test predictions identical with and without it); the variance floor is binding for 80 % of the
state variances; the test set is small and single-person; all three still errors are in the two clips flagged by the QC check. Details in
`HMM_Activity_Recognition.ipynb` (section 9) and `report/Formative2_HMM_Report.pdf`.

## Video evidence (placeholders: replace with public links)

* Standing: **[VIDEO LINK: standing]**
* Walking: **[VIDEO LINK: walking]**
* Jumping: **[VIDEO LINK: jumping]**
* Still: **[VIDEO LINK: still]**

The local files are in `Data/videos/`; see `VIDEO_TODO.md` for the problems found and the re-recording / upload checklist.
