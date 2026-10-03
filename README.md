# CSE-307 Term Paper, Track 2: Learned Disk Scheduler Selector

Mhamuda Shafiq Shamme | ID 202414093 | Section B | Batch CSE-24

A small random-forest classifier looks at a window of 32 pending disk
requests and predicts which of FCFS, SCAN, C-SCAN or SSTF will give the
smallest total head movement. It also reports a confidence score (the forest's
top class probability). I test it on a timeline whose workload changes twice
(sequential -> random -> bursty) and check whether the confidence is calibrated.

Repository: https://github.com/Sham-me09/cse307-disk-scheduler-selector

## Files
- `schedulers.py`  FCFS, SCAN, C-SCAN, SSTF (returns total seek distance in tracks)
- `workloads.py`   sequential / random / bursty window generators + 5 features
- `make_dataset.py`  step 1: builds the labelled training table `results/dataset.csv`
- `train_model.py`   step 2: trains and compares the models, saves `results/model.joblib`
- `run_experiments.py`  step 3: shifted-timeline test, calibration, figures
- `results/`       dataset, model, `results.json`, `model_comparison.json`, figures (PDF + PNG)
- `report/`        LaTeX source of the paper

## Setup and run (Windows, Kali/Linux, macOS: same commands)
```
python -m venv venv
# Windows:  venv\Scripts\activate          Kali/Linux:  source venv/bin/activate
pip install -r requirements.txt
python make_dataset.py
python train_model.py
python run_experiments.py      # about a minute
```
Seeds are fixed, so the numbers in the report can be reproduced exactly.
(On Kali use `python3` if `python` is not found.)

## Setup in short
- 200 cylinders, head starts every window at track 100, 32 requests per window,
  all requests of a window treated as pending at once. Seek time is taken as
  proportional to distance.
- SCAN and C-SCAN go all the way to the disk edge (textbook version). The
  C-SCAN return jump is counted. I checked the four functions on the
  Silberschatz textbook queue (98,183,37,122,14,124,65,67, head 53):
  FCFS 640, SSTF 236, C-SCAN 382. SCAN gives 331 because mine sweeps upward first
  (the textbook sweeps downward first and gets 236).
- Features: variance of track positions, mean jump between consecutive requests,
  fraction of ascending steps, burstiness (coefficient of variation of the
  arrival gaps), distance of the request centre from the head.
- Label of a window = the scheduler with the smallest seek total on it.
- Training: 400 windows per workload type. Test: a 60-window timeline
  (20 sequential, 20 random, 20 bursty), repeated for 30 independent seeds.
- Second experiment: the same, but the classifier never sees bursty windows
  in training.

## AI assistance disclosure
I used an AI assistant (Claude) to help write the code skeleton and the LaTeX
template. The experimental design choices, the interpretation of results and
the written analysis are mine, and I can explain every part of the code. I reviewed and modified the code as needed, and I am responsible for the experimental design, results, interpretation, and written analysis presented in this report.

