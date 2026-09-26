# Reproducing the Milestone 3 Experiments

Disclaimer: We used ChatGPT to help us create the md files - as a writing assistant, to help us turn our main ideas into clearer and more formal English.

This document explains how to reproduce the Milestone 3 experiments from the submitted repository. The package includes the prepared model inputs, trained checkpoints, saved configurations, result tables, and the local Qwen model assets required for replay. The complete raw transaction corpus is not duplicated in the repository; model replay does not require it.

## Environment

The measured experiments were executed on Windows using Python 3.12.14, PyTorch 2.8.0 (CPU), and scikit-learn 1.7.1.

From `02_code`, create an isolated Python environment and install the required dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pytest -q
```

The fully resolved dependency versions are stored in `09_reproducibility/environment_freeze.txt`. On another operating system, the matching CPU PyTorch build should be installed. The supplied `llama.cpp` runtime is the Windows x64 build. Small numerical differences in neural-model scores may occur across CPUs or numerical libraries even when fixed seeds are used.

## Replay the frozen model checkpoints

To regenerate Test and External Test predictions from the frozen checkpoints:

```powershell
.\.venv\Scripts\python.exe reproduce.py --output ..\..\milestone3_replay
```

The command creates a new sibling directory and does not overwrite the submitted results. The Validation-selected thresholds stored with the frozen runs remain unchanged during replay.

The regenerated results can be compared with the submitted tables and files under `06_results`.

## Replay with the local LLM arbitrator

The local arbitration stage uses Qwen2.5-1.5B-Instruct through `llama.cpp`.

Open a second terminal in `02_code` and start the local server:

```powershell
.\start_local_llm.ps1
```

Then run:

```powershell
.\.venv\Scripts\python.exe reproduce.py --output ..\..\milestone3_replay_with_llm --with-llm
```

The server listens only on `127.0.0.1:8765`; no cloud API is required.

Each LLM request contains selected numeric window features, Train reference medians, and the four detector decisions. Source name, tool name, recording identifier, and the true class label are not included in the prompt.

The arbitration policy permits at most 32 LLM requests per split and at most 2 per recording. Candidate ordering is determined by a deterministic hash of `sample_id`, independent of the true label or observed error. If a candidate is not selected, a request fails, a timeout occurs, or the response is invalid, the cascade decision is preserved.

The recorded experiment generated 2 LLM requests on Validation, 6 on Test, and 13 on External Test. These are the complete requests selected by the frozen policy.

## Retrain the declared experiment set

To repeat the declared model-training experiments:

```powershell
.\.venv\Scripts\python.exe reproduce.py --output ..\..\milestone3_retrained --retrain --with-llm
```

The LLM server must already be running when `--with-llm` is used.

The retraining workflow executes:

- 16 one-factor sensitivity configurations across the four model families;
- 2 evidence-driven Isolation Forest refinements;
- 4 additional neural seed runs.

Model and threshold selection uses Validation only. Seed 42 is the locked held-out evaluation seed for the neural models. The declared configurations are stored in `03_config`, and the completed run outputs and learning curves are stored in `06_results/runs`.

## Rebuild prepared inputs from canonical DNS transactions

The preparation pipeline starts from parsed canonical DNS transaction CSV files:

```powershell
.\.venv\Scripts\python.exe -m dnsids.prepare --raw-dir 'PATH_TO_CANONICAL_CSVS' --output ..\..\rebuilt_inputs
```

The expected files are:

- `train_transactions.csv`
- `validation_transactions.csv`
- `test_transactions.csv`
- `external_test_transactions.csv`

An optional `--column-map adapter.json` can map alternative column names to the canonical schema.

The preparation stage checks labels and recording boundaries, verifies that recording groups do not cross splits, fits learned preprocessing on Train only, and then applies the frozen transforms to all splits. It produces the aligned tabular, sequence, and graph representations used by the four model families.

The primary tabular representation contains the accepted 42-feature core. The selected refined Isolation Forest uses a smaller 29-feature subset after the Validation-driven removal of volume/payload and timing features.

The pipeline starts from parsed DNS transaction tables rather than arbitrary PCAP files.

## Reading prediction outputs

Every prediction row contains `sample_id` for alignment across model outputs.

`label=1` represents malicious traffic.

For the model scores:

- Random Forest produces a supervised malicious-class score.
- Isolation Forest uses the negative of `score_samples`, so a higher value indicates a more anomalous sample.
- BiLSTM and GraphSAGE produce sigmoid scores; these scores are not treated as calibrated probabilities.
- The cascade produces a binary operational decision rather than a continuous ranking score.

The External Test split contains 7,547 malicious windows and no benign samples. Therefore, malicious recall and miss counts are reported for that split, while binary F1, FPR, specificity, precision, and ROC-AUC are left undefined rather than reported as zero.

Recording-level evaluation uses the 95th percentile of window scores within each recording, with separate thresholds selected on Validation.

## Verification and recovery

Completed training runs are marked by `COMPLETE.json`. A fresh inference replay should always use a new output directory rather than overwrite the submitted evidence.

The local LLM responses are stored in JSONL logs so an interrupted arbitration run can resume while preserving completed outcomes.

`verify_package.py` checks the saved experiment package, including split disjointness, representation alignment, model-selection locks, completed training runs, and cascade population accounting.

## Recreate the figures

To recreate the submitted figures from the saved result tables:

```powershell
.\.venv\Scripts\python.exe make_figures.py --output ..\..\milestone3_figures
```

The plotting script writes PNG and vector PDF versions to a new output directory and does not modify the submitted results.
