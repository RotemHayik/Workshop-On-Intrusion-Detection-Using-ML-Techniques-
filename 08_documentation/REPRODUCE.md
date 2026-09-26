# Reproduce the Milestone 3 experiments

Start with the report in `01_report` and the measured tables in `06_results`. All eligible prepared samples, trained checkpoints and the local Qwen weights are included. A replay requires no Drive access. The 2.9 million canonical raw transactions remain in the accepted Part 2 source location; representative raw windows are included here.

## Environment

The measured run used Windows, Python 3.12.14, PyTorch 2.8.0 CPU and scikit-learn 1.7.1. From `02_code`, create an isolated Python 3.12 environment and install dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pytest -q
```

The full resolved dependency versions are in `09_reproducibility/environment_freeze.txt`. On another OS, install the matching CPU PyTorch wheel; the supplied llama.cpp executable is Windows x64. Exact neural scores can vary with numerical libraries or CPU instructions despite deterministic seeds.

## Replay frozen checkpoints

```powershell
.\.venv\Scripts\python.exe reproduce.py --output ..\..\milestone3_replay
```

This creates a new sibling folder and regenerates Test and External Test predictions from the frozen checkpoints. It refuses an existing destination and does not overwrite this submission. Compare `06_results/final/*/metrics.csv` in the new folder with this package. Every checkpoint's Validation threshold remains fixed.

## Run the local LLM as well

In a separate terminal opened at `02_code`, start the supplied runtime:

```powershell
.\start_local_llm.ps1
```

Then run:

```powershell
.\.venv\Scripts\python.exe reproduce.py --output ..\..\milestone3_replay_with_llm --with-llm
```

The server listens only on 127.0.0.1:8765. No cloud API key is used. Stop the server with Ctrl+C after the replay. Requests contain numeric features, Train reference medians and detector decisions; source, tool, recording identifiers and true labels are excluded. The sample identifier is stored only in the local audit log.

The policy allocates at most 32 requests per split and 2 per recording. Deterministic hashes select candidates without labels. All unselected candidates retain their cascade decisions. Invalid JSON, unsupported labels, HTTP failure or timeout also retains that decision and is counted. The measured run used 2 Validation, 6 Test and 13 External Test requests. These are the complete selected requests under the declared policy, not a hand-picked subset of successful responses.

## Retrain and repeat all experiments

```powershell
.\.venv\Scripts\python.exe reproduce.py --output ..\..\milestone3_retrained --retrain --with-llm
```

The LLM server must already be running for `--with-llm`. This executes 16 sensitivity configurations, 2 evidence-driven IF refinements and 4 neural seed repeats. Selection uses Validation only. Seed 42 is the locked Test seed regardless of the other seeds' scores. The copied settings are in `03_config`; actual per-run settings and learning curves are in `06_results/runs`.

## Rebuild prepared inputs from canonical raw transactions

```powershell
.\.venv\Scripts\python.exe -m dnsids.prepare --raw-dir 'PATH_TO_ROTEM_CANONICAL_CSVS' --output ..\..\rebuilt_inputs
```

Supply `train_transactions.csv`, `validation_transactions.csv`, `test_transactions.csv` and `external_test_transactions.csv`. A `--column-map adapter.json` option maps vendor field names to canonical fields. Ingestion checks labels, recording boundaries and required columns. The builder checks that recording IDs do not cross splits, fits scaling only on Train, then applies the fixed transforms and the accepted 42-feature roster to all splits. It writes to a new folder. To use a rebuilt dataset with the model modules, place its `prepared` folder and scaler/schema files under a new experiment root's `04_data` folder.

This entry point starts from parsed DNS transaction CSVs, the accepted boundary of Part 2. It does not claim to parse arbitrary PCAP formats. The inherited Part 2 transformations are retained in `02_code/part2_reference`; do not run their old top-level entry points, which preserve original project paths for provenance. `dnsids.prepare` overrides those paths and exposes the portable interface.

The regression tests reconstruct an actual raw Validation window and compare its sequence channels, mask, graph nodes, topology and edge attributes with the frozen arrays. All 52 tabular features were additionally reconstructed for 20 representative error windows. See `04_data/raw_examples/*/reconstruction_checks.csv`.

## Reading the results correctly

Every prediction row carries `sample_id` for alignment. `label=1` means malicious. IF scores are negative `score_samples` values, so higher means more anomalous. LSTM and GraphSAGE output sigmoid scores; these are not calibrated probabilities. The cascade emits a binary operational score; its AUROC is therefore based on that binary score, not comparable to the ranking quality of a continuous detector score.

The external split has 7,547 malicious windows and no benign samples. Its recall and miss counts are reported; binary accuracy, precision, F1, FPR and ROC-AUC are intentionally blank. A blank metric is not zero. Recording-level scores use the 95th percentile with separately frozen Validation thresholds. Cluster-bootstrap intervals use whole recordings, and exact binomial recording intervals supplement degenerate zero-error bootstrap intervals.

## Recovery and audit

Training resumes by skipping a run only when its `COMPLETE.json` exists. Cached final predictions are reused within a run; use a new reproduction folder for fresh inference. Local LLM JSONL logs allow interrupted requests to resume without losing successful or failed outcomes. Remove neither locks nor evidence files in the delivered folder. `verify_package.py` verifies the delivered inventory and cross-representation contracts without training.


## Recreate the figures

From `02_code`, run the following to recreate all six figures as PNG and vector PDF in a new sibling folder:

```powershell
.\.venv\Scripts\python.exe make_figures.py --output ..\..\milestone3_figures
```

The plotting command reads the delivered result tables and refuses an existing output folder. The measured checkpoint replay preserved all decisions across ten files, with a maximum score difference of 4.45e-16 from floating-point arithmetic; see `09_reproducibility/checkpoint_replay_comparison.csv`.
