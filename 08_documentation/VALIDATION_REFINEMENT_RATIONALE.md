# Validation-Driven Isolation Forest Refinement

Disclaimer: We used ChatGPT to help us create the md files - as a writing assistant, to help us turn our main ideas into clearer and more formal English.

## Purpose

The Isolation Forest refinements were introduced to reduce Validation false positives while preserving a leakage-safe evaluation protocol. All refinement decisions were based on Train and Validation evidence only. Held-out Test and External Test results were not used to select the final Isolation Forest configuration.

The Isolation Forest remained a benign-only anomaly detector throughout the process and was fitted using recording-balanced benign Train data.

## Baseline Validation error pattern

The baseline Isolation Forest produced 7 false positives on Validation. All 7 belonged to the same CTU normal recording.

The error analysis showed a consistent acquisition-related pattern:

- all 7 false-positive windows exceeded the maximum benign Train value for transaction-byte mean;
- all 7 exceeded the benign Train maximum for transaction-byte standard deviation;
- all 7 exceeded the benign Train maximum for transaction-byte 95th percentile;
- all 7 exceeded the benign Train maximum for response-payload mean;
- 5 of the 7 also exceeded the benign Train maximum for interarrival-time 95th percentile.

The median query-name length of these false-positive windows was 18.75 characters, compared with 14.20 in benign Train and 40.00 in malicious Train.

These observations suggested that packet-volume and timing features were capturing differences in background acquisition conditions in addition to possible attack-related behavior. This observation motivated a bounded refinement experiment; it was not treated as proof of causation.

## Refinement sequence

The original Isolation Forest baseline used:

- 300 trees;
- `max_samples=256`;
- all 42 core tabular features;
- benign Train samples only.

The first sensitivity change increased `max_samples` from 256 to 1,024. Validation false positives decreased from 7 to 3 while malicious recall remained 100%.

Two further feature-removal hypotheses were then tested, both using `max_samples=1024`.

### Refinement 1: remove packet-volume features

Five features were removed:

- `transaction_bytes_mean`
- `transaction_bytes_std`
- `transaction_bytes_p95`
- `request_payload_mean`
- `response_payload_mean`

This reduced the active feature set from 42 to 37.

### Refinement 2: additionally remove timing features

The five volume/payload features above were removed together with:

- `response_time_median`
- `response_time_std`
- `response_time_p95`
- `iat_mean`
- `iat_std`
- `iat_median`
- `iat_p95`
- `iat_cv`

This reduced the active Isolation Forest feature set from 42 to 29.

The lexical, entropy, recurrence, query-type, and response-code feature groups remained available to the detector.

## Validation results

The Validation progression was:

| Configuration | Active features | Validation FP | Validation FN | Validation F1 |
|---|---:|---:|---:|---:|
| Baseline IF | 42 | 7 | 0 | 99.703% |
| `max_samples=1024` | 42 | 3 | 0 | 99.872% |
| Remove volume/payload | 37 | 2 | 0 | 99.915% |
| Remove volume/payload + timing | 29 | 1 | 0 | 99.957% |

All four configurations retained 100% Validation malicious recall.

The 29-feature refinement was selected using the same predeclared operating-point rule: maximize Validation malicious recall subject to FPR <= 1%, then prefer lower FPR. Test performance was not used to choose among these candidates.

## Held-out interpretation

After the final Isolation Forest configuration was frozen, held-out evaluation showed a trade-off rather than a uniform improvement.

On the internal Test split, the refinement reduced false positives from 10 to 5, but false negatives increased from 2 to 7. Consequently, F1 remained almost unchanged.

On the malicious-only External Test split, the refined model produced more misses than the baseline Isolation Forest.

These results are intentionally reported without post-Test retuning. The refinement succeeded at its Validation objective of reducing false alarms, but it also demonstrated that removing acquisition-sensitive volume and timing information can discard useful attack evidence.

## Related model-selection observations

The same Validation-only discipline was applied to the other model families.

Random Forest and GraphSAGE produced no Validation errors in their executed sensitivity configurations. The baseline BiLSTM produced 1 Validation false positive; the wider and deeper BiLSTM variants produced 5 and 4, while the lower-learning-rate variant tied the baseline at 1. The baseline architecture was therefore retained according to the predeclared tie rule.

No additional architecture search was introduced after inspecting held-out Test results.
