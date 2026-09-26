# Experimental Design, Execution Notes, and Limitations

Disclaimer: We used ChatGPT to help us create the md files - as a writing assistant, to help us turn our main ideas into clearer and more formal English.

## Starting point

Milestone 3 starts from the frozen outputs of Milestone 2: the recording-level data splits, the accepted feature roster, and the aligned tabular, sequence, and graph representations.

The production experiments use recording-disjoint Train, Validation, Test, and External Test partitions. Learned preprocessing is fitted on Train only. Validation is used for tuning, early stopping, threshold selection, and refinement decisions. Test and External Test are reserved for held-out evaluation.

## Executed sensitivity study

The available hardware was a CPU-only system with four physical cores, eight logical threads, and approximately 8 GB of RAM. For that environment, the study used a bounded one-factor-at-a-time sensitivity design rather than an exhaustive Cartesian grid.

Four configurations were executed for each primary model family.

### Random Forest

- baseline: 300 trees, maximum depth 12, minimum leaf size 5;
- depth increased to 20;
- minimum leaf size increased to 10;
- number of trees increased to 500.

### Isolation Forest

- baseline: 300 trees, `max_samples=256`, all features;
- `max_samples=1024`;
- `max_features=0.5`;
- 500 trees.

Two additional Validation-driven Isolation Forest feature refinements were run after the baseline error analysis. Their rationale is documented separately in `VALIDATION_REFINEMENT_RATIONALE.md`.

### Bidirectional LSTM

- baseline: hidden size 32, 1 bidirectional layer, dropout 0.2, learning rate 0.001;
- hidden size increased to 64;
- learning rate reduced to 0.0003;
- depth increased to 2 layers.

### GraphSAGE

- baseline: hidden size 32, 2 layers, dropout 0.2, mean pooling, learning rate 0.001;
- hidden size increased to 64;
- learning rate reduced to 0.0003;
- depth increased to 3 layers.

The executed configurations are defined in `03_config/experiments.json`.

## Neural training protocol

The neural models use AdamW with gradient clipping and a maximum of 24 epochs.

Early stopping is based on Validation AUPRC with:

- patience: 8 epochs;
- minimum improvement: 0.00001.

The operating threshold is selected separately after training by maximizing Validation malicious recall subject to FPR <= 1%. Ties are resolved by lower FPR and then the stricter threshold.

This separates epoch selection from operating-point selection.

Three fixed seeds were evaluated for each selected neural architecture: 42, 17, and 2026. Seed 42 remained the locked held-out evaluation seed; the additional seeds were used only to characterize Validation stability.

## Sequence implementation note

An early packed-sequence BiLSTM profiling run was stopped after one epoch because the CPU implementation required approximately 136 seconds per epoch.

The final implementation groups minibatch items by exact valid sequence length and executes the dense LSTM only over real timesteps. The interrupted profiling run is not included in the reported experiment table.

Padding-invariance checks verify that padded timesteps do not affect the backward-direction representation. Completed final BiLSTM runs required approximately 10–30 seconds per epoch depending on architecture size and concurrent CPU workload.

## Graph representation scope

The GraphSAGE implementation uses:

- node attributes;
- within-window graph topology;
- mean neighborhood aggregation;
- normalized hidden vectors;
- mean graph pooling.

Stored edge attributes are not consumed by the baseline GraphSAGE model.

The graph is a canonical within-window proxy linking clients, DNS servers, base domains, and subdomains. It is not a reproduction of GraphTunnel's full recursive DNS-resolution graph. Therefore, comparisons with GraphTunnel should be interpreted as comparisons between graph-based behavioral representations rather than exact architecture reproductions.

## Computational measurements

Reported training and inference times are descriptive measurements from the project environment rather than controlled microbenchmarks.

The local LLM ran concurrently with some neural seed experiments, so its request latency can include CPU contention. Detector inference timings also include model and data loading.

The cascade was evaluated by routing aligned, precomputed detector outputs. This validates the decision logic and classification outcomes but does not establish a measured end-to-end latency saving for a live deployment that conditionally skips neural inference.

## Cascade design

The hybrid pipeline was frozen before held-out evaluation.

Stage 1 uses the Random Forest and the refined Isolation Forest. If their binary decisions agree and the Random Forest score is at an extreme operating band (`<=0.2` or `>=0.8`), the Random Forest decision is accepted.

All remaining windows proceed to Stage 2, where the frozen Random Forest, BiLSTM, and GraphSAGE binary decisions are combined by majority vote.

Stage 2 windows are eligible for local LLM arbitration under a fixed request budget.

The internal Test result improved from 4 Random Forest false negatives to 1 cascade false negative while maintaining zero cascade false positives. On the malicious-only External Test split, the cascade produced 16 false negatives compared with 9 for the standalone Random Forest. The cascade is therefore not interpreted as uniformly superior across data distributions.

## Local LLM arbitration

The local arbitration model is Qwen2.5-1.5B-Instruct, executed through `llama.cpp` on CPU.

The frozen policy allows:

- at most 32 requests per split;
- at most 2 requests per recording;
- deterministic candidate ordering by hashed `sample_id`;
- preservation of the cascade decision on timeout, invalid response, unsupported label, or budget exhaustion.

The measured run contained:

- 2 Validation requests;
- 6 Test requests;
- 13 External Test requests.

All 21 saved responses were valid JSON.

Across these 21 arbitrated cases, the cascade decision was correct on 17 before LLM arbitration and 19 after arbitration. On Test, the LLM introduced one harmful override and corrected none. On External Test, it corrected three cascade misses and introduced no new errors.

The explanations also showed a strong tendency to rely on query-name length: `query_name_len` appeared in the evidence of all 21 saved responses. This observation is reported as a behavioral limitation of the small local model. No controlled prompt-content ablation was included in the completed experiment, so the causal sensitivity of the LLM to removing individual prompt components remains unmeasured.

Because the Validation arbitration sample is very small, automatic LLM overrides should not be treated as production-validated behavior. A larger, source-diverse, independently held-out arbitration set would be required before deployment.

## Class imbalance and sampling

No synthetic traffic was generated.

Class imbalance and recording dominance were handled using class/recording weighting or balanced sampling while leaving Validation and held-out populations unchanged.

The Isolation Forest was trained only on benign Train data, balanced across benign recordings.

## Important limitations

The experiment should be interpreted with the following constraints:

1. **Source-label confounding remains present.** Most benign Train windows originate from one capture ecosystem, so model performance may partially reflect source-specific properties.
2. **External Test is malicious-only.** It can measure malicious recall and miss counts but cannot estimate specificity, false-positive rate, or binary F1 on unseen benign traffic.
3. **GraphSAGE uses a proxy graph.** The implementation does not contain the complete recursive resolver paths used by GraphTunnel.
4. **Validation saturation limits hyperparameter conclusions.** Random Forest and GraphSAGE achieved perfect Validation operating metrics across the tested configurations, so the experiment cannot establish a unique optimal complexity for those models.
5. **LLM arbitration evidence is small.** Only 21 total requests were executed under the frozen budget, and only 2 were from Validation.
6. **Runtime observations are descriptive.** They were collected on the project CPU system and should not be interpreted as hardware-independent benchmarks.

These limitations are retained in the documentation so that the reported held-out results are not overstated.
