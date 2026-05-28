# Codex Iteration Bundle

This folder is a snapshot of repo files created or touched during the recent
feature-extraction, causal-forest, and agentic-search experiments. Paths under
this folder mirror their repo-relative locations.

## New Experiment Files

- `example_configs/agentic_explicit_feature_forest_config.json`
- `oci/inference/agentic_explicit_feature_forest.py`
- `oci/models/byte_cnn_extractor.py`
- `oci/models/candidate_variable_text_svd_extractor.py`
- `oci/models/causal_purity_hash_extractor.py`
- `oci/models/neural_causal_hash_gate.py`
- `oci/models/neural_mention_slot_extractor.py`
- `oci/models/text_marker_extractor.py`
- `oci/models/token_hash_embedding_extractor.py`
- `oci/models/token_windowing.py`
- `oracle_experiment_scripts/run_agentic_explicit_feature_forest.py`
- `oracle_experiment_scripts/run_candidate_variable_text_svd_forest.py`
- `oracle_experiment_scripts/run_neural_causal_hash_gate_probe.py`
- `oracle_experiment_scripts/run_neural_mention_slot_forest.py`
- `oracle_experiment_scripts/run_residual_modifier_forest.py`
- `tests/test_agentic_feature_search.py`
- `tests/test_candidate_variable_text_svd_extractor.py`
- `tests/test_neural_mention_slot_extractor.py`

## Modified Integration Snapshots

- `README.md`
- `oci/config.py`
- `oci/extraction/cache.py`
- `oci/extraction/explicit_features.py`
- `oci/inference/__init__.py`
- `oci/inference/applied.py`
- `oci/models/causal_text.py`
- `oci/models/causal_text_forest.py`
- `oci/models/contrastive_causal_text_forest.py`
- `oci/models/extractor_factory.py`
- `oci/models/frozen_llm_pooler_extractor.py`
- `oci/models/hidden_state_cache.py`
- `oci/models/propensity_model.py`
- `oci/training/contrastive_effect.py`
- `oracle_experiment_scripts/run_oracle_experiments.py`
- `oracle_experiment_scripts/run_oracle_xw_rlearner_forest_experiments.py`
- `tests/test_extractors.py`
