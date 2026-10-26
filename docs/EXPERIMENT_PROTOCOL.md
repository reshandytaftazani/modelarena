# Experiment protocol

This document describes the implemented data and partition boundaries through Day 4. Model fitting, preprocessing, durable evaluation-scope locks, worker lifecycle and final evaluation remain later work. The local split manifest is an integrity record, not a completed experiment.

## Configure before splitting

1. Parse a private UTF-8 CSV using `validate_csv`. Preserve ordered raw records and canonical content identity; expose structural information only.
2. Supply a frozen `DatasetConfig`: task, target, ordered selected features and their declared types. Binary classification requires an explicit positive label; regression requires a finite, variable numeric target. No feature statistics or category inventories are inferred from the complete dataset.
3. Resolve feature exclusions using knowledge of prediction-time availability. In particular, exclude `duration` for Bank Marketing pre-call prediction. Neither random splitting nor a warning establishes that a feature is scientifically valid.
4. Validate all records. Repeated selected typed features require explicit risk acknowledgement, including complete repeats and differing labels. Invalid records remain invalid after acknowledgement. No rows are silently removed. Detailed duplicate statistics are training-only.

Random splitting assumes independent observations. Repeated subjects, households, groups, or temporal observations can violate that assumption even if all exact rows differ. Duplicate acknowledgement does not establish independence. Grouped and temporal evaluation are unsupported; use a scientifically appropriate protocol outside this MVP rather than interpreting these random-split results as reliable.

## Freeze the holdout and folds

`create_split(parsed, config, RunConfig()) -> SplitManifest` defaults to split seed 42. The model seed is separate and does not affect partition creation. Fixed row identities are zero-based positions in the original parsed record order; persisted identities are sorted within each partition. Do not reorder rows before applying these identities.

The default holdout size is `max(2, ceil(0.2 * row_count))`. Training contains the remaining rows. Binary classification uses a shuffled stratified split and requires both labels on both sides. It rejects infeasible counts or allocations; it never silently switches to unstratified splitting or searches alternative seeds. Regression uses a shuffled split with at least two holdout rows. No held-out feature statistics or scores influence these decisions. Full-data target validity and label counts are used only for integrity and stratification.

Classification stratification uses a stable ordering of original label strings, independent of the user-confirmed positive-label polarity. Changing which label maps to 1 cannot change holdout, outer-fold or inner-fold row assignments. Changing feature selection likewise does not change row assignments for the same source, target, task and split seed; configuration digests still record these choices.

Outer folds use shuffled `StratifiedKFold` for classification and shuffled `KFold` for regression, with the split seed. Evaluate feasibility in descending order 5, 4, 3, 2 and freeze the largest feasible scheme. Every training row appears in outer validation exactly once; each fold's fit and validation sets are disjoint and exhaust the training partition. No outer or inner partition contains a holdout row.

For each outer fit partition, create a separate MLP early-stopping split:

- The default inner validation size is `ceil(0.2 * outer_fit_count)`; require at least two inner fitting and two inner validation rows.
- Classification requires both labels in every outer validation, inner fitting and inner validation set. Require at least two observations per label in an outer fit before considering its inner split. If the default validation allocation fails, use the smallest larger validation size that preserves both labels on both sides. Binary search bounds this search; it does not scan thousands of allocations for imbalanced uploads.
- Regression uses the same size minimums, including at least two outer validation rows for R2. The size minimum does not guarantee target variance; undefined metrics will need explicit handling in the later metrics implementation.
- Each inner seed is `(split_seed + 101 * (fold_index + 1)) % 2**32`, with zero-based fold indices. Persist the seed, exact identities and an expansion flag. Expansion means the chosen size exceeds the unadjusted ceiling of 20%.
- Reject the complete configuration if no 5/4/3/2 scheme satisfies all outer and inner requirements. Do not drop the MLP, a label or required validation rows to make an experiment appear feasible.

Two-fold candidates are implemented and tested. With the current minimum of two rows in each inner partition, a feasible two-fold scheme may always be dominated by a feasible larger scheme. Tests demonstrate real five-, four- and three-fold selections and two-fold feasibility directly; they do not fabricate a dataset that selects two folds or relax the safety policy.

## Persist, verify and reuse

`write_manifest(manifest, private_path)` writes canonical sorted JSON with a terminal LF. It flushes and closes a same-directory temporary file, then atomically links it into place without replacing an existing record. An identical retry is idempotent. A different record at the same path raises `split_conflict` and preserves the original bytes. Publication failures clean up their temporary file. The local filesystem must support hard links; an unsupported publication fails rather than falling back to unsafe overwrite behavior. Runtime paths must be service-owned, private and outside Git or public static directories.

The manifest stores schema version, dataset/configuration SHA-256 identities, row count, split seed, splitter algorithm and scikit-learn version, exact training/holdout/outer/inner identities, partition hashes, fold adaptation reason/warning, inner seeds/expansion flags and a manifest checksum. It contains no raw rows, category examples or analytical summaries. The manifest checksum covers all fields except itself. Partition hashes cover the relevant ordered raw records, columns, row identities and partition name, so a change confined to holdout records cannot change the training partition hash. The complete dataset fingerprint still changes. Equivalent parser formatting preserves canonical identity; row reordering changes it.

`read_manifest(private_path)` bounds reads to 16 MB, accepts JSON only, rejects duplicate fields and malformed contracts, validates all partition relationships, then verifies the checksum. Corrupted row counts do not drive allocation. Hashes detect corruption; they do not authenticate an attacker-controlled record. Never deserialize pickle/joblib/model artifacts supplied by a user.

`training_from_manifest(configured, manifest) -> TrainingView` verifies the checksum, exact dataset/configuration identity and aligned row count before copying only the persisted training rows. Use this boundary for `profile_training`. It rejects mismatched/reordered source identities or changed configurations. The training view retains no parent dataset or holdout reference. Tests guard held-out feature indexing and verify statistics against the persisted training identities.

Persisted indices are authoritative on reload; do not regenerate them with a different package version. Byte-identical recreation is tested within the current locked environment. Cross-version recreation is not promised, and no numeric model-score reproducibility claim is made before model evaluation exists.

## Subsequent experiment boundaries

Split storage does not yet implement the durable dataset-content/target/task evaluation-scope lock. Later repository and finalization work must look up and reuse the original split for that scope. Feature subsets, positive-label changes, primary metric changes and model seeds must not create a new holdout opportunity. `create_split` is a trusted creation function, not a client endpoint for repeated reseeding. Never overwrite a frozen split or present changed data as the same identity.

Day 5 must fit preprocessing inside each appropriate fitting partition, with explicit all-missing-column and allocation policies. MLP preprocessing and gradient updates use inner fitting rows only; early stopping uses inner validation, and scoring uses outer validation. Classical candidates use the same outer validation folds. Training-only profile statistics must never fit preprocessing.

Before experiments, freeze the task's primary CV metric and binary positive label. Rank by training-portion CV only. Later finalization freezes the selected model, refits on the full training partition, and evaluates once on the locked holdout with a durable audit trail. No holdout evaluation, model fitting or release milestone is claimed by Day 4.

## Verification

```powershell
uv run --locked pytest tests/unit/test_splits.py -q
uv run --locked pytest tests/unit/test_profile.py tests/unit/test_validation.py -q
uv run --locked pytest -q
uv lock --check
git diff --check
```

See [validation evidence](VALIDATION.md) for executed results and known limitations. Generated manifests and dataset artifacts remain private runtime files, excluded from Git.
