# ModelArena — Project Specification v0.1

## Mission
Build a local-first, CPU-only Automated ML Experimentation and Evaluation Platform for a solo intermediate developer in 28 days (~2–3 focused hours/day). The product must teach sound ML fundamentals, not just wrap algorithms. No paid LLM APIs, GPU, cloud services, or external model providers required.

## Scope and users
- User: student/developer experimenting with tabular datasets locally.
- MVP supports binary classification and regression only.
- Input: user-supplied CSV with selected target column; shipped example datasets.
- Output: reproducible runs, CV-based leaderboard, learning curves, locked holdout evaluation, downloadable local experiment report and trusted model artifacts.
- Not in MVP: multiclass, time series, multi-user, distributed training, cloud deployment, live inference hosting, autonomous LLM agents.

## Required stack
React + TypeScript + Vite, Tailwind CSS, shadcn/ui, Recharts; FastAPI + Pydantic; pandas + NumPy; scikit-learn; PyTorch CPU; MLflow local tracking; SQLite metadata; pytest + httpx; uv or pip with locked versions. Prefer a simple local process worker and durable SQLite job states; avoid Redis/Celery/Kubernetes.

## P0 features
1. CSV upload, validation, dataset fingerprint/hash, column/target selection.
2. Dataset profiling: dimensions, dtypes, missingness, target distribution, class imbalance, basic feature stats.
3. Holdout split default 80/20, stratified for classification where feasible; keep holdout isolated from all selection/tuning.
4. Leakage-safe preprocessing *inside* training folds: numeric imputation/scaling, categorical imputation/one-hot with unknown handling. Fit transforms on training fold only.
5. Classical models per task: classification DummyClassifier, LogisticRegression, RandomForestClassifier, HistGradientBoostingClassifier; regression DummyRegressor, Ridge, RandomForestRegressor, HistGradientBoostingRegressor.
6. PyTorch CPU MLP per task: configurable baseline input -> 64 -> ReLU -> 32 -> ReLU -> output; Adam LR 1e-3, batch size 32, max 100 epochs, early stopping patience 10; BCEWithLogitsLoss for binary classification, MSELoss for regression. For MLP, preprocessing and early-stopping validation must be fit/split within each CV training fold, never on CV validation or holdout data. Ensure reproducible seeds and save training/validation loss curves.
7. Comparable evaluation via 5-fold CV on training portion when class/sample counts permit; classification F1, precision, recall, ROC-AUC, average precision/PR-AUC, confusion matrix where appropriate; regression MAE, RMSE, R2, residual diagnostics. Record training duration. Handle undefined metrics explicitly.
8. MLflow tracking of configs, versions, seed, dataset fingerprint, metrics, artifacts; SQLite for dataset/job metadata. Reproducible experiment manifest.
9. Leaderboard ranked by preselected primary CV metric, with baseline comparison and mean/std across folds. Do not rank by holdout test results.
10. Separate local worker process for jobs, single active job default; persistent queued/running/completed/failed/interrupted states, cancel/interrupt support, progress polling.
11. Final model selection based on CV; refit on full training portion; evaluate once on holdout. Avoid repeated test peeking and preserve evaluation audit trail.
12. Local dashboard: upload -> configure -> train -> monitor -> compare -> inspect -> final evaluation -> report.
13. Tests for data leakage, splitting, metrics, worker lifecycle, invalid datasets, and end-to-end flow.

## P1 if time remains
- RandomizedSearchCV capped at 12 configurations/model; ensure tuning fully nested within training partition and no holdout usage.
- Permutation importance for selected models, calculated on suitable validation data and clearly labeled.
- Trusted local model export with metadata and versioning.

## P2 stretch only
- Optional rule-based or local LLM explanation of *observed* metrics; never required for MVP. Must not invent causes or imply a paid API requirement.

## Example datasets
- Breast Cancer Wisconsin Diagnostic: sklearn load_breast_cancer; 569 rows, 30 numeric features; binary classification.
- California Housing: sklearn fetch_california_housing; ~20,640 rows, 8 numeric features; regression; document any first-time download requirement.
- UCI Bank Marketing: mixed types and imbalance; ~45,211 rows full version; stretch integration dataset. For pre-call prediction, exclude `duration` (known only after call).

## Evaluation protocol
- Before experiments, select task-specific primary metric (classification F1 by default; for strongly imbalanced problems consider average precision; regression RMSE or MAE, with direction explicit).
- Training portion CV only. All preprocessing, feature selection, tuning, and MLP early stopping are fold-local. Holdout untouched until a chosen model is frozen.
- For classification, report confusion matrix on holdout and avoid interpreting accuracy alone for imbalanced data.
- Compare training time as secondary metric. Three-seed stability experiment is recommended; explicitly distinguish changing model seed vs changing split seed.
- Baseline must be beaten on at least the demonstration datasets by the selected metric; do not invent an accuracy threshold.
- Save seed, fold IDs/split hashes, hyperparameters, library versions, dataset fingerprint, code revision where possible, and metrics. Define numeric tolerance for deterministic reruns.

## Resource budget
- Local laptop, CPU only, ideally 8–16GB RAM.
- Default one training worker; bounded CPU threads, epoch/time/config caps.
- Initial upload cap 20MB / 50k rows (tune after profiling). For tiny datasets, insufficient minority class counts, high cardinality, or massive dense matrices, fail safely or adapt with documented policy.
- Target classical baseline run on Breast Cancer within ~5 minutes on reference laptop, measured not promised.
- No mandatory paid services. Electricity and optional hosting excluded.

## Security and failure handling
- Reject corrupt/empty CSV, invalid target, one-class classification target, missing target labels, unsupported types, path traversal, and oversized files.
- Warn for ID-like features, target leakage, data available only after prediction, and suspiciously perfect CV results.
- Handle unseen categories, constant columns, NaN/inf features, high cardinality, memory limits, NaN loss, failed/cancelled/interrupted jobs, restarts, duplicate concurrent jobs.
- Never deserialize untrusted pickle/joblib/model artifacts. Keep uploaded files and artifacts outside publicly served directories; validate filenames/paths.
- Define cleanup/retention behavior for uploads and artifacts.

## Delivery plan
Week 1 days 1–7: repo/test setup, loader, validator, profiler, splitting, preprocessing, baseline, integration tests.
Week 2 days 8–14: classical ensembles, evaluation, MLflow, limited tuning if feasible, leaderboard API, leakage tests.
Week 3 days 15–21: PyTorch Dataset/DataLoader, classifier/regressor MLP, early stopping, learning curves, seed studies, CPU profiling.
Week 4 days 22–28: FastAPI worker lifecycle, React upload/configuration/progress/leaderboard/plots, final evaluation, reports, E2E, README/demo.
The plan is a target, not permission to skip testing or compromise leakage safety. Defer P1/P2 first if delayed.

## Acceptance criteria / Definition of Done
- Valid CSV upload, profiling, task/target configuration.
- Both task types run through all required classical families and PyTorch MLP.
- Fold-local preprocessing and MLP validation verified by automated tests.
- CV leaderboard correct and holdout excluded from model selection.
- One final holdout evaluation for selected model; audit trail saved.
- MLflow records metrics, config, fingerprint, seeds, artifacts.
- Worker job lifecycle and interruption/restart behavior tested.
- MLP training/validation loss charts available.
- All critical tests pass; a deterministic rerun matches metrics within documented tolerance.
- Fully local end-to-end demo and setup README, experiment report, limitations and known risks.

## Suggested repository
apps/web/src/{components,pages,features,lib}; apps/api/{routers,schemas,services,workers,database}; ml/{data,preprocessing,models,training,evaluation,tracking}; tests/{unit,integration,e2e,fixtures}; experiments/{configs,reports}; scripts; docs/{architecture.md,experiment_protocol.md,learning_notes,model_cards}; data and artifacts gitignored; pyproject.toml, .gitignore, README.md.

## Planning questions Codex must resolve explicitly
- How is a durable worker process started/stopped on Windows PowerShell and other platforms?
- How are fold-local MLP preprocessing and early stopping implemented without CV leakage?
- How are split/fold identities persisted and the holdout evaluation gated?
- How are MLflow and SQLite responsibilities separated?
- How are job cancellation, partial artifacts, and restart reconciliation handled?
- What is the simplest MVP report format and trusted export policy?
- What measurable acceptance test verifies each P0 feature?
