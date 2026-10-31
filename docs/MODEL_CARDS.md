# Model cards — local tabular baselines

## Intended use and limitations

ModelArena compares CPU-only binary classification and regression models on
tabular data for education and experimentation. It is not validated for medical,
financial, legal, safety-critical or other consequential decisions. Evaluation
is meaningful only when rows are approximately independent and representative
of future use. Random splits do not protect against repeated people, households,
groups, temporal dependence, distribution shift, or features known only after
the prediction. Duplicate warnings and acknowledgement do not prove independence.

For classification, the user explicitly chooses the positive class. Imbalance
can make accuracy misleading: inspect precision, recall, F1, average precision,
ROC-AUC, and the confusion matrix, noting undefined metrics. The fixed 0.5
threshold is not optimized on the holdout. For regression, inspect MAE, RMSE,
R² and residuals, including units and outliers. CV results used to select among
models can be optimistic; the locked holdout is reserved for final evaluation.

## Classical families

Dummy (prior/mean) is a reference, not an assumed useful predictor. Logistic
regression or Ridge provides a linear baseline; random forests and histogram
gradient boosting represent nonlinear alternatives. Each fold learns its own
imputation, scaling and category vocabulary. Final refit constructs a new
preprocessor and model and fits the entire training partition, not holdout rows.

## CPU MLP

The default network has two hidden layers (64 and 32 ReLU units), Adam at
0.001 and batch size 32. Classification uses logits/BCE and regression uses
MSE. During CV, each outer training fold has a separate inner early-stopping
split; preprocessing is fitted on inner-fit rows only. Final refit has no
validation split: it uses fresh weights and preprocessing on all training rows
for `ceil(median(best epochs))` across **all complete successful** frozen CV
folds, bounded to 1–100. Final training loss is separate from the per-fold
inner-validation curves; no final validation curve is implied.

Dense one-hot encoding, category cardinality, and network training consume CPU
time and memory. Existing input, matrix, epoch and worker limits can reject a
run; fitting a fixture does not establish performance on a laptop workload.
Runs are seed-controlled but numerical identity across library versions,
operating systems or CPUs is not guaranteed.

## Artifacts and provenance

Day 18 creates private, locally generated model/preprocessor files with hashes,
selection and split identities, CV-evidence digest, epoch policy, training loss
and software provenance. Verification accepts only a handle issued within the
writing process and checks the bytes; it does not load the model. The live
in-process model can be used by the future final evaluator. Checksums do not make
unknown pickle safe or guarantee portability. There is no external model import,
model download/export, or standalone inference feature in this work. Local
owner or same-process hostile code is outside the provenance boundary.
