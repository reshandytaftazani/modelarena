"""Fresh fold-local transformers and the shared guarded dense output boundary.

Fit only on an isolated outer-fit (or MLP inner-fit) TrainingView. This module
has no dataset/holdout reader. Schema declarations come from configuration;
statistics and vocabularies come exclusively from fitting rows. Downstream
models must use FoldPreprocessor, rather than calling the raw factory directly.
"""

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from ml.contracts import FeatureSchema, FeatureType, ResourceLimits, TrainingView


def _category_token(value: object) -> str | float:
    # A namespace prefix separates all user strings from the imputer sentinel.
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return np.nan
    return "value:" + str(value)


def _categorical_tokens(values: object) -> NDArray:
    array = np.asarray(values, dtype=object)
    return np.array([[_category_token(cell) for cell in row] for row in array], dtype=object)


def make_preprocessor(schema: FeatureSchema) -> ColumnTransformer:
    """Return a fresh unfitted transformer; callers must guard allocations.

    Numeric empty columns become zero, constants scale to zero; categorical
    missing cells become a separate sentinel. Numeric columns precede encoded
    categorical/boolean columns, each in declared order. Unknowns are all-zero.
    Sparse output is retained where possible until the guarded dense boundary.
    """
    if not isinstance(schema, FeatureSchema):
        raise TypeError("Explicit feature schema required")
    numeric = [column.name for column in schema.columns if column.dtype == FeatureType.NUMERIC]
    categorical = [column.name for column in schema.columns if column.dtype != FeatureType.NUMERIC]
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("tokens", FunctionTransformer(_categorical_tokens, feature_names_out="one-to-one")),
            ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__", keep_empty_features=True)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float64)),
        ]), categorical))
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=1.0, n_jobs=1)


@dataclass(frozen=True, slots=True)
class ResourceEstimate:
    encoded_features: int
    category_counts: tuple[tuple[str, int], ...]
    projected_bytes: int


def guard_allocation(fit_rows: int, transform_rows: int, width: int,
                     limits: ResourceLimits) -> int:
    """Bound live fit + transform buffers, including float64 copies and cast.

    Reserve (8-byte intermediate + 8-byte copy + 4-byte cast) * 3 safety factor
    per cell for both partitions. This is a conservative admission estimate,
    not a hard RSS quota, and assumes serial transforms with one retained fit
    matrix. Callers must release prior validation matrices before the next one.
    """
    if any(type(n) is not int or n < 0 for n in (fit_rows, transform_rows, width)) or not width:
        raise ValueError("Invalid matrix dimensions")
    if width > limits.max_encoded_features:
        raise ValueError("Encoded feature width exceeds resource limit")
    projected = (fit_rows + transform_rows) * width * (8 + 8 + 4) * 3
    if projected > limits.preprocessing_bytes:
        raise ValueError("Projected preprocessing memory exceeds resource limit")
    return projected


def _validate_view(data: TrainingView) -> None:
    if not isinstance(data, TrainingView):
        raise TypeError("Preprocessing requires an isolated TrainingView")
    for index, column in enumerate(data.config.features.columns):
        for row in data.rows:
            value = row[index]
            if value is None:
                continue
            if column.dtype == FeatureType.NUMERIC:
                if type(value) not in (float, int) or not math.isfinite(value):
                    raise ValueError("Expected finite numeric feature cells")
            elif column.dtype == FeatureType.BOOLEAN:
                if type(value) is not bool:
                    raise ValueError("Expected boolean feature cells")
            elif not isinstance(value, str):
                raise ValueError("Expected categorical string feature cells")


def estimate_resources(data: TrainingView, limits: ResourceLimits = ResourceLimits()) -> ResourceEstimate:
    """Inspect fitting rows only; count missing as an encoded category if seen."""
    _validate_view(data)
    width = 0
    counts = []
    usable = False
    for index, column in enumerate(data.config.features.columns):
        observed: set[str] = set()
        missing = False
        for row in data.rows:
            value = row[index]
            if value is None:
                missing = True
            else:
                usable = True
                if column.dtype != FeatureType.NUMERIC:
                    observed.add(str(_category_token(value)))
                    if len(observed) + int(missing) > limits.max_categories:
                        raise ValueError(f"Training category limit exceeded: {column.name}")
        if column.dtype == FeatureType.NUMERIC:
            width += 1
        else:
            count = len(observed) + int(missing)
            if count > limits.max_categories:
                raise ValueError(f"Training category limit exceeded: {column.name}")
            counts.append((column.name, count))
            width += count
    if not usable:
        raise ValueError("No usable observed training feature")
    # Fit can allocate both a working transform and the returned fit matrix.
    projected = guard_allocation(len(data.rows), len(data.rows), width, limits)
    return ResourceEstimate(width, tuple(counts), projected)


def _frame(data: TrainingView) -> pd.DataFrame:
    return pd.DataFrame({
        column.name: pd.Series([row[i] for row in data.rows],
                               dtype=np.float64 if column.dtype == FeatureType.NUMERIC else object)
        for i, column in enumerate(data.config.features.columns)
    })


class FoldPreprocessor:
    """Guarded shared path for classical models and later CPU MLP conversion.

    Refit creates a fresh transformer and invalidates previous fitted state.
    Transform never learns categories or statistics, and never reads targets.
    Only float64 and explicitly requested float32 outputs are supported.
    """

    def __init__(self, limits: ResourceLimits = ResourceLimits()) -> None:
        self.limits = limits
        self._fitted = False

    def fit(self, data: TrainingView) -> "FoldPreprocessor":
        self._fitted = False
        estimate = estimate_resources(data, self.limits)
        transformer = make_preprocessor(data.config.features)
        transformer.fit(_frame(data))
        width = len(transformer.get_feature_names_out())
        guard_allocation(len(data.rows), len(data.rows), width, self.limits)
        if width != estimate.encoded_features:
            raise ValueError("Realized encoded width differs from training estimate")
        self.transformer_ = transformer
        self.estimate_ = estimate
        self.fit_row_ids_ = data.row_ids
        self._config = data.config
        self._fit_rows = len(data.rows)
        self._width = width
        self._fitted = True
        return self

    def transform(self, data: TrainingView, *, dtype: object = np.float64) -> NDArray:
        if not self._fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        _validate_view(data)
        if data.config != self._config:
            raise ValueError("Transform configuration/schema differs from fit")
        if dtype not in (np.float64, np.float32):
            raise ValueError("Only float64 or float32 output supported")
        guard_allocation(self._fit_rows, len(data.rows), self._width, self.limits)
        encoded = self.transformer_.transform(_frame(data))
        actual_rows, actual_width = encoded.shape
        guard_allocation(self._fit_rows, actual_rows, actual_width, self.limits)
        if actual_rows != len(data.rows) or actual_width != self._width:
            raise ValueError("Realized matrix shape/width differs from fit")
        result = encoded.toarray() if sparse.issparse(encoded) else np.asarray(encoded)
        if not np.isfinite(result).all():
            raise ValueError("Preprocessing output must be finite")
        if dtype == np.float32 and np.any(np.abs(result) > np.finfo(np.float32).max):
            raise ValueError("Preprocessing output exceeds finite float32 range")
        return result.astype(dtype, copy=False)

    def fit_transform(self, data: TrainingView, *, dtype: object = np.float64) -> NDArray:
        return self.fit(data).transform(data, dtype=dtype)
