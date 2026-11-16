"""Automatic feature typing and preprocessing for arbitrary tabular / text data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

MISSING = "__missing__"


# Module-level functions (not lambdas) so fitted pipelines can be pickled.
def to_str_frame(X):
    df = pd.DataFrame(X).astype(object)
    return df.where(df.notna(), MISSING).astype(str)


def to_str_series(x):
    s = x if isinstance(x, pd.Series) else pd.Series(np.ravel(x))
    return s.astype(object).where(s.notna(), "").astype(str)


def infer_schema(X: pd.DataFrame) -> dict[str, str]:
    """Classify each column as 'numeric', 'categorical' or 'text'; drop empty and ID-like columns."""
    schema: dict[str, str] = {}
    for col in X.columns:
        s = X[col]
        nn = s.dropna()
        if len(nn) == 0:
            continue
        if pd.api.types.is_bool_dtype(s):
            schema[col] = "categorical"
            continue
        if pd.api.types.is_numeric_dtype(s):
            schema[col] = "numeric"
            continue
        if pd.to_numeric(nn, errors="coerce").notna().mean() > 0.95:
            schema[col] = "numeric"
            continue
        st = nn.astype(str)
        avg_words = st.str.split().str.len().mean()
        if avg_words >= 3 or st.str.len().mean() > 40:
            schema[col] = "text"
        elif len(st) > 20 and st.nunique() / len(st) > 0.9:
            continue  # unique per row: an identifier, not a feature
        else:
            schema[col] = "categorical"
    return schema


def prepare_frame(data, schema: dict[str, str]) -> pd.DataFrame:
    """Align raw rows to a schema: add missing columns, drop unknown ones, coerce types."""
    df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(list(data))
    out = {}
    for col, kind in schema.items():
        s = df[col] if col in df.columns else pd.Series(np.nan, index=df.index)
        if kind == "numeric":
            out[col] = pd.to_numeric(s, errors="coerce").astype(float)
        else:
            out[col] = s.astype(object)
    return pd.DataFrame(out, index=df.index)


def build_preprocessor(schema: dict[str, str], dense: bool) -> ColumnTransformer:
    num = [c for c, k in schema.items() if k == "numeric"]
    cat = [c for c, k in schema.items() if k == "categorical"]
    txt = [c for c, k in schema.items() if k == "text"]
    transformers = []
    if num:
        transformers.append(("num", Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]), num))
    if cat:
        transformers.append(("cat", Pipeline([
            ("str", FunctionTransformer(to_str_frame)),
            ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=3,
                                     sparse_output=not dense)),
        ]), cat))
    for c in txt:
        transformers.append((f"text_{c}", Pipeline([
            ("str", FunctionTransformer(to_str_series)),
            ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
        ]), c))
    return ColumnTransformer(transformers, sparse_threshold=0.0 if dense else 0.3)


def reference_stats(Xp: pd.DataFrame, schema: dict[str, str]) -> dict:
    """Snapshot of the training distribution, used later for drift detection."""
    numeric, categorical = {}, {}
    for col, kind in schema.items():
        s = Xp[col].dropna()
        if len(s) == 0:
            continue
        if kind == "numeric":
            numeric[col] = {"mean": float(s.mean()), "std": float(s.std(ddof=0))}
        elif kind == "categorical":
            categorical[col] = s.astype(str).value_counts().index[:1000].tolist()
    return {"numeric": numeric, "categorical": categorical}
