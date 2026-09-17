"""Pandas dtype compatibility helpers.

Pandas 3 infers text columns as the dedicated ``str`` dtype. Unlike the
legacy ``object`` dtype, it rejects non-string assignments. The data-cleaning
and ML code intentionally converts some text columns to numeric values, so we
normalise inferred string columns to ``object`` at each public entry point.

This module intentionally does not import pandas. Importing it from ``app.py``
therefore preserves the app's lazy-loading behaviour for data libraries.
"""


def normalize_string_columns(df):
    """Convert pandas string extension columns to object columns in-place.

    The dtype names cover pandas 2.x's optional ``StringDtype`` and pandas
    3.x's default ``str`` dtype. ``object`` is appropriate because a column
    can later be transformed from text into numeric values.
    """
    for column in df.columns:
        series = df[column]
        if getattr(series.dtype, "name", None) in {"str", "string", "String"}:
            df[column] = series.astype(object)
    return df
