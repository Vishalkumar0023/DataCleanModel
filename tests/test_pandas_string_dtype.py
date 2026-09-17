"""Regression tests for pandas 3's default strict string dtype."""

import pandas as pd

from data_pipeline.data_cleaner import DataCleaner
from data_pipeline.data_loader import DataLoader
from data_pipeline.model_trainer import ModelTrainer


def test_text_columns_can_be_converted_to_numbers_with_str_dtype(tmp_path):
    future_options = getattr(pd.options, "future", None)
    original_infer_string = getattr(future_options, "infer_string", None)
    if future_options is not None and original_infer_string is not None:
        future_options.infer_string = True
    try:
        source = pd.DataFrame(
            {
                "plan": pd.Series(["basic", "pro", "basic", "pro"], dtype="string"),
                "amount": ["$10", "$20", "$15", "$25"],
                "revenue": [10.0, 20.0, 15.0, 25.0],
            }
        )
        assert source["plan"].dtype.name in {"str", "string"}

        path = tmp_path / "subscriptions.csv"
        source.to_csv(path, index=False)
        loaded = DataLoader().load(str(path))
        assert loaded["plan"].dtype == object

        cleaner = DataCleaner(loaded)
        cleaner.clean_numeric_text(columns=["amount"])
        cleaned = cleaner.get_cleaned_data()
        assert cleaned["amount"].dtype.kind == "f"

        trainer = ModelTrainer(cleaned, target_col="revenue", problem_type="regression")
        trainer._encode_categoricals()
        assert trainer.df["plan"].dtype.kind == "f"
    finally:
        if future_options is not None and original_infer_string is not None:
            future_options.infer_string = original_infer_string
