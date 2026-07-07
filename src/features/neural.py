from __future__ import annotations

import numpy as np
import pandas as pd


class WalmartNeuralPreprocessor:
    """
    Converts a cleaned Walmart panel dataframe into a neural-model-ready panel.

    Responsibilities:
    - create series_id for Store-Dept grouping
    - encode Store, Dept, Type as categorical IDs
    - compute target scaling statistics from known history only
    - scale continuous covariates
    - keep known-future covariates available for window datasets
    """

    def __init__(
        self,
        target_col: str = "Weekly_Sales",
        series_cols: tuple[str, str] = ("Store", "Dept"),
        categorical_cols: tuple[str, ...] = ("Store", "Dept", "Type"),
        static_real_cols: tuple[str, ...] = ("Size",),
        continuous_covariate_cols: tuple[str, ...] = (
            "Temperature",
            "Fuel_Price",
            "CPI",
            "Unemployment",
            "MarkDown1",
            "MarkDown2",
            "MarkDown3",
            "MarkDown4",
            "MarkDown5",
            "total_markdown",
            "abs_total_markdown",
            "positive_markdown_sum",
            "negative_markdown_sum",
            "markdown_missing_count",
            "Week_sin",
            "Week_cos",
        ),
        binary_covariate_cols: tuple[str, ...] = (
            "IsHoliday",
            "IsSuperBowl",
            "IsLaborDay",
            "IsThanksgiving",
            "IsChristmas",
            "has_markdown_signal",
            "markdown_available_period",
            "MarkDown1_was_missing",
            "MarkDown2_was_missing",
            "MarkDown3_was_missing",
            "MarkDown4_was_missing",
            "MarkDown5_was_missing",
        ),
    ):
        self.target_col = target_col
        self.series_cols = list(series_cols)
        self.categorical_cols = list(categorical_cols)
        self.static_real_cols = list(static_real_cols)
        self.continuous_covariate_cols = list(continuous_covariate_cols)
        self.binary_covariate_cols = list(binary_covariate_cols)

        self.is_fitted_ = False

    def fit(self, history_df: pd.DataFrame) -> "WalmartNeuralPreprocessor":
        """
        Fit using known sales history only.

        For validation:
            history_df = train_part after base preprocessing

        For final test:
            history_df = full train.csv after base preprocessing
        """
        df = history_df.copy()
        df["Date"] = pd.to_datetime(df["Date"])

        self._require_columns(df, self.series_cols + [self.target_col])

        target_df = df[df[self.target_col].notna()].copy()

        if target_df.empty:
            raise ValueError("history_df must contain known Weekly_Sales values.")

        # categorical mappings
        # 0 is reserved for unknown categories
        self.category_maps_ = {}

        for col in self.categorical_cols:
            if col not in df.columns:
                continue

            values = sorted(df[col].dropna().unique())
            self.category_maps_[col] = {
                value: idx + 1 for idx, value in enumerate(values)
            }

        # Store-Dept target statistics
        series_stats = (
            target_df.groupby(self.series_cols)[self.target_col]
            .agg(["mean", "std"])
            .reset_index()
            .rename(columns={"mean": "series_target_mean", "std": "series_target_std"})
        )

        # Department fallback statistics
        dept_stats = (
            target_df.groupby("Dept")[self.target_col]
            .agg(["mean", "std"])
            .reset_index()
            .rename(columns={"mean": "dept_target_mean", "std": "dept_target_std"})
        )

        # Store fallback statistics
        store_stats = (
            target_df.groupby("Store")[self.target_col]
            .agg(["mean", "std"])
            .reset_index()
            .rename(columns={"mean": "store_target_mean", "std": "store_target_std"})
        )

        self.series_stats_ = series_stats
        self.dept_stats_ = dept_stats
        self.store_stats_ = store_stats

        self.global_target_mean_ = float(target_df[self.target_col].mean())
        self.global_target_std_ = float(target_df[self.target_col].std())

        if not np.isfinite(self.global_target_std_) or self.global_target_std_ == 0:
            self.global_target_std_ = 1.0

        # continuous covariate scaling
        candidate_continuous_cols = (
            self.static_real_cols + self.continuous_covariate_cols
        )

        self.continuous_cols_ = [
            col for col in candidate_continuous_cols if col in df.columns
        ]

        self.continuous_stats_ = {}

        for col in self.continuous_cols_:
            values = pd.to_numeric(df[col], errors="coerce")

            mean = float(values.mean())
            std = float(values.std())

            if not np.isfinite(mean):
                mean = 0.0

            if not np.isfinite(std) or std == 0:
                std = 1.0

            self.continuous_stats_[col] = {"mean": mean, "std": std}

        self.binary_cols_ = [
            col for col in self.binary_covariate_cols if col in df.columns
        ]

        self.static_cat_cols_ = [
            f"{col}_id" for col in self.categorical_cols if col in self.category_maps_
        ]

        self.static_real_cols_ = [
            f"{col}_scaled" for col in self.static_real_cols if col in self.continuous_cols_
        ]

        self.known_future_real_cols_ = []

        for col in self.continuous_covariate_cols:
            if col in self.continuous_cols_:
                self.known_future_real_cols_.append(f"{col}_scaled")

        for col in self.binary_cols_:
            self.known_future_real_cols_.append(col)

        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform train, validation or test dataframe.

        If Weekly_Sales exists, creates Weekly_Sales_scaled.
        If Weekly_Sales is absent or NaN, leaves target_scaled as NaN.
        """
        if not self.is_fitted_:
            raise RuntimeError("Call fit(...) before transform(...).")

        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"])

        self._require_columns(df, self.series_cols)

        # internal grouping key
        df["series_id"] = (
            df["Store"].astype(str) + "_" + df["Dept"].astype(str)
        )

        # Categorical IDs (later used for embedding layers instead of OHE)
        # unknown categories become 0
        for col, mapping in self.category_maps_.items():
            if col not in df.columns:
                raise ValueError(f"Missing categorical column during transform: {col}")

            df[f"{col}_id"] = df[col].map(mapping).fillna(0).astype(np.int64)

        # attach target scaling statistics
        df = df.merge(self.series_stats_, on=self.series_cols, how="left")
        df = df.merge(self.dept_stats_, on="Dept", how="left")
        df = df.merge(self.store_stats_, on="Store", how="left")

        # fallbacks
        df["target_mean"] = (
            df["series_target_mean"]
            .fillna(df["dept_target_mean"])
            .fillna(df["store_target_mean"])
            .fillna(self.global_target_mean_)
        )

        df["target_std"] = (
            df["series_target_std"]
            .fillna(df["dept_target_std"])
            .fillna(df["store_target_std"])
            .fillna(self.global_target_std_)
        )

        df["target_std"] = df["target_std"].replace(0, self.global_target_std_)
        df["target_std"] = df["target_std"].fillna(self.global_target_std_)

        helper_cols = [
            "series_target_mean",
            "series_target_std",
            "dept_target_mean",
            "dept_target_std",
            "store_target_mean",
            "store_target_std",
        ]

        df = df.drop(columns=[col for col in helper_cols if col in df.columns])

        # scale target if present
        scaled_target_col = f"{self.target_col}_scaled"

        if self.target_col in df.columns:
            df[scaled_target_col] = np.nan
            mask = df[self.target_col].notna()

            df.loc[mask, scaled_target_col] = (
                (df.loc[mask, self.target_col] - df.loc[mask, "target_mean"])
                / df.loc[mask, "target_std"]
            )
        else:
            df[scaled_target_col] = np.nan

        # scale continuous covariates
        for col, stats in self.continuous_stats_.items():
            if col not in df.columns:
                df[col] = np.nan

            values = pd.to_numeric(df[col], errors="coerce").fillna(stats["mean"])
            df[f"{col}_scaled"] = (values - stats["mean"]) / stats["std"]

        # clean binary covariates
        for col in self.binary_cols_:
            if col not in df.columns:
                df[col] = 0.0

            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(float)

        return df.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)

    def inverse_transform_target(
        self,
        y_scaled: np.ndarray,
        target_mean: np.ndarray,
        target_std: np.ndarray,
    ) -> np.ndarray:
        return y_scaled * target_std + target_mean

    def get_dataset_columns(self) -> dict[str, list[str] | str]:
        if not self.is_fitted_:
            raise RuntimeError("Call fit(...) before get_dataset_columns().")

        return {
            "target_col": f"{self.target_col}_scaled",
            "series_col": "series_id",
            "static_cat_cols": self.static_cat_cols_,
            "static_real_cols": self.static_real_cols_,
            "known_future_real_cols": self.known_future_real_cols_,
        }

    @staticmethod
    def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
        missing = [col for col in columns if col not in df.columns]

        if missing:
            raise ValueError(f"Missing required columns: {missing}")