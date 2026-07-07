import pandas as pd
import numpy as np


class WalmartBasePreprocessor:
    """
    Shared preprocessing steps for all model types (tree, neural, statistical):
    merge, missing value handling, calendar and holiday features.
    """

    HOLIDAY_DATES = {
        "IsSuperBowl": ["2010-02-12", "2011-02-11", "2012-02-10", "2013-02-08"],
        "IsLaborDay": ["2010-09-10", "2011-09-09", "2012-09-07", "2013-09-06"],
        "IsThanksgiving": ["2010-11-26", "2011-11-25", "2012-11-23", "2013-11-29"],
        "IsChristmas": ["2010-12-31", "2011-12-30", "2012-12-28", "2013-12-27"],
    }

    MARKDOWN_COLS = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
    MARKDOWN_START_DATE = pd.Timestamp("2011-11-11")

    def __init__(self):
        self.stores_df = None
        self.features_df = None
        self._is_fitted = False

    def fit(self, stores_df: pd.DataFrame, features_df: pd.DataFrame):
        self.stores_df = stores_df.copy()

        self.features_df = features_df.copy()
        self.features_df["Date"] = pd.to_datetime(self.features_df["Date"])

        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("Call fit() before transform().")

        out = df.copy()
        out["Date"] = pd.to_datetime(out["Date"])

        out = self._merge(out)
        out = self._handle_missing_values(out)
        out = self._add_calendar_features(out)
        out = self._add_holiday_features(out)

        return out

    def _merge(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)

        merged = df.merge(self.stores_df, on="Store", how="left")
        merged = merged.merge(
            self.features_df, on=["Store", "Date", "IsHoliday"], how="left"
        )

        assert len(merged) == before, "Merge changed row count, check join keys."
        return merged

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["Store", "Dept", "Date"])

        # CPI/Unemployment ffill within each store
        for col in ["CPI", "Unemployment"]:
            df[col] = df.groupby("Store")[col].ffill()

        # MarkDown missing indicators before 0-fill, plus a NaN-preserved
        # "raw" version for tree models that can natively handle missingness
        for col in self.MARKDOWN_COLS:
            df[f"{col}_was_missing"] = df[col].isna().astype(int)
            df[f"{col}_raw"] = df[col].where(df[f"{col}_was_missing"] == 0, np.nan)

        df[self.MARKDOWN_COLS] = df[self.MARKDOWN_COLS].fillna(0)

        # Aggregate markdown features
        df["total_markdown"] = df[self.MARKDOWN_COLS].sum(axis=1)
        df["abs_total_markdown"] = df[self.MARKDOWN_COLS].abs().sum(axis=1)
        df["positive_markdown_sum"] = df[self.MARKDOWN_COLS].clip(lower=0).sum(axis=1)
        df["negative_markdown_sum"] = df[self.MARKDOWN_COLS].clip(upper=0).sum(axis=1)
        df["has_markdown_signal"] = (df["abs_total_markdown"] > 0).astype(int)
        df["markdown_missing_count"] = df[
            [f"{c}_was_missing" for c in self.MARKDOWN_COLS]
        ].sum(axis=1)
        df["markdown_available_period"] = (
            df["Date"] >= self.MARKDOWN_START_DATE
        ).astype(int)

        return df

    def _add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)

        df["Week_sin"] = np.sin(2 * np.pi * df["WeekOfYear"] / 52)
        df["Week_cos"] = np.cos(2 * np.pi * df["WeekOfYear"] / 52)

        return df

    def _add_holiday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for flag_name, dates in self.HOLIDAY_DATES.items():
            df[flag_name] = df["Date"].isin(pd.to_datetime(dates)).astype(int)

        return df