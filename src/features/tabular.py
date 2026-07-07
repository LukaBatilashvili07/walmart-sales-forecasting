import pandas as pd
import numpy as np


class WalmartTabularFeatureEngineer:
    """
    Tree-model-specific feature engineering: Type encoding, lag/rolling features.

    Uses a leakage-safe fit/transform_train/transform_future interface instead of
    a plain fit/transform, because lag/rolling features and the dept-level fallback
    mean depend on Weekly_Sales history — computing them on data that includes the
    future/validation target would leak information backwards in time.
    """

    TYPE_MAPPING = {"A": 0, "B": 1, "C": 2}
    LAG_WEEKS = [1, 4, 52]

    def __init__(self):
        self.dept_avg_sales = None
        self.history_df = None
        self._is_fitted = False

    def fit(self, history_df: pd.DataFrame):
        # history_df must contain Store, Dept, Date, Weekly_Sales and only rows whose Weekly_Sales is known.
        required_cols = {"Store", "Dept", "Date", "Weekly_Sales"}
        missing = required_cols - set(history_df.columns)
        if missing:
            raise ValueError(f"history_df is missing columns: {missing}")

        self.history_df = history_df[["Store", "Dept", "Date", "Weekly_Sales"]].copy()
        self.history_df["Date"] = pd.to_datetime(self.history_df["Date"])

        # Dept-level fallback average, only from known history
        self.dept_avg_sales = self.history_df.groupby("Dept")["Weekly_Sales"].mean()

        self._is_fitted = True
        return self

    def transform_train(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform the same data the engineer was fit on (the train split)."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() before transform_train()")

        out = df.copy()
        out = self._add_type_encoding(out)
        out = self._add_lag_rolling(out, future_df=None)
        return out

    def transform_future(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform validation/test data whose Weekly_Sales is unknown (or must be
        treated as unknown). Lag/rolling features are computed using the history
        passed to fit(), never using df's own Weekly_Sales.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before transform_future()")

        out = df.copy()
        out = self._add_type_encoding(out)
        out = self._add_lag_rolling(out, future_df=out)
        return out

    def _add_type_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        df["Type_encoded"] = df["Type"].map(self.TYPE_MAPPING)

        dummies = pd.get_dummies(df["Type"], prefix="Type")
        for col in ["Type_A", "Type_B", "Type_C"]:
            df[col] = dummies[col].astype(int) if col in dummies.columns else 0

        return df

    def _add_lag_rolling(self, df: pd.DataFrame, future_df) -> pd.DataFrame:
        before = len(df)

        if future_df is None:
            # Training case: lag/rolling computed on history's own sequence
            combined = self.history_df.copy()
            combined["is_target"] = 1
        else:
            # Future case: history + future timeline, future's Weekly_Sales blanked out
            future_part = future_df[["Store", "Dept", "Date"]].copy()
            future_part["Weekly_Sales"] = np.nan
            future_part["is_target"] = 1

            history_part = self.history_df.copy()
            history_part["is_target"] = 0

            combined = pd.concat([history_part, future_part], axis=0)

        combined = combined.sort_values(["Store", "Dept", "Date"])

        for lag in self.LAG_WEEKS:
            combined[f"Sales_lag_{lag}"] = (
                combined.groupby(["Store", "Dept"])["Weekly_Sales"].shift(lag)
            )

        combined["_sales_shifted"] = (
            combined.groupby(["Store", "Dept"])["Weekly_Sales"].shift(1)
        )
        combined["Sales_roll_mean_4"] = (
            combined.groupby(["Store", "Dept"])["_sales_shifted"]
            .transform(lambda s: s.rolling(4, min_periods=1).mean())
        )
        combined["Sales_roll_std_4"] = (
            combined.groupby(["Store", "Dept"])["_sales_shifted"]
            .transform(lambda s: s.rolling(4, min_periods=1).std())
        )
        combined["Sales_roll_mean_12"] = (
            combined.groupby(["Store", "Dept"])["_sales_shifted"]
            .transform(lambda s: s.rolling(12, min_periods=1).mean())
        )

        feature_cols = [f"Sales_lag_{l}" for l in self.LAG_WEEKS] + [
            "Sales_roll_mean_4", "Sales_roll_std_4", "Sales_roll_mean_12"
        ]

        result_features = combined.loc[
            combined["is_target"] == 1, ["Store", "Dept", "Date"] + feature_cols
        ]

        out = df.merge(result_features, on=["Store", "Dept", "Date"], how="left")

        # Dept-level fallback for any remaining NaN
        mean_cols = [f"Sales_lag_{l}" for l in self.LAG_WEEKS] + [
            "Sales_roll_mean_4", "Sales_roll_mean_12"
        ]
        for col in mean_cols:
            out[col] = out[col].fillna(out["Dept"].map(self.dept_avg_sales))

        out["Sales_roll_std_4"] = out["Sales_roll_std_4"].fillna(0)

        assert len(out) == before, "Lag/rolling merge changed row count."
        return out