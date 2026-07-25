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
    LAG_WEEKS = [1, 2, 3, 4, 52]

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

        self.store_avg_sales = self.history_df.groupby("Store")["Weekly_Sales"].mean()
        self.store_dept_avg_sales = (
            self.history_df.groupby(["Store", "Dept"])["Weekly_Sales"].mean()
        )
        self.global_avg_sales = self.history_df["Weekly_Sales"].mean()

        # Precompute same-week-across-years aggregates for the seasonal average feature.
        hist_calendar = self.history_df.copy()
        hist_calendar["WeekOfYear"] = hist_calendar["Date"].dt.isocalendar().week.astype(int)
        hist_calendar["Year"] = hist_calendar["Date"].dt.isocalendar().year

        self._seasonal_group = (
            hist_calendar.groupby(["Store", "Dept", "WeekOfYear"])["Weekly_Sales"]
            .agg(_grp_sum="sum", _grp_count="count")
            .reset_index()
        )

        self._seasonal_group_year = (
            hist_calendar.groupby(["Store", "Dept", "WeekOfYear", "Year"])["Weekly_Sales"]
            .sum()
            .rename("_own_year_sum")
            .reset_index()
        )
        self._seasonal_group_year["_own_year_present"] = 1

        self._is_fitted = True
        return self

    def transform_train(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform the same data the engineer was fit on (the train split)."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() before transform_train()")

        out = df.copy()
        out = self._add_type_encoding(out)
        out = self._add_lag_rolling(out, future_df=None)
        out = self._add_seasonal_avg(out)
        out = self._add_target_encoding(out)
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
        out = self._add_seasonal_avg(out)
        out = self._add_target_encoding(out)
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
        combined["Sales_diff_1"] = (
            combined.groupby(["Store", "Dept"])["Weekly_Sales"].diff(1)
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
            "Sales_roll_mean_4", "Sales_roll_std_4", "Sales_roll_mean_12", "Sales_diff_1"
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
        out["Sales_diff_1"] = out["Sales_diff_1"].fillna(0)

        assert len(out) == before, "Lag/rolling merge changed row count."
        return out

    def _add_seasonal_avg(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Same-week-across-years seasonal average, leave-one-out safe.
        """
        before = len(df)
        out = df.copy()

        if "WeekOfYear" not in out.columns:
            out["WeekOfYear"] = pd.to_datetime(out["Date"]).dt.isocalendar().week.astype(int)
        if "Year" not in out.columns:
            out["Year"] = pd.to_datetime(out["Date"]).dt.year

        out = out.merge(self._seasonal_group, on=["Store", "Dept", "WeekOfYear"], how="left")
        out = out.merge(
            self._seasonal_group_year, on=["Store", "Dept", "WeekOfYear", "Year"], how="left"
        )

        out["_own_year_sum"] = out["_own_year_sum"].fillna(0.0)
        out["_own_year_present"] = out["_own_year_present"].fillna(0)

        adj_sum = out["_grp_sum"] - out["_own_year_present"] * out["_own_year_sum"]
        adj_count = out["_grp_count"] - out["_own_year_present"]

        out["Sales_seasonal_avg"] = np.where(adj_count > 0, adj_sum / adj_count, np.nan)
        out["Sales_seasonal_avg"] = out["Sales_seasonal_avg"].fillna(out["Dept"].map(self.dept_avg_sales))

        out = out.drop(columns=["_grp_sum", "_grp_count", "_own_year_sum", "_own_year_present"])

        assert len(out) == before, "Seasonal avg merge changed row count."
        return out

    def _add_target_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        store_dept_key = list(zip(out["Store"], out["Dept"]))
        out["StoreDept_avg_sales"] = pd.Series(store_dept_key, index=out.index).map(
            self.store_dept_avg_sales.to_dict()
        )
        out["Store_avg_sales"] = out["Store"].map(self.store_avg_sales)
        out["Dept_avg_sales"] = out["Dept"].map(self.dept_avg_sales)

        # fallback chain: StoreDept -> Dept -> global (ახალი store-dept კომბინაცია training-ში რომ არ ყოფილიყო)
        out["StoreDept_avg_sales"] = out["StoreDept_avg_sales"].fillna(out["Dept_avg_sales"])
        out["StoreDept_avg_sales"] = out["StoreDept_avg_sales"].fillna(self.global_avg_sales)
        out["Store_avg_sales"] = out["Store_avg_sales"].fillna(self.global_avg_sales)
        out["Dept_avg_sales"] = out["Dept_avg_sales"].fillna(self.global_avg_sales)

        return out