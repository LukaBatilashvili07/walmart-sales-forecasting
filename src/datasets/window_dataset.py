from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class WalmartTrainingWindowDataset(Dataset):
    """
    Sliding-window dataset for neural training.

    Each sample is:

        past_target          = previous context_length weeks
        future_target        = next prediction_length weeks
        past_known_reals     = covariates for the past window
        future_known_reals   = covariates for the prediction window
        static_categoricals  = Store_id, Dept_id, Type_id
        static_reals         = Size_scaled, etc.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        context_length: int = 52,
        prediction_length: int = 39,
        target_col: str = "Weekly_Sales_scaled",
        series_col: str = "series_id",
        static_cat_cols: list[str] | tuple[str, ...] = (),
        static_real_cols: list[str] | tuple[str, ...] = (),
        known_future_real_cols: list[str] | tuple[str, ...] = (),
        drop_nan_targets: bool = True,
    ):
        self.df = df.copy()
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.target_col = target_col
        self.series_col = series_col
        self.static_cat_cols = list(static_cat_cols)
        self.static_real_cols = list(static_real_cols)
        self.known_future_real_cols = list(known_future_real_cols)
        self.drop_nan_targets = drop_nan_targets

        self._require_columns(
            self.df,
            [self.series_col, "Date", self.target_col]
            + self.static_cat_cols
            + self.static_real_cols
            + self.known_future_real_cols
            + ["target_mean", "target_std"],
        )

        # group by series_id for efficient indexing
        self.groups: list[pd.DataFrame] = []

        # list of (group_idx, start_idx) for each valid window
        self.samples: list[tuple[int, int]] = []

        self._build_index()

    def _build_index(self) -> None:
        total_length = self.context_length + self.prediction_length

        for _, group in self.df.groupby(self.series_col, sort=False):
            group = group.sort_values("Date").reset_index(drop=True)

            # skip if not enough
            if len(group) < total_length: 
                continue

            group_idx = len(self.groups)
            self.groups.append(group)

            target_values = pd.to_numeric(
                group[self.target_col],
                errors="coerce",
            ).to_numpy(dtype=np.float32)

            for start in range(0, len(group) - total_length + 1):
                context_end = start + self.context_length
                pred_end = context_end + self.prediction_length

                if self.drop_nan_targets:
                    target_window = target_values[start:pred_end]

                    if np.isnan(target_window).any():
                        continue

                self.samples.append((group_idx, start))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        group_idx, start = self.samples[idx]

        group = self.groups[group_idx]

        context_end = start + self.context_length
        pred_end = context_end + self.prediction_length

        past_slice = group.iloc[start:context_end]
        future_slice = group.iloc[context_end:pred_end]

        # arbitrary, statics are same for all rows in the window
        static_row = group.iloc[context_end - 1]

        past_target = self._series_to_tensor(past_slice[self.target_col])
        future_target = self._series_to_tensor(future_slice[self.target_col])

        past_known_reals = self._frame_to_float_tensor(
            past_slice,
            self.known_future_real_cols,
        )

        future_known_reals = self._frame_to_float_tensor(
            future_slice,
            self.known_future_real_cols,
        )

        static_categoricals = self._row_to_long_tensor(
            static_row,
            self.static_cat_cols,
        )

        static_reals = self._row_to_float_tensor(
            static_row,
            self.static_real_cols,
        )

        target_mean = torch.tensor(
            float(static_row["target_mean"]),
            dtype=torch.float32,
        )

        target_std = torch.tensor(
            float(static_row["target_std"]),
            dtype=torch.float32,
        )

        return {
            "past_target": past_target,
            "future_target": future_target,
            "past_known_reals": past_known_reals,
            "future_known_reals": future_known_reals,
            "static_categoricals": static_categoricals,
            "static_reals": static_reals,
            "target_mean": target_mean,
            "target_std": target_std,
        }

    @staticmethod
    def _series_to_tensor(series: pd.Series) -> torch.Tensor:
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float32)
        return torch.tensor(values, dtype=torch.float32)

    @staticmethod
    def _frame_to_float_tensor(df: pd.DataFrame, cols: list[str]) -> torch.Tensor:
        if not cols:
            return torch.empty((len(df), 0), dtype=torch.float32)

        values = df[cols].to_numpy(dtype=np.float32)
        return torch.tensor(values, dtype=torch.float32)

    @staticmethod
    def _row_to_float_tensor(row: pd.Series, cols: list[str]) -> torch.Tensor:
        if not cols:
            return torch.empty((0,), dtype=torch.float32)

        values = row[cols].to_numpy(dtype=np.float32)
        return torch.tensor(values, dtype=torch.float32)

    @staticmethod
    def _row_to_long_tensor(row: pd.Series, cols: list[str]) -> torch.Tensor:
        if not cols:
            return torch.empty((0,), dtype=torch.long)

        values = row[cols].to_numpy(dtype=np.int64)
        return torch.tensor(values, dtype=torch.long)

    @staticmethod
    def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
        missing = [col for col in columns if col not in df.columns]

        if missing:
            raise ValueError(f"Missing required columns: {missing}")


class WalmartForecastWindowDataset(Dataset):

    """
    Direct forecasting dataset for validation/test.

    It creates one forecast sample per Store-Dept series in future_df.

    For known Store-Dept pairs:
        past_target = last context_length known sales from history_df

    For unseen Store-Dept pairs:
        past_target = zeros in scaled space
        This means "use the fallback average level" after inverse scaling.
    """

    def __init__(
        self,
        history_df: pd.DataFrame,
        future_df: pd.DataFrame,
        context_length: int = 52,
        prediction_length: int = 39,
        target_col: str = "Weekly_Sales_scaled",
        series_col: str = "series_id",
        static_cat_cols: list[str] | tuple[str, ...] = (),
        static_real_cols: list[str] | tuple[str, ...] = (),
        known_future_real_cols: list[str] | tuple[str, ...] = (),
    ):
        self.history_df = history_df.copy()
        self.future_df = future_df.copy()

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.target_col = target_col
        self.series_col = series_col
        self.static_cat_cols = list(static_cat_cols)
        self.static_real_cols = list(static_real_cols)
        self.known_future_real_cols = list(known_future_real_cols)

        self._require_columns(
            self.history_df,
            [self.series_col, "Date", self.target_col]
            + self.known_future_real_cols,
        )

        self._require_columns(
            self.future_df,
            [self.series_col, "Store", "Dept", "Date"]
            + self.static_cat_cols
            + self.static_real_cols
            + self.known_future_real_cols
            + ["target_mean", "target_std"],
        )

        self.history_groups = {
            series_id: group.sort_values("Date").reset_index(drop=True)
            for series_id, group in self.history_df.groupby(self.series_col, sort=False)
        }

        self.future_groups: list[pd.DataFrame] = []
        self._build_index()

    def _build_index(self) -> None:
        for series_id, group in self.future_df.groupby(self.series_col, sort=False):
            group = group.sort_values("Date").reset_index(drop=True)

            if len(group) != self.prediction_length:
                raise ValueError(
                    f"Expected {self.prediction_length} future rows for series_id={series_id}, "
                    f"got {len(group)}."
                )

            self.future_groups.append(group)

    def __len__(self) -> int:
        return len(self.future_groups)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        future_group = self.future_groups[idx]
        series_id = future_group[self.series_col].iloc[0]

        history_group = self.history_groups.get(series_id)

        if history_group is None:
            history_group = pd.DataFrame(columns=self.history_df.columns)

        past_target = self._last_context_vector(
            history_group,
            self.target_col,
            self.context_length,
        )

        past_known_reals = self._last_context_matrix(
            history_group,
            self.known_future_real_cols,
            self.context_length,
        )

        future_known_reals = self._frame_to_float_tensor(
            future_group,
            self.known_future_real_cols,
        )

        static_row = future_group.iloc[0]

        static_categoricals = self._row_to_long_tensor(
            static_row,
            self.static_cat_cols,
        )

        static_reals = self._row_to_float_tensor(
            static_row,
            self.static_real_cols,
        )

        if self.target_col in future_group.columns:
            future_target = self._series_to_tensor(future_group[self.target_col])
        else:
            future_target = torch.full(
                (self.prediction_length,),
                float("nan"),
                dtype=torch.float32,
            )

        target_mean = torch.tensor(
            float(static_row["target_mean"]),
            dtype=torch.float32,
        )

        target_std = torch.tensor(
            float(static_row["target_std"]),
            dtype=torch.float32,
        )

        store = torch.tensor(int(static_row["Store"]), dtype=torch.long)
        dept = torch.tensor(int(static_row["Dept"]), dtype=torch.long)

        return {
            "past_target": past_target,
            "future_target": future_target,
            "past_known_reals": past_known_reals,
            "future_known_reals": future_known_reals,
            "static_categoricals": static_categoricals,
            "static_reals": static_reals,
            "target_mean": target_mean,
            "target_std": target_std,
            "store": store,
            "dept": dept,
        }

    def get_future_index(self) -> pd.DataFrame:
        rows = []

        for group in self.future_groups:
            rows.append(group[["Store", "Dept", "Date"]].copy())

        if not rows:
            return pd.DataFrame(columns=["Store", "Dept", "Date"])

        return pd.concat(rows, axis=0).reset_index(drop=True)

    @staticmethod
    def _last_context_vector(
        df: pd.DataFrame,
        col: str,
        context_length: int,
    ) -> torch.Tensor:
        if df.empty or col not in df.columns:
            return torch.zeros(context_length, dtype=torch.float32)

        values = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(
            dtype=np.float32
        )

        if len(values) >= context_length:
            values = values[-context_length:]
        else:
            pad = np.zeros(context_length - len(values), dtype=np.float32)
            values = np.concatenate([pad, values], axis=0)

        return torch.tensor(values, dtype=torch.float32)

    @staticmethod
    def _last_context_matrix(
        df: pd.DataFrame,
        cols: list[str],
        context_length: int,
    ) -> torch.Tensor:
        if not cols:
            return torch.empty((context_length, 0), dtype=torch.float32)

        if df.empty:
            return torch.zeros((context_length, len(cols)), dtype=torch.float32)

        values = df[cols].tail(context_length).to_numpy(dtype=np.float32)

        if len(values) < context_length:
            pad = np.zeros(
                (context_length - len(values), len(cols)),
                dtype=np.float32,
            )
            values = np.vstack([pad, values])

        return torch.tensor(values, dtype=torch.float32)

    @staticmethod
    def _series_to_tensor(series: pd.Series) -> torch.Tensor:
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float32)
        return torch.tensor(values, dtype=torch.float32)

    @staticmethod
    def _frame_to_float_tensor(df: pd.DataFrame, cols: list[str]) -> torch.Tensor:
        if not cols:
            return torch.empty((len(df), 0), dtype=torch.float32)

        values = df[cols].to_numpy(dtype=np.float32)
        return torch.tensor(values, dtype=torch.float32)

    @staticmethod
    def _row_to_float_tensor(row: pd.Series, cols: list[str]) -> torch.Tensor:
        if not cols:
            return torch.empty((0,), dtype=torch.float32)

        values = row[cols].to_numpy(dtype=np.float32)
        return torch.tensor(values, dtype=torch.float32)

    @staticmethod
    def _row_to_long_tensor(row: pd.Series, cols: list[str]) -> torch.Tensor:
        if not cols:
            return torch.empty((0,), dtype=torch.long)

        values = row[cols].to_numpy(dtype=np.int64)
        return torch.tensor(values, dtype=torch.long)

    @staticmethod
    def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
        missing = [col for col in columns if col not in df.columns]

        if missing:
            raise ValueError(f"Missing required columns: {missing}")