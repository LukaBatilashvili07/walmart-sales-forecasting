from __future__ import annotations
import math

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class WalmartTrainingWindowDataset(Dataset):
    """
    Sliding-window dataset for neural training.

    Adapted for faster sliding window, panda -> numpy conversion once.

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
        self.groups: list[dict[str, np.ndarray | float]] = []

        # list of (group_idx, start_idx) for each valid window
        self.samples: list[tuple[int, int]] = []

        self._build_index()

    def _build_index(self) -> None:
        total_length = self.context_length + self.prediction_length

        self.df["Date"] = pd.to_datetime(self.df["Date"])
        self.df = self.df.sort_values([self.series_col, "Date"]).reset_index(drop=True)

        for _, group in self.df.groupby(self.series_col, sort=False):
            group = group.sort_values("Date").reset_index(drop=True)

            # skip if not enough
            if len(group) < total_length:
                continue

            target_values = pd.to_numeric(
                group[self.target_col],
                errors="coerce",
            ).to_numpy(dtype=np.float32)

            if self.known_future_real_cols:
                known_reals = group[self.known_future_real_cols].to_numpy(dtype=np.float32)
            else:
                known_reals = np.empty((len(group), 0), dtype=np.float32)

            static_row = group.iloc[0]

            if self.static_cat_cols:
                static_cats = static_row[self.static_cat_cols].to_numpy(dtype=np.int64)
            else:
                static_cats = np.empty((0,), dtype=np.int64)

            if self.static_real_cols:
                static_reals = static_row[self.static_real_cols].to_numpy(dtype=np.float32)
            else:
                static_reals = np.empty((0,), dtype=np.float32)

            target_mean = float(static_row["target_mean"])
            target_std = float(static_row["target_std"])

            group_idx = len(self.groups)

            self.groups.append(
                {
                    "target": target_values,
                    "known_reals": known_reals,
                    "static_cats": static_cats,
                    "static_reals": static_reals,
                    "target_mean": target_mean,
                    "target_std": target_std,
                }
            )

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
        
        target = group["target"]
        known_reals = group["known_reals"]

        past_target = target[start:context_end]
        future_target = target[context_end:pred_end]

        past_known_reals = known_reals[start:context_end]
        future_known_reals = known_reals[context_end:pred_end]

        return {
            "past_target": torch.from_numpy(past_target).float(),
            "future_target": torch.from_numpy(future_target).float(),
            "past_known_reals": torch.from_numpy(past_known_reals).float(),
            "future_known_reals": torch.from_numpy(future_known_reals).float(),
            "static_categoricals": torch.from_numpy(group["static_cats"]).long(),
            "static_reals": torch.from_numpy(group["static_reals"]).float(),
            "target_mean": torch.tensor(group["target_mean"], dtype=torch.float32),
            "target_std": torch.tensor(group["target_std"], dtype=torch.float32),
        }

    @staticmethod
    def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
        missing = [col for col in columns if col not in df.columns]

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

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
    
class WalmartPrecomputedTrainingWindowDataset:
    """
    General precomputed window dataset for neural training.

    Works for:
    - target-only models: DLinear, N-BEATS
    - covariate models: DLinear-X, PatchTST, TFT

    It precomputes all 52 -> 39 windows once as tensors.
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
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.target_col = target_col
        self.series_col = series_col
        self.static_cat_cols = list(static_cat_cols)
        self.static_real_cols = list(static_real_cols)
        self.known_future_real_cols = list(known_future_real_cols)
        self.drop_nan_targets = drop_nan_targets

        self._require_columns(
            df,
            [series_col, "Date", target_col]
            + self.static_cat_cols
            + self.static_real_cols
            + self.known_future_real_cols
            + ["target_mean", "target_std"],
        )

        self.tensors = self._build_tensors(df)

    def _build_tensors(self, df: pd.DataFrame) -> dict[str, torch.Tensor]:
        total_length = self.context_length + self.prediction_length

        work = df.copy()
        work["Date"] = pd.to_datetime(work["Date"])
        work = work.sort_values([self.series_col, "Date"]).reset_index(drop=True)

        past_targets = []
        future_targets = []
        past_known_reals = []
        future_known_reals = []
        static_categoricals = []
        static_reals = []
        target_means = []
        target_stds = []

        for _, group in work.groupby(self.series_col, sort=False):
            group = group.sort_values("Date").reset_index(drop=True)

            if len(group) < total_length:
                continue

            target_values = pd.to_numeric(
                group[self.target_col],
                errors="coerce",
            ).to_numpy(dtype=np.float32)

            if self.known_future_real_cols:
                known_values = group[self.known_future_real_cols].to_numpy(dtype=np.float32)
            else:
                known_values = np.empty((len(group), 0), dtype=np.float32)

            static_row = group.iloc[0]

            if self.static_cat_cols:
                static_cat_values = static_row[self.static_cat_cols].to_numpy(dtype=np.int64)
            else:
                static_cat_values = np.empty((0,), dtype=np.int64)

            if self.static_real_cols:
                static_real_values = static_row[self.static_real_cols].to_numpy(dtype=np.float32)
            else:
                static_real_values = np.empty((0,), dtype=np.float32)

            target_mean = float(static_row["target_mean"])
            target_std = float(static_row["target_std"])

            for start in range(0, len(group) - total_length + 1):
                context_end = start + self.context_length
                pred_end = context_end + self.prediction_length

                target_window = target_values[start:pred_end]

                if self.drop_nan_targets and np.isnan(target_window).any():
                    continue

                past_targets.append(target_values[start:context_end])
                future_targets.append(target_values[context_end:pred_end])

                past_known_reals.append(known_values[start:context_end])
                future_known_reals.append(known_values[context_end:pred_end])

                static_categoricals.append(static_cat_values)
                static_reals.append(static_real_values)

                target_means.append(target_mean)
                target_stds.append(target_std)

        if not past_targets:
            raise ValueError("No valid precomputed training windows were created.")

        return {
            "past_target": torch.tensor(np.stack(past_targets), dtype=torch.float32),
            "future_target": torch.tensor(np.stack(future_targets), dtype=torch.float32),
            "past_known_reals": torch.tensor(np.stack(past_known_reals), dtype=torch.float32),
            "future_known_reals": torch.tensor(np.stack(future_known_reals), dtype=torch.float32),
            "static_categoricals": torch.tensor(np.stack(static_categoricals), dtype=torch.long),
            "static_reals": torch.tensor(np.stack(static_reals), dtype=torch.float32),
            "target_mean": torch.tensor(target_means, dtype=torch.float32),
            "target_std": torch.tensor(target_stds, dtype=torch.float32),
        }

    def __len__(self) -> int:
        return self.tensors["past_target"].shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            key: value[idx]
            for key, value in self.tensors.items()
        }

    @staticmethod
    def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
        missing = [col for col in columns if col not in df.columns]

        if missing:
            raise ValueError(f"Missing required columns: {missing}")


class FastTensorDataLoader:
    """
    Fast mini-batch iterator over precomputed tensors.

    Avoids PyTorch DataLoader's per-sample __getitem__ and dictionary collation.
    """

    def __init__(
        self,
        tensors: dict[str, torch.Tensor],
        batch_size: int = 256,
        shuffle: bool = False,
        drop_last: bool = False,
    ):
        self.tensors = tensors
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        lengths = {key: value.shape[0] for key, value in tensors.items()}
        unique_lengths = set(lengths.values())

        if len(unique_lengths) != 1:
            raise ValueError(f"All tensors must have same first dimension. Got: {lengths}")

        self.n_samples = next(iter(unique_lengths))

    def __len__(self) -> int:
        if self.drop_last:
            return self.n_samples // self.batch_size

        return math.ceil(self.n_samples / self.batch_size)

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(self.n_samples)
        else:
            indices = torch.arange(self.n_samples)

        for start in range(0, self.n_samples, self.batch_size):
            end = start + self.batch_size

            if end > self.n_samples and self.drop_last:
                break

            batch_idx = indices[start:end]

            yield {
                key: value[batch_idx]
                for key, value in self.tensors.items()
            }

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
        
class WalmartPrecomputedForecastWindowDataset:
    """
    General precomputed forecast dataset for validation/test-style forecasting.

    Creates one 52 -> 39 forecast sample per Store-Dept future group.

    Works for:
    - target-only models: DLinear, N-BEATS
    - covariate models: DLinear-X, PatchTST, TFT
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
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.target_col = target_col
        self.series_col = series_col
        self.static_cat_cols = list(static_cat_cols)
        self.static_real_cols = list(static_real_cols)
        self.known_future_real_cols = list(known_future_real_cols)

        self._require_columns(
            history_df,
            [series_col, "Date", target_col] + self.known_future_real_cols,
        )

        self._require_columns(
            future_df,
            [series_col, "Store", "Dept", "Date"]
            + self.static_cat_cols
            + self.static_real_cols
            + self.known_future_real_cols
            + ["target_mean", "target_std"],
        )

        self.tensors, self.future_index = self._build_tensors(history_df, future_df)

    def _build_tensors(
        self,
        history_df: pd.DataFrame,
        future_df: pd.DataFrame,
    ) -> tuple[dict[str, torch.Tensor], pd.DataFrame]:
        history = history_df.copy()
        future = future_df.copy()

        history["Date"] = pd.to_datetime(history["Date"])
        future["Date"] = pd.to_datetime(future["Date"])

        history = history.sort_values([self.series_col, "Date"]).reset_index(drop=True)
        future = future.sort_values([self.series_col, "Date"]).reset_index(drop=True)

        history_groups = {
            series_id: group.sort_values("Date").reset_index(drop=True)
            for series_id, group in history.groupby(self.series_col, sort=False)
        }

        past_targets = []
        future_targets = []
        past_known_reals = []
        future_known_reals = []
        static_categoricals = []
        static_reals = []
        target_means = []
        target_stds = []
        stores = []
        depts = []
        index_rows = []

        for series_id, future_group in future.groupby(self.series_col, sort=False):
            future_group = future_group.sort_values("Date").reset_index(drop=True)

            if len(future_group) != self.prediction_length:
                raise ValueError(
                    f"Expected {self.prediction_length} future rows for "
                    f"series_id={series_id}, got {len(future_group)}."
                )

            history_group = history_groups.get(series_id)

            if history_group is None:
                history_group = pd.DataFrame(columns=history.columns)

            past_target = self._last_context_vector(
                history_group,
                self.target_col,
                self.context_length,
            )

            past_known = self._last_context_matrix(
                history_group,
                self.known_future_real_cols,
                self.context_length,
            )

            if self.known_future_real_cols:
                future_known = future_group[self.known_future_real_cols].to_numpy(
                    dtype=np.float32
                )
                future_known = np.nan_to_num(future_known, nan=0.0)
            else:
                future_known = np.empty(
                    (self.prediction_length, 0),
                    dtype=np.float32,
                )

            static_row = future_group.iloc[0]

            if self.static_cat_cols:
                static_cat = static_row[self.static_cat_cols].to_numpy(dtype=np.int64)
            else:
                static_cat = np.empty((0,), dtype=np.int64)

            if self.static_real_cols:
                static_real = static_row[self.static_real_cols].to_numpy(dtype=np.float32)
                static_real = np.nan_to_num(static_real, nan=0.0)
            else:
                static_real = np.empty((0,), dtype=np.float32)

            if self.target_col in future_group.columns:
                future_target = pd.to_numeric(
                    future_group[self.target_col],
                    errors="coerce",
                ).to_numpy(dtype=np.float32)
            else:
                future_target = np.full(
                    self.prediction_length,
                    np.nan,
                    dtype=np.float32,
                )

            past_targets.append(past_target)
            future_targets.append(future_target)
            past_known_reals.append(past_known)
            future_known_reals.append(future_known)
            static_categoricals.append(static_cat)
            static_reals.append(static_real)
            target_means.append(float(static_row["target_mean"]))
            target_stds.append(float(static_row["target_std"]))
            stores.append(int(static_row["Store"]))
            depts.append(int(static_row["Dept"]))

            index_rows.append(future_group[["Store", "Dept", "Date"]].copy())

        if not past_targets:
            raise ValueError("No precomputed forecast samples were created.")

        tensors = {
            "past_target": torch.tensor(np.stack(past_targets), dtype=torch.float32),
            "future_target": torch.tensor(np.stack(future_targets), dtype=torch.float32),
            "past_known_reals": torch.tensor(np.stack(past_known_reals), dtype=torch.float32),
            "future_known_reals": torch.tensor(np.stack(future_known_reals), dtype=torch.float32),
            "static_categoricals": torch.tensor(np.stack(static_categoricals), dtype=torch.long),
            "static_reals": torch.tensor(np.stack(static_reals), dtype=torch.float32),
            "target_mean": torch.tensor(target_means, dtype=torch.float32),
            "target_std": torch.tensor(target_stds, dtype=torch.float32),
            "store": torch.tensor(stores, dtype=torch.long),
            "dept": torch.tensor(depts, dtype=torch.long),
        }

        future_index = pd.concat(index_rows, axis=0).reset_index(drop=True)

        return tensors, future_index

    def __len__(self) -> int:
        return self.tensors["past_target"].shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            key: value[idx]
            for key, value in self.tensors.items()
        }

    def get_future_index(self) -> pd.DataFrame:
        return self.future_index.copy()

    @staticmethod
    def _last_context_vector(
        df: pd.DataFrame,
        col: str,
        context_length: int,
    ) -> np.ndarray:
        if df.empty or col not in df.columns:
            return np.zeros(context_length, dtype=np.float32)

        values = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(
            dtype=np.float32
        )

        if len(values) >= context_length:
            values = values[-context_length:]
        else:
            pad = np.zeros(context_length - len(values), dtype=np.float32)
            values = np.concatenate([pad, values], axis=0)

        return values.astype(np.float32)

    @staticmethod
    def _last_context_matrix(
        df: pd.DataFrame,
        cols: list[str],
        context_length: int,
    ) -> np.ndarray:
        if not cols:
            return np.empty((context_length, 0), dtype=np.float32)

        if df.empty:
            return np.zeros((context_length, len(cols)), dtype=np.float32)

        values = df[cols].tail(context_length).to_numpy(dtype=np.float32)
        values = np.nan_to_num(values, nan=0.0)

        if len(values) < context_length:
            pad = np.zeros(
                (context_length - len(values), len(cols)),
                dtype=np.float32,
            )
            values = np.vstack([pad, values])

        return values.astype(np.float32)

    @staticmethod
    def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
        missing = [col for col in columns if col not in df.columns]

        if missing:
            raise ValueError(f"Missing required columns: {missing}")