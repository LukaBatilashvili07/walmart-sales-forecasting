import pandas as pd

# Split the dataframe into training and validation sets based on the last n weeks
def last_n_weeks_split(df, n_weeks=39, date_col="Date"):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    unique_dates = sorted(df[date_col].unique())

    if len(unique_dates) <= n_weeks:
        raise ValueError("Not enough dates for requested validation size.")

    valid_dates = unique_dates[-n_weeks:]
    train_dates = unique_dates[:-n_weeks]

    train_part = df[df[date_col].isin(train_dates)].copy()
    valid_part = df[df[date_col].isin(valid_dates)].copy()

    return train_part, valid_part

def calendar_aligned_split(
    df: pd.DataFrame,
    valid_start: str = "2011-11-04",
    valid_end: str = "2012-07-27",
    date_col: str = "Date",
):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    valid_start = pd.Timestamp(valid_start)
    valid_end = pd.Timestamp(valid_end)

    train_part = df[df[date_col] < valid_start].copy()
    valid_part = df[
        (df[date_col] >= valid_start)
        & (df[date_col] <= valid_end)
    ].copy()

    unique_valid_dates = sorted(valid_part[date_col].unique())

    if len(unique_valid_dates) != 39:
        raise ValueError(
            f"Expected 39 validation dates, got {len(unique_valid_dates)}. "
            f"Range: {valid_start.date()} to {valid_end.date()}"
        )

    return train_part, valid_part