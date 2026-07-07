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