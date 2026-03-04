import pandas as pd

def remove_nulls(df):
    return df.dropna()

def normalise_columns(df):
    df.columns = df.columns.str.lower().str.strip()
    return df