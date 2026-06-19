import pandas as pd
import os

def load_raw_data():
    file_path = "dataset/household_power_consumption.txt"

    df = pd.read_csv(
        file_path,
        sep=';',
        low_memory=False,
        na_values=['?', '']
    )

    print(f" Raw dataset loaded: {len(df):,} records, {len(df.columns)} columns")
    print(f" (Same file Module 1 used — now we clean it)")


  

    # Store missing counts BEFORE any cleaning (for before/after chart in step11)
    missing_before = df.isnull().sum()
    print(f" Module 1 found {missing_before.sum():,} total missing values — now we fix them.")

    return df, missing_before