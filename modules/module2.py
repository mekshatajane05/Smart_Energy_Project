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

import pandas as pd

def parse_datetime(df):
    print("\n[STEP 2] Merging Date + Time columns into Datetime index...")
    print(f" Module 1 showed Date range: {df['Date'].min()} → {df['Date'].max()}")

    df['Datetime'] = pd.to_datetime(
        df['Date'] + ' ' + df['Time'],
        format='%d/%m/%Y %H:%M:%S',
        errors='coerce'
    )

    df.drop(columns=['Date', 'Time'], inplace=True)
    df.set_index('Datetime', inplace=True)
    df.sort_index(inplace=True)

    nat_count = df.index.isna().sum()
    print(f" Unparseable timestamps: {nat_count}")
    if nat_count > 0:
        df = df[~df.index.isna()]
        print(f" Removed {nat_count} bad timestamp rows.")

    print(f" Datetime index set: {df.index.min()} → {df.index.max()}")
    return df