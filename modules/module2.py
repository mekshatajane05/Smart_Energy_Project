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

def convert_to_numeric(df, numeric_cols):
    print("\n[STEP 3] Converting columns to numeric (Module 1 flagged these as 'object')...")

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    print(" All measurement columns converted to float.")
    return df

def remove_duplicates(df):
    print("\n[STEP 4] Checking for duplicate timestamps...")

    duplicates = df.index.duplicated().sum()
    print(f" Duplicate timestamps found: {duplicates}")

    if duplicates > 0:
        df = df[~df.index.duplicated(keep='first')]
        print(f" Duplicates removed. Remaining: {len(df):,} records")
    else:
        print(" No duplicates. Clean!")

    return df
def fill_missing_values(df, numeric_cols, missing_before):
    print("\n[STEP 5] Filling missing values...")
    print(f" Before (from Module 1's analysis): {missing_before[numeric_cols].sum():,} missing cells")

    df.ffill(inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)

    missing_after = df.isnull().sum()
    print(f" After cleaning: {missing_after.sum()} missing cells remaining")

    return df, missing_after