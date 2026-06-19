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

def cap_outliers(df, numeric_cols):
    print("\n[STEP 6] Capping outliers using IQR method...")
    print(" (Module 1's df.describe() showed suspicious min/max — now fixing)")

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower=lower, upper=upper)

        print(f"   {col}: {n_outliers:,} outliers capped → bounds [{lower:.3f}, {upper:.3f}]")

    return df

def resample_data(df):
    print("\n[STEP 7] Resampling data (minute → hourly and daily)...")

    agg_dict = {
        'Global_active_power':   'mean',
        'Global_reactive_power': 'mean',
        'Voltage':               'mean',
        'Global_intensity':      'mean',
        'Sub_metering_1':        'sum',
        'Sub_metering_2':        'sum',
        'Sub_metering_3':        'sum'
    }

    df_hourly = df.resample('h').agg(agg_dict)
    df_hourly.dropna(inplace=True)

    df_daily = df.resample('D').agg(agg_dict)
    df_daily.dropna(inplace=True)

    print(f" Raw (minute-level) : {len(df):,} records")
    print(f" Hourly resampled   : {len(df_hourly):,} records  ← used for LSTM (Module 5)")
    print(f" Daily resampled    : {len(df_daily):,} records   ← used for Linear Regression (Module 4)")

    return df_hourly, df_daily

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def scale_features(df_hourly, numeric_cols):
    print("\n[STEP 8] Scaling features to [0, 1] range using MinMaxScaler...")

    scaler = MinMaxScaler()
    df_hourly_scaled = df_hourly.copy()
    df_hourly_scaled[numeric_cols] = scaler.fit_transform(df_hourly[numeric_cols])

    print(f" Scaling done. Value range: [{df_hourly_scaled[numeric_cols].min().min():.2f}, "
          f"{df_hourly_scaled[numeric_cols].max().max():.2f}]")

    # Save scaler params so Module 5 can inverse_transform predictions
    scaler_params = pd.DataFrame({
        'feature':  numeric_cols,
        'data_min': scaler.data_min_,
        'data_max': scaler.data_max_,
        'scale':    scaler.scale_,
        'min':      scaler.min_
    })

    return df_hourly_scaled, scaler_params

def split_data(df_hourly_scaled):
    print("\n[STEP 9] Splitting into Train / Validation / Test (70/15/15)...")

    total     = len(df_hourly_scaled)
    train_end = int(total * 0.70)
    val_end   = int(total * 0.85)

    df_train = df_hourly_scaled.iloc[:train_end]
    df_val   = df_hourly_scaled.iloc[train_end:val_end]
    df_test  = df_hourly_scaled.iloc[val_end:]

    print(f" Total : {total:,} hourly records")
    print(f" Train : {len(df_train):,}  ({df_train.index.min().date()} → {df_train.index.max().date()})")
    print(f" Val   : {len(df_val):,}   ({df_val.index.min().date()} → {df_val.index.max().date()})")
    print(f" Test  : {len(df_test):,}   ({df_test.index.min().date()} → {df_test.index.max().date()})")

    return df_train, df_val, df_test

import os

def save_processed_files(df_hourly, df_daily, df_hourly_scaled,
                          df_train, df_val, df_test, scaler_params):

    output_dir = os.path.join("data", "processed")
    os.makedirs(output_dir, exist_ok=True)

    df_hourly.to_csv(os.path.join(output_dir, "hourly_data.csv"))
    df_daily.to_csv(os.path.join(output_dir, "daily_data.csv"))
    df_hourly_scaled.to_csv(os.path.join(output_dir, "hourly_scaled.csv"))
    df_train.to_csv(os.path.join(output_dir, "train_set.csv"))
    df_val.to_csv(os.path.join(output_dir, "val_set.csv"))
    df_test.to_csv(os.path.join(output_dir, "test_set.csv"))
    scaler_params.to_csv(os.path.join(output_dir, "scaler_params.csv"), index=False)

    print("\n Files saved to data/processed/:")
    print("   hourly_data.csv      ← Module 3 loads this")
    print("   daily_data.csv       ← Module 4 (Linear Regression) loads this")
    print("   hourly_scaled.csv    ← Module 5 (LSTM) loads this")
    print("   train_set.csv        ← Module 5 training")
    print("   val_set.csv          ← Module 5 validation")
    print("   test_set.csv         ← Module 6 evaluation")
    print("   scaler_params.csv    ← Module 5 uses to reverse predictions")

    import matplotlib.pyplot as plt
import numpy as np
import os

def generate_visualizations(df_hourly, df_daily, df_train, df_val, df_test,
                              missing_before, missing_after, numeric_cols, device_mapping):

    print("\n[STEP 11] Generating Module 2 visualizations...")

    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 11))
    fig2.suptitle('MODULE 2: Data Cleaning and Preprocessing\n(Continues from Module 1 EDA)',
                   fontsize=15, fontweight='bold')

    short_labels = [c.replace('Global_', 'G_').replace('Sub_metering_', 'Sub_') for c in numeric_cols]

    # Chart 1: Before vs After Missing Values
    ax1 = axes2[0, 0]
    x = np.arange(len(numeric_cols))
    width = 0.35
    ax1.bar(x - width/2, missing_before[numeric_cols], width,
            label='Before (Module 1 found)', color='#FF6B6B', edgecolor='black')
    ax1.bar(x + width/2, missing_after[numeric_cols], width,
            label='After (Module 2 fixed)', color='#4CAF50', edgecolor='black')
    ax1.set_title('Missing Values: Before vs After Cleaning')
    ax1.set_xlabel('Column')
    ax1.set_ylabel('Missing Count')
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_labels, rotation=30, ha='right', fontsize=8)
    ax1.legend(fontsize=8)
    ax1.grid(axis='y', alpha=0.3)

    # Chart 2: Cleaned Hourly Power (first 30 days)
    ax2 = axes2[0, 1]
    plot_data = df_hourly['Global_active_power'].iloc[:24*30]
    ax2.plot(plot_data.index, plot_data.values, color='steelblue', linewidth=0.8, alpha=0.85)
    ax2.fill_between(plot_data.index, plot_data.values, alpha=0.15, color='steelblue')
    ax2.set_title('Cleaned Hourly Global Active Power (First 30 Days)')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Power (kW)')
    ax2.tick_params(axis='x', rotation=30)
    ax2.grid(alpha=0.3)

    # Chart 3: Daily Device Stacked Area
    ax3 = axes2[1, 0]
    daily_sub = df_daily[['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']].iloc[:180]
    ax3.stackplot(
        daily_sub.index,
        daily_sub['Sub_metering_1'],
        daily_sub['Sub_metering_2'],
        daily_sub['Sub_metering_3'],
        labels=[f'Kitchen', f'Laundry', f'HVAC'],
        colors=['#FF6B6B', '#4ECDC4', '#45B7D1'],
        alpha=0.75
    )
    ax3.set_title('Daily Device Energy — Stacked Area (6 Months)')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Energy (Wh)')
    ax3.legend(loc='upper left', fontsize=8)
    ax3.tick_params(axis='x', rotation=30)
    ax3.grid(alpha=0.2)

    # Chart 4: Train / Val / Test Split
    ax4 = axes2[1, 1]
    col = 'Global_active_power'
    ax4.plot(df_train.index, df_train[col], color='#2196F3', linewidth=0.5,
             label=f'Train — {len(df_train):,} hrs (70%)')
    ax4.plot(df_val.index,   df_val[col],   color='#FF9800', linewidth=0.5,
             label=f'Validation — {len(df_val):,} hrs (15%)')
    ax4.plot(df_test.index,  df_test[col],  color='#F44336', linewidth=0.5,
             label=f'Test — {len(df_test):,} hrs (15%)')
    ax4.set_title('Train / Validation / Test Split (Scaled, 0–1)')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Scaled Power')
    ax4.legend(fontsize=8)
    ax4.tick_params(axis='x', rotation=30)
    ax4.grid(alpha=0.3)

    plt.tight_layout()

    output_dir = os.path.join("data", "processed")
    viz_path = os.path.join(output_dir, "module2_cleaning_visualization.png")
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    print(f" Visualization saved: '{viz_path}'")
    plt.show()