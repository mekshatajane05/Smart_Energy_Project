
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
 
 
# ─────────────────────────────────────────────────────────
# STEP 1: Load the dataset
# ─────────────────────────────────────────────────────────
# The dataset uses semicolons (;) as the separator instead of commas.
# Missing values in this dataset are marked as '?' or left blank,
# so we tell pandas to treat both as NaN (missing).
file_path = "dataset/household_power_consumption.txt"
 
try:
    df = pd.read_csv(
        file_path,
        sep=';',
        low_memory=False,
        na_values=['?', '']
    )
 
    print(" Dataset loaded successfully!")
    print(f" Total records: {len(df):,}")
    print(f" Total columns: {len(df.columns)}")
 
except FileNotFoundError:
    print(" ERROR: Dataset file not found!")
    print("Please download from: https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption")
    print("Save as 'household_power_consumption.txt' inside the data/raw/ folder")
    exit()
 
 
# ─────────────────────────────────────────────────────────
# STEP 2: Initial Data Exploration
# ─────────────────────────────────────────────────────────
# Before doing anything else, we look at the basic shape and
# structure of the dataset to understand what we're working with.
 
print("\n Dataset Shape:")
print(f"   Rows: {df.shape[0]:,}")
print(f"   Columns: {df.shape[1]}")
 
print("\nColumn Names and Data Types:")
print(df.dtypes)
 
print("\nFirst 5 records:")
print(df.head())
 
print("\nDataset Info:")
print(df.info())
 
print("\nStatistical Summary:")
print(df.describe())
 
 
# ─────────────────────────────────────────────────────────
# STEP 3: Data Integrity Check
# ─────────────────────────────────────────────────────────
# Here we check for missing values, confirm columns are the
# correct data type, and look at the overall date range covered
# by the dataset.
 
print("\n Missing Values Analysis:")
missing_data = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': df.isnull().sum(),
    'Missing_Percentage': (df.isnull().sum() / len(df) * 100).round(2)
})
missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
 
if len(missing_data) > 0:
    print(missing_data.to_string(index=False))
else:
    print("    No missing values found!")
 
# Store missing values for later comparison in Module 2 (before vs after cleaning)
missing_before = df.isnull().sum()
 
print("\n Data Type Check:")
numeric_cols_all = ['Global_active_power', 'Global_reactive_power', 'Voltage',
                     'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
 
for col in numeric_cols_all:
    if df[col].dtype == 'object':
        print(f"    {col} is stored as object (should be numeric)")
 
print("\n Date/Time Range:")
print(f"   Start: {df['Date'].min()}")
print(f"   End: {df['Date'].max()}")
 
 
# ─────────────────────────────────────────────────────────
# STEP 4: Device-Level Energy Organization
# ─────────────────────────────────────────────────────────
# The dataset groups energy usage into 3 "sub-meters", each
# representing a category of household devices. We map each
# sub-meter to a friendly device name and print basic stats
# for each one.
 
print("\n Understanding Sub-metering Data:")
print("   Sub_metering_1: Kitchen (dishwasher, microwave, oven)")
print("   Sub_metering_2: Laundry (washing machine, dryer, refrigerator)")
print("   Sub_metering_3: HVAC/Climate control (water heater, AC)")
 
device_mapping = {
    'Sub_metering_1': 'Kitchen',
    'Sub_metering_2': 'Laundry',
    'Sub_metering_3': 'HVAC'
}
 
print("\n Device-wise Basic Statistics:")
for col, device in device_mapping.items():
    if col in df.columns:
        numeric_values = pd.to_numeric(df[col], errors='coerce')
        print(f"\n   {device} ({col}):")
        print(f"      Mean: {numeric_values.mean():.2f} Wh")
        print(f"      Max: {numeric_values.max():.2f} Wh")
        print(f"      Min: {numeric_values.min():.2f} Wh")
        print(f"      Std Dev: {numeric_values.std():.2f} Wh")
 
 
# ─────────────────────────────────────────────────────────
# STEP 5: Exploratory Visualizations
# ─────────────────────────────────────────────────────────
# We sample the first 10,000 rows for plotting — this is enough
# data to clearly see patterns without making matplotlib slow.
 
df_sample = df.head(10000).copy()
 
# Convert sub-meter / power columns to numeric for plotting
for col in numeric_cols_all:
    df_sample[col] = pd.to_numeric(df_sample[col], errors='coerce')
 
# Create a 2x2 grid of charts for Module 1
fig1, axes1 = plt.subplots(2, 2, figsize=(15, 10))
fig1.suptitle('MODULE 1: Exploratory Data Analysis', fontsize=16, fontweight='bold')
 
# --- Chart 1: Missing values heatmap (first 1000 records) ---
ax1 = axes1[0, 0]
missing_matrix = df.head(1000).isnull()
ax1.imshow(missing_matrix, cmap='RdYlGn_r', aspect='auto', interpolation='none')
ax1.set_title('Missing Values Pattern (First 1000 records)')
ax1.set_xlabel('Columns')
ax1.set_ylabel('Records')
ax1.set_xticks(range(len(df.columns)))
ax1.set_xticklabels(df.columns, rotation=45, ha='right', fontsize=8)
 
# --- Chart 2: Global Active Power distribution ---
ax2 = axes1[0, 1]
df_sample['Global_active_power'].dropna().hist(bins=50, ax=ax2, color='skyblue', edgecolor='black')
ax2.set_title('Global Active Power Distribution')
ax2.set_xlabel('Power (kW)')
ax2.set_ylabel('Frequency')
ax2.grid(alpha=0.3)
 
# --- Chart 3: Sub-metering comparison (average energy per device) ---
ax3 = axes1[1, 0]
sub_meters = ['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
means = [df_sample[col].mean() for col in sub_meters]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
bars = ax3.bar(['Kitchen', 'Laundry', 'HVAC'], means, color=colors, edgecolor='black')
ax3.set_title('Average Energy by Device Type')
ax3.set_ylabel('Average Energy (Wh)')
ax3.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width() / 2., height,
              f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
 
# --- Chart 4: Data completeness by column ---
ax4 = axes1[1, 1]
completeness = ((len(df) - df.isnull().sum()) / len(df) * 100)
completeness.plot(kind='barh', ax=ax4, color='green', edgecolor='black')
ax4.set_title('Data Completeness by Column')
ax4.set_xlabel('Completeness (%)')
ax4.axvline(x=95, color='red', linestyle='--', label='95% threshold')
ax4.legend()
ax4.grid(axis='x', alpha=0.3)
 
plt.tight_layout()
 
# Save the chart into data/processed/ so outputs stay organized
output_dir = os.path.join("data", "processed")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "module1_eda_visualization.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n Module 1 visualization saved as '{output_path}'")
plt.show()
 
print("\n MODULE 1 COMPLETE — proceed to Module 2 (Data Cleaning and Preprocessing)")
