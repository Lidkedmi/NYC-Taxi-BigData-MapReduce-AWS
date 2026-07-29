"""
NYC Taxi - Data Preparation for MapReduce KMeans
Reads the 3 parquet files, cleans them, computes 10 normalized features,
and writes:
  - data/processed/taxi_full.csv     (all trips, for AWS EMR)
  - data/processed/taxi_sample.csv   (50k trips, for local testing)
Also writes the normalization stats so results can be interpreted later.
"""

import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join("data", "raw")
OUT_DIR = os.path.join("data", "processed")
SAMPLE_SIZE = 50_000
RANDOM_SEED = 42

os.makedirs(OUT_DIR, exist_ok=True)

FEATURES = [
    'PULocationID', 'DOLocationID',
    'pickup_hour', 'pickup_dayofweek',
    'trip_distance', 'trip_duration',
    'fare_amount', 'tip_amount',
    'tip_percentage', 'passenger_count'
]

print("=" * 60)
print("NYC Taxi - Data Preparation")
print("=" * 60)

# ---- Load all 3 months ----
print("\n[1/5] Loading parquet files...")
frames = []
for month in ["01", "02", "03"]:
    path = os.path.join(DATA_DIR, f"yellow_tripdata_2023-{month}.parquet")
    frames.append(pd.read_parquet(path))
    print(f"   loaded month {month}")
df = pd.concat(frames, ignore_index=True)
print(f"   total: {len(df):,} rows")

# ---- Clean ----
print("\n[2/5] Cleaning...")
initial = len(df)
df['trip_duration'] = (
    df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']
).dt.total_seconds() / 60
df = df[
    (df['trip_distance'] > 0) & (df['trip_distance'] < 100) &
    (df['fare_amount'] > 0) & (df['fare_amount'] < 500) &
    (df['trip_duration'] > 1) & (df['trip_duration'] < 180) &
    (df['passenger_count'] > 0) & (df['passenger_count'] <= 6) &
    (df['PULocationID'].between(1, 265)) &
    (df['DOLocationID'].between(1, 265))
]
df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
df['pickup_dayofweek'] = df['tpep_pickup_datetime'].dt.dayofweek
df['tip_percentage'] = (df['tip_amount'] / df['fare_amount']).clip(0, 1)
print(f"   {initial:,} -> {len(df):,} rows after cleaning")

# ---- Normalize (z-score) ----
print("\n[3/5] Normalizing features...")
X = df[FEATURES].astype(float).copy()
means = X.mean()
stds = X.std().replace(0, 1)
X_norm = (X - means) / stds

# save normalization stats for later interpretation
stats = pd.DataFrame({'feature': FEATURES,
                      'mean': means.values,
                      'std': stds.values})
stats.to_csv(os.path.join(OUT_DIR, "normalization_stats.csv"), index=False)
print(f"   saved normalization_stats.csv")

# ---- Write full CSV ----
print("\n[4/5] Writing full CSV (this may take a minute)...")
full_path = os.path.join(OUT_DIR, "taxi_full.csv")
X_norm.to_csv(full_path, index=False, header=False, float_format='%.6f')
print(f"   wrote {full_path}  ({len(X_norm):,} rows)")

# ---- Write sample CSV ----
print("\n[5/5] Writing sample CSV...")
sample = X_norm.sample(n=min(SAMPLE_SIZE, len(X_norm)), random_state=RANDOM_SEED)
sample_path = os.path.join(OUT_DIR, "taxi_sample.csv")
sample.to_csv(sample_path, index=False, header=False, float_format='%.6f')
print(f"   wrote {sample_path}  ({len(sample):,} rows)")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
print(f"Full dataset:  {full_path}")
print(f"Sample:        {sample_path}")
print("Each row = one trip = 10 comma-separated normalized numbers.")