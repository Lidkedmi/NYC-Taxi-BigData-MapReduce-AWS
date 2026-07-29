"""
NYC Taxi Clustering - Cluster Inspection (k=4)
Purpose: Run KMeans with the optimal k=4 and characterize each cluster
         so we can give every cluster a meaningful name.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = os.path.join("data", "raw")
SAMPLE_SIZE = 150_000
BEST_K = 4
RANDOM_SEED = 42

print("=" * 65)
print(f"NYC Taxi Cluster Inspection  -  k = {BEST_K}")
print("=" * 65)

# ============================================================
# LOAD + CLEAN  (3 months for a richer picture)
# ============================================================
print("\n[1/5] Loading all 3 months...")
frames = []
for month in ["01", "02", "03"]:
    f = os.path.join(DATA_DIR, f"yellow_tripdata_2023-{month}.parquet")
    frames.append(pd.read_parquet(f))
df = pd.concat(frames, ignore_index=True)
print(f"   Loaded {len(df):,} total rows")

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

# ============================================================
# SAMPLE + NORMALIZE + CLUSTER
# ============================================================
print(f"\n[3/5] Sampling {SAMPLE_SIZE:,} rows and clustering...")
sample = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED).copy()

features = [
    'PULocationID', 'DOLocationID',
    'pickup_hour', 'pickup_dayofweek',
    'trip_distance', 'trip_duration',
    'fare_amount', 'tip_amount',
    'tip_percentage', 'passenger_count'
]
X = sample[features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

km = KMeans(n_clusters=BEST_K, random_state=RANDOM_SEED, n_init=10)
sample['cluster'] = km.fit_predict(X_scaled)

# ============================================================
# CHARACTERIZE EACH CLUSTER
# ============================================================
print("\n[4/5] Cluster profiles (average feature values):")
print("=" * 65)

summary = sample.groupby('cluster')[features].mean().round(2)
sizes = sample.groupby('cluster').size()
summary.insert(0, 'count', sizes)
summary.insert(1, 'pct', (sizes / len(sample) * 100).round(1))

# print transposed so each cluster is a column - easier to compare
print(summary.T.to_string())
print("=" * 65)

# load zone names to interpret PULocationID
zone_path = os.path.join(DATA_DIR, "taxi_zone_lookup.csv")
if os.path.exists(zone_path):
    zones = pd.read_csv(zone_path)
    zone_map = dict(zip(zones['LocationID'], zones['Zone']))
    boro_map = dict(zip(zones['LocationID'], zones['Borough']))
    print("\nMost common pickup zone per cluster:")
    for c in range(BEST_K):
        sub = sample[sample['cluster'] == c]
        top_pu = sub['PULocationID'].mode()[0]
        top_do = sub['DOLocationID'].mode()[0]
        print(f"  Cluster {c}:  pickup-> {zone_map.get(top_pu,'?')} "
              f"({boro_map.get(top_pu,'?')})   "
              f"dropoff-> {zone_map.get(top_do,'?')}")

# ============================================================
# VISUALIZE
# ============================================================
print("\n[5/5] Generating cluster visualization...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# hour distribution per cluster
for c in range(BEST_K):
    sub = sample[sample['cluster'] == c]
    axes[0, 0].hist(sub['pickup_hour'], bins=24, alpha=0.5, label=f'Cluster {c}')
axes[0, 0].set_title('Pickup Hour Distribution by Cluster')
axes[0, 0].set_xlabel('Hour of day'); axes[0, 0].legend()

# distance vs fare scatter
colors = ['steelblue', 'darkorange', 'green', 'crimson', 'purple', 'brown']
for c in range(BEST_K):
    sub = sample[sample['cluster'] == c].sample(min(2000, len(sample[sample['cluster']==c])))
    axes[0, 1].scatter(sub['trip_distance'], sub['fare_amount'],
                       s=5, alpha=0.4, c=colors[c], label=f'Cluster {c}')
axes[0, 1].set_title('Trip Distance vs Fare')
axes[0, 1].set_xlabel('Distance (mi)'); axes[0, 1].set_ylabel('Fare ($)')
axes[0, 1].legend()

# cluster sizes
axes[1, 0].bar([f'C{c}' for c in range(BEST_K)], sizes.values,
               color=colors[:BEST_K])
axes[1, 0].set_title('Cluster Sizes')
axes[1, 0].set_ylabel('Trip count')

# day of week per cluster
for c in range(BEST_K):
    sub = sample[sample['cluster'] == c]
    counts = sub['pickup_dayofweek'].value_counts(normalize=True).sort_index()
    axes[1, 1].plot(counts.index, counts.values, 'o-', label=f'Cluster {c}')
axes[1, 1].set_title('Day of Week Distribution by Cluster')
axes[1, 1].set_xlabel('0=Mon ... 6=Sun')
axes[1, 1].set_xticks(range(7)); axes[1, 1].legend()

plt.tight_layout()
out_path = os.path.join("report", "cluster_profiles_k4.png")
plt.savefig(out_path, dpi=120, bbox_inches='tight')
print(f"   Saved to: {out_path}")

print("\n" + "=" * 65)
print("INSPECTION COMPLETE")
print("=" * 65)
print("\nNext: read the cluster profiles above. Each cluster should look")
print("DIFFERENT from the others. Tell Claude the numbers and we'll name them.")