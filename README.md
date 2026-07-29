תערוך לי את זה שזה יהיה יפה 
# NYC Taxi Mobility Pattern Clustering
### Iterative Distributed K-Means on Apache Hadoop & AWS EMR

![Java](https://img.shields.io/badge/Java-8-orange?style=for-the-badge&logo=java)
![Framework](https://img.shields.io/badge/Hadoop-3.2.1-yellow?style=for-the-badge&logo=apachehadoop)
![Cloud](https://img.shields.io/badge/AWS-EMR%20%2F%20S3-blue?style=for-the-badge&logo=amazon-aws)
![Data](https://img.shields.io/badge/Dataset-8.77M%20Records-green?style=for-the-badge)

**Course:** Distributed Big Data Processing in Cloud Technologies (72070)  
**Institution:** HIT – Holon Institute of Technology  
**Authors:** Lidor Kedmi, Daniel Shkuri, Ben Meir  
**Academic Term:** Spring 2026  

---

## 📋 1. Project Overview

This project implements an iterative **K-Means clustering pipeline using MapReduce on Apache Hadoop** to group New York City yellow taxi trip records into four distinct behavioral categories. Each K-Means iteration operates as a complete, decoupled MapReduce job; a driver program orchestrates execution loops and evaluates convergence thresholds. 

The system was developed and validated on a multi-node distributed local **Hadoop cluster (Docker)** and seamlessly deployed onto **AWS Elastic MapReduce (EMR)** to demonstrate end-to-end cloud portability.

---

## 📊 2. Dataset Information

The project utilizes official NYC Yellow Taxi trip records published by the NYC Taxi and Limousine Commission (TLC) in Parquet format.

* **Data Source:** [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)[cite: 13]
* **Target Datasets (Yellow Taxi, Jan–Mar 2023):** [Jan 2023](https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet) | [Feb 2023](https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-02.parquet) | [Mar 2023](https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet)[cite: 13]
* **Volume:** 9,384,487 raw records $\rightarrow$ **8,771,772 cleaned valid records** (~851 MB uncompressed CSV payload).

---

## 🧩 3. Features & Data Schema

Each trip is represented by 10 numerical features, normalized via **z-score standard scaling** to prevent distance metric distortion:

| # | Feature | Description |
|---|---|---|
| 1 | `PULocationID` | TLC Pickup Zone ID (1–265) |
| 2 | `DOLocationID` | TLC Drop-off Zone ID (1–265) |
| 3 | `pickup_hour` | Hour of day (0–23)[cite: 13] |
| 4 | `pickup_dayofweek` | Day of week (0=Mon ... 6=Sun)[cite: 13] |
| 5 | `trip_distance` | Trip distance in miles[cite: 13] |
| 6 | `trip_duration` | Trip duration in minutes (derived)[cite: 13] |
| 7 | `fare_amount` | Base fare in USD[cite: 13] |
| 8 | `tip_amount` | Tip amount in USD[cite: 13] |
| 9 | `tip_percentage` | Tip ratio (`tip_amount / fare_amount`)[cite: 13] |
| 10 | `passenger_count` | Number of passengers[cite: 13] |

---

## 🛠 4. Code Base & Project Structure

```text
├── pom.xml                   # Maven build configuration (Java 8, Hadoop 3.2.1, Shade plugin)[cite: 13]
├── prepare_data.py           # Ingests Parquet, cleans data, applies z-score normalization[cite: 13]
├── make_centroids.py         # Generates initial centroid file (k=4)[cite: 13]
├── validate_clustering.py    # Determines optimal k via Silhouette and Elbow methods[cite: 13]
├── inspect_clusters.py       # Plots and analyzes final cluster profiles[cite: 13]
└── src/
    └── main/java/com/taxi/kmeans/
        ├── Point.java        # 10D Writable vector class handling serialization & arithmetic[cite: 13]
        ├── KMeansMapper.java   # Assigns trip points to nearest centroid[cite: 13]
        ├── KMeansCombiner.java # Local aggregation optimization prior to shuffle phase[cite: 13]
        ├── KMeansReducer.java  # Re-computes updated centroid coordinates[cite: 13]
        └── KMeansDriver.java   # Orchestrates iteration loops & evaluates convergence[cite: 13]

#⚙️ 5. Step-by-Step Execution Guide
### Phase 0: Data Preprocessing (Python)
Prerequisites: Python 3.x with pandas, pyarrow, scikit-learn, matplotlib[cite: 13].

BASH
# 1. Install dependencies[cite: 13]
pip install pandas pyarrow scikit-learn matplotlib

# 2. Place downloaded Parquet files into data/raw/ directory[cite: 13]

# 3. Clean and normalize dataset -> generates taxi_full.csv[cite: 13]
python prepare_data.py

# 4. Generate initial centroids file (initial_centroids.txt)[cite: 13]
python make_centroids.py

### Phase A: Compilation & Build (Java / Maven)
Prerequisites: JDK 8+, Apache Maven[cite: 13].

BASH

# Compile and create shaded fat-JAR (~12KB application JAR)[cite: 13]
mvn clean package

# Output artifact: target/taxi-kmeans.jar[cite: 13]

Phase B: Distributed Local Execution (Hadoop Multi-Node Docker)
Runs on a 7-container Docker cluster (NameNode, ResourceManager, NodeManager, HistoryServer, 3x DataNodes)[cite: 13].

# 1. Verify cluster containers are running[cite: 13]
docker ps

# 2. Copy artifacts to NameNode container[cite: 13]
docker cp target/taxi-kmeans.jar namenode:/tmp/taxi-kmeans.jar
docker cp data/processed/taxi_full.csv namenode:/tmp/taxi_full.csv
docker cp data/processed/initial_centroids.txt namenode:/tmp/initial_centroids.txt

# 3. Enter NameNode container shell[cite: 13]
docker exec -it namenode bash

# 4. Load dataset into HDFS (replicated 3x across DataNodes)[cite: 13]
hdfs dfs -mkdir -p /taxi/input
hdfs dfs -put /tmp/taxi_full.csv /taxi/input/taxi_full.csv
hdfs dfs -put /tmp/initial_centroids.txt /taxi/input/initial_centroids.txt

# 5. Execute MapReduce K-Means Driver[cite: 13]
hadoop jar /tmp/taxi-kmeans.jar com.taxi.kmeans.KMeansDriver \
    /taxi/input/taxi_full.csv \
    /taxi/input/initial_centroids.txt \
    /taxi/output_full \
    20

# 6. View final output centroids[cite: 13]
hdfs dfs -cat /taxi/output_full/iteration_8/part-r-00000


Phase C: Cloud Execution on AWS EMRThe application uses dynamic file system resolution (path.getFileSystem(conf)), allowing the exact same JAR to run interchangeably on HDFS or Amazon S3 paths.
1. Upload Artifacts to Amazon S3: Upload taxi-kmeans.jar to s3://<bucket>/jar/ and data files to s3://<bucket>/input/[cite: 13].
2. Launch EMR Cluster: Deploy AWS EMR (7.x, Core Hadoop) with 1 Primary and 2 Core nodes (m5.xlarge)[cite: 13].
3. Submit Custom JAR Step:JAR Location: s3://<bucket>/jar/taxi-kmeans.jar[cite: 13]Arguments:

s3://<bucket>/input/taxi_full.csv
s3://<bucket>/input/initial_centroids.txt
s3://<bucket>/output
20
```[cite: 13]

4. Retrieve Results: Upon step completion, final centroids are stored in s3://<bucket>/output/iteration_N/[cite: 13].

📈 6. Analytical Results & Cluster ProfilesFor optimal cluster separation ($k=4$), the distributed pipeline identified four distinct mobility patterns[cite: 13]:
1. Airport / Long-Distance Trips: ~13.2 miles, ~$52.3 fare, high tip ratio (JFK/LGA/EWR transfers).
2. Group / Multi-Passenger Rides: Average 4+ passengers, mid-week volume peaks.
3. Standard City Trips (High Tip): Short trips (~1.9 miles), high tip ratio (~24%).
4. Short City Trips (Low Tip / Cash): Short distance (~2.4 miles), lower recorded tip ratio (~16%, cash payments).
