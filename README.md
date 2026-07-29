# Distributed Big Data Clustering Pipeline on AWS EMR (NYC Taxi Dataset)

![Java](https://img.shields.io/badge/Java-Hadoop%20MapReduce-red?style=for-the-badge)
![Cloud](https://img.shields.io/badge/AWS-EMR%20%2F%20S3-orange?style=for-the-badge)
![Data](https://img.shields.io/badge/Dataset-8.77M%20Records-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Validation-Python%20%2F%20Silhouette-green?style=for-the-badge)

Distributed, iterative K-Means clustering pipeline built with Java MapReduce to process and analyze 8.77 million urban transportation records (~851 MB) on multi-node Hadoop clusters and AWS EMR.

---

## 📌 Executive Summary

Urban mobility systems generate vast spatial-temporal datasets that exceed the memory and processing limits of single-node machines. This project presents an end-to-end Big Data solution that processes **8,771,772 trip records** from the New York City Taxi and Limousine Commission (TLC). 

By leveraging an iterative **MapReduce architecture** deployed on **Apache Hadoop** and **Amazon EMR (Elastic MapReduce)**, the system partitions computation horizontally to discover distinct behavioral patterns (such as airport transfers, group trips, and tipping anomalies) without sampling bias.

---

## 🏗 Architecture & Big Data Pipeline
