# Privacy-Preserving Misbehavior Detection in VANETs Using Horizontal Federated Learning

**An Empirical Ablation Study of Data Heterogeneity, Aggregation Strategies, and Client Dropout**

**Muradi Al-Otaibi**
Computer Engineering Department, King Fahd University of Petroleum and Minerals (KFUPM), Dhahran, Saudi Arabia

![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0-orange)
![Dataset](https://img.shields.io/badge/Dataset-VeReMi-green)

---

## Overview

Vehicular Ad Hoc Networks (VANETs) let vehicles exchange Basic Safety Messages (BSMs) containing position, speed, acceleration, and heading. Misbehaving nodes can falsify this information and create unsafe conditions. Centralized machine-learning detection requires collecting sensitive mobility data in one place, which raises privacy, communication, and scalability concerns.

This repository accompanies a paper that **systematically evaluates Horizontal Federated Learning (HFL)** for multiclass misbehavior detection on the **VeReMi** dataset. The goal is **not** to propose a new FL algorithm, but to run a controlled empirical study of existing methods.

## Research Questions / What Was Evaluated

| Dimension | Settings |
|---|---|
| **Learning paradigm** | Centralized Learning (CL) vs. Horizontal Federated Learning |
| **Aggregation algorithm** | FedAvg, FedProx (µ = 0.01) |
| **Aggregation topology** | Flat vs. Hierarchical (Flat FedAvg, Flat FedProx, Hierarchical FedAvg, Hierarchical FedProx) |
| **Data distribution** | IID and strong quantity-skew non-IID |
| **Number of clients** | 5, 10, 15, 20 |
| **Client dropout** | 10%, 20%, 40% (20-client non-IID setup) |
| **Metrics** | Accuracy, Macro-F1, per-class TPR / confusion matrices, communication cost (MB), execution time, rounds to convergence |

## Key Findings

- **IID data:** all methods (CL, FedAvg, FedProx, flat and hierarchical) reach **accuracy = 1.0000**.
- **Non-IID data:** CL performs best in every setting. No federated configuration closes the gap to CL.
- **Best federated method:** Flat FedAvg under quantity skew, trailing CL by **0.83–2.54 pp in accuracy** but **4.8–7.4 pp in Macro-F1**.
- **FedAvg vs. FedProx:** FedAvg outperforms FedProx in every quantity-skew setting.
- **Hierarchical aggregation:** does not improve accuracy in most settings and increases communication cost (e.g., **5.19 MB → 20.76 MB** for FedAvg with 15 clients).
- **Client dropout:** accuracy is stable, with changes of at most **0.14 pp**.
- **Overall:** Flat FedAvg offers the best trade-off in most remaining settings.

> Results are averaged over two seeds (42 and 100) because of the computational cost of each run, so confidence intervals are indicative only.

## Dataset

[VeReMi extension](https://www.kaggle.com/datasets) (OMNeT++ / Veins on the Luxembourg SUMO Traffic scenario, LuST), using Type-3 BSM data with 23 features.

The task is formulated as **6-class classification**: benign behavior plus five position-falsification attacks:

- Constant Position
- Constant Position Offset
- Random Position
- Random Position Offset
- Eventually Stop

The data is **not** included in this repository because of its size. The partitioned data is available via the Google Drive link in the [Data Availability](#data-availability) section below.


The data is **not** included in this repository because of its size. The partitioned data is available via the Google Drive link in the [Data Availability](#data-availability) section below.

🔧 **Data pre-processing code** (raw VeReMi JSON → cleaned CSV) is available in a separate repository: [muradialotabi/Data-Pre-Processing](https://github.com/muradialotabi/Data-Pre-Processing).

## Data Availability

The partitioned datasets (Quantity-skew non-IID) are too large to be hosted on GitHub. They can be downloaded from Google Drive:

📦 **[Download the data (Google Drive)](https://drive.google.com/drive/folders/17BUsj6rkDRB3EgGo840wUHDqs6MHhUge)**


