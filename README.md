# Federated Learning for Vehicular Misbehavior Detection

This repository contains experiments on machine-learning-based misbehavior detection in Vehicular Ad Hoc Networks (VANETs) using the VeReMi dataset. The experiments compare centralized training with federated learning under different data-partitioning strategies, client configurations, and client-availability conditions.

## Experiment groups

| Experiment | Purpose |
|---|---|
| Centralized Learning | Establish a centralized baseline for comparison with federated learning. |
| IID | Evaluate FL when clients have similarly distributed data. |
| Non-IID | Evaluate FL when client data distributions differ. |
| Quantity Skew | Evaluate FL when clients hold different quantities of training data. |
| Client Dropout | Evaluate FL when a proportion of clients is unavailable. |

## Federated-learning methods and architectures

The experiments investigate FedAvg and FedProx. Where implemented, both flat and hierarchical coordination are considered. In a hierarchical vehicular-network architecture, RSUs can act as intermediate aggregators between vehicle-side clients and the central server. Refer to the individual scripts to identify the exact method and architecture used in each run.

## Quantity Skew label adjustment

In the Quantity Skew experiments, the label value **9 was changed to 5**. This refers to changing a label value in the target-label encoding; it does not mean that the number of classes was reduced from nine to five. The exact mapping should be verified against the preprocessing code.

## Client configurations

The FL experiments investigate client counts of 5, 10, 15, and 20 where configured. The Client Dropout experiments use a 20-client setup and examine dropout rates of 10%, 20%, and 40%, corresponding to 2, 4, and 8 unavailable clients.

## Evaluation

The experiments use classification metrics such as accuracy, precision, recall, and F1-score. Training time and/or communication-round time are also recorded where available. For multi-class results, document the averaging method used for precision, recall, and F1-score.

## Reproducibility

For each experiment, check the associated script and configuration for the dataset path, feature preprocessing, label mapping, train/test split, client count, data partitioning strategy, model settings, optimizer, aggregation method, communication rounds, and random seed. Use the implementation and saved result files as the source of truth for exact parameter values.
