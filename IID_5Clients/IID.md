# IID Data Distribution

## Objective
The IID (Independent and Identically Distributed) experiment evaluates federated learning when the data assigned to clients is sampled so that each client has approximately the same underlying data distribution.

## Setup
The VeReMi dataset is partitioned among the participating clients using an IID strategy. Each client trains the local model on its own partition and sends model updates to the federated aggregation process. The experiments investigate different client counts (5, 10, 15, and 20, where configured).

## Why this experiment matters
IID is a useful reference setting because clients have broadly similar data distributions. It helps assess the FL workflow under a comparatively balanced distribution and provides a baseline for understanding the effect of data heterogeneity.

## What to report
Report the client count, partition sizes, model and training configuration, aggregation method, classification metrics, and training time. Compare these results with the corresponding Non-IID and Quantity Skew experiments only when their label space and evaluation setup are aligned.
