# Federated Learning Methods

This repository investigates federated-learning configurations for vehicular misbehavior detection. The exact methods included in each run should be verified from the corresponding experiment script.

## FedAvg
Federated Averaging (FedAvg) is an aggregation method in which participating clients train models locally and send their model parameters or updates to an aggregator. The aggregator combines the received client models, commonly using client dataset sizes as aggregation weights, to produce the next global model.

## FedProx
FedProx extends the local optimization objective with a proximal term that discourages local models from moving too far from the current global model. It is designed to help address statistical heterogeneity across clients. The proximal coefficient (often written as μ) is a configuration parameter and should be reported from the actual experiment code.

## Flat and hierarchical coordination
In flat FL, client updates are sent to a central server for aggregation. In hierarchical FL, intermediate aggregators (such as RSUs in a vehicular-network architecture) can aggregate updates from nearby vehicle clients before communicating with a higher-level server. Describe the exact hierarchy and aggregation stages implemented in the code; do not assume that every experiment uses both architectures.

## Reporting
For every run, record the algorithm, coordination architecture, client count, local training settings, aggregation settings, number of communication rounds, and evaluation metrics.
