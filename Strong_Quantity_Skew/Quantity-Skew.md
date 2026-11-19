# Quantity Skew Experiment

## Objective
The Quantity Skew experiment evaluates federated learning when clients have unequal amounts of training data.

## Setup
The VeReMi dataset is partitioned so that the quantity of data assigned to each client differs. This creates an imbalance in local dataset sizes: some clients hold more training examples than others. The experiment investigates how unequal local data volumes affect local training and global model performance.

The experiment series includes different client counts where configured (5, 10, 15, and 20). Refer to the corresponding scripts and configuration files for the exact quantity-skew distribution.

## Label-value adjustment
In the Quantity Skew experiments, the label value **9 was changed to 5**. This is a change to the label value used in the target-label encoding; it does **not** mean that the classification task was changed from nine classes to five classes. The label mapping implemented in the preprocessing code should be treated as the definitive mapping when interpreting the experiment.

## Why this experiment matters
In practical federated-learning deployments, vehicles may generate or retain different amounts of data. Quantity imbalance can affect the influence of each client during aggregation and may change the overall learning behavior.

## What to report
Include the number of clients, each client's sample count (or the distribution/range), the quantity-skew generation method, the exact label mapping (including the change from label value 9 to 5), model and FL settings, evaluation metrics, and training time. When comparing results with other experiments, verify that their label encodings and evaluation protocols are consistent.
