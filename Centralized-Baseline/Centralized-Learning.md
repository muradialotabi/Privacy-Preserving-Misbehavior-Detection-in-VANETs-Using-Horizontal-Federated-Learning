# Centralized Learning Baseline

## Objective
The centralized-learning experiment provides a baseline against which the federated-learning approaches can be evaluated.

## Setup
In centralized learning, the training data is collected into a single training process rather than being kept separately at multiple FL clients. A model is trained using the centralized training dataset, and its performance is evaluated using the same evaluation protocol intended for the corresponding FL comparison.

## Why this experiment matters
The centralized baseline helps quantify the performance difference between training with centrally available data and training collaboratively through federated learning. It provides a reference point for discussing the impact of distributed data, client-level partitioning, and federated aggregation.

## Fair-comparison considerations
For a fair comparison, align the model architecture, feature preprocessing, target-label definition, train/test split, evaluation metrics, and relevant training settings wherever the experiment design allows. In particular, do not directly compare a five-label Quantity Skew result with a nine-label centralized result without explaining the label-space difference or evaluating both under a common label definition.

## What to report
Document the dataset split, label configuration, model settings, optimizer, training procedure, evaluation metrics, and training time. State clearly which FL experiment or partitioning setup the centralized result is intended to serve as a baseline for.
