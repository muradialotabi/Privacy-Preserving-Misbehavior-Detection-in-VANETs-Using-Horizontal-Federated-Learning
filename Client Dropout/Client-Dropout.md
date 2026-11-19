# Client Dropout Experiment

## Objective
The Client Dropout experiment evaluates the robustness of federated learning when some clients are unavailable and cannot participate in a communication round.

## Setup
The dropout experiments use a 20-client configuration and examine client-dropout rates of **10%, 20%, and 40%**. These rates correspond to 2, 4, and 8 unavailable clients, respectively, when calculated from 20 clients.

The remaining available clients participate in local training and contribute updates according to the aggregation procedure implemented in the experiment.

## Why this experiment matters
In vehicular networks, clients may disconnect, move out of coverage, experience communication failures, or otherwise become temporarily unavailable. Client dropout therefore represents an important practical condition for evaluating the reliability of FL.

## What to report
Specify whether the unavailable clients are selected randomly or according to another rule, whether the same clients remain unavailable across rounds or the set changes, the number of participating clients per round, and the aggregation behavior when clients are missing. Report the evaluation metrics and training/round time for each dropout rate.
