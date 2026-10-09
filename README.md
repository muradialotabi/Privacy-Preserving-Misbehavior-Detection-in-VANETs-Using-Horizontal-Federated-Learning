# Privacy-Preserving Misbehavior Detection in VANETs Using Horizontal Federated Learning

This repository contains the experimental implementation for the research paper:

**Privacy-Preserving Misbehavior Detection in Vehicular Ad-Hoc Networks Using Horizontal Federated Learning: An Empirical Study of Data Partitioning Strategies**

## Overview

This project investigates Horizontal Federated Learning (HFL) for privacy-preserving misbehavior detection in Vehicular Ad-Hoc Networks (VANETs) using the VeReMi dataset. It evaluates the effects of data partitioning strategies, client scalability, federated optimization algorithms, and aggregation architectures on model performance.

## Experimental Setup

* **Dataset:** VeReMi
* **Model:** Multilayer Perceptron (MLP) with three hidden layers (128, 64, and 32 neurons)
* **Federated Learning Algorithms:** FedAvg and FedProx
* **Architectures:** Flat Federated Learning and Hierarchical Federated Learning
* **Data Distributions:** IID, Non-IID, and Strong Quantity Skew
* **Client Configurations:** 5, 10, 15, and 20 clients
* **Hierarchical Coordination:** Clients coordinated through Roadside Units (RSUs)
* **Additional Experiment:** Client dropout with 20 clients

## Evaluation Metrics

The experiments evaluate the following metrics:

* Accuracy
* Precision
* Recall
* F1-score
* Training time per communication round

## Research Objectives

The primary objectives are to investigate how data heterogeneity, client scalability, federated optimization algorithms, and hierarchical coordination affect misbehavior detection performance in distributed vehicular environments.

## Research Resources

*Research Paper:
* **Paper** :[]
* **Dataset**:[https://drive.google.com/drive/folders/17BUsj6rkDRB3EgGo840wUHDqs6MHhUge?usp=sharing]
* **Google Colab Notebook**: [https://drive.google.com/drive/folders/1zIEHkR8zBFMaun0az6RBy6f3AjZ1-UHV?usp=sharing]


## Development Environment

The experimental implementation uses Python and can be developed or executed using:

* **Google Colab:** For cloud-based notebook execution.
* **PyCharm:** For local development, debugging, and experiment management implemented using Python 3.12 and TensorFlow 2.19.0 with the Keras API.

## Technologies

* Python
* Machine Learning
* Horizontal Federated Learning
* FedAvg and FedProx
* VeReMi Dataset
* Data Analysis and Visualization

## Repository Structure

The repository includes experimental configurations, training procedures, evaluation scripts, and results. The exact structure depends on the files uploaded to the repository.

## Citation

If you use this implementation in your research, please cite the associated paper.



*Privacy-Preserving Misbehavior Detection in Vehicular Ad-Hoc Networks Using Horizontal Federated Learning: An Empirical Study of Data Partitioning Strategies*

Full citation details will be added when available.
