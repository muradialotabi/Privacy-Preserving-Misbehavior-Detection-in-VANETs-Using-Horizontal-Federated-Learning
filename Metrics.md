# Evaluation Metrics

The experiments use classification metrics to evaluate vehicular misbehavior detection.

- **Accuracy:** The proportion of all evaluated examples that are classified correctly.
- **Precision:** The proportion of examples predicted as a class that truly belong to that class.
- **Recall:** The proportion of examples belonging to a class that are correctly detected.
- **F1-score:** The harmonic mean of precision and recall. For multi-class classification, state whether the reported value is macro, micro, or weighted F1.
- **Training/round time:** The elapsed training time, or the time required for a communication/training round, as defined by the experiment implementation.

For multi-class results, document the averaging method used for precision, recall, and F1. Also state the test set and label configuration. Metrics from different label spaces (such as 9 classes versus 5 classes) should not be treated as directly equivalent without a clearly explained common evaluation setup.
