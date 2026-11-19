
# ============================================================
# CENTRALIZED BASELINE
# FULL CODE WITH CONFUSION MATRIX & CLASS-WISE METRICS
# Prints results after each seed
# Compatible with FL Benchmark Comparative Framework
# ============================================================

import os
import time
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# 1. CONFIGURATION & HYPERPARAMETERS
# ============================================================

SEEDS = [42, 101]

GLOBAL_ROUNDS = 10
LOCAL_EPOCHS = 3
BATCH_SIZE = 256
LEARNING_RATE = 0.001
TEST_SIZE = 0.20

# Hidden Layer Architecture
HIDDEN_LAYERS = [128, 64, 32]

# Label Mapping
label_mapping = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    9: 5
}

# ============================================================
# FILE PATHS
# ============================================================

file_paths = [

r'C:\Users\Murad\PycharmProjects\federated-learning-tutorial\extreme_label_skew_5clients\client_1_extreme_noniid.csv' ,
r'C:\Users\Murad\PycharmProjects\federated-learning-tutorial\extreme_label_skew_5clients\client_2_extreme_noniid.csv',
r'C:\Users\Murad\PycharmProjects\federated-learning-tutorial\extreme_label_skew_5clients\client_3_extreme_noniid.csv',
r'C:\Users\Murad\PycharmProjects\federated-learning-tutorial\extreme_label_skew_5clients\client_4_extreme_noniid.csv',
r'C:\Users\Murad\PycharmProjects\federated-learning-tutorial\extreme_label_skew_5clients\client_5_extreme_noniid.csv'

]
# Output Directory
output_dir = r'C:\Users\Murad\PycharmProjects\federated-learning-tutorial'
os.makedirs(output_dir, exist_ok=True)

# ============================================================
# 2. DATA LOADING & PREPROCESSING
# ============================================================

def load_data(paths):
    start_time = time.time()
    df_list = []

    for file_path in paths:
        try:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)

                if not df.empty:
                    df_list.append(df)
                else:
                    print(f"Warning: Empty file -> {file_path}")
            else:
                print(f"Warning: File not found -> {file_path}")

        except Exception as e:
            print(f"Error reading '{file_path}': {e}")

    if not df_list:
        raise ValueError(
            "No data files were loaded. Check file paths."
        )

    combined_df = pd.concat(df_list, ignore_index=True)

    print(f"Data loading time: {time.time() - start_time:.4f} seconds")
    print(f"Total dataset samples: {len(combined_df):,}")

    return combined_df


data = load_data(file_paths)

# Validate label column
if 'label' not in data.columns:
    raise ValueError("The dataset must contain a 'label' column.")

# Map labels
data['label'] = data['label'].map(label_mapping)

if data['label'].isna().any():
    raise ValueError(
        "Unknown labels detected in dataset after mapping."
    )

# Features and labels
X_raw = data.drop('label', axis=1).values.astype(np.float32)
y = data['label'].values.astype(np.int32)

# Handle NaN and infinite values
X_raw = np.nan_to_num(
    X_raw,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

# Global train/test split
# Fixed split across seeds for fair model initialization comparison
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw,
    y,
    test_size=TEST_SIZE,
    random_state=42,
    stratify=y
)

# Standardization: fit on training data only
scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_train_raw
).astype(np.float32)

X_test = scaler.transform(
    X_test_raw
).astype(np.float32)

input_shape = (X_train.shape[1],)

# Use fixed class labels 0-5
class_labels = np.array(sorted(label_mapping.values()))
num_classes = len(class_labels)

if not np.isin(y_train, class_labels).all():
    raise ValueError("Unexpected class labels in training data.")

print("\n" + "=" * 100)
print("DATASET INFORMATION")
print("=" * 100)
print(f"Total samples      : {len(y):,}")
print(f"Training samples   : {len(y_train):,}")
print(f"Testing samples    : {len(y_test):,}")
print(f"Number of features : {X_train.shape[1]}")
print(f"Number of classes  : {num_classes}")
print(f"Classes            : {class_labels.tolist()}")

print("\nTraining class distribution:")
print(pd.Series(y_train).value_counts().sort_index().to_string())

print("\nTesting class distribution:")
print(pd.Series(y_test).value_counts().sort_index().to_string())

# ============================================================
# 3. MODEL BUILDER
# ============================================================

def create_mlp_model(input_shape, num_classes, hidden_layers):

    model = tf.keras.Sequential()

    model.add(
        tf.keras.layers.Input(shape=input_shape)
    )

    for units in hidden_layers:
        model.add(
            tf.keras.layers.Dense(
                units,
                activation='relu'
            )
        )

    model.add(
        tf.keras.layers.Dense(
            num_classes,
            activation='softmax'
        )
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    )

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ============================================================
# 4. SINGLE RUN EXPERIMENT FUNCTION
# ============================================================

def run_fl_experiment(mode, seed):
    """
    Executes centralized training for a given seed.

    GLOBAL_ROUNDS and LOCAL_EPOCHS are retained to make the
    convergence output comparable in structure with the FL
    benchmark framework.

    Each round continues training the same centralized model.
    """

    # Set seeds
    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Create a fresh model for each seed
    tf.keras.backend.clear_session()

    model = create_mlp_model(
        input_shape,
        num_classes,
        HIDDEN_LAYERS
    )

    experiment_start = time.time()

    round_accuracy = []
    round_f1 = []
    round_precision = []
    round_recall = []
    round_time = []

    epochs_per_round = LOCAL_EPOCHS

    # --------------------------------------------------------
    # Training rounds
    # --------------------------------------------------------

    for r in range(GLOBAL_ROUNDS):

        round_start = time.time()

        model.fit(
            X_train,
            y_train,
            epochs=epochs_per_round,
            batch_size=BATCH_SIZE,
            verbose=0
        )

        elapsed_round = time.time() - round_start

        # Evaluate on the held-out test set
        y_pred_probs = model.predict(
            X_test,
            batch_size=BATCH_SIZE,
            verbose=0
        )

        y_pred = np.argmax(
            y_pred_probs,
            axis=1
        )

        acc = accuracy_score(
            y_test,
            y_pred
        )

        prec = precision_score(
            y_test,
            y_pred,
            labels=class_labels,
            average='macro',
            zero_division=0
        )

        rec = recall_score(
            y_test,
            y_pred,
            labels=class_labels,
            average='macro',
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            labels=class_labels,
            average='macro',
            zero_division=0
        )

        round_accuracy.append(acc)
        round_precision.append(prec)
        round_recall.append(rec)
        round_f1.append(f1)
        round_time.append(elapsed_round)

    total_time = time.time() - experiment_start

    # --------------------------------------------------------
    # Final classification report
    # --------------------------------------------------------

    class_report = classification_report(
        y_test,
        y_pred,
        labels=class_labels,
        output_dict=True,
        zero_division=0
    )

    # --------------------------------------------------------
    # Final normalized confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=class_labels
    )

    row_sums = cm.sum(
        axis=1,
        keepdims=True
    )

    cm_norm = np.divide(
        cm.astype(np.float64),
        row_sums,
        out=np.zeros_like(
            cm,
            dtype=np.float64
        ),
        where=row_sums != 0
    )

    # --------------------------------------------------------
    # Return all results
    # --------------------------------------------------------

    return {
        'mode': mode,
        'seed': seed,

        'accuracy': round_accuracy[-1],
        'macro_precision': round_precision[-1],
        'macro_recall': round_recall[-1],
        'macro_f1': round_f1[-1],

        'comm_mb': 0.0,
        'time_sec': total_time,

        'class_report': class_report,
        'confusion_matrix': cm,
        'cm_norm': cm_norm,

        'round_accuracy': round_accuracy,
        'round_f1': round_f1,
        'round_precision': round_precision,
        'round_recall': round_recall,
        'round_time': round_time
    }


# ============================================================
# 5. MAIN BENCHMARK EXECUTION LOOP
# PRINT RESULTS AFTER EACH SEED
# ============================================================

mode = "Centralized-Baseline"

all_summary_results = []
all_convergence_results = []
detailed_class_metrics = {mode: []}

acc_vals = []
f1_vals = []
prec_vals = []
rec_vals = []
comm_vals = []
time_vals = []

print("\n" + "#" * 100)
print(f"RUNNING METHOD: {mode}")
print(f"Seeds: {SEEDS}")
print("#" * 100)

for seed in SEEDS:

    print("\n" + "=" * 100)
    print(f"STARTING SEED: {seed}")
    print("=" * 100)

    # Run experiment
    res = run_fl_experiment(
        mode,
        seed
    )

    # --------------------------------------------------------
    # A. COLLECT OVERALL METRICS
    # --------------------------------------------------------

    acc_vals.append(res['accuracy'])
    f1_vals.append(res['macro_f1'])
    prec_vals.append(res['macro_precision'])
    rec_vals.append(res['macro_recall'])
    comm_vals.append(res['comm_mb'])
    time_vals.append(res['time_sec'])

    detailed_class_metrics[mode].append(res)

    # --------------------------------------------------------
    # B. PRINT FINAL METRICS FOR THIS SEED
    # --------------------------------------------------------

    print("\n" + "-" * 100)
    print(f"FINAL PERFORMANCE METRICS - SEED {seed}")
    print("-" * 100)

    print(f"Accuracy         : {res['accuracy']:.6f}")
    print(f"Macro Precision  : {res['macro_precision']:.6f}")
    print(f"Macro Recall     : {res['macro_recall']:.6f}")
    print(f"Macro F1-score   : {res['macro_f1']:.6f}")
    print(f"Communication MB : {res['comm_mb']:.4f}")
    print(f"Execution Time   : {res['time_sec']:.2f} seconds")

    # --------------------------------------------------------
    # C. PRINT ROUND-BY-ROUND CONVERGENCE
    # --------------------------------------------------------

    df_seed_rounds = pd.DataFrame({
        'Round': range(1, GLOBAL_ROUNDS + 1),
        'Accuracy': res['round_accuracy'],
        'Macro_Precision': res['round_precision'],
        'Macro_Recall': res['round_recall'],
        'Macro_F1': res['round_f1'],
        'Round_Time_Sec': res['round_time']
    })

    print("\n" + "-" * 100)
    print(f"CONVERGENCE METRICS - SEED {seed}")
    print("-" * 100)

    print(
        df_seed_rounds.round(6).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # D. PRINT CLASS-WISE METRICS
    # --------------------------------------------------------

    report_df = pd.DataFrame(
        res['class_report']
    ).T

    print("\n" + "-" * 100)
    print(f"CLASSIFICATION REPORT - SEED {seed}")
    print("-" * 100)

    print(
        report_df.round(6).to_string()
    )

    # --------------------------------------------------------
    # E. PRINT RAW CONFUSION MATRIX
    # --------------------------------------------------------

    df_raw_cm = pd.DataFrame(
        res['confusion_matrix'],
        index=[f'Class {i}' for i in class_labels],
        columns=[f'Class {i}' for i in class_labels]
    )

    print("\n" + "-" * 100)
    print(f"RAW CONFUSION MATRIX - SEED {seed}")
    print("-" * 100)

    print(df_raw_cm.to_string())

    # --------------------------------------------------------
    # F. PRINT NORMALIZED CONFUSION MATRIX
    # --------------------------------------------------------

    df_seed_cm = pd.DataFrame(
        res['cm_norm'],
        index=[f'Class {i}' for i in class_labels],
        columns=[f'Class {i}' for i in class_labels]
    )

    print("\n" + "-" * 100)
    print(f"NORMALIZED CONFUSION MATRIX - SEED {seed}")
    print("-" * 100)

    print(
        df_seed_cm.round(6).to_string()
    )

    # --------------------------------------------------------
    # G. SAVE ROUND RESULTS IN MEMORY
    # --------------------------------------------------------

    for r in range(GLOBAL_ROUNDS):

        all_convergence_results.append({
            'Method': mode,
            'Seed': seed,
            'Round': r + 1,
            'Accuracy': res['round_accuracy'][r],
            'Macro_F1': res['round_f1'][r],
            'Macro_Precision': res['round_precision'][r],
            'Macro_Recall': res['round_recall'][r],
            'Round_Time_Sec': res['round_time'][r]
        })

    print("\n" + "=" * 100)
    print(f"FINISHED SEED {seed}")
    print("=" * 100)

    # --------------------------------------------------------
    # H. RUNNING SUMMARY AFTER EACH SEED
    # --------------------------------------------------------

    print(f"\nRUNNING SUMMARY AFTER SEED {seed}")

    running_summary = pd.DataFrame({
        'Metric': [
            'Accuracy',
            'Macro Precision',
            'Macro Recall',
            'Macro F1',
            'Communication MB',
            'Execution Time (sec)'
        ],
        'Mean': [
            np.mean(acc_vals),
            np.mean(prec_vals),
            np.mean(rec_vals),
            np.mean(f1_vals),
            np.mean(comm_vals),
            np.mean(time_vals)
        ],
        'Std': [
            np.std(acc_vals),
            np.std(prec_vals),
            np.std(rec_vals),
            np.std(f1_vals),
            np.std(comm_vals),
            np.std(time_vals)
        ]
    })

    print(
        running_summary.round(6).to_string(
            index=False
        )
    )

# ============================================================
# 6. AGGREGATE SUMMARY ACROSS ALL SEEDS
# ============================================================

all_summary_results.append({
    'Method': mode,

    'Accuracy_Mean': np.mean(acc_vals),
    'Accuracy_Std': np.std(acc_vals),

    'Macro_F1_Mean': np.mean(f1_vals),
    'Macro_F1_Std': np.std(f1_vals),

    'Macro_Precision_Mean': np.mean(prec_vals),
    'Macro_Precision_Std': np.std(prec_vals),

    'Macro_Recall_Mean': np.mean(rec_vals),
    'Macro_Recall_Std': np.std(rec_vals),

    'Comm_Cost_MB_Mean': np.mean(comm_vals),
    'Comm_Cost_MB_Std': np.std(comm_vals),

    'Exec_Time_Sec_Mean': np.mean(time_vals),
    'Exec_Time_Sec_Std': np.std(time_vals)
})

# ============================================================
# 7. SAVE SUMMARY AND CONVERGENCE RESULTS
# ============================================================

df_summary = pd.DataFrame(
    all_summary_results
)

df_convergence = pd.DataFrame(
    all_convergence_results
)

summary_path = os.path.join(
    output_dir,
    'centralized_baseline_summary.csv'
)

convergence_path = os.path.join(
    output_dir,
    'centralized_baseline_convergence.csv'
)

df_summary.to_csv(
    summary_path,
    index=False
)

df_convergence.to_csv(
    convergence_path,
    index=False
)

# ============================================================
# 8. CLASS-WISE PERFORMANCE EXPORT
# ============================================================

reports = [
    run['class_report']
    for run in detailed_class_metrics[mode]
]

class_rows = []

for c in class_labels:

    c_str = str(c)

    p_vals = [
        report[c_str]['precision']
        for report in reports
    ]

    r_vals = [
        report[c_str]['recall']
        for report in reports
    ]

    f_vals = [
        report[c_str]['f1-score']
        for report in reports
    ]

    s_vals = [
        report[c_str]['support']
        for report in reports
    ]

    class_rows.append({
        'Method': mode,
        'Class': f'Class {c}',

        'Support_Mean': np.mean(s_vals),

        'Precision_Mean': np.mean(p_vals),
        'Precision_Std': np.std(p_vals),

        'Recall_Mean': np.mean(r_vals),
        'Recall_Std': np.std(r_vals),

        'F1_Mean': np.mean(f_vals),
        'F1_Std': np.std(f_vals)
    })

df_class = pd.DataFrame(
    class_rows
)

class_path = os.path.join(
    output_dir,
    'centralized_baseline_class_wise_performance.csv'
)

df_class.to_csv(
    class_path,
    index=False
)

# ============================================================
# 9. CONFUSION MATRIX EXPORT (CSV & PNG)
# ============================================================

matrices = [
    run['cm_norm']
    for run in detailed_class_metrics[mode]
]

# Average normalized confusion matrix across seeds
avg_cm = np.mean(
    matrices,
    axis=0
)

df_cm = pd.DataFrame(
    avg_cm,
    index=[f'Class {i}' for i in class_labels],
    columns=[f'Class {i}' for i in class_labels]
)

cm_csv_path = os.path.join(
    output_dir,
    'centralized_baseline_normalized_confusion_matrix.csv'
)

df_cm.to_csv(
    cm_csv_path
)

# ------------------------------------------------------------
# Generate average normalized confusion matrix plot
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(9, 8)
)

im = ax.imshow(
    avg_cm,
    interpolation='nearest',
    cmap='Blues',
    vmin=0,
    vmax=1
)

cbar = ax.figure.colorbar(
    im,
    ax=ax
)

cbar.ax.set_ylabel(
    'Normalized Value',
    rotation=-90,
    va="bottom"
)

ax.set(
    xticks=np.arange(num_classes),
    yticks=np.arange(num_classes),

    xticklabels=[
        f'Class {i}' for i in class_labels
    ],

    yticklabels=[
        f'Class {i}' for i in class_labels
    ],

    ylabel='True Label',
    xlabel='Predicted Label',

    title=(
        'Centralized Baseline\n'
        'Average Normalized Confusion Matrix'
    )
)

plt.setp(
    ax.get_xticklabels(),
    rotation=45,
    ha='right',
    rotation_mode='anchor'
)

threshold = avg_cm.max() / 2.0

for i in range(num_classes):
    for j in range(num_classes):

        ax.text(
            j,
            i,
            f'{avg_cm[i, j]:.2f}',
            ha='center',
            va='center',
            color=(
                'white'
                if avg_cm[i, j] > threshold
                else 'black'
            ),
            fontsize=11
        )

fig.tight_layout()

cm_png_path = os.path.join(
    output_dir,
    'centralized_baseline_normalized_confusion_matrix.png'
)

plt.savefig(
    cm_png_path,
    dpi=300,
    bbox_inches='tight'
)

plt.close(fig)

# ============================================================
# 10. FINAL SUMMARY OUTPUT
# ============================================================

print("\n" + "=" * 100)
print("CENTRALIZED BASELINE FINAL SUMMARY (Mean +/- Std across seeds)")
print("=" * 100)

print(
    df_summary.round(6).to_string(
        index=False
    )
)

print("\n" + "=" * 100)
print("CLASS-WISE PERFORMANCE SUMMARY")
print("=" * 100)

print(
    df_class.round(6).to_string(
        index=False
    )
)

print("\n" + "=" * 100)
print("AVERAGE NORMALIZED CONFUSION MATRIX")
print("=" * 100)

print(
    df_cm.round(6).to_string()
)

print("\n" + "=" * 100)
print("SAVED FILES")
print("=" * 100)

print(f"1. Summary CSV            : {summary_path}")
print(f"2. Convergence CSV        : {convergence_path}")
print(f"3. Class-wise Metrics CSV : {class_path}")
print(f"4. Confusion Matrix CSV   : {cm_csv_path}")
print(f"5. Confusion Matrix PNG   : {cm_png_path}")

print("\nAll centralized baseline experiments completed successfully.")
