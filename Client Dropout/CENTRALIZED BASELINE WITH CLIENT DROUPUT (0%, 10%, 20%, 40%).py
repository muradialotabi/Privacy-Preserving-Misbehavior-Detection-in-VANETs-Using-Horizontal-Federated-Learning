# ============================================================
# CENTRALIZED BASELINE WITH CLIENT DROUPUT (0%, 10%, 20%, 40%)
# FULL CODE WITH CONFUSION MATRIX & CLASS-WISE METRICS PER DROPOUT
# Prints results after each seed and exports artifacts per dropout rate
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

SEEDS = [42, 100]
DROPOUT_RATES = [ 0.10, 0.20, 0.40]  # 0%, 10%, 20%, 40%

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
    5: 5
}

# ============================================================
# FILE PATHS (10 Clients)
# ============================================================

# ============================================================
# FILE PATHS (20 Clients)
# ============================================================

file_paths = [
    r'C:\Users\Murad\PycharmProjects\federated-learning-tutorial\Strong_Quantity_Skew\20_Clients\client_data_1.csv',
    r'C:\Users\Murad\PycharmProjects\federated-learning-tutorial\Strong_Quantity_Skew\20_Clients\client_data_2.csv',
    r'C:\Users\Murad\PycharmProjects\federated-learning-tutorial\Strong_Quantity_Skew\20_Clients\client_data_3.csv',
    r'C:\Users\Murad\PycharmProjects\federated-learning-tutorial\Strong_Quantity_Skew\20_Clients\client_data_4.csv',
    r'C:\Users\Murad\PycharmProjects\federated-learning-tutorial\Strong_Quantity_Skew\20_Clients\client_data_5.csv',
    r'C:\Users\Murad\PycharmProjects\federated-learning-tutorial\Strong_Quantity_Skew\20_Clients\client_data_6.csv',
    r'C:\Users\Murad\PycharmProjects\federated-learning-tutorial\Strong_Quantity_Skew\20_Clients\client_data_7.csv',
    r'C:\Users\Murad\PycharmProjects\federated-learning-tutorial\Strong_Quantity_Skew\20_Clients\client_data_8.csv',
    r'C:\Users\Murad\PycharmProjects\federated-learning-tutorial\Strong_Quantity_Skew\20_Clients\client_data_9.csv',
    r'C:\Users\Murad\PycharmProjects\federated-learning-tutorial\Strong_Quantity_Skew\20_Clients\client_data_10.csv',
    r'C:\Users\Murad\PycharmProjects\federated-learning-tutorial\Strong_Quantity_Skew\20_Clients\client_data_11.csv',
    r'C:\Users\Murad\PycharmProjects\federated-learning-tutorial\Strong_Quantity_Skew\20_Clients\client_data_12.csv',
    r'C:\Users\Murad\PycharmProjects\federated-learning-tutorial\Strong_Quantity_Skew\20_Clients\client_data_13.csv',
    r'C:\Users\Murad\PycharmProjects\federated-learning-tutorial\Strong_Quantity_Skew\20_Clients\client_data_14.csv',
    r'C:\Users\Murad\PycharmProjects\federated-learning-tutorial\Strong_Quantity_Skew\20_Clients\client_data_15.csv',
    r'C:\Users\Murad\PycharmProjects\federated-learning-tutorial\Strong_Quantity_Skew\20_Clients\client_data_16.csv',
    r'C:\Users\Murad\PycharmProjects\federated-learning-tutorial\Strong_Quantity_Skew\20_Clients\client_data_17.csv',
    r'C:\Users\Murad\PycharmProjects\federated-learning-tutorial\Strong_Quantity_Skew\20_Clients\client_data_18.csv',
    r'C:\Users\Murad\PycharmProjects\federated-learning-tutorial\Strong_Quantity_Skew\20_Clients\client_data_19.csv',
    r'C:\Users\Murad\PycharmProjects\federated-learning-tutorial\Strong_Quantity_Skew\20_Clients\client_data_20.csv'
]

# Output Directory
output_dir = r'C:\Users\Murad\PycharmProjects\federated-learning-tutorial\Centralized_Dropout_Results'
os.makedirs(output_dir, exist_ok=True)

# Fixed class labels
class_labels = np.array(sorted(label_mapping.values()))
num_classes = len(class_labels)


# ============================================================
# 2. MODEL BUILDER
# ============================================================

def create_mlp_model(input_shape, num_classes, hidden_layers):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))

    for units in hidden_layers:
        model.add(tf.keras.layers.Dense(units, activation='relu'))

    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ============================================================
# 3. SINGLE RUN EXPERIMENT FUNCTION
# ============================================================

def run_centralized_experiment(X_train, y_train, X_test, y_test, input_shape, mode_name, seed):
    """
    Executes centralized training for a given dataset split and seed.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    tf.keras.backend.clear_session()
    model = create_mlp_model(input_shape, num_classes, HIDDEN_LAYERS)

    experiment_start = time.time()

    round_accuracy, round_f1, round_precision, round_recall, round_time = [], [], [], [], []
    epochs_per_round = LOCAL_EPOCHS

    for r in range(GLOBAL_ROUNDS):
        round_start = time.time()

        model.fit(
            X_train, y_train,
            epochs=epochs_per_round,
            batch_size=BATCH_SIZE,
            verbose=0
        )

        elapsed_round = time.time() - round_start

        y_pred_probs = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, labels=class_labels, average='macro', zero_division=0)
        rec = recall_score(y_test, y_pred, labels=class_labels, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, labels=class_labels, average='macro', zero_division=0)

        round_accuracy.append(acc)
        round_precision.append(prec)
        round_recall.append(rec)
        round_f1.append(f1)
        round_time.append(elapsed_round)

    total_time = time.time() - experiment_start

    class_report = classification_report(
        y_test, y_pred, labels=class_labels, output_dict=True, zero_division=0
    )

    cm = confusion_matrix(y_test, y_pred, labels=class_labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm.astype(np.float64),
        row_sums,
        out=np.zeros_like(cm, dtype=np.float64),
        where=row_sums != 0
    )

    return {
        'mode': mode_name,
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
# 4. MAIN EXPERIMENT LOOP ACROSS ALL DROPOUT RATES
# ============================================================

total_clients = len(file_paths)
all_summary_results = []
all_convergence_results = []

print("\n" + "#" * 100)
print(f"STARTING CENTRALIZED CLIENT DROPOUT BENCHMARK")
print(f"Total Clients: {total_clients} | Dropout Rates: {[f'{int(d * 100)}%' for d in DROPOUT_RATES]}")
print(f"Seeds: {SEEDS}")
print("#" * 100)

for dropout_rate in DROPOUT_RATES:
    dropout_pct = int(dropout_rate * 100)
    num_dropped = int(np.round(total_clients * dropout_rate))
    num_participating = total_clients - num_dropped
    mode_name = f"Centralized-Baseline-Dropout-{dropout_pct}pct"

    print("\n" + "=" * 100)
    print(f"RUNNING DROPOUT CONFIGURATION: {dropout_pct}% DROPOUT ({num_participating}/{total_clients} Clients Active)")
    print("=" * 100)

    # --------------------------------------------------------
    # Select participating client files deterministically
    # --------------------------------------------------------
    participating_paths = file_paths[:num_participating]

    print(f"Loading data from {len(participating_paths)} active client file(s)...")
    df_list = [pd.read_csv(p) for p in participating_paths if os.path.exists(p)]
    combined_df = pd.concat(df_list, ignore_index=True)

    combined_df['label'] = combined_df['label'].map(label_mapping)
    X_raw = combined_df.drop('label', axis=1).values.astype(np.float32)
    y = combined_df['label'].values.astype(np.int32)
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

    # Split and Scale
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_test = scaler.transform(X_test_raw).astype(np.float32)
    input_shape = (X_train.shape[1],)

    acc_vals, f1_vals, prec_vals, rec_vals, comm_vals, time_vals = [], [], [], [], [], []
    seed_runs = []

    for seed in SEEDS:
        print(f"\n---> Starting Seed {seed} for {dropout_pct}% Dropout")
        res = run_centralized_experiment(X_train, y_train, X_test, y_test, input_shape, mode_name, seed)

        acc_vals.append(res['accuracy'])
        f1_vals.append(res['macro_f1'])
        prec_vals.append(res['macro_precision'])
        rec_vals.append(res['macro_recall'])
        comm_vals.append(res['comm_mb'])
        time_vals.append(res['time_sec'])

        seed_runs.append(res)

        # Print metrics for this seed
        print(
            f"Seed {seed} Results -> Acc: {res['accuracy']:.6f} | Macro F1: {res['macro_f1']:.6f} | Time: {res['time_sec']:.2f}s")

        # Save round-by-round convergence data
        for r in range(GLOBAL_ROUNDS):
            all_convergence_results.append({
                'Dropout_Rate': f"{dropout_pct}%",
                'Method': mode_name,
                'Seed': seed,
                'Round': r + 1,
                'Accuracy': res['round_accuracy'][r],
                'Macro_F1': res['round_f1'][r],
                'Macro_Precision': res['round_precision'][r],
                'Macro_Recall': res['round_recall'][r],
                'Round_Time_Sec': res['round_time'][r]
            })

    # Record overall summary metrics for this dropout rate
    all_summary_results.append({
        'Dropout_Rate': f"{dropout_pct}%",
        'Total_Clients': total_clients,
        'Dropped_Clients': num_dropped,
        'Participating_Clients': num_participating,
        'Method': mode_name,
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
    # EXPORT CLASS-WISE METRICS FOR THIS DROPOUT RATE
    # ============================================================
    reports = [run['class_report'] for run in seed_runs]
    class_rows = []

    for c in class_labels:
        c_str = str(c)
        p_vals = [rep[c_str]['precision'] for rep in reports]
        r_vals = [rep[c_str]['recall'] for rep in reports]
        f_vals = [rep[c_str]['f1-score'] for rep in reports]
        s_vals = [rep[c_str]['support'] for rep in reports]

        class_rows.append({
            'Dropout_Rate': f"{dropout_pct}%",
            'Class': f'Class {c}',
            'Support_Mean': np.mean(s_vals),
            'Precision_Mean': np.mean(p_vals),
            'Precision_Std': np.std(p_vals),
            'Recall_Mean': np.mean(r_vals),
            'Recall_Std': np.std(r_vals),
            'F1_Mean': np.mean(f_vals),
            'F1_Std': np.std(f_vals)
        })

    df_class_dropout = pd.DataFrame(class_rows)
    class_path = os.path.join(output_dir, f'class_wise_metrics_dropout_{dropout_pct}pct.csv')
    df_class_dropout.to_csv(class_path, index=False)

    print(f"\n--- Class-Wise Metrics ({dropout_pct}% Dropout) ---")
    print(df_class_dropout.round(6).to_string(index=False))

    # ============================================================
    # EXPORT CONFUSION MATRIX FOR THIS DROPOUT RATE (CSV & PNG)
    # ============================================================
    avg_cm = np.mean([run['cm_norm'] for run in seed_runs], axis=0)
    df_cm = pd.DataFrame(
        avg_cm,
        index=[f'Class {i}' for i in class_labels],
        columns=[f'Class {i}' for i in class_labels]
    )

    cm_csv_path = os.path.join(output_dir, f'normalized_confusion_matrix_dropout_{dropout_pct}pct.csv')
    df_cm.to_csv(cm_csv_path)

    print(f"\n--- Average Normalized Confusion Matrix ({dropout_pct}% Dropout) ---")
    print(df_cm.round(6).to_string())

    # Plot & Save PNG
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(avg_cm, interpolation='nearest', cmap='viridis', vmin=0, vmax=1)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Normalized Value', rotation=-90, va="bottom")

    ax.set(
        xticks=np.arange(num_classes),
        yticks=np.arange(num_classes),
        xticklabels=[f'Class {i}' for i in class_labels],
        yticklabels=[f'Class {i}' for i in class_labels],
        ylabel='True Label',
        xlabel='Predicted Label',
        title=f'Centralized Baseline ({dropout_pct}% Client Dropout)\nNormalized Confusion Matrix'
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    threshold = avg_cm.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j, i, f'{avg_cm[i, j]:.2f}',
                ha='center', va='center',
                color='black' if avg_cm[i, j] > threshold else 'white',
                fontsize=11
            )

    fig.tight_layout()
    cm_png_path = os.path.join(output_dir, f'confusion_matrix_dropout_{dropout_pct}pct.png')
    plt.savefig(cm_png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Artifacts saved for {dropout_pct}% Dropout: {cm_csv_path} & {cm_png_path}")

# ============================================================
# 5. GLOBAL EXPORT FOR SUMMARY & CONVERGENCE
# ============================================================

df_summary_all = pd.DataFrame(all_summary_results)
df_convergence_all = pd.DataFrame(all_convergence_results)

summary_path = os.path.join(output_dir, 'centralized_dropout_summary_all.csv')
convergence_path = os.path.join(output_dir, 'centralized_dropout_convergence_all.csv')

df_summary_all.to_csv(summary_path, index=False)
df_convergence_all.to_csv(convergence_path, index=False)

print("\n" + "=" * 100)
print("FINAL SUMMARY ACROSS ALL DROPOUT RATES (Mean +/- Std)")
print("=" * 100)
print(df_summary_all.round(6).to_string(index=False))

print(f"\nAll experiments finished. Global summary saved to: {summary_path}")