import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# Suppress TensorFlow log noise & oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

NUM_CLIENTS = 5
GLOBAL_ROUNDS = 10
LOCAL_EPOCHS = 3
BATCH_SIZE = 256
LEARNING_RATE = 0.001
TEST_SIZE = 0.20

# Hierarchical Topology
HFL_CLUSTERS = {
    'Station_1': [0],
    'Station_2': [1],
    'Station_3': [2],
    'Station_4': [3],
    'Station_5': [4]
}

# ============================================================
# 2. DATASET PATHS & SETUP
# ============================================================


file_paths = [

r'C:\Users\Murad\PycharmProjects\federated-learning-tutorial\extreme_label_skew_5clients\client_1_extreme_noniid.csv',
r'C:\Users\Murad\PycharmProjects\federated-learning-tutorial\extreme_label_skew_5clients\client_2_extreme_noniid.csv',
r'C:\Users\Murad\PycharmProjects\federated-learning-tutorial\extreme_label_skew_5clients\client_3_extreme_noniid.csv',
r'C:\Users\Murad\PycharmProjects\federated-learning-tutorial\extreme_label_skew_5clients\client_4_extreme_noniid.csv',
r'C:\Users\Murad\PycharmProjects\federated-learning-tutorial\extreme_label_skew_5clients\client_5_extreme_noniid.csv'


]

output_dir = r'C:\Users\Murad\PycharmProjects\federated-learning-tutorial\FedAvg_Results'
os.makedirs(output_dir, exist_ok=True)

label_mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 9: 5}


# ============================================================
# 3. LOAD & PREPROCESS DATA (NO DATA LEAKAGE)
# ============================================================

def load_client_data(paths):
    client_dfs = []
    for client_id, path in enumerate(paths):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File does not exist:\n{path}")
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError(f"Client {client_id + 1} dataset is empty.")
        if 'label' not in df.columns:
            raise ValueError(f"'label' column missing in Client {client_id + 1}")
        client_dfs.append(df)
    return client_dfs


print("=" * 75)
print("LOADING DATASETS & PREPROCESSING (FEDAVG)")
print("=" * 75)

client_dfs = load_client_data(file_paths)

X_train_list, y_train_list = [], []
X_test_list, y_test_list = [], []

for client_id, df in enumerate(client_dfs):
    df = df.copy()
    df['label'] = df['label'].map(label_mapping)

    if df['label'].isna().any():
        raise ValueError(f"Unknown labels detected in Client {client_id + 1}.")

    X_c_raw = np.nan_to_num(df.drop('label', axis=1).values.astype(np.float32))
    y_c = df['label'].values.astype(np.int32)

    # Split per client first to prevent test data leakage
    X_c_tr_raw, X_c_te_raw, y_c_tr, y_c_te = train_test_split(
        X_c_raw, y_c, test_size=TEST_SIZE, random_state=42, stratify=y_c
    )

    X_train_list.append(X_c_tr_raw)
    y_train_list.append(y_c_tr)
    X_test_list.append(X_c_te_raw)
    y_test_list.append(y_c_te)

# Fit scaler strictly on combined training data
scaler = StandardScaler()
scaler.fit(np.vstack(X_train_list))

X_test = scaler.transform(np.vstack(X_test_list)).astype(np.float32)
y_test = np.hstack(y_test_list)

client_data_list = [
    (scaler.transform(tr_x).astype(np.float32), tr_y)
    for tr_x, tr_y in zip(X_train_list, y_train_list)
]

input_shape = (X_test.shape[1],)
num_classes = len(np.unique(y_test))


# ============================================================
# 4. MODEL & AGGREGATION
# ============================================================

def create_mlp_model(input_shape, num_classes):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])


MODEL_SIZE_MB = (create_mlp_model(input_shape, num_classes).count_params() * 4) / (1024 * 1024)


def aggregate_weights(weights_list, sizes_list):
    total_samples = sum(sizes_list)
    aggregated_weights = []
    for layer_idx in range(len(weights_list[0])):
        layer_avg = np.zeros_like(weights_list[0][layer_idx])
        for client_idx in range(len(weights_list)):
            factor = sizes_list[client_idx] / total_samples
            layer_avg += factor * weights_list[client_idx][layer_idx]
        aggregated_weights.append(layer_avg)
    return aggregated_weights


def train_client_fedavg(global_weights, X_c, y_c):
    model = create_mlp_model(input_shape, num_classes)
    model.set_weights(global_weights)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy'
    )
    model.fit(X_c, y_c, epochs=LOCAL_EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    return model.get_weights()


def evaluate_model(model):
    probabilities = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    accuracy = accuracy_score(y_test, predictions)
    macro_precision = precision_score(y_test, predictions, average='macro', zero_division=0)
    macro_recall = recall_score(y_test, predictions, average='macro', zero_division=0)
    macro_f1 = f1_score(y_test, predictions, average='macro', zero_division=0)

    class_report = classification_report(
        y_test,
        predictions,
        labels=np.arange(num_classes),
        output_dict=True,
        zero_division=0
    )

    cm = confusion_matrix(y_test, predictions, labels=np.arange(num_classes))
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm.astype(np.float64),
        row_sums,
        out=np.zeros_like(cm, dtype=np.float64),
        where=row_sums != 0
    )

    return {
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'class_report': class_report,
        'cm_norm': cm_norm
    }


# ============================================================
# 5. EXPERIMENT RUNNER
# ============================================================

def run_experiment(method, seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

    global_model = create_mlp_model(input_shape, num_classes)
    total_comm_mb = 0.0
    experiment_start = time.time()

    round_accuracy, round_f1, round_precision, round_recall, round_time = [], [], [], [], []

    print(f"\n" + "-" * 75)
    print(f"  Starting Experiment Mode: {method} | Seed: {seed}")
    print("-" * 75)

    for round_idx in range(GLOBAL_ROUNDS):
        round_start = time.time()
        global_weights = global_model.get_weights()

        if method == 'Flat-FedAvg':
            local_weights, local_sizes = [], []
            for client_idx in range(NUM_CLIENTS):
                X_c, y_c = client_data_list[client_idx]
                client_weights = train_client_fedavg(global_weights, X_c, y_c)
                local_weights.append(client_weights)
                local_sizes.append(len(X_c))
                total_comm_mb += MODEL_SIZE_MB * 2  # Client <-> Cloud

            new_global_weights = aggregate_weights(local_weights, local_sizes)

        elif method == 'Hier-FedAvg':
            station_weights, station_sizes = [], []
            for station, client_indices in HFL_CLUSTERS.items():
                station_local_weights, station_local_sizes = [], []
                for client_idx in client_indices:
                    X_c, y_c = client_data_list[client_idx]
                    client_weights = train_client_fedavg(global_weights, X_c, y_c)
                    station_local_weights.append(client_weights)
                    station_local_sizes.append(len(X_c))
                    total_comm_mb += MODEL_SIZE_MB * 2  # Client <-> Station

                station_agg_weights = aggregate_weights(station_local_weights, station_local_sizes)
                station_weights.append(station_agg_weights)
                station_sizes.append(sum(station_local_sizes))
                total_comm_mb += MODEL_SIZE_MB * 2  # Station <-> Cloud

            new_global_weights = aggregate_weights(station_weights, station_sizes)

        global_model.set_weights(new_global_weights)
        round_res = evaluate_model(global_model)

        elapsed_round = time.time() - round_start

        round_accuracy.append(round_res['accuracy'])
        round_f1.append(round_res['macro_f1'])
        round_precision.append(round_res['macro_precision'])
        round_recall.append(round_res['macro_recall'])
        round_time.append(elapsed_round)

        # ----------------------------------------------------
        # PRINT ROUND SUMMARY
        # ----------------------------------------------------
        print(f"  [Round {round_idx + 1:02d}/{GLOBAL_ROUNDS:02d}] "
              f"Time: {elapsed_round:6.2f}s | "
              f"Acc: {round_res['accuracy'] * 100:6.2f}% | "
              f"F1: {round_res['macro_f1']:6.4f} | "
              f"Precision: {round_res['macro_precision']:6.4f} | "
              f"Recall: {round_res['macro_recall']:6.4f} | "
              f"Comm: {total_comm_mb:8.2f} MB")

    final_result = evaluate_model(global_model)
    total_time = time.time() - experiment_start

    return {
        'accuracy': final_result['accuracy'],
        'macro_precision': final_result['macro_precision'],
        'macro_recall': final_result['macro_recall'],
        'macro_f1': final_result['macro_f1'],
        'comm_mb': total_comm_mb,
        'time_sec': total_time,
        'class_report': final_result['class_report'],
        'cm_norm': final_result['cm_norm'],
        'round_accuracy': round_accuracy,
        'round_f1': round_f1,
        'round_precision': round_precision,
        'round_recall': round_recall,
        'round_time': round_time
    }


# ============================================================
# 6. EXECUTE COMBINED BENCHMARK
# ============================================================

methods = ['Flat-FedAvg', 'Hier-FedAvg']
summary_results = []
all_convergence_results = []
detailed_class_metrics = {m: [] for m in methods}

print("\n" + "=" * 75)
print("STARTING BENCHMARK: FLAT-FEDAVG VS HIER-FEDAVG")
print("=" * 75)

for method in methods:
    print(f"\n>>> Running Method: {method}")
    accs, precs, recs, f1s, comms, times = [], [], [], [], [], []

    for seed in SEEDS:
        res = run_experiment(method, seed)

        accs.append(res['accuracy'])
        precs.append(res['macro_precision'])
        recs.append(res['macro_recall'])
        f1s.append(res['macro_f1'])
        comms.append(res['comm_mb'])
        times.append(res['time_sec'])

        detailed_class_metrics[method].append(res)

        for r in range(GLOBAL_ROUNDS):
            all_convergence_results.append({
                'Method': method,
                'Seed': seed,
                'Round': r + 1,
                'Accuracy': res['round_accuracy'][r],
                'Macro_F1': res['round_f1'][r],
                'Macro_Precision': res['round_precision'][r],
                'Macro_Recall': res['round_recall'][r],
                'Round_Time_Sec': res['round_time'][r]
            })

    summary_results.append({
        'Method': method,
        'Accuracy_Mean': np.mean(accs),
        'Accuracy_Std': np.std(accs),
        'Macro_Precision_Mean': np.mean(precs),
        'Macro_Precision_Std': np.std(precs),
        'Macro_Recall_Mean': np.mean(recs),
        'Macro_Recall_Std': np.std(recs),
        'Macro_F1_Mean': np.mean(f1s),
        'Macro_F1_Std': np.std(f1s),
        'Comm_Cost_MB_Mean': np.mean(comms),
        'Comm_Cost_MB_Std': np.std(comms),
        'Exec_Time_Sec_Mean': np.mean(times),
        'Exec_Time_Sec_Std': np.std(times)
    })

# Save Summaries
df_summary = pd.DataFrame(summary_results)
df_summary.to_csv(os.path.join(output_dir, 'fedavg_benchmark_summary.csv'), index=False)

df_convergence = pd.DataFrame(all_convergence_results)
df_convergence.to_csv(os.path.join(output_dir, 'fedavg_benchmark_convergence.csv'), index=False)


# ============================================================
# 7. EXPORT CLASS-WISE METRICS & CONFUSION MATRICES
# ============================================================

for method in methods:
    mode_slug = method.lower().replace('-', '_')

    # --- Class-Wise Metrics Export ---
    reports = [run['class_report'] for run in detailed_class_metrics[method]]
    class_rows = []

    for c in range(num_classes):
        c_str = str(c)
        p_vals = [r[c_str]['precision'] for r in reports]
        r_vals = [r[c_str]['recall'] for r in reports]
        f_vals = [r[c_str]['f1-score'] for r in reports]
        s_vals = [r[c_str]['support'] for r in reports]

        class_rows.append({
            'Method': method,
            'Class': f'Class {c}',
            'Support': np.mean(s_vals),
            'Precision_Mean': np.mean(p_vals),
            'Precision_Std': np.std(p_vals),
            'Recall_Mean': np.mean(r_vals),
            'Recall_Std': np.std(r_vals),
            'F1_Mean': np.mean(f_vals),
            'F1_Std': np.std(f_vals)
        })

    df_class = pd.DataFrame(class_rows)
    df_class.to_csv(os.path.join(output_dir, f'{mode_slug}_class_wise_performance.csv'), index=False)

    # --- Confusion Matrix Export (CSV & PNG) ---
    matrices = [run['cm_norm'] for run in detailed_class_metrics[method]]
    avg_cm = np.mean(matrices, axis=0)

    df_cm = pd.DataFrame(
        avg_cm,
        index=[f'Class {i}' for i in range(num_classes)],
        columns=[f'Class {i}' for i in range(num_classes)]
    )
    df_cm.to_csv(os.path.join(output_dir, f'{mode_slug}_normalized_confusion_matrix.csv'))

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(avg_cm, interpolation='nearest', cmap='Blues')
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Normalized Value', rotation=-90, va="bottom")

    ax.set(
        xticks=np.arange(num_classes),
        yticks=np.arange(num_classes),
        xticklabels=[f'Class {i}' for i in range(num_classes)],
        yticklabels=[f'Class {i}' for i in range(num_classes)],
        ylabel='True Label',
        xlabel='Predicted Label',
        title=f'{method}\nAverage Normalized Confusion Matrix'
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    threshold = avg_cm.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j, i, f'{avg_cm[i, j]:.2f}',
                ha='center', va='center',
                color='white' if avg_cm[i, j] > threshold else 'black',
                fontsize=11
            )

    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{mode_slug}_normalized_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# 8. GENERATE COMPARISON PLOTS
# ============================================================

plt.figure(figsize=(10, 6))
for method in methods:
    df_m = df_convergence[df_convergence['Method'] == method]
    avg_acc_per_round = df_m.groupby('Round')['Accuracy'].mean()
    plt.plot(avg_acc_per_round.index, avg_acc_per_round.values, label=method, marker='o')

plt.title('Accuracy Convergence: Flat-FedAvg vs Hier-FedAvg')
plt.xlabel('Global Round')
plt.ylabel('Test Accuracy')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fedavg_convergence_comparison.png'), dpi=300)
plt.close()


# ============================================================
# 9. DISPLAY FINAL COMPARISON SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("FINAL COMPARISON SUMMARY (Flat-FedAvg vs. Hier-FedAvg)")
print("=" * 75)
print(df_summary.to_string(index=False))

print("\n" + "=" * 75)
print(f"[FINISHED] All results successfully saved to directory:\n{output_dir}")
print("=" * 75)