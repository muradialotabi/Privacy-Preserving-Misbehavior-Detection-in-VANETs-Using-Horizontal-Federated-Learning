# ============================================================
# FEDERATED LEARNING BENCHMARK: FLAT-FEDPROX vs. HIER-FEDPROX
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
FEDPROX_MU = 0.01
TEST_SIZE = 0.20

# Hierarchical topology
HFL_CLUSTERS = {
    'Station_1': [0],
    'Station_2': [1],
    'Station_3': [2],
    'Station_4': [3],
    'Station_5': [4]
}

# ============================================================
# 2. DATASET PATHS & OUTPUT DIRECTORY
# ============================================================

file_paths = [
    r'C:\Users\Murad\PycharmProjects\federated-learning-tutorial\Strong_Quantity_Skew\10_Clients\client_data_1.csv',
    r'C:\Users\Murad\PycharmProjects\federated-learning-tutorial\Strong_Quantity_Skew\10_Clients\client_data_2.csv',
    r'C:\Users\Murad\PycharmProjects\federated-learning-tutorial\Strong_Quantity_Skew\10_Clients\client_data_3.csv',
    r'C:\Users\Murad\PycharmProjects\federated-learning-tutorial\Strong_Quantity_Skew\10_Clients\client_data_4.csv',
    r'C:\Users\Murad\PycharmProjects\federated-learning-tutorial\Strong_Quantity_Skew\10_Clients\client_data_5.csv' ,
    r'C:\Users\Murad\PycharmProjects\federated-learning-tutorial\Strong_Quantity_Skew\10_Clients\client_data_6.csv',
    r'C:\Users\Murad\PycharmProjects\federated-learning-tutorial\Strong_Quantity_Skew\10_Clients\client_data_7.csv',
    r'C:\Users\Murad\PycharmProjects\federated-learning-tutorial\Strong_Quantity_Skew\10_Clients\client_data_8.csv',
    r'C:\Users\Murad\PycharmProjects\federated-learning-tutorial\Strong_Quantity_Skew\10_Clients\client_data_9.csv',
    r'C:\Users\Murad\PycharmProjects\federated-learning-tutorial\Strong_Quantity_Skew\10_Clients\client_data_10.csv'


]



NUM_CLIENTS = len(file_paths)
output_dir = r'C:\Users\Murad\PycharmProjects\federated-learning-tutorial'
os.makedirs(output_dir, exist_ok=True)

label_mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}


# ============================================================
# 3. DATA LOADING & PREPROCESSING
# ============================================================

def load_client_data(paths):
    client_dfs = []
    for client_id, path in enumerate(paths):
        if not os.path.exists(path):
            raise FileNotFoundError(f"\nFile does not exist:\n{path}")
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError(f"Client {client_id + 1} dataset is empty.")
        if 'label' not in df.columns:
            raise ValueError(f"'label' column missing in Client {client_id + 1}")
        client_dfs.append(df)
    return client_dfs


client_dfs = load_client_data(file_paths)

# Combine and setup global distribution
combined_df = pd.concat(client_dfs, ignore_index=True)
combined_df['label'] = combined_df['label'].map(label_mapping)

if combined_df['label'].isna().any():
    raise ValueError("Unknown labels detected after label mapping.")

X_full_raw = combined_df.drop('label', axis=1).values.astype(np.float32)
y_full = combined_df['label'].values.astype(np.int32)
X_full_raw = np.nan_to_num(X_full_raw, nan=0.0, posinf=0.0, neginf=0.0)

# Global scaler fitting
scaler = StandardScaler()
scaler.fit(X_full_raw)

# Global Train/Test Split
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_full_raw,
    y_full,
    test_size=TEST_SIZE,
    random_state=42,
    stratify=y_full
)

X_train = scaler.transform(X_train_raw).astype(np.float32)
X_test = scaler.transform(X_test_raw).astype(np.float32)

input_shape = (X_train.shape[1],)
num_classes = len(np.unique(y_full))

# Client-level split
client_data_list = []
for client_id, df in enumerate(client_dfs):
    df = df.copy()
    df['label'] = df['label'].map(label_mapping)
    X_c_raw = df.drop('label', axis=1).values.astype(np.float32)
    y_c = df['label'].values.astype(np.int32)
    X_c_raw = np.nan_to_num(X_c_raw, nan=0.0, posinf=0.0, neginf=0.0)
    X_c = scaler.transform(X_c_raw).astype(np.float32)
    client_data_list.append((X_c, y_c))


# ============================================================
# 4. MODEL & TRAIN UTILITIES
# ============================================================

def create_mlp_model(input_shape, num_classes):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])


def calculate_model_size_mb(model):
    return (model.count_params() * 4) / (1024 * 1024)


MODEL_SIZE_MB = calculate_model_size_mb(create_mlp_model(input_shape, num_classes))


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


@tf.function(reduce_retracing=True)
def train_step_fedprox(model, optimizer, x_batch, y_batch, global_weights, mu):
    with tf.GradientTape() as tape:
        predictions = model(x_batch, training=True)
        classification_loss = tf.keras.losses.sparse_categorical_crossentropy(y_batch, predictions)
        classification_loss = tf.reduce_mean(classification_loss)

        prox_term = tf.constant(0.0, dtype=tf.float32)
        for local_var, global_var in zip(model.trainable_variables, global_weights):
            prox_term += tf.reduce_sum(tf.square(local_var - global_var))

        total_loss = classification_loss + (mu / 2.0) * prox_term

    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return total_loss


def train_client_fedprox(client_model, optimizer, global_weights, X_c, y_c):
    client_model.set_weights(global_weights)
    global_weights_tf = [tf.convert_to_tensor(w, dtype=tf.float32) for w in global_weights]

    dataset = (
        tf.data.Dataset
        .from_tensor_slices((X_c, y_c))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    for epoch in range(LOCAL_EPOCHS):
        for x_batch, y_batch in dataset:
            train_step_fedprox(
                client_model,
                optimizer,
                x_batch,
                y_batch,
                global_weights_tf,
                tf.constant(FEDPROX_MU, dtype=tf.float32)
            )

    return client_model.get_weights()


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
# 5. EXPERIMENT RUNNER (FLAT VS. HIERARCHICAL)
# ============================================================

def run_fl_experiment(mode, seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

    global_model = create_mlp_model(input_shape, num_classes)
    client_worker_model = create_mlp_model(input_shape, num_classes)
    client_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    client_optimizer.build(client_worker_model.trainable_variables)

    total_comm_mb = 0.0
    experiment_start = time.time()

    round_accuracy, round_f1, round_precision, round_recall, round_time = [], [], [], [], []

    for round_idx in range(GLOBAL_ROUNDS):
        round_start = time.time()
        global_weights = global_model.get_weights()

        if mode == 'Flat-FedProx':
            client_weights_list, client_sizes_list = [], []

            for client_idx in range(NUM_CLIENTS):
                X_c, y_c = client_data_list[client_idx]
                client_weights = train_client_fedprox(client_worker_model, client_optimizer, global_weights, X_c, y_c)

                client_weights_list.append(client_weights)
                client_sizes_list.append(len(X_c))

                # Client <-> Server payload
                total_comm_mb += (MODEL_SIZE_MB * 2)

            new_global_weights = aggregate_weights(client_weights_list, client_sizes_list)

        elif mode == 'Hier-FedProx':
            station_weights, station_sizes = [], []

            for station, client_indices in HFL_CLUSTERS.items():
                station_local_weights, station_local_sizes = [], []

                for client_idx in client_indices:
                    X_c, y_c = client_data_list[client_idx]
                    client_weights = train_client_fedprox(client_worker_model, client_optimizer, global_weights, X_c,
                                                          y_c)

                    station_local_weights.append(client_weights)
                    station_local_sizes.append(len(X_c))

                    # Client <-> Station
                    total_comm_mb += (MODEL_SIZE_MB * 2)

                station_agg_weights = aggregate_weights(station_local_weights, station_local_sizes)
                station_weights.append(station_agg_weights)
                station_sizes.append(sum(station_local_sizes))

                # Station <-> Global Server
                total_comm_mb += (MODEL_SIZE_MB * 2)

            new_global_weights = aggregate_weights(station_weights, station_sizes)

        global_model.set_weights(new_global_weights)
        round_result = evaluate_model(global_model)

        round_accuracy.append(round_result['accuracy'])
        round_f1.append(round_result['macro_f1'])
        round_precision.append(round_result['macro_precision'])
        round_recall.append(round_result['macro_recall'])

        elapsed_round = time.time() - round_start
        round_time.append(elapsed_round)

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

methods = ['Flat-FedProx', 'Hier-FedProx']
all_summary_results = []
all_convergence_results = []
detailed_class_metrics = {m: [] for m in methods}

print("\n" + "=" * 75)
print("STARTING COMBINED BENCHMARK: FLAT-FEDPROX VS HIER-FEDPROX")
print("=" * 75)

for mode in methods:
    print(f"\n>>> Running Method: {mode}")
    acc_vals, f1_vals, prec_vals, rec_vals, comm_vals, time_vals = [], [], [], [], [], []

    for seed in SEEDS:
        print(f"  Executing Seed {seed}...")
        res = run_fl_experiment(mode, seed)

        acc_vals.append(res['accuracy'])
        f1_vals.append(res['macro_f1'])
        prec_vals.append(res['macro_precision'])
        rec_vals.append(res['macro_recall'])
        comm_vals.append(res['comm_mb'])
        time_vals.append(res['time_sec'])

        detailed_class_metrics[mode].append(res)

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
        'Exec_Time_Sec_Std': np.std(time_vals)  # Corrected variable name
    })

# Save Summaries
df_summary = pd.DataFrame(all_summary_results)
df_summary.to_csv(os.path.join(output_dir, 'fl_comparison_summary.csv'), index=False)

df_convergence = pd.DataFrame(all_convergence_results)
df_convergence.to_csv(os.path.join(output_dir, 'fl_comparison_convergence.csv'), index=False)

# ============================================================
# 7. GENERATE COMPARISON PLOTS
# ============================================================

# Convergence Comparison Plot
plt.figure(figsize=(10, 6))
for mode in methods:
    df_m = df_convergence[df_convergence['Method'] == mode]
    avg_acc_per_round = df_m.groupby('Round')['Accuracy'].mean()
    plt.plot(avg_acc_per_round.index, avg_acc_per_round.values, label=mode, marker='o')

plt.title('Accuracy Convergence: Flat-FedProx vs Hier-FedProx')
plt.xlabel('Global Round')
plt.ylabel('Test Accuracy')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'convergence_comparison.png'), dpi=300)
plt.close()

# ============================================================
# 8. DISPLAY FINAL COMPARISON SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("FINAL COMPARISON SUMMARY (Flat-FedProx vs. Hier-FedProx)")
print("=" * 75)
print(df_summary.to_string(index=False))

print("\n" + "=" * 75)
print(f"[FINISHED] All results successfully saved to directory:\n{output_dir}")
print("=" * 75)