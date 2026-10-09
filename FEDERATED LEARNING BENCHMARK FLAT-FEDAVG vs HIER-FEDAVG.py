import os
import time
import random
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
# إيقاف تحذيرات oneDNN وضبط مستوى سجلات TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
    r'C:\Users\Murad\PycharmProjects\federated-learning-tutorial\HFL_NonIID_5Clients\client_data_1.csv',
    r'C:\Users\Murad\PycharmProjects\federated-learning-tutorial\HFL_NonIID_5Clients\client_data_2.csv',
    r'C:\Users\Murad\PycharmProjects\federated-learning-tutorial\HFL_NonIID_5Clients\client_data_3.csv',
    r'C:\Users\Murad\PycharmProjects\federated-learning-tutorial\HFL_NonIID_5Clients\client_data_4.csv',
    r'C:\Users\Murad\PycharmProjects\federated-learning-tutorial\HFL_NonIID_5Clients\client_data_5.csv'
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
        client_dfs.append(df)
    return client_dfs


client_dfs = load_client_data(file_paths)

X_train_list, y_train_list = [], []
X_test_list, y_test_list = [], []

for client_id, df in enumerate(client_dfs):
    df = df.copy()
    df['label'] = df['label'].map(label_mapping)

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
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='sparse_categorical_crossentropy')
    model.fit(X_c, y_c, epochs=LOCAL_EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    return model.get_weights()


def evaluate_model(model):
    probabilities = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    return {
        'accuracy': accuracy_score(y_test, predictions),
        'macro_precision': precision_score(y_test, predictions, average='macro', zero_division=0),
        'macro_recall': recall_score(y_test, predictions, average='macro', zero_division=0),
        'macro_f1': f1_score(y_test, predictions, average='macro', zero_division=0),
        'class_report': classification_report(y_test, predictions, labels=np.arange(num_classes), output_dict=True,
                                              zero_division=0),
        'cm_norm': confusion_matrix(y_test, predictions, labels=np.arange(num_classes), normalize='true')
    }


# ============================================================
# 5. EXPERIMENT RUNNER
# ============================================================

def run_experiment(method, seed):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

    global_model = create_mlp_model(input_shape, num_classes)
    total_comm_mb = 0.0
    experiment_start = time.time()

    round_accuracy, round_f1 = [], []

    for round_idx in range(GLOBAL_ROUNDS):
        global_weights = global_model.get_weights()

        if method == 'Flat-FedAvg':
            local_weights, local_sizes = [], []
            for client_idx in range(NUM_CLIENTS):
                X_c, y_c = client_data_list[client_idx]
                client_weights = train_client_fedavg(global_weights, X_c, y_c)
                local_weights.append(client_weights)
                local_sizes.append(len(X_c))
                total_comm_mb += MODEL_SIZE_MB * 2

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
        round_accuracy.append(round_res['accuracy'])
        round_f1.append(round_res['macro_f1'])

    final_result = evaluate_model(global_model)
    return {
        'accuracy': final_result['accuracy'], 'macro_precision': final_result['macro_precision'],
        'macro_recall': final_result['macro_recall'], 'macro_f1': final_result['macro_f1'],
        'comm_mb': total_comm_mb, 'time_sec': time.time() - experiment_start,
        'round_accuracy': round_accuracy, 'round_f1': round_f1
    }


# ============================================================
# 6. EXECUTION
# ============================================================

methods = ['Flat-FedAvg', 'Hier-FedAvg']
summary_results = []

for method in methods:
    accs, precs, recs, f1s, comms, times = [], [], [], [], [], []
    for seed in SEEDS:
        res = run_experiment(method, seed)
        accs.append(res['accuracy'])
        precs.append(res['macro_precision'])
        recs.append(res['macro_recall'])
        f1s.append(res['macro_f1'])
        comms.append(res['comm_mb'])
        times.append(res['time_sec'])

    summary_results.append({
        'Method': method,
        'Accuracy_Mean': np.mean(accs), 'Accuracy_Std': np.std(accs),
        'Macro_Precision_Mean': np.mean(precs), 'Macro_Precision_Std': np.std(precs),
        'Macro_Recall_Mean': np.mean(recs), 'Macro_Recall_Std': np.std(recs),
        'Macro_F1_Mean': np.mean(f1s), 'Macro_F1_Std': np.std(f1s),
        'Comm_Cost_MB_Mean': np.mean(comms), 'Exec_Time_Sec_Mean': np.mean(times)
    })

pd.DataFrame(summary_results).to_csv(os.path.join(output_dir, 'fedavg_benchmark_summary.csv'), index=False)
print("FedAvg Benchmark Execution Finished Successfully!")