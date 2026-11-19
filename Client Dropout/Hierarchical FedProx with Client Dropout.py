# ============================================================
# Hierarchical FedProx with Client Dropout
# Strong Quantity Skew - 20 Clients
#
# Dropout Scenarios:
# 10% -> 18 active / 2 dropped
# 20% -> 16 active / 4 dropped
# 40% -> 12 active / 8 dropped
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

NUM_CLIENTS = 20

GLOBAL_ROUNDS = 10

LOCAL_EPOCHS = 3

BATCH_SIZE = 256

LEARNING_RATE = 0.001

FEDPROX_MU = 0.01

TEST_SIZE = 0.20

# ============================================================
# 2. CLIENT DROPOUT SCENARIOS
# ============================================================

DROPOUT_SCENARIOS = {
    0.10: {
        'num_dropped': 2,
        'num_active': 18
    },

    0.20: {
        'num_dropped': 4,
        'num_active': 16
    },

    0.40: {
        'num_dropped': 8,
        'num_active': 12
    }
}

# Safety checks
for dropout_rate, scenario in DROPOUT_SCENARIOS.items():

    expected_dropped = int(
        NUM_CLIENTS * dropout_rate
    )

    expected_active = (
        NUM_CLIENTS - expected_dropped
    )

    if (
        scenario['num_dropped']
        != expected_dropped
    ):
        raise ValueError(
            f"Invalid dropout configuration "
            f"for {dropout_rate * 100:.0f}%."
        )

    if (
        scenario['num_active']
        != expected_active
    ):
        raise ValueError(
            f"Invalid active-client configuration "
            f"for {dropout_rate * 100:.0f}%."
        )

# ============================================================
# 3. HIERARCHICAL TOPOLOGY
# ============================================================

HFL_CLUSTERS = {

    'Station_1': [
        0, 1, 2, 3
    ],

    'Station_2': [
        4, 5, 6, 7
    ],

    'Station_3': [
        8, 9, 10, 11
    ],

    'Station_4': [
        12, 13, 14, 15
    ],

    'Station_5': [
        16, 17, 18, 19
    ]
}

# ============================================================
# Safety check
# ============================================================

all_cluster_clients = [

    client_idx

    for clients in HFL_CLUSTERS.values()

    for client_idx in clients
]

if sorted(all_cluster_clients) != list(
    range(NUM_CLIENTS)
):

    raise ValueError(
        "HFL cluster topology does not correctly "
        "cover all 20 clients."
    )

# ============================================================
# 4. DATASET PATHS
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

if len(file_paths) != NUM_CLIENTS:

    raise ValueError(
        f"Expected {NUM_CLIENTS} client files, "
        f"but found {len(file_paths)}."
    )

# ============================================================
# 5. OUTPUT DIRECTORY
# ============================================================

output_dir = (
    r'C:\Users\Murad\PycharmProjects'
    r'\federated-learning-tutorial'
)

os.makedirs(
    output_dir,
    exist_ok=True
)

# ============================================================
# 6. LABEL MAPPING
# ============================================================

label_mapping = {

    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5

}

# ============================================================
# 7. LOAD CLIENT DATA
# ============================================================

def load_client_data(paths):

    client_dfs = []

    for client_id, path in enumerate(paths):

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"\nFile does not exist:\n{path}"
            )

        df = pd.read_csv(path)

        if df.empty:

            raise ValueError(
                f"Client {client_id + 1} "
                f"dataset is empty."
            )

        if 'label' not in df.columns:

            raise ValueError(
                f"'label' column missing in "
                f"Client {client_id + 1}"
            )

        client_dfs.append(df)

    return client_dfs


client_dfs = load_client_data(
    file_paths
)

# ============================================================
# 8. COMBINE DATASET
# ============================================================

combined_df = pd.concat(
    client_dfs,
    ignore_index=True
)

combined_df['label'] = (
    combined_df['label']
    .map(label_mapping)
)

if combined_df['label'].isna().any():

    raise ValueError(
        "Unknown labels detected after "
        "label mapping."
    )

X_full_raw = (
    combined_df
    .drop('label', axis=1)
    .values
    .astype(np.float32)
)

y_full = (
    combined_df['label']
    .values
    .astype(np.int32)
)

X_full_raw = np.nan_to_num(
    X_full_raw,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

# ============================================================
# 9. GLOBAL TRAIN / TEST SPLIT
# ============================================================

X_train_raw, X_test_raw, y_train, y_test = (
    train_test_split(
        X_full_raw,
        y_full,
        test_size=TEST_SIZE,
        random_state=42,
        stratify=y_full
    )
)

# ============================================================
# 10. STANDARDIZATION
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_train_raw
).astype(np.float32)

X_test = scaler.transform(
    X_test_raw
).astype(np.float32)

input_shape = (
    X_train.shape[1],
)

num_classes = len(
    np.unique(y_full)
)

# ============================================================
# 11. CREATE CLIENT DATA
# ============================================================

client_data_list = []

for client_id, df in enumerate(
    client_dfs
):

    df = df.copy()

    df['label'] = (
        df['label']
        .map(label_mapping)
    )

    if df['label'].isna().any():

        raise ValueError(
            f"Unknown labels in Client "
            f"{client_id + 1}"
        )

    X_c_raw = (
        df
        .drop('label', axis=1)
        .values
        .astype(np.float32)
    )

    y_c = (
        df['label']
        .values
        .astype(np.int32)
    )

    X_c_raw = np.nan_to_num(
        X_c_raw,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    X_c = scaler.transform(
        X_c_raw
    ).astype(np.float32)

    client_data_list.append(
        (X_c, y_c)
    )

# ============================================================
# 12. DATA INFORMATION
# ============================================================

print("\n" + "=" * 80)

print(
    "HIER-FEDPROX WITH CLIENT DROPOUT"
)

print("=" * 80)

print(
    f"Total samples             : "
    f"{len(X_full_raw)}"
)

print(
    f"Training samples          : "
    f"{len(X_train)}"
)

print(
    f"Testing samples           : "
    f"{len(X_test)}"
)

print(
    f"Number of features        : "
    f"{input_shape[0]}"
)

print(
    f"Number of classes         : "
    f"{num_classes}"
)

print(
    f"Number of clients         : "
    f"{NUM_CLIENTS}"
)

print(
    f"Number of stations        : "
    f"{len(HFL_CLUSTERS)}"
)

print("\nDropout scenarios:")

for rate, scenario in DROPOUT_SCENARIOS.items():

    print(
        f"  {rate * 100:.0f}% -> "
        f"{scenario['num_dropped']} dropped / "
        f"{scenario['num_active']} active"
    )

print(
    f"\nFedProx MU                : "
    f"{FEDPROX_MU}"
)

print("\nClient distribution:")

for client_idx, (
    X_c,
    y_c
) in enumerate(client_data_list):

    unique, counts = np.unique(
        y_c,
        return_counts=True
    )

    distribution = {
        int(k): int(v)
        for k, v in zip(
            unique,
            counts
        )
    }

    print(
        f"Client {client_idx + 1:02d}: "
        f"{len(X_c)} samples | "
        f"{distribution}"
    )

print("\nHierarchical topology:")

for station, client_indices in (
    HFL_CLUSTERS.items()
):

    print(
        f"{station}: "
        f"{[i + 1 for i in client_indices]}"
    )

print("\nTest-set distribution:")

unique_test, counts_test = np.unique(
    y_test,
    return_counts=True
)

for cls, count in zip(
    unique_test,
    counts_test
):

    print(
        f"Class {cls}: {count}"
    )

# ============================================================
# 13. MODEL
# ============================================================

def create_mlp_model(
    input_shape,
    num_classes
):

    return tf.keras.Sequential([

        tf.keras.layers.Input(
            shape=input_shape
        ),

        tf.keras.layers.Dense(
            128,
            activation='relu'
        ),

        tf.keras.layers.Dense(
            64,
            activation='relu'
        ),

        tf.keras.layers.Dense(
            32,
            activation='relu'
        ),

        tf.keras.layers.Dense(
            num_classes,
            activation='softmax'
        )
    ])

# ============================================================
# 14. MODEL SIZE
# ============================================================

def calculate_model_size_mb(model):

    total_params = model.count_params()

    return (
        total_params * 4
    ) / (
        1024 * 1024
    )


MODEL_SIZE_MB = (
    calculate_model_size_mb(
        create_mlp_model(
            input_shape,
            num_classes
        )
    )
)

print("\n" + "=" * 80)

print(
    f"MODEL SIZE: "
    f"{MODEL_SIZE_MB:.4f} MB"
)

print("=" * 80)

# ============================================================
# 15. WEIGHT AGGREGATION
# ============================================================

def aggregate_weights(
    weights_list,
    sizes_list
):

    if len(weights_list) == 0:

        raise ValueError(
            "weights_list is empty."
        )

    total_samples = sum(
        sizes_list
    )

    if total_samples <= 0:

        raise ValueError(
            "Total number of samples "
            "must be > 0."
        )

    aggregated_weights = []

    for layer_idx in range(
        len(weights_list[0])
    ):

        layer_avg = np.zeros_like(
            weights_list[0][layer_idx]
        )

        for client_idx in range(
            len(weights_list)
        ):

            factor = (
                sizes_list[client_idx]
                / total_samples
            )

            layer_avg += (
                factor
                * weights_list[
                    client_idx
                ][layer_idx]
            )

        aggregated_weights.append(
            layer_avg
        )

    return aggregated_weights

# ============================================================
# 16. FEDPROX TRAINING STEP
# ============================================================

@tf.function(
    reduce_retracing=True
)
def train_step_fedprox(
    model,
    optimizer,
    x_batch,
    y_batch,
    global_weights,
    mu
):

    with tf.GradientTape() as tape:

        predictions = model(
            x_batch,
            training=True
        )

        classification_loss = (
            tf.keras.losses
            .sparse_categorical_crossentropy(
                y_batch,
                predictions
            )
        )

        classification_loss = (
            tf.reduce_mean(
                classification_loss
            )
        )

        prox_term = tf.constant(
            0.0,
            dtype=tf.float32
        )

        for local_var, global_var in zip(
            model.trainable_variables,
            global_weights
        ):

            prox_term += tf.reduce_sum(
                tf.square(
                    local_var - global_var
                )
            )

        total_loss = (
            classification_loss
            + (mu / 2.0)
            * prox_term
        )

    gradients = tape.gradient(
        total_loss,
        model.trainable_variables
    )

    optimizer.apply_gradients(
        zip(
            gradients,
            model.trainable_variables
        )
    )

    return total_loss

# ============================================================
# 17. FEDPROX CLIENT TRAINING
# ============================================================

def train_client_fedprox(
    global_weights,
    X_c,
    y_c
):

    model = create_mlp_model(
        input_shape,
        num_classes
    )

    model.set_weights(
        global_weights
    )

    optimizer = (
        tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        )
    )

    optimizer.build(
        model.trainable_variables
    )

    global_weights_tf = [

        tf.convert_to_tensor(
            w,
            dtype=tf.float32
        )

        for w in global_weights

    ]

    dataset = (
        tf.data.Dataset
        .from_tensor_slices(
            (X_c, y_c)
        )
        .batch(BATCH_SIZE)
        .prefetch(
            tf.data.AUTOTUNE
        )
    )

    for epoch in range(
        LOCAL_EPOCHS
    ):

        for x_batch, y_batch in dataset:

            train_step_fedprox(
                model,
                optimizer,
                x_batch,
                y_batch,
                global_weights_tf,
                tf.constant(
                    FEDPROX_MU,
                    dtype=tf.float32
                )
            )

    weights = model.get_weights()

    del model
    del optimizer
    del dataset

    return weights

# ============================================================
# 18. EVALUATION
# ============================================================

def evaluate_model(model):

    probabilities = model.predict(
        X_test,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    macro_precision = precision_score(
        y_test,
        predictions,
        average='macro',
        zero_division=0
    )

    macro_recall = recall_score(
        y_test,
        predictions,
        average='macro',
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average='macro',
        zero_division=0
    )

    class_report = classification_report(
        y_test,
        predictions,
        labels=np.arange(
            num_classes
        ),
        output_dict=True,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=np.arange(
            num_classes
        )
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

    return {

        'accuracy':
            accuracy,

        'macro_precision':
            macro_precision,

        'macro_recall':
            macro_recall,

        'macro_f1':
            macro_f1,

        'class_report':
            class_report,

        'cm_norm':
            cm_norm
    }

# ============================================================
# 19. SELECT DROPPED CLIENTS
# ============================================================
#
# IMPORTANT:
# This function is deterministic.
#
# The same seed + round produces the same shuffled
# client ordering for every dropout scenario.
#
# Therefore:
#
# 10% dropped = first 2
# 20% dropped = first 4
# 40% dropped = first 8
#
# This also makes the dropout sets nested.
# ============================================================

def select_dropped_clients(
    seed,
    round_idx,
    num_dropped_clients
):

    dropout_rng = random.Random(
        seed * 1000 + round_idx
    )

    shuffled_clients = list(
        range(NUM_CLIENTS)
    )

    dropout_rng.shuffle(
        shuffled_clients
    )

    dropped_clients = sorted(
        shuffled_clients[
            :num_dropped_clients
        ]
    )

    participating_clients = [

        client_idx

        for client_idx in range(
            NUM_CLIENTS
        )

        if client_idx
        not in dropped_clients

    ]

    return (
        dropped_clients,
        participating_clients
    )

# ============================================================
# 20. RUN HIER-FEDPROX EXPERIMENT
# ============================================================

def run_fl_experiment(
    seed,
    dropout_rate
):

    if dropout_rate not in DROPOUT_SCENARIOS:

        raise ValueError(
            f"Unsupported dropout rate: "
            f"{dropout_rate}"
        )

    scenario = (
        DROPOUT_SCENARIOS[
            dropout_rate
        ]
    )

    num_dropped_clients = (
        scenario['num_dropped']
    )

    num_participating_clients = (
        scenario['num_active']
    )

    os.environ['PYTHONHASHSEED'] = str(
        seed
    )

    np.random.seed(seed)

    random.seed(seed)

    tf.random.set_seed(seed)

    # --------------------------------------------------------
    # Global model
    # --------------------------------------------------------

    global_model = create_mlp_model(
        input_shape,
        num_classes
    )

    total_comm_mb = 0.0

    experiment_start = time.time()

    round_accuracy = []
    round_f1 = []
    round_precision = []
    round_recall = []
    round_time = []

    dropout_history = []
    station_history = []

    # ========================================================
    # GLOBAL ROUNDS
    # ========================================================

    for round_idx in range(
        GLOBAL_ROUNDS
    ):

        round_start = time.time()

        print(
            f"\n      Global Round "
            f"{round_idx + 1}/"
            f"{GLOBAL_ROUNDS}"
        )

        # ----------------------------------------------------
        # Select dropped clients
        # ----------------------------------------------------

        (
            dropped_clients,
            participating_clients
        ) = select_dropped_clients(

            seed,

            round_idx,

            num_dropped_clients

        )

        if len(dropped_clients) != (
            num_dropped_clients
        ):

            raise RuntimeError(
                "Incorrect number of dropped clients."
            )

        if len(participating_clients) != (
            num_participating_clients
        ):

            raise RuntimeError(
                "Incorrect number of participating clients."
            )

        dropped_display = [

            i + 1

            for i in dropped_clients

        ]

        participating_display = [

            i + 1

            for i in participating_clients

        ]

        print(
            f"      Dropout Rate: "
            f"{dropout_rate * 100:.0f}%"
        )

        print(
            f"      Dropped Clients: "
            f"{dropped_display}"
        )

        print(
            f"      Participating Clients: "
            f"{participating_display}"
        )

        dropout_history.append({

            'Dropout_Rate':
                dropout_rate,

            'Dropout_Percent':
                dropout_rate * 100,

            'Seed':
                seed,

            'Round':
                round_idx + 1,

            'Dropped_Clients':
                ', '.join(
                    map(
                        str,
                        dropped_display
                    )
                ),

            'Participating_Clients':
                ', '.join(
                    map(
                        str,
                        participating_display
                    )
                ),

            'Dropped_Count':
                len(
                    dropped_clients
                ),

            'Participating_Count':
                len(
                    participating_clients
                )
        })

        # ----------------------------------------------------
        # Global weights
        # ----------------------------------------------------

        global_weights = (
            global_model.get_weights()
        )

        station_weights = []
        station_sizes = []

        # ====================================================
        # STATION LEVEL
        # ====================================================

        for station, client_indices in (
            HFL_CLUSTERS.items()
        ):

            station_clients = [

                client_idx

                for client_idx
                in client_indices

                if client_idx
                not in dropped_clients

            ]

            dropped_station_clients = [

                client_idx

                for client_idx
                in client_indices

                if client_idx
                in dropped_clients

            ]

            print(
                f"\n        {station}"
            )

            print(
                f"          Active clients: "
                f"{[i + 1 for i in station_clients]}"
            )

            if dropped_station_clients:

                print(
                    f"          Dropped clients: "
                    f"{[i + 1 for i in dropped_station_clients]}"
                )

            else:

                print(
                    "          Dropped clients: None"
                )

            # ------------------------------------------------
            # If ALL clients in a station are dropped
            # ------------------------------------------------

            if len(station_clients) == 0:

                print(
                    f"          {station} has no "
                    f"participating clients."
                )

                station_history.append({

                    'Dropout_Rate':
                        dropout_rate,

                    'Seed':
                        seed,

                    'Round':
                        round_idx + 1,

                    'Station':
                        station,

                    'Active_Clients':
                        '',

                    'Dropped_Clients':
                        ', '.join(
                            str(i + 1)
                            for i in
                            dropped_station_clients
                        ),

                    'Active_Count':
                        0,

                    'Station_Samples':
                        0,

                    'Station_Used':
                        False

                })

                continue

            station_local_weights = []

            station_local_sizes = []

            # ------------------------------------------------
            # Client training
            # ------------------------------------------------

            for client_idx in station_clients:

                X_c, y_c = (
                    client_data_list[
                        client_idx
                    ]
                )

                client_weights = (
                    train_client_fedprox(

                        global_weights,

                        X_c,

                        y_c

                    )
                )

                station_local_weights.append(
                    client_weights
                )

                station_local_sizes.append(
                    len(X_c)
                )

                # --------------------------------------------
                # Client <-> Station
                #
                # Global -> Client
                # Client -> Station
                # --------------------------------------------

                total_comm_mb += (
                    MODEL_SIZE_MB * 2
                )

            # ------------------------------------------------
            # Station aggregation
            # ------------------------------------------------

            station_agg_weights = (
                aggregate_weights(

                    station_local_weights,

                    station_local_sizes

                )
            )

            station_sample_count = sum(
                station_local_sizes
            )

            station_weights.append(
                station_agg_weights
            )

            station_sizes.append(
                station_sample_count
            )

            # ------------------------------------------------
            # Station <-> Global Server
            # ------------------------------------------------

            total_comm_mb += (
                MODEL_SIZE_MB * 2
            )

            station_history.append({

                'Dropout_Rate':
                    dropout_rate,

                'Seed':
                    seed,

                'Round':
                    round_idx + 1,

                'Station':
                    station,

                'Active_Clients':
                    ', '.join(

                        str(i + 1)

                        for i in
                        station_clients

                    ),

                'Dropped_Clients':
                    ', '.join(

                        str(i + 1)

                        for i in
                        dropped_station_clients

                    ),

                'Active_Count':
                    len(
                        station_clients
                    ),

                'Station_Samples':
                    station_sample_count,

                'Station_Used':
                    True

            })

        # ====================================================
        # GLOBAL AGGREGATION
        # ====================================================

        if len(station_weights) == 0:

            raise RuntimeError(
                "No station has participating clients."
            )

        new_global_weights = (
            aggregate_weights(

                station_weights,

                station_sizes

            )
        )

        global_model.set_weights(
            new_global_weights
        )

        # ====================================================
        # ROUND EVALUATION
        # ====================================================

        round_result = evaluate_model(
            global_model
        )

        round_accuracy.append(
            round_result['accuracy']
        )

        round_f1.append(
            round_result['macro_f1']
        )

        round_precision.append(
            round_result['macro_precision']
        )

        round_recall.append(
            round_result['macro_recall']
        )

        elapsed_round = (
            time.time()
            - round_start
        )

        round_time.append(
            elapsed_round
        )

        print(
            f"\n      Round "
            f"{round_idx + 1:02d} | "
            f"Accuracy: "
            f"{round_result['accuracy']:.4f} | "
            f"Macro-F1: "
            f"{round_result['macro_f1']:.4f} | "
            f"Time: "
            f"{elapsed_round:.2f}s"
        )

    # ========================================================
    # FINAL EVALUATION
    # ========================================================

    final_result = evaluate_model(
        global_model
    )

    total_time = (
        time.time()
        - experiment_start
    )

    del global_model

    return {

        'dropout_rate':
            dropout_rate,

        'num_dropped':
            num_dropped_clients,

        'num_active':
            num_participating_clients,

        'accuracy':
            final_result['accuracy'],

        'macro_precision':
            final_result['macro_precision'],

        'macro_recall':
            final_result['macro_recall'],

        'macro_f1':
            final_result['macro_f1'],

        'comm_mb':
            total_comm_mb,

        'time_sec':
            total_time,

        'class_report':
            final_result['class_report'],

        'cm_norm':
            final_result['cm_norm'],

        'round_accuracy':
            round_accuracy,

        'round_f1':
            round_f1,

        'round_precision':
            round_precision,

        'round_recall':
            round_recall,

        'round_time':
            round_time,

        'dropout_history':
            dropout_history,

        'station_history':
            station_history

    }

# ============================================================
# 21. RUN BENCHMARK
# ============================================================

summary_results = []

detailed_class_metrics = []

convergence_results = []

dropout_results = []

station_results = []

print("\n" + "=" * 80)

print(
    "HIER-FEDPROX CLIENT DROPOUT EXPERIMENT"
)

print("=" * 80)

print(
    f"Clients                : "
    f"{NUM_CLIENTS}"
)

print(
    f"Stations               : "
    f"{len(HFL_CLUSTERS)}"
)

print(
    "Clients / Station      : 4"
)

print("\nDropout Scenarios:")

for rate, scenario in (
    DROPOUT_SCENARIOS.items()
):

    print(
        f"  {rate * 100:.0f}% -> "
        f"{scenario['num_dropped']} dropped / "
        f"{scenario['num_active']} active"
    )

print(
    f"\nSeeds                  : "
    f"{SEEDS}"
)

print(
    f"Global Rounds          : "
    f"{GLOBAL_ROUNDS}"
)

print(
    f"Local Epochs           : "
    f"{LOCAL_EPOCHS}"
)

print(
    f"FedProx MU             : "
    f"{FEDPROX_MU}"
)

print("=" * 80)

# ============================================================
# RUN ALL DROPOUT SCENARIOS
# ============================================================

for dropout_rate, scenario in (
    DROPOUT_SCENARIOS.items()
):

    num_dropped = (
        scenario['num_dropped']
    )

    num_active = (
        scenario['num_active']
    )

    print("\n\n" + "=" * 80)

    print(
        f"STARTING DROPOUT SCENARIO: "
        f"{dropout_rate * 100:.0f}%"
    )

    print(
        f"Dropped Clients / Round : "
        f"{num_dropped}"
    )

    print(
        f"Active Clients / Round  : "
        f"{num_active}"
    )

    print("=" * 80)

    accuracy_values = []

    f1_values = []

    precision_values = []

    recall_values = []

    communication_values = []

    time_values = []

    # ========================================================
    # RUN SEEDS
    # ========================================================

    for seed in SEEDS:

        print("\n" + "=" * 60)

        print(
            f"DROPOUT "
            f"{dropout_rate * 100:.0f}% "
            f"| SEED {seed}"
        )

        print("=" * 60)

        result = run_fl_experiment(

            seed,

            dropout_rate

        )

        accuracy_values.append(
            result['accuracy']
        )

        f1_values.append(
            result['macro_f1']
        )

        precision_values.append(
            result['macro_precision']
        )

        recall_values.append(
            result['macro_recall']
        )

        communication_values.append(
            result['comm_mb']
        )

        time_values.append(
            result['time_sec']
        )

        detailed_class_metrics.append({

            'Dropout_Rate':
                dropout_rate,

            'Seed':
                seed,

            'class_report':
                result['class_report'],

            'cm_norm':
                result['cm_norm']

        })

        dropout_results.extend(
            result['dropout_history']
        )

        station_results.extend(
            result['station_history']
        )

        # ====================================================
        # CONVERGENCE
        # ====================================================

        for round_idx in range(
            GLOBAL_ROUNDS
        ):

            dropout_info = (
                result[
                    'dropout_history'
                ][round_idx]
            )

            convergence_results.append({

                'Method':
                    'Hier-FedProx-Client-Dropout',

                'Dropout_Rate':
                    dropout_rate,

                'Dropout_Percent':
                    dropout_rate * 100,

                'Seed':
                    seed,

                'Round':
                    round_idx + 1,

                'Accuracy':
                    result[
                        'round_accuracy'
                    ][round_idx],

                'Macro_F1':
                    result[
                        'round_f1'
                    ][round_idx],

                'Macro_Precision':
                    result[
                        'round_precision'
                    ][round_idx],

                'Macro_Recall':
                    result[
                        'round_recall'
                    ][round_idx],

                'Round_Time_Sec':
                    result[
                        'round_time'
                    ][round_idx],

                'Dropped_Clients':
                    dropout_info[
                        'Dropped_Clients'
                    ],

                'Participating_Clients':
                    dropout_info[
                        'Participating_Clients'
                    ],

                'Dropped_Count':
                    dropout_info[
                        'Dropped_Count'
                    ],

                'Participating_Count':
                    dropout_info[
                        'Participating_Count'
                    ]

            })

        print(
            f"\nFinal Accuracy: "
            f"{result['accuracy']:.4f}"
        )

        print(
            f"Final Macro-F1: "
            f"{result['macro_f1']:.4f}"
        )

        print(
            f"Communication: "
            f"{result['comm_mb']:.2f} MB"
        )

        print(
            f"Execution Time: "
            f"{result['time_sec']:.2f} sec"
        )

    # ========================================================
    # SUMMARY FOR THIS DROPOUT SCENARIO
    # ========================================================

    summary_results.append({

        'Method':
            'Hier-FedProx-Client-Dropout',

        'Num_Clients':
            NUM_CLIENTS,

        'Num_Stations':
            len(HFL_CLUSTERS),

        'Dropout_Rate':
            dropout_rate,

        'Dropped_Clients_Per_Round':
            num_dropped,

        'Participating_Clients_Per_Round':
            num_active,

        'Accuracy_Mean':
            np.mean(
                accuracy_values
            ),

        'Accuracy_Std':
            np.std(
                accuracy_values
            ),

        'Macro_F1_Mean':
            np.mean(
                f1_values
            ),

        'Macro_F1_Std':
            np.std(
                f1_values
            ),

        'Macro_Precision_Mean':
            np.mean(
                precision_values
            ),

        'Macro_Precision_Std':
            np.std(
                precision_values
            ),

        'Macro_Recall_Mean':
            np.mean(
                recall_values
            ),

        'Macro_Recall_Std':
            np.std(
                recall_values
            ),

        'Comm_Cost_MB_Mean':
            np.mean(
                communication_values
            ),

        'Comm_Cost_MB_Std':
            np.std(
                communication_values
            ),

        'Exec_Time_Sec_Mean':
            np.mean(
                time_values
            ),

        'Exec_Time_Sec_Std':
            np.std(
                time_values
            )

    })

# ============================================================
# 22. SUMMARY
# ============================================================

df_summary = pd.DataFrame(
    summary_results
)

summary_path = os.path.join(

    output_dir,

    'hier_fedprox_client_dropout_summary_all_scenarios.csv'

)

df_summary.to_csv(
    summary_path,
    index=False
)

# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n" + "=" * 100)

print(
    "HIER-FEDPROX CLIENT DROPOUT SUMMARY"
)

print(
    "(Mean ± Std across seeds)"
)

print("=" * 100)

summary_display = df_summary.copy()

summary_display[
    'Dropout_Rate'
] = (
    summary_display[
        'Dropout_Rate'
    ] * 100
)

print(
    summary_display[
        [
            'Dropout_Rate',
            'Dropped_Clients_Per_Round',
            'Participating_Clients_Per_Round',
            'Accuracy_Mean',
            'Accuracy_Std',
            'Macro_Precision_Mean',
            'Macro_Precision_Std',
            'Macro_Recall_Mean',
            'Macro_Recall_Std',
            'Macro_F1_Mean',
            'Macro_F1_Std',
            'Comm_Cost_MB_Mean',
            'Comm_Cost_MB_Std',
            'Exec_Time_Sec_Mean',
            'Exec_Time_Sec_Std'
        ]
    ]
    .round(4)
    .to_string(index=False)
)

# ============================================================
# 23. CONVERGENCE CSV
# ============================================================

df_convergence = pd.DataFrame(
    convergence_results
)

convergence_path = os.path.join(

    output_dir,

    'hier_fedprox_client_dropout_convergence_all_scenarios.csv'

)

df_convergence.to_csv(
    convergence_path,
    index=False
)

# ============================================================
# 24. DROPOUT LOG
# ============================================================

df_dropout = pd.DataFrame(
    dropout_results
)

dropout_log_path = os.path.join(

    output_dir,

    'hier_fedprox_client_dropout_log_all_scenarios.csv'

)

df_dropout.to_csv(
    dropout_log_path,
    index=False
)

# ============================================================
# 25. STATION LOG
# ============================================================

df_station = pd.DataFrame(
    station_results
)

station_log_path = os.path.join(

    output_dir,

    'hier_fedprox_client_dropout_station_log_all_scenarios.csv'

)

df_station.to_csv(
    station_log_path,
    index=False
)

# ============================================================
# 26. CLASS-WISE PERFORMANCE
# ============================================================

print("\n" + "=" * 100)

print(
    "HIER-FEDPROX CLIENT DROPOUT "
    "CLASS-WISE PERFORMANCE"
)

print("=" * 100)

class_rows = []

for dropout_rate in DROPOUT_SCENARIOS.keys():

    scenario_runs = [

        run

        for run in detailed_class_metrics

        if run[
            'Dropout_Rate'
        ] == dropout_rate

    ]

    reports = [

        run['class_report']

        for run in scenario_runs

    ]

    for c in range(
        num_classes
    ):

        c_str = str(c)

        p_vals = [

            r[c_str]['precision']

            for r in reports

        ]

        r_vals = [

            r[c_str]['recall']

            for r in reports

        ]

        f_vals = [

            r[c_str]['f1-score']

            for r in reports

        ]

        s_vals = [

            r[c_str]['support']

            for r in reports

        ]

        class_rows.append({

            'Dropout_Rate':
                dropout_rate,

            'Class':
                f'Class {c}',

            'Support':
                np.mean(
                    s_vals
                ),

            'Precision_Mean':
                np.mean(
                    p_vals
                ),

            'Precision_Std':
                np.std(
                    p_vals
                ),

            'Recall_Mean':
                np.mean(
                    r_vals
                ),

            'Recall_Std':
                np.std(
                    r_vals
                ),

            'F1_Mean':
                np.mean(
                    f_vals
                ),

            'F1_Std':
                np.std(
                    f_vals
                )

        })

df_class = pd.DataFrame(
    class_rows
)

print(
    df_class.round(4)
    .to_string(index=False)
)

class_path = os.path.join(

    output_dir,

    'hier_fedprox_client_dropout_class_wise_all_scenarios.csv'

)

df_class.to_csv(
    class_path,
    index=False
)

# ============================================================
# 27. CONFUSION MATRICES
# ============================================================
#
# IMPORTANT:
# We create a separate average confusion matrix
# for each dropout scenario.
#
# This is more meaningful than averaging 10%, 20%,
# and 40% together.
# ============================================================

confusion_paths = []

for dropout_rate in DROPOUT_SCENARIOS.keys():

    scenario_matrices = [

        run['cm_norm']

        for run in detailed_class_metrics

        if run[
            'Dropout_Rate'
        ] == dropout_rate

    ]

    avg_cm = np.mean(

        scenario_matrices,

        axis=0

    )

    rate_label = int(
        dropout_rate * 100
    )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    df_cm = pd.DataFrame(

        avg_cm,

        index=[
            f'Class {i}'
            for i in range(
                num_classes
            )
        ],

        columns=[
            f'Class {i}'
            for i in range(
                num_classes
            )
        ]

    )

    cm_csv_path = os.path.join(

        output_dir,

        f'hier_fedprox_client_dropout_'
        f'{rate_label}pct_'
        f'normalized_confusion_matrix.csv'

    )

    df_cm.to_csv(
        cm_csv_path
    )

    # --------------------------------------------------------
    # PNG
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(9, 8)
    )

    im = ax.imshow(

        avg_cm,

        interpolation='nearest',

        cmap='Blues'

    )

    cbar = ax.figure.colorbar(

        im,

        ax=ax

    )

    cbar.ax.set_ylabel(

        'Normalized Value',

        rotation=-90,

        va='bottom'

    )

    ax.set(

        xticks=np.arange(
            num_classes
        ),

        yticks=np.arange(
            num_classes
        ),

        xticklabels=[
            f'Class {i}'
            for i in range(
                num_classes
            )
        ],

        yticklabels=[
            f'Class {i}'
            for i in range(
                num_classes
            )
        ],

        ylabel='True Label',

        xlabel='Predicted Label',

        title=(

            f'Hier-FedProx with '
            f'{rate_label}% Client Dropout\n'

            f'Average Normalized '
            f'Confusion Matrix'

        )

    )

    plt.setp(

        ax.get_xticklabels(),

        rotation=45,

        ha='right',

        rotation_mode='anchor'

    )

    threshold = (
        avg_cm.max() / 2.0
    )

    for i in range(
        num_classes
    ):

        for j in range(
            num_classes
        ):

            ax.text(

                j,

                i,

                f'{avg_cm[i, j]:.2f}',

                ha='center',

                va='center',

                color=(

                    'white'

                    if avg_cm[i, j]
                    > threshold

                    else 'black'

                ),

                fontsize=11

            )

    fig.tight_layout()

    image_path = os.path.join(

        output_dir,

        f'hier_fedprox_client_dropout_'
        f'{rate_label}pct_'
        f'normalized_confusion_matrix.png'

    )

    plt.savefig(

        image_path,

        dpi=300,

        bbox_inches='tight'

    )

    plt.close(fig)

    confusion_paths.append({

        'dropout_rate':
            dropout_rate,

        'csv':
            cm_csv_path,

        'png':
            image_path

    })

# ============================================================
# 28. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 100)

print(
    "[FINISHED] Hier-FedProx Client Dropout "
    "experiment completed successfully."
)

print("=" * 100)

print("\nSaved files:")

print(
    f"\n1. Summary:\n"
    f"{summary_path}"
)

print(
    f"\n2. Convergence:\n"
    f"{convergence_path}"
)

print(
    f"\n3. Dropout Log:\n"
    f"{dropout_log_path}"
)

print(
    f"\n4. Station Log:\n"
    f"{station_log_path}"
)

print(
    f"\n5. Class-wise Performance:\n"
    f"{class_path}"
)

print("\n6. Confusion Matrices:")

for item in confusion_paths:

    print(
        f"\n   {item['dropout_rate'] * 100:.0f}%:"
    )

    print(
        f"   CSV: {item['csv']}"
    )

    print(
        f"   PNG: {item['png']}"
    )

print("\n" + "=" * 100)

print(
    "EXPERIMENT SETTINGS"
)

print("=" * 100)

print(
    f"Total Clients        : "
    f"{NUM_CLIENTS}"
)

print(
    f"Total Stations       : "
    f"{len(HFL_CLUSTERS)}"
)

print(
    "Dropout Scenarios    : 10%, 20%, 40%"
)

print(
    "10%                  : "
    "2 dropped / 18 active"
)

print(
    "20%                  : "
    "4 dropped / 16 active"
)

print(
    "40%                  : "
    "8 dropped / 12 active"
)

print(
    "Samples unchanged    : YES"
)

print(
    "Data redistribution  : NO"
)

print(
    "Strong Quantity Skew : YES"
)

print(
    "Dropout level        : CLIENT"
)

print(
    "Station aggregation  : ACTIVE CLIENTS ONLY"
)

print(
    f"FedProx MU           : "
    f"{FEDPROX_MU}"
)

print("=" * 100)