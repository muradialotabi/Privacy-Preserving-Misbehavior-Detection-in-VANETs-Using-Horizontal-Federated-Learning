# ============================================================
# STRONG QUANTITY SKEW PARTITIONING ONLY
# NO TRAINING
# NO MODEL
# NO FEDAVG
# NO FEDPROX
# ============================================================

import os
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42

TEST_SIZE = 0.20

TOTAL_CLIENT_DATA = 1_853_190

OUTPUT_ROOT = "Strong_Quantity_Skew"

LABEL_COLUMN = "label"


# ============================================================
# 2. ORIGINAL DATA FILES
# ============================================================

DATA_FILES = [
    r"ConstPosFullPathes.csv",
    r"ConstPosOffsetFullPathes.csv",
    r"EventalStopFullPathes.csv",
    r"RandomPosFullPathes.csv",
    r"RandomPosOffsetFullPathes.csv"
]


# ============================================================
# 3. LABEL MAPPING
# ============================================================

LABEL_MAPPING = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    9: 5
}


# ============================================================
# 4. CLIENT WEIGHTS
# ============================================================

# Strong Quantity Skew
#
# Larger weight  -> more samples
# Smaller weight -> fewer samples
#
# Total data is fixed at 1,853,190
# ============================================================

QUANTITY_WEIGHTS = {

    5: [
        1.80,
        1.35,
        1.00,
        0.65,
        0.50
    ],

    10: [
        2.00,
        1.70,
        1.50,
        1.30,
        1.15,
        1.00,
        0.85,
        0.70,
        0.55,
        0.45
    ],

    15: [
        2.00,
        1.80,
        1.65,
        1.50,
        1.40,
        1.30,
        1.20,
        1.10,
        1.00,
        0.90,
        0.80,
        0.70,
        0.60,
        0.50,
        0.40
    ],

    20: [
        2.00,
        1.90,
        1.80,
        1.70,
        1.60,
        1.50,
        1.40,
        1.30,
        1.20,
        1.10,
        1.00,
        0.95,
        0.90,
        0.85,
        0.80,
        0.75,
        0.70,
        0.65,
        0.60,
        0.55
    ]
}


# ============================================================
# 5. RANDOM SEEDS
# ============================================================

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# 6. CALCULATE EXACT CLIENT SIZES
# ============================================================

def calculate_client_sizes(total_samples, weights):

    weights = np.array(
        weights,
        dtype=float
    )

    proportions = (
        weights / weights.sum()
    )

    raw_sizes = (
        proportions * total_samples
    )

    sizes = np.floor(
        raw_sizes
    ).astype(int)

    # Fix rounding
    remaining = (
        total_samples - sizes.sum()
    )

    fractional = (
        raw_sizes - sizes
    )

    order = np.argsort(
        -fractional
    )

    for i in range(remaining):

        sizes[
            order[i % len(order)]
        ] += 1

    return sizes.tolist()


# ============================================================
# 7. LOAD DATA
# ============================================================

print("\n" + "=" * 80)
print("LOADING DATA")
print("=" * 80)

dataframes = []

for file_path in DATA_FILES:

    print(
        f"\nLoading: {file_path}"
    )

    df = pd.read_csv(
        file_path
    )

    print(
        f"Samples: {len(df):,}"
    )

    dataframes.append(df)


# ============================================================
# 8. COMBINE DATA
# ============================================================

combined_df = pd.concat(
    dataframes,
    ignore_index=True
)

print("\n" + "=" * 80)
print("COMBINED DATA")
print("=" * 80)

print(
    f"Total samples: "
    f"{len(combined_df):,}"
)


# ============================================================
# 9. MAP LABELS
# ============================================================

combined_df[LABEL_COLUMN] = (
    combined_df[LABEL_COLUMN]
    .map(LABEL_MAPPING)
)

combined_df = (
    combined_df
    .dropna(
        subset=[LABEL_COLUMN]
    )
    .reset_index(drop=True)
)

combined_df[LABEL_COLUMN] = (
    combined_df[LABEL_COLUMN]
    .astype(int)
)


# ============================================================
# 10. TRAIN / TEST SPLIT
# ============================================================
#
# This is ONLY for preparing the data.
#
# No training happens here.
#
# Client data comes ONLY from train_df.
#
# ============================================================

print("\n" + "=" * 80)
print("TRAIN / TEST SPLIT")
print("=" * 80)

train_df, test_df = train_test_split(

    combined_df,

    test_size=TEST_SIZE,

    random_state=SEED,

    stratify=combined_df[LABEL_COLUMN]
)

train_df = train_df.reset_index(
    drop=True
)

test_df = test_df.reset_index(
    drop=True
)

print(
    f"Train: {len(train_df):,}"
)

print(
    f"Test : {len(test_df):,}"
)


# ============================================================
# 11. CHECK ENOUGH TRAINING DATA
# ============================================================

if TOTAL_CLIENT_DATA > len(train_df):

    raise ValueError(
        f"Not enough training data.\n"
        f"Required: {TOTAL_CLIENT_DATA:,}\n"
        f"Available: {len(train_df):,}"
    )


# ============================================================
# 12. CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_ROOT,
    exist_ok=True
)


# ============================================================
# 13. SAVE GLOBAL TRAIN / TEST
# ============================================================

train_df.to_csv(

    os.path.join(
        OUTPUT_ROOT,
        "global_train.csv"
    ),

    index=False
)


test_df.to_csv(

    os.path.join(
        OUTPUT_ROOT,
        "global_test.csv"
    ),

    index=False
)


# ============================================================
# 14. FUNCTION:
#     CREATE QUANTITY SKEW CLIENTS
# ============================================================

def create_quantity_skew(

    train_data,
    client_sizes,
    seed

):

    rng = np.random.default_rng(
        seed
    )

    # --------------------------------------------------------
    # Create a separate shuffled pool for every class
    # --------------------------------------------------------

    class_pools = {}

    for label in range(6):

        class_df = train_data[
            train_data[LABEL_COLUMN] == label
        ].copy()

        indices = np.arange(
            len(class_df)
        )

        rng.shuffle(
            indices
        )

        class_pools[label] = (
            class_df
            .iloc[indices]
            .reset_index(drop=True)
        )


    # --------------------------------------------------------
    # Global class proportions
    # --------------------------------------------------------

    class_proportions = {}

    total_train = len(train_data)

    for label in range(6):

        class_proportions[label] = (

            len(class_pools[label])
            / total_train

        )


    # --------------------------------------------------------
    # Track used samples
    # --------------------------------------------------------

    used = {
        label: 0
        for label in range(6)
    }


    clients = []


    # ========================================================
    # CREATE CLIENTS
    # ========================================================

    for client_id, target_size in enumerate(

        client_sizes,

        start=1

    ):

        # ----------------------------------------------------
        # Calculate number of samples from each class
        # ----------------------------------------------------

        raw = {

            label:
            target_size *
            class_proportions[label]

            for label in range(6)

        }


        allocation = {

            label:
            int(np.floor(raw[label]))

            for label in range(6)

        }


        # ----------------------------------------------------
        # Correct rounding
        # ----------------------------------------------------

        difference = (

            target_size
            - sum(allocation.values())

        )


        fractional_order = sorted(

            range(6),

            key=lambda label:
            raw[label] -
            allocation[label],

            reverse=True

        )


        for i in range(difference):

            label = fractional_order[
                i % 6
            ]

            allocation[label] += 1


        # ----------------------------------------------------
        # Check availability
        # ----------------------------------------------------

        for label in range(6):

            available = (

                len(
                    class_pools[label]
                )
                -
                used[label]

            )

            if allocation[label] > available:

                raise ValueError(

                    f"Not enough Class {label} "
                    f"samples for Client {client_id}."

                )


        # ----------------------------------------------------
        # Extract data
        # ----------------------------------------------------

        parts = []


        for label in range(6):

            count = allocation[label]

            start = used[label]

            end = (
                start + count
            )

            part = (

                class_pools[label]
                .iloc[start:end]
                .copy()

            )

            parts.append(
                part
            )

            used[label] = end


        # ----------------------------------------------------
        # Combine client data
        # ----------------------------------------------------

        client_df = pd.concat(

            parts,

            ignore_index=True

        )


        # ----------------------------------------------------
        # Shuffle client
        # ----------------------------------------------------

        client_df = (

            client_df
            .sample(
                frac=1,
                random_state=seed + client_id
            )
            .reset_index(drop=True)

        )


        clients.append(
            client_df
        )


    return clients


# ============================================================
# 15. GENERATE 5 / 10 / 15 / 20 CLIENTS
# ============================================================

summary = []


for num_clients in [5, 10, 15, 20]:

    print("\n\n" + "=" * 80)

    print(
        f"{num_clients} CLIENTS"
    )

    print("=" * 80)


    # --------------------------------------------------------
    # Calculate exact sizes
    # --------------------------------------------------------

    client_sizes = calculate_client_sizes(

        TOTAL_CLIENT_DATA,

        QUANTITY_WEIGHTS[num_clients]

    )


    # --------------------------------------------------------
    # Verify total
    # --------------------------------------------------------

    assert sum(
        client_sizes
    ) == TOTAL_CLIENT_DATA


    # --------------------------------------------------------
    # Print sizes
    # --------------------------------------------------------

    print("\nClient sizes:")


    for client_id, size in enumerate(

        client_sizes,

        start=1

    ):

        print(

            f"Client {client_id:02d}: "
            f"{size:,} samples"

        )


    print(
        f"\nTotal: "
        f"{sum(client_sizes):,}"
    )


    # --------------------------------------------------------
    # Generate clients
    # --------------------------------------------------------

    clients = create_quantity_skew(

        train_data=train_df,

        client_sizes=client_sizes,

        seed=SEED + num_clients

    )


    # --------------------------------------------------------
    # Create folder
    # --------------------------------------------------------

    client_folder = os.path.join(

        OUTPUT_ROOT,

        f"{num_clients}_Clients"

    )


    os.makedirs(

        client_folder,

        exist_ok=True

    )


    # --------------------------------------------------------
    # Save each client
    # --------------------------------------------------------

    for client_id, client_df in enumerate(

        clients,

        start=1

    ):

        file_path = os.path.join(

            client_folder,

            f"client_data_{client_id}.csv"

        )


        client_df.to_csv(

            file_path,

            index=False

        )


        # Summary

        summary.append({

            "num_clients":
                num_clients,

            "client_id":
                client_id,

            "samples":
                len(client_df)

        })


        print(

            f"Saved Client {client_id}: "
            f"{len(client_df):,}"

        )


# ============================================================
# 16. SAVE SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    summary
)


summary_path = os.path.join(

    OUTPUT_ROOT,

    "quantity_skew_summary.csv"

)


summary_df.to_csv(

    summary_path,

    index=False

)


# ============================================================
# 17. FINAL CHECK
# ============================================================

print("\n\n" + "=" * 80)

print(
    "QUANTITY SKEW PARTITIONING COMPLETED"
)

print("=" * 80)


print(
    f"\nTotal data per experiment: "
    f"{TOTAL_CLIENT_DATA:,}"
)


for num_clients in [5, 10, 15, 20]:

    subset = summary_df[
        summary_df["num_clients"]
        == num_clients
    ]

    sizes = subset[
        "samples"
    ].tolist()


    print("\n" + "-" * 60)

    print(
        f"{num_clients} Clients"
    )

    print(
        f"Minimum: {min(sizes):,}"
    )

    print(
        f"Maximum: {max(sizes):,}"
    )

    print(
        f"Average: {np.mean(sizes):,.1f}"
    )

    print(
        f"Max/Min: "
        f"{max(sizes) / min(sizes):.2f}"
    )


print("\n" + "=" * 80)

print("NO TRAINING WAS PERFORMED.")

print(
    "Only dataset partitioning and CSV generation."
)

print("=" * 80)

print(
    "\nOutput folder:"
)

print(
    os.path.abspath(
        OUTPUT_ROOT
    )
)