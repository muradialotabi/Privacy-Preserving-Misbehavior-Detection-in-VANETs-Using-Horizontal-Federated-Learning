import os
import pandas as pd
import random

# Define file paths
file_paths = [
    'ConstPosFullPathes.csv',
    'ConstPosOffsetFullPathes.csv',
    'EventalStopFullPathes.csv',
    'RandomPosFullPathes.csv',
    'RandomPosOffsetFullPathes.csv'
]


def load_data(file_paths):
    df_list = []
    for file_path in file_paths:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            if not df.empty:
                df_list.append(df)
    return pd.concat(df_list, ignore_index=True) if df_list else None


def apply_quantity_skew(data):
    adjusted_data = pd.DataFrame()
    senders = data['sender'].unique()

    for sender in senders:
        sender_data = data[data['sender'] == sender]
        skew_factor = 1 + (hash(sender) % 5) * 0.1  # Dynamic skew factor
        n_samples = int(len(sender_data) * skew_factor)

        adjusted_data = pd.concat([adjusted_data, sender_data.sample(n=n_samples, random_state=42)])

    return adjusted_data


def distribute_to_clients(data, num_clients, participation_rate=0.8):
    client_partitions = [pd.DataFrame() for _ in range(num_clients)]
    adjusted_data = apply_quantity_skew(data)  # Apply quantity skew

    senders = adjusted_data['sender'].unique()

    # Randomly select clients to participate in this round
    participating_clients = random.sample(range(num_clients), k=int(num_clients * participation_rate))

    for client in participating_clients:
        client_data = pd.DataFrame()
        for sender in senders:
            sender_data = adjusted_data[adjusted_data['sender'] == sender]
            n_samples = len(sender_data)  # Take all available samples for the client
            client_data = pd.concat([client_data, sender_data.sample(n=n_samples, random_state=client)])
        client_partitions[client] = client_data.sample(frac=1).reset_index(drop=True)  # Shuffle client data

    return client_partitions, participating_clients


# Main Execution
data = load_data(file_paths)
if data is not None:
    num_clients = 5
    participation_rate = 0.8  # 80% of clients will participate
    client_data, participating_clients = distribute_to_clients(data, num_clients, participation_rate)

    for client in range(num_clients):
        if client in participating_clients:
            client_data[client].to_csv(f'client_data_skewed_{client + 1}.csv', index=False)
else:
    print("Error: No data loaded.")