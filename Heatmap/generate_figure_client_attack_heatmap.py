import os
import matplotlib

# Fix for PyCharm Matplotlib backend compatibility issue (tostring_rgb error)
matplotlib.use('TkAgg')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# Configuration & Setup
# ==========================================

# Label Mapping
label_mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 9: 5}

# Class Names for Plot Legend / Axes
class_names = {
    0: "Normal (0)",
    1: "Constant Position Attack",
    2: "Constant Offset Attack",
    3: "Random Offset Attack",
    4: "Random Position Attack",
    5: "Eventual Stop Attack"
}

file_paths = [
    r'C:\Users\Murad\PycharmProjects\federated-learning-tutorial\mid_heterogeneous_noniid_5clients\client_1_noniid.csv',
    r'C:\Users\Murad\PycharmProjects\federated-learning-tutorial\mid_heterogeneous_noniid_5clients\client_2_noniid.csv',
    r'C:\Users\Murad\PycharmProjects\federated-learning-tutorial\mid_heterogeneous_noniid_5clients\client_3_noniid.csv',
    r'C:\Users\Murad\PycharmProjects\federated-learning-tutorial\mid_heterogeneous_noniid_5clients\client_4_noniid.csv',
    r'C:\Users\Murad\PycharmProjects\federated-learning-tutorial\mid_heterogeneous_noniid_5clients\client_5_noniid.csv'

]

# Column name representing target labels in your CSV files
LABEL_COL = 'label'  # Adjust if your CSV column is named 'Label', 'class', or 'target'

# Output plot filename
OUTPUT_IMAGE_PATH = 'client_attack_heatmap.png'

# ==========================================
# Data Processing
# ==========================================

client_data = {}

for idx, path in enumerate(file_paths, start=1):
    client_name = f"Client {idx}"
    if os.path.exists(path):
        df = pd.read_csv(path)

        # Map raw labels using defined label_mapping
        df['mapped_label'] = df[LABEL_COL].map(label_mapping)

        # Count frequency of each mapped label
        counts = df['mapped_label'].value_counts()
        client_data[client_name] = counts
    else:
        print(f"Warning: File not found - {path}")

# Combine counts into DataFrame (Rows: Clients, Columns: Classes)
dist_df = pd.DataFrame(client_data).fillna(0).T
dist_df.rename(columns=class_names, inplace=True)

# Convert raw counts to percentage per client (each client row sums to 100%)
pct_df = dist_df.div(dist_df.sum(axis=1), axis=0) * 100

print("\n--- Attack Distribution (Percentages per Client) ---")
print(pct_df.round(2))

# ==========================================
# Visualization & Saving
# ==========================================

plt.figure(figsize=(10, 6))

sns.heatmap(
    pct_df,
    annot=True,  # Display values in cells
    fmt=".1f",  # Display 1 decimal place
    cmap="YlGnBu",  # Color gradient palette
    cbar_kws={'label': 'Distribution Percentage (%)'},
    linewidths=0.5,  # Gridline thickness between cells
    linecolor='whitesmoke'
)


plt.title(
    'Extreme Heterogeneity Non-IID Data Distribution 5 Clients Attack Class Distribution (%)',
    fontsize=14,
    fontname='Times New Roman',
    fontweight='bold',
    pad=15
)

plt.xlabel('Attack Class', fontsize=12, fontname='Times New Roman')
plt.ylabel('Client', fontsize=12, fontname='Times New Roman')

plt.xticks(rotation=30, ha='right', fontname='Times New Roman')
plt.yticks(rotation=0, fontname='Times New Roman')

plt.tight_layout()
# Save high-resolution image for research paper
plt.savefig(OUTPUT_IMAGE_PATH, dpi=300, bbox_inches='tight')
print(f"\nHeatmap successfully saved to: {os.path.abspath(OUTPUT_IMAGE_PATH)}")

# Show interactive plot window
plt.show()