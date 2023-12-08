import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os

# Load the Local Feature Importance DataFrames
df_unnormalized = pd.read_csv("Local_Feature_Importance_DataFrame/Local_Feature_Importance_unnormalized.csv")
df_z_score = pd.read_csv("Local_Feature_Importance_DataFrame/Local_Feature_Importance_z-score.csv")
df_min_max = pd.read_csv("Local_Feature_Importance_DataFrame/Local_Feature_Importance_min-max.csv")

# Fcreate and plot a heatmap with focus on non-zero values
def plot_heatmap(df, normalization_type, save_folder):
    plt.figure(figsize=(12, 8))
    
    # Focus on non-zero values
    df_nonzero = df[df != 0]
    
    # Use a diverging color map and log-scale (found tutorial online)
    sns.heatmap(df_nonzero, cmap="RdBu_r", annot=False, norm=LogNorm())
    
    plt.title(f"Local Feature Importance Heatmap ({normalization_type})")
    plt.xlabel("Pixel Index")
    plt.ylabel("Sample Index")
    
    # save the plot
    save_path = os.path.join(save_folder, f"ANALYSIS_PLOT_Local_Feature_Importance_{normalization_type}_DataFrame.png")
    plt.savefig(save_path)
    plt.close()
    
    print(f"Plot saved at: {save_path}")

# Define the folder for saving plots
plot_save_folder = "ANALYSIS_PLOT_for_DateFrames"


# Plot heatmaps for each normalization type and save them
plot_heatmap(df_unnormalized, "Unnormalized", plot_save_folder)
plot_heatmap(df_z_score, "Z-score", plot_save_folder)
plot_heatmap(df_min_max, "Min-Max", plot_save_folder)