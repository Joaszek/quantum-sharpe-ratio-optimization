import pandas as pd
import matplotlib.pyplot as plt
import ast
import os

# Paths
RESULTS_PATH = "results/sliding_window_sharpe_results.csv"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Load and parse results
df = pd.read_csv(RESULTS_PATH)
df["selected_assets"] = df["selected_assets"].apply(lambda x: list(map(float, ast.literal_eval(x))))

# Plot 1: Objective Value Over Time
plt.figure(figsize=(12, 5))
plt.plot(df["window_end"], df["objective_value"], marker='o')
plt.title("Objective Value Over Time (Optimized by QAOA)")
plt.xlabel("Window End")
plt.ylabel("Objective Value")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "objective_value.png"))
plt.close()

# Plot 2: Realized Profit Over Time
plt.figure(figsize=(12, 5))
plt.plot(df["window_end"], df["realized_profit"], marker='o', color='green')
plt.title("Realized Profit Over Time")
plt.xlabel("Window End")
plt.ylabel("Realized Profit")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "realized_profit.png"))
plt.close()

# Plot 3: Selected Assets Over Time
plt.figure(figsize=(12, 5))
asset_labels = ["Apple", "CocaCola", "Google"]
for i in range(3):
    plt.plot(df["window_end"], df["selected_assets"].apply(lambda x: x[i]), label=asset_labels[i])
plt.title("Selected Assets Over Time")
plt.xlabel("Window End")
plt.ylabel("Selected (1) or Not (0)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "selected_assets.png"))
plt.close()

print("Plots saved successfully in the 'plots/' directory.")
