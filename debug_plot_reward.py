import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file
file_path = "rewards_summary.csv"
df = pd.read_csv(file_path)

# Deduce the column names from the CSV file
columns = df.columns
parameter_names = set([
    "_".join(col.split("_")[:-1]) for col in columns if col.endswith(("_min", "_25%", "_median", "_mean", "_75%", "_max"))
])
parameter_names = list(parameter_names)
parameter_names.sort()

# Create a plot with cumulative mean and quantiles
plt.figure(figsize=(10, 6))

# Define line styles and colors
line_styles = ['-', '--', "--"]
line_colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
line_alphas = [1,0.5,0.5]

# Plot each parameter
for i, param in enumerate(parameter_names):
    for j, stat in enumerate(["mean", "min", "max"]):
        # Reset cumulative sum every 500 time steps
        label = param if stat == "mean" else None
        plt.plot(df[f"{param}_{stat}"], label=label, linestyle=line_styles[j], color=line_colors[i], alpha=line_alphas[j])

plt.title("Cumulative Rewards Over Time")
plt.xlabel("Time Step")
plt.ylabel("Cumulative Mean Reward")
plt.xticks([df.shape[0]/8 *i  for i in range(9)], [df.iloc[int(df.shape[0]/(8.00001) *i)]["epoch"] for i in range(9)])
plt.legend()
plt.grid(True)
plt.show()