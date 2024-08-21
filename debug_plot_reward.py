import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file
file_path = "rewards_summary.csv"
df = pd.read_csv(file_path)

# Create a plot with cumulative mean and quantiles
plt.figure(figsize=(10, 6))

# Define line styles
mean_line_style = '-'
quantile_line_style = '--'

# Plot for "pos"
plt.plot(df["pos_mean"].cumsum(), label="pos", linestyle=mean_line_style)
plt.plot(df["pos_25%"].cumsum(), label="pos 25%", linestyle=quantile_line_style, color='blue', alpha=0.5)
plt.plot(df["pos_75%"].cumsum(), label="pos 75%", linestyle=quantile_line_style, color='blue', alpha=0.5)

# Plot for "safety"
plt.plot(df["safety_mean"].cumsum(), label="safety", linestyle=mean_line_style)
plt.plot(df["safety_25%"].cumsum(), label="safety 25%", linestyle=quantile_line_style, color='orange', alpha=0.5)
plt.plot(df["safety_75%"].cumsum(), label="safety 75%", linestyle=quantile_line_style, color='orange', alpha=0.5)

# Plot for "angvel"
plt.plot(df["angvel_mean"].cumsum(), label="angvel", linestyle=mean_line_style)
plt.plot(df["angvel_25%"].cumsum(), label="angvel 25%", linestyle=quantile_line_style, color='green', alpha=0.5)
plt.plot(df["angvel_75%"].cumsum(), label="angvel 75%", linestyle=quantile_line_style, color='green', alpha=0.5)

# Plot for "battery_use_on_step"
plt.plot(df["battery_use_on_step_mean"].cumsum(), label="battery_use_on_step", linestyle=mean_line_style)
plt.plot(df["battery_use_on_step_25%"].cumsum(), label="battery_use_on_step 25%", linestyle=quantile_line_style, color='red', alpha=0.5)
plt.plot(df["battery_use_on_step_75%"].cumsum(), label="battery_use_on_step 75%", linestyle=quantile_line_style, color='red', alpha=0.5)

# Plot for "orientation"
plt.plot(df["orientation_mean"].cumsum(), label="orientation", linestyle=mean_line_style)
plt.plot(df["orientation_25%"].cumsum(), label="orientation 25%", linestyle=quantile_line_style, color='purple', alpha=0.5)
plt.plot(df["orientation_75%"].cumsum(), label="orientation 75%", linestyle=quantile_line_style, color='purple', alpha=0.5)

plt.title("Cumulative Rewards Over Time")
plt.xlabel("Time Step")
plt.ylabel("Cumulative Mean Reward")
plt.legend()
plt.grid(True)
plt.show()
