import matplotlib.pyplot as plt

# Algorithms and  their metrics
algorithms = [
    "Random Committee", "Hoeffding Tree", "Logit Boost",
    "Rep Tree", "Bagging", "LWL"
]

# RMSE and MAE values
rmse = [0.0546, 0, 0.0685, 0.0912, 0.0594, 0.2169]
mae = [0.0073, 0, 0.006, 0.0103, 0.0073, 0.1063]

# Accuracy values
accuracies = [98.89, 100.00, 98.41, 99.05, 99.21, 98.57]

# RMSE & MAE Line Plot 
plt.figure(figsize=(10, 6))
plt.plot(algorithms, rmse, marker='o', label="RMSE", color='royalblue')
plt.plot(algorithms, mae, marker='s', label="MAE", color='orange')

plt.title("RMSE and MAE Error Comparison", fontsize=14, fontweight='bold')
plt.xlabel("Algorithms", fontsize=12)
plt.ylabel("Error Value", fontsize=12)
plt.xticks(rotation=20)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()

# Save image
plt.savefig("static/images/rmse.jpg")
plt.show()

#  Accuracy Bar Chart
plt.figure(figsize=(10, 6))
bars = plt.bar(algorithms, accuracies, color='skyblue', edgecolor='black')

# Add value labels on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.3, f'{yval}%', ha='center', fontsize=10)

plt.title("Model Accuracy Comparison", fontsize=14, fontweight='bold')
plt.xlabel("Algorithms", fontsize=12)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.ylim(95, 101)
plt.xticks(rotation=20)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()

# Save image 
plt.savefig("static/images/algorithm_accuracy.jpg")
plt.show()
