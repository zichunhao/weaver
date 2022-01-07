import numpy as np
import matplotlib.pyplot as plt

import os

model_addr = "/Users/raghav/Downloads/pyg_ef_test"
training_losses = np.loadtxt(f"{model_addr}_training_losses.txt")
validation_losses = np.loadtxt(f"{model_addr}_validation_losses.txt")

plt.plot(training_losses, label="Training Loss")
plt.plot(validation_losses, label="Validation Loss")
plt.legend()
plt.title("PyG Edge Features Model")
plt.xlabel("# Epochs")
plt.ylabel("Loss")
plt.savefig("../plots/loss_plots/pyg_ef_test.pdf")


model_addr = "/Users/raghav/Downloads/pyg_hetero_test"
training_losses = np.loadtxt(f"{model_addr}_training_losses.txt")
validation_losses = np.loadtxt(f"{model_addr}_validation_losses.txt")

plt.plot(training_losses, label="Training Loss")
plt.plot(validation_losses, label="Validation Loss")
plt.legend()
plt.title("PyG Heterogeneous Model")
plt.xlabel("# Epochs")
plt.ylabel("Loss")
plt.savefig("../plots/loss_plots/pyg_hetero_test.pdf")
