import numpy as np
import matplotlib.pyplot as plt

import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-t",
    dest="train_loss",
    type=str,
    default="models/dnn/ak8_dnn_classification_hwwVSQCD_coli_training_losses.txt",
    help="train loss txt file",
)
parser.add_argument(
    "-v",
    dest="val_loss",
    type=str,
    default="models/dnn/ak8_dnn_classification_hwwVSQCD_coli_validation_losses.txt",
    help="validation loss txt file",
)
parser.add_argument(
    "--model-path",
    dest="val_loss",
    type=str,
    default="models/dnn/ak8_dnn_classification_hwwVSQCD_coli",
    help="validation loss txt file",
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    default="loss",
    help="output file with extension",
)
parser.add_argument(
    "--tag",
    type=str,
    default="PN model",
    help="",
)
args = parser.parse_args()

# training_losses = np.loadtxt(args.train_loss)
# validation_losses = np.loadtxt(args.val_loss)

training_losses = np.loadtxt(f"{args.model_path}_training_losses.txt")
validation_losses = np.loadtxt(f"{args.model_path}_validation_losses.txt")

plt.plot(training_losses, label="Training Loss")
plt.plot(validation_losses, label="Validation Loss")
plt.legend()
plt.title(args.tag)
plt.xlabel("# Epochs")
plt.ylabel("Loss")

plt.savefig(args.output)
plt.close()

# plt.savefig("../plots/loss_plots/pyg_ef_test.pdf")

# model_addr = "/Users/raghav/Downloads/pyg_hetero_test"
# training_losses = np.loadtxt(f"{model_addr}_training_losses-2.txt")
# validation_losses = np.loadtxt(f"{model_addr}_validation_losses-2.txt")
#
# plt.plot(training_losses, label="Training Loss")
# plt.plot(validation_losses, label="Validation Loss")
# plt.legend()
# plt.title("PyG Heterogeneous Model")
# plt.xlabel("# Epochs")
# plt.ylabel("Loss")
# plt.savefig("../plots/loss_plots/pyg_hetero_test.pdf")
#
#
# model_addr = "/Users/raghav/Downloads/pyg_2"
# training_losses = np.loadtxt(f"{model_addr}_training_losses.txt")
# validation_losses = np.loadtxt(f"{model_addr}_validation_losses.txt")
#
# plt.plot(training_losses, label="Training Loss")
# plt.plot(validation_losses, label="Validation Loss")
# plt.legend()
# plt.title("PyG Model")
# plt.xlabel("# Epochs")
# plt.ylabel("Loss")
# plt.savefig("../plots/loss_plots/pyg_2.pdf")
