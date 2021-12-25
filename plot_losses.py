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
    "-o",
    "--output",
    type=str,
    default="loss",
    help="output file",
)
parser.add_argument(
    "--tag",
    type=str,
    default="PN model",
    help="",
)
args = parser.parse_args()

training_losses = np.loadtxt(args.train_loss)
validation_losses = np.loadtxt(args.val_loss)

plt.plot(training_losses, label="Training Loss")
plt.plot(validation_losses, label="Validation Loss")
plt.legend()
plt.title(args.tag)
plt.xlabel("# Epochs")
plt.ylabel("Loss")
plt.savefig(f"plots/loss_plots/{args.output}.png")
