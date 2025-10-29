import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
from src.generateData import generateSyntheticData
from src.models import TF
from src.train import train_model


# Set random seeds for reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# Parameters for the data generation
N = 10       # Number of sequences
d = 16       # Dimension of the input space
n = 100      # Number of tokens in each sequence

# Generate synthetic data
inputs, outputs, exact_L2norm = generateSyntheticData(N, d, n, param_norm = True)

# Define the model
model = TF(d, h=1, tau=1.0, causal=False, depth=1)

# Train the model
epochs = 50000
lr = 1e-3
weight_decay = 1e-4
training_losses = train_model(model, inputs, outputs, epochs=epochs, lr=lr, weight_decay=weight_decay, batch_size=N)

# Plot training loss
plt.rcParams.update({
    "font.size": 10,            # Base font size
    "axes.labelsize": 10,       # Axis label size
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "lines.linewidth": 1.5,
    "lines.markersize": 4,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "legend.frameon": False,
    "pdf.fonttype": 42          # Use TrueType fonts in PDF
})

# Create figure
fig, ax = plt.subplots(figsize=(4.5, 3))  # Suitable for 1-column figure

# Plot reference line
ax.hlines(weight_decay * exact_L2norm, 0, len(training_losses), 
          colors='red', linestyles='--', label='Tikhonov threshold')

# Plot training loss (log-log)
ax.loglog(training_losses, label='Training Loss', color='blue')

# Axis labels
ax.set_xlabel('epochs')
ax.set_ylabel('loss')

# Add legend
ax.legend()

# Layout and save
plt.tight_layout()
plt.savefig(os.path.join(r"figs", "figure1a.pdf"))
plt.close()