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
weight_decays = [1e-1, 1e-2, 1e-3]
lr = 1e-3
epochs = [5000, 10000, 50000]
minimal_losses = []

# Train the model for each weight decay and store the minimal loss
for i in range(len(weight_decays)):
    training_losses = train_model(model, inputs, outputs, epochs=epochs[i], lr=lr, weight_decay=weight_decays[i], batch_size=N)
    minimal_losses.append(min(training_losses))

# Create figure for minimal loss vs weight decay
fig, ax = plt.subplots(figsize=(4.5, 3)) 

plt.rcParams.update({
    "legend.frameon": False,
})

adjusted_weight_decays = [wd / weight_decays[0] * minimal_losses[0] for wd in weight_decays]

# Plot reference slope=1 line
ax.loglog(weight_decays, adjusted_weight_decays, linestyle='--', color='gray', label='Linear Scaling')

# Plot minimal loss against log(weight_decay)
ax.loglog(weight_decays, minimal_losses, marker='o', label='Minimal Training Loss', color='blue')


# Axis labels
ax.set_xlabel('$\epsilon$')
ax.set_ylabel('minimum loss')

# Add legend
ax.legend()

# Layout and save
plt.tight_layout()
plt.savefig(os.path.join("figs", "figure1b.pdf"))
plt.close()