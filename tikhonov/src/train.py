import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

def hausdorff_to_singleton_loss(y_pred, y, reduction="mean"):
    """
    y_pred: (B, n, d)  model outputs (tokens)
    y:      (B, d) or (B, 1, d)  singleton targets
    returns: scalar loss = mean over batch of max_{token} ||x - y||^2
    """
    if y.dim() == 3 and y.size(1) == 1:
        y = y[:, 0, :]                     # (B, d)
    assert y_pred.dim() == 3 and y.dim() == 2
    # Euclidean distances from every token to y
    d2 = (y_pred - y.unsqueeze(1))**2      # (B, n, d)
    d2 = d2.sum(dim=-1)                    # (B, n)
    # Hausdorff to singleton = max over tokens
    per_ex_loss = d2.max(dim=1).values     # (B,)
    return per_ex_loss.mean() if reduction == "mean" else per_ex_loss


def train_model(model, samples, targets, epochs=1000, lr=1e-3, weight_decay=1e-3,
                batch_size=2, device='cpu'):
    """
    samples: list/array/tensor of shape (B, n, d)
    targets: list/array/tensor of shape (B, d) or (B, 1, d)
    """
    model = model.to(device)

    # Build dataset & loader (assume samples/targets already tensors on CPU)
    train_dataset = list(zip(samples, targets))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    training_losses = []
    for epoch in range(epochs):
        model.train()
        epoch_losses = []

        for x, y in train_loader:
            x = x.to(device)               # (B, n, d_in)
            y = y.to(device)               # (B, d_out) or (B, 1, d_out) with d_out == model d
            optimizer.zero_grad()

            y_pred = model(x)              # (B, n, d)
            loss = hausdorff_to_singleton_loss(y_pred, y, reduction="mean")
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        training_losses.append(train_loss)
        print(f'Epoch {epoch+1}/{epochs} - Training Loss: {train_loss:.6f}')

    return training_losses