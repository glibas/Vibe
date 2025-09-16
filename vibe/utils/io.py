"""Model saving and loading utilities."""

import torch
import os

def save_model(model, path):
    """Save model to file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_model(model, path, device='cpu'):
    """Load model from file."""
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    return model