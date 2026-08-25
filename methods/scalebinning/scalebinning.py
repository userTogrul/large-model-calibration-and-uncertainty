"""
    Scale-binning calibration method.
    Implementation of the scaling-binning calibrator from the paper:
    "Verified Uncertainty Calibration" (https://arxiv.org/pdf/1909.10155)
"""
# Standard Lib
import warnings

# External Lib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np

class ScaleBinning(nn.Module):
    def __init__(self, num_bins=10):
        """
        Initialize the Scale-binning calibrator.
        
        Parameters:
        -----------
        num_bins : int
            Number of bins to divide the predictions into.
        """
        super().__init__()
        self.num_bins = num_bins
        # Platt scaling parameters
        self.scale = nn.Parameter(torch.ones(1) * 0.5)  # prevent extreme values
        self.bias = nn.Parameter(torch.zeros(1))
        # Storage for bin statistics
        self.bin_boundaries = None
        self.bin_values = None
        
    def _platt_scaling(self, probabilities):
        """Apply Platt scaling transformation."""
        return F.sigmoid(probabilities * self.scale + self.bias)
        
    def _compute_bin_boundaries(self, probabilities):
        """Compute bin boundaries to have equal number of points per bin."""
        with torch.no_grad():
            # Compute quantiles for uniform mass binning
            self.bin_boundaries = torch.quantile(
                probabilities.flatten(),
                torch.linspace(0, 1, self.num_bins + 1),
            )
            # Ensure unique boundaries
            self.bin_boundaries = torch.unique(self.bin_boundaries)
            # Update num_bins if we got fewer unique boundaries
            self.num_bins = len(self.bin_boundaries) - 1
    
    def _compute_bin_values(self, probabilities, targets=None):
        """
        Compute average of Platt scaled values in each bin.
        If targets are provided, compute average of targets instead (for training).
        """
        bin_sums = torch.zeros(self.num_bins, device=probabilities.device)
        bin_counts = torch.zeros(self.num_bins, device=probabilities.device)
        
        scaled_probs = self._platt_scaling(probabilities)
        values_to_bin = targets if targets is not None else scaled_probs
        
        for i in range(self.num_bins):
            mask = (scaled_probs >= self.bin_boundaries[i]) & (scaled_probs <= self.bin_boundaries[i + 1])
            bin_sums[i] = values_to_bin[mask].sum()
            bin_counts[i] = mask.sum()
            
        # Compute average value per bin, avoiding division by zero
        safe_counts = torch.clamp(bin_counts, min=1)
        self.bin_values = bin_sums / safe_counts
        
    def forward(self, probabilities: torch.FloatTensor) -> torch.FloatTensor:
        """
        Apply scale-binning calibration to the input probabilities.
        
        Parameters:
        -----------
        probabilities : torch.FloatTensor
            Input probabilities to calibrate.
            
        Returns:
        --------
        torch.FloatTensor
            Calibrated probabilities.
        """
        if self.bin_boundaries is None or self.bin_values is None:
            raise RuntimeError("Model must be trained before inference")
            
        # First apply Platt scaling
        scaled_probs = self._platt_scaling(probabilities)
        
        # Assign each probability to a bin
        calibrated_probs = torch.zeros_like(scaled_probs)
        for i in range(self.num_bins):
            mask = (scaled_probs >= self.bin_boundaries[i]) & (scaled_probs <= self.bin_boundaries[i + 1])
            calibrated_probs[mask] = self.bin_values[i]
            
        return calibrated_probs
    
    def train_parameters(
        self,
        train_probabilities: torch.FloatTensor,
        train_targets: torch.FloatTensor,
        learning_rate: float = 0.01,
        batch_size: int = 64,
        num_steps: int = 200,
    ):
        """
        Train the Scale-binning calibrator using a two-step process:
        1. Train Platt scaling parameters
        2. Compute bin values based on the scaled probabilities
        
        Parameters:
        -----------
        train_probabilities : torch.FloatTensor
            Uncalibrated probabilities from the model.
        train_targets : torch.FloatTensor
            Ground truth targets.
        learning_rate : float
            Learning rate for optimization.
        batch_size : int
            Batch size for training.
        num_steps : int
            Number of training steps.
        """
        # Step 1: Train Platt scaling parameters
        optimizer = optim.SGD([self.scale, self.bias], lr=learning_rate)
        loss_fn = nn.MSELoss()
        
        best_scale = self.scale.clone().detach()
        best_bias = self.bias.clone().detach()
        best_loss = float('inf')
        
        train_dataloader = DataLoader(
            TensorDataset(train_probabilities, train_targets),
            batch_size=batch_size,
            shuffle=True
        )
        
        with tqdm(total=num_steps, desc="Training Platt scaling") as pbar:
            for i, (inputs, targets) in enumerate(loop_dataloader(train_dataloader)):
                if i >= num_steps:
                    break
                    
                outputs = self._platt_scaling(inputs)
                loss = loss_fn(outputs, targets)
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                pbar.set_description(
                    f"Step #{i+1} - Loss: {loss.detach().cpu().item():.4f}"
                )
                pbar.update(1)
                
                if loss.detach().cpu().item() < best_loss:
                    best_loss = loss.detach().cpu().item()
                    best_scale = self.scale.clone().detach()
                    best_bias = self.bias.clone().detach()
        
        # Use the best parameters found during training
        with torch.no_grad():
            self.scale.copy_(best_scale)
            self.bias.copy_(best_bias)
        
        # Step 2: Compute bin boundaries and values
        self._compute_bin_boundaries(self._platt_scaling(train_probabilities))
        self._compute_bin_values(train_probabilities, train_targets)
        
        print(f"Training completed. Best Platt scaling loss: {best_loss:.4f}")
