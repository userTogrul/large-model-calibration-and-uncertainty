"""
    Source:
        https://github.com/gpleiss/temperature_scaling.git
        Based on results from [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599).
"""
# Script for Temperature Scaling
# Standard Lib
import argparse
import os
import json
import time
import copy
import warnings
# EXTERNAL LIB
import numpy as np
import pandas as pd
from tqdm import tqdm # mind the modules for importing tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer
from src.data import loop_dataloader
import dill

class TemperatureScaling(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
    
    def forward(self, raw_probabilities: torch.FloatTensor,) -> torch.FloatTensor:
        # Add numerical stability checks
        raw_probabilities = torch.clamp(raw_probabilities, min=1e-7, max=1.0)
        log_probabilities = torch.log(raw_probabilities)
        log_probabilities = torch.nan_to_num(log_probabilities, nan=0.0)
        # Temperature should be positive to maintain monotonicity
        temperature = torch.clamp(self.temperature, min=1e-6, max=100.0)
        return torch.exp(log_probabilities / temperature)
         
    def train_parameters(
        self,
        train_probabilities: torch.FloatTensor,
        train_targets: torch.FloatTensor,
        learning_rate: float,
        batch_size: int=64,
        num_steps: int=200,
    ):
        """
            Train the Temperature Scaler

            Parameters:
            ----------------------------------------------------
            train_probabilities: torch.FloatTensor
                Raw probabilities from the model
            train_targets: torch.FloatTensor
                Target probabilities to calibrate against
            learning_rate: float
                Learning rate for optimization
            batch_size: int
                Batch size for training
            num_steps: int
                Number of optimization steps
        """
        optimizer = optim.AdamW(self.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()
        best_temperature = self.temperature.clone().detach()
        best_loss = float('inf')
        train_dataloader = DataLoader(
            TensorDataset(train_probabilities, train_targets),
            batch_size=batch_size,
            shuffle=True,  # Add shuffling for better training
        )
        
        with tqdm(total=num_steps) as pbar:
            for i, (inputs, targets) in enumerate(loop_dataloader(train_dataloader)): 
                if i >= num_steps:  # Changed from > to >= for exact number of steps
                    break

                outputs = self.forward(inputs)
                loss = loss_fn(outputs, targets)
                
                # Check for invalid loss
                if not torch.isfinite(loss):
                    warnings.warn("Non-finite loss detected, skipping step")
                    continue
                    
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                current_loss = loss.detach().cpu().item()
                pbar.set_description(
                    f"Step #{i+1} - Loss: {current_loss:.4f}"
                )
                pbar.update(1)

                if current_loss < best_loss:
                    best_loss = current_loss
                    best_temperature = self.temperature.clone().detach()

        # Set the best temperature found during training
        with torch.no_grad():
            self.temperature.data = torch.clamp(best_temperature, min=1e-6, max=100.0)
            
        print(f"Training completed. Best training Loss: {best_loss:.4f}, Temperature: {self.temperature.item():.4f}")
