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
        log_probabilities = torch.log(raw_probabilities)
        log_probabilities = torch.nan_to_num(log_probabilities, nan=0.0)
        return torch.exp(log_probabilities / self.temperature)
         
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
            train_correctness: torch.LongTensor
                Whether the generated sequence is correct or not.
        """
        optimizer = optim.AdamW(self.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()
        best_temperature =  self.temperature.clone().detach()
        best_loss = float('inf')
        train_dataloader = DataLoader(
            TensorDataset(train_probabilities, train_targets), # no need to wrap to TensorDataset for one item
            batch_size=batch_size,
        )
        with tqdm(total=num_steps) as pbar:
            for i, (inputs, targets) in enumerate(loop_dataloader(train_dataloader)): 

                if i > num_steps:
                    break

                # stabilize the temperature to avoid very small values
                outputs = self.forward(inputs)
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
                    best_temperature = self.temperature.clone().detach()

        # Clamp temperature to avoid instability
        with torch.no_grad():
            self.temperature = self.temperature.copy_(best_temperature)
            self.temperature.clamp_(min=1e-6) # to avoid extremely small temperature
        print(f"Training for Temperature Scaling completed. Best training Loss: {best_loss:.4f}")
