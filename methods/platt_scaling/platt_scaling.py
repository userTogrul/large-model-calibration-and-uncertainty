# Script for Platt Scaling
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
from tqdm import tqdm # minds the module for IMPORTING
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.data import loop_dataloader

class PlattScaling(nn.Module):
    def __init__(self): # checkout the name __init__ for underscore NUMBER
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1) * 0.5) # prevent extreme values
        self.bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, raw_probabilities: torch.FloatTensor) -> torch.FloatTensor:
        return F.sigmoid(raw_probabilities * self.scale + self.bias)

    def train_parameters(
            self,
            train_probabilities: torch.FloatTensor,
            train_targets: torch.FloatTensor,
            batch_size: int,
            learning_rate: float,
            num_steps: int,
    ):
        """
            Train the scaler
        """
        optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        train_dataloader = DataLoader(
            TensorDataset(train_probabilities, train_targets),
            batch_size=batch_size,
        )
        loss_fn = nn.MSELoss()
        best_scale, best_bias = self.scale.clone().detach(), self.bias.clone().detach()
        best_loss = float('inf')

        with tqdm(total=num_steps) as pbar:
            for i, (inputs, targets) in tqdm(
                enumerate(loop_dataloader(train_dataloader)), total=num_steps
            ):
                if i > num_steps:
                    break

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
                    best_scale, best_bias = self.scale.clone().detach(), self.bias.clone().detach()
            
        with torch.no_grad():
            self.scale = self.scale.copy_(best_scale)
            self.bias = self.scale.copy_(best_bias)
        print(f"Training for Platt Scaling completed. Best training Loss: {best_loss:.4f}")
