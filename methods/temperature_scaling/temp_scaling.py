"""
    Source:
        https://github.com/gpleiss/temperature_scaling.git
        Based on results from [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599).
"""
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
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.data import loop_dataloader

class TemperatureScaling(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
    
    def forward(self, raw_probabilities: torch.FloatTensor) -> torch.FloatTensor:
        return F.softmax(raw_probabilities / self.temperature, dim=1)
    
    def train_parameters(
        self,
        train_probabilities: torch.FloatTensor,
        train_targets: torch.FloatTensor,
        batch_size: int,
        learning_rate: float,
        num_steps: int,
    ):
        optimizer = optim.lbfgs([self.temperature], lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        train_dataloader = DataLoader(
            TensorDataset(train_probabilities, train_targets),
            batch_size=batch_size,
        )
        best_temperature = self.temperature.clone().detach()
        best_loss = float("inf")
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
        
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_temperature = self.temperature.clone().detach()
        
        with torch.no_grad():
            self.temperature = self.temperature.copy_(best_temperature) # inplace update
        print(f"Training for Temperature Scaling completed. Best training Loss: {best_loss:.4f}")

