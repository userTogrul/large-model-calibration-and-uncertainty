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
from tqdm import tqdm # mind the modules for importing tqdm
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
        return F.softmax(raw_probabilities / self.temperature, dim=-1) # can be negative
    
    def train_parameters(
        self,
        train_probabilities: torch.FloatTensor,
        train_targets: torch.FloatTensor,
        batch_size: int,
        learning_rate: float,
        num_steps: int,
    ):
        optimizer = optim.LBFGS([self.temperature], lr=learning_rate, max_iter=num_steps)
        loss_fn = nn.NLLLoss()
        train_dataloader = DataLoader(
            TensorDataset(train_probabilities, train_targets.type(torch.LongTensor)),
            batch_size=batch_size,
        )
        with tqdm(total=num_steps) as pbar:
            for i, (inputs, targets) in enumerate(loop_dataloader(train_dataloader)): 
                if i >= 1:
                    break

                def closure(): # necessary for LBFGS optimizer   
                    optimizer.zero_grad()
                    outputs = self.forward(inputs)
                    log_outputs = torch.log(outputs)
                    loss = loss_fn(log_outputs, targets)
                    loss.backward()
                    return loss
                optimizer.step(closure)
        
        with torch.no_grad():
            self.temperature.clamp_(min=1e-6) # make sure temperature is positive
        print(f"Training for Temperature Scaling completed. Best temperature: {self.temperature.item():.4f}")

