#!/usr/bin/env python3

"""
Defines:
  1) LinearRegressionModel(nn.Module): A simple linear regression model.
  2) TukeyBiweightLoss(nn.Module): A custom robust loss function based on
     Tukey’s biweight (bisquare) function.

Usage Example (in a separate training script):

  import torch
  from model import LinearRegressionModel, TukeyBiweightLoss

  # Create a linear model with e.g. 5 input features
  model = LinearRegressionModel(input_dim=5)
  
  # Create the Tukey's biweight loss with c=4.6851
  criterion = TukeyBiweightLoss(c=4.6851, reduction='mean')
  
  optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

  # Suppose we have X of shape (batch_size, 5), y of shape (batch_size,)
  # typical training loop snippet:
  for epoch in range(100):
      optimizer.zero_grad()
      preds = model(X)  # shape (batch_size,1)
      loss = criterion(preds, y)  # scalar
      loss.backward()
      optimizer.step()
  
"""

import torch
import torch.nn as nn

class LinearRegressionModel(nn.Module):
    """
    A simple linear regression model: y = X * w + b
    for input_dim features -> output_dim=1.
    """
    def __init__(self, input_dim=1):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=False)  # PyTorch built-in linear layer

    def forward(self, x):
        """
        x: shape (batch_size, input_dim)
        returns: shape (batch_size, 1)
        """
        return self.linear(x)


class TukeyBiweightLoss(nn.Module):
    """
    Custom robust loss based on Tukey’s biweight (bisquare) function.
    
    The biweight function for each residual r with cutoff c is:
    
        rho(r) = c^2 / 6 * [1 - (1 - (r/c)^2)^3],       if |r| < c
                 c^2 / 6,                             if |r| >= c
    
    We apply this elementwise to residuals (pred - target).
    
    Arguments:
      c (float): the tuning constant (default ~4.6851 in robust stats)
      reduction (str): 'mean' or 'sum'. If 'mean', average over the batch.
    """
    def __init__(self, c=4.6851, reduction='mean'):
        super().__init__()
        self.c = c
        self.reduction = reduction

    def forward(self, pred, target):
        """
        pred: shape (batch_size, 1) or (batch_size,)
        target: shape (batch_size,) or (batch_size,1)
        
        Returns a scalar (if reduction='mean' or 'sum') 
        or a tensor of shape (batch_size,) if no reduction.
        """
        # Ensure pred, target are same shape
        if pred.dim() > 1 and pred.shape[1] == 1:
            pred = pred.squeeze(dim=1)
        if target.dim() > 1 and target.shape[1] == 1:
            target = target.squeeze(dim=1)
        
        # residuals
        r = pred - target  # shape (batch_size,)
        abs_r = torch.abs(r)

        # mask for which residuals are < c
        mask = (abs_r < self.c)
        
        # squared ratio (r/c)^2
        ratio_sq = (r / self.c)**2
        
        # Tukey’s biweight piecewise function
        # if |r| < c: c^2/6 * [1 - (1 - ratio_sq)^3]
        # if |r| >= c: c^2/6
        inside = 1.0 - ratio_sq  # (batch_size,)
        inside_cube = inside**3  # (batch_size,)
        
        val_in = (self.c**2 / 6.0) * (1.0 - inside_cube)
        val_out = torch.full_like(r, self.c**2 / 6.0)
        
        # piecewise
        loss_vals = torch.where(mask, val_in, val_out)  # shape (batch_size,)
        
        if self.reduction == 'mean':
            return torch.mean(loss_vals)
        elif self.reduction == 'sum':
            return torch.sum(loss_vals)
        else:
            # 'none' -> return the full tensor of losses
            return loss_vals