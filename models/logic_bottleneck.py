import torch
import torch.nn as nn

class LogicBottleneck(nn.Module):
    """
    Differentiable logic layer for binary CA patch inference.
    Learns 9 neighbor weights and two thresholds for birth/survival.
    """
    def __init__(self, alpha: float = 10.0):
        super().__init__()
        # weight per neighbor position (3x3 patch flattened)
        self.w = nn.Parameter(torch.randn(9))
        # thresholds for birth and survival
        self.b_birth = nn.Parameter(torch.tensor(3.0))
        self.b_surv  = nn.Parameter(torch.tensor(2.0))
        # sharpness for sigmoid
        self.alpha = alpha

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        patches: Tensor[B,9] with 0/1 entries
        returns: Tensor[B] in [0,1] representing next state probability
        """
        # Linear combination of neighbors
        s = patches.float().matmul(self.w)
        # survival if current cell alive
        surv = torch.sigmoid(self.alpha * (s - self.b_surv))
        # birth if current cell dead
        birth = torch.sigmoid(self.alpha * (s - self.b_birth))
        # combine; clamp to [0,1]
        out = torch.clamp(surv + birth, 0.0, 1.0)
        return out
