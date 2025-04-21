import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel

class DiffusionUNet(nn.Module):
    """
    Conditional U-Net for predicting noise in CA board transitions.
    Wraps HuggingFace UNet2DConditionModel for 1-channel binary grids.
    """
    def __init__(self):
        super().__init__()
        # Initialize a small UNet with depth 2
        self.unet = UNet2DConditionModel(
            sample_size=32,
            in_channels=1,
            out_channels=1,
            down_block_types=("DownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "UpBlock2D"),
            block_out_channels=(64, 128),
            cross_attention_dim=32,
        )

    def forward(self, x, timestep, cond):
        # x: [B,1,32,32] noisy board
        # cond: [B,1,32,32] clean board_t as condition
        return self.unet(x, timestep, encoder_hidden_states=cond).sample
