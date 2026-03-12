import torch
import torch.nn as nn

class HKA(nn.Module):
    """Simple single-path HKA"""
    def __init__(self, channels: int):
        super().__init__()
        self.gap_estimator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, 1)
        )
        self._init_weights()
    
    def _init_weights(self):
        for m in self.gap_estimator.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x):
        return torch.sigmoid(self.gap_estimator(x)).view(-1, 1, 1, 1)

class ViDAConvAdapter(nn.Module):
    """Simple ViDA without LayerNorm"""
    def __init__(self, channels: int, high_dim: int = 32, low_dim: int = 4):
        super().__init__()
        self.h_down = nn.Conv2d(channels, high_dim, 1, bias=False)
        self.h_up = nn.Conv2d(high_dim, channels, 1, bias=False)
        self.l_down = nn.Conv2d(channels, low_dim, 1, bias=False)
        self.l_up = nn.Conv2d(low_dim, channels, 1, bias=False)
        self.hka = HKA(channels)
        self._init_vida_weights()
    
    def _init_vida_weights(self):
        # High-rank: identity
        nn.init.orthogonal_(self.h_down.weight, gain=0.1)
        with torch.no_grad():
            h_down_matrix = self.h_down.weight.data.squeeze().t()
            h_up_init = torch.linalg.pinv(h_down_matrix).t()
            self.h_up.weight.data = h_up_init.unsqueeze(-1).unsqueeze(-1)
        # Low-rank: zero
        nn.init.normal_(self.l_down.weight, 0.0, 0.001)
        nn.init.normal_(self.l_up.weight, 0.0, 0.001)
    
    def forward(self, x):
        fh = self.h_up(self.h_down(x))
        fl = self.l_up(self.l_down(x))
        lambda_h = self.hka(x)
        lambda_l = 1.0 - lambda_h
        return x + lambda_h * fh + lambda_l * fl
