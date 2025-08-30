import torch.nn as nn
import torch

class QualityWeight(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        # x: (batch_size, input_dim)
        # 输出： (batch_size,)
        return self.mlp(x).squeeze(-1)


class QualityWeightFlatten(nn.Module):
    def __init__(self, num_patch, in_dim=5, hidden_dim=32):
        super().__init__()
        self.num_patch = num_patch
        self.in_dim = in_dim
        self.mlp = nn.Sequential(
            nn.Linear(num_patch * in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        # x: [batch_size, num_patch, in_dim]
        B, N, D = x.shape
        assert N == self.num_patch and D == self.in_dim
        x_flat = x.view(B, N * D)    # [B, num_patch*in_dim]
        return self.mlp(x_flat).squeeze(-1)  # [B]
    
class QualityWeightFlatten1(nn.Module):
    def __init__(self, num_patch, in_dim=5, hidden_dim=8):
        super().__init__()
        self.num_patch = num_patch
        self.in_dim = in_dim
        self.mlp = nn.Sequential(
            nn.Linear(num_patch * in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        # x: [batch_size, num_patch, in_dim]
        B, N, D = x.shape
        assert N == self.num_patch and D == self.in_dim
        x_flat = x.view(B, N * D)    # [B, num_patch*in_dim]
        return self.mlp(x_flat).squeeze(-1)  # [B]


class QualityWeightAttention(nn.Module):
    def __init__(self, num_patch, in_dim=5):
        super().__init__()
        self.num_patch = num_patch
        self.in_dim = in_dim

        # 每个 patch 的打分权重通过 attention logits 计算
        self.attn_layer = nn.Sequential(
            nn.Linear(in_dim, 1),
        )

    def forward(self, x):
        # x: [B, N, D]
        B, N, D = x.shape
        assert N == self.num_patch and D == self.in_dim

        attn_logits = self.attn_layer(x).squeeze(-1)  # [B, N]
        attn_weights = torch.softmax(attn_logits, dim=1)  # [B, N]

        # score: ∑ (attn_weight_i * score_i)
        # 原始得分就是 softmax 权重后的加权平均
        weighted_score = torch.sum(attn_weights * torch.sum(x * torch.arange(1, D+1).float().to(x.device), dim=-1), dim=-1)
        return weighted_score  # [B]
