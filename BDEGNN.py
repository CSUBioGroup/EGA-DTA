import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, Set2Set, LayerNorm

# ==================== 基础组件 ====================

class SwiGLU(nn.Module):
    def __init__(self, dim_in, dim_out, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(dim_in, dim_out)
        self.w2 = nn.Linear(dim_in, dim_out)
        self.w3 = nn.Linear(dim_out, dim_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate = F.silu(self.w1(x))
        val = self.w2(x)
        out = gate * val
        return self.dropout(self.w3(out))


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return x + self.block(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, C = x.shape
        # 兼容性处理：增加序列维度
        x = x.unsqueeze(1) # [B, 1, C]
        B, N, _ = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out).squeeze(1)


class LearnableFusion(nn.Module):
    def __init__(self, num_features=3):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_features))

    def forward(self, features):
        weights = F.softmax(self.weights, dim=0)
        return sum(w * f for w, f in zip(weights, features))


# ==================== 蛋白质编码器 (适配 3B 特征) ====================
class AdvancedProteinEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        # SwiGLU 作为瓶颈层：2560 -> 1024
        self.input_proj = SwiGLU(input_dim, hidden_dim, dropout)

        self.res_blocks = nn.Sequential(
            ResidualBlock(hidden_dim, dropout),
            ResidualBlock(hidden_dim, dropout)
        )

        self.out_proj = nn.Linear(hidden_dim, output_dim)
        self.ln_final = nn.LayerNorm(output_dim)

    def forward(self, xt, device):
        xt = xt.to(device)
        if xt.dim() > 2:
            xt = xt.squeeze(1) # [B, esm_dim]

        x = self.input_proj(xt)
        x = self.res_blocks(x)
        x = self.out_proj(x)
        x = self.ln_final(x)
        return x


# ==================== 经典架构回归版 ====================
class EnhancedGATNet(nn.Module):
    def __init__(self, n_output=1, num_features_xd=72, num_features_xt=25,
                 output_dim=512, dropout=0.1,
                 use_precomputed_esm2=True, use_esm2=False,
                 esm_dim=2560, # 默认适配 3B
                 ecfp_dim=2048, 
                 gnn_hidden_dim=128,
                 gnn_heads=8,
                 fusion_strategy='full'
                 ):

        super().__init__()
        self.fusion_strategy = fusion_strategy
        self.output_dim = output_dim

        # ---------- 1. GNN 分支 ----------
        self.conv1 = TransformerConv(num_features_xd, gnn_hidden_dim, heads=gnn_heads,
                                     concat=True, dropout=dropout, edge_dim=2, beta=True)
        self.ln1 = LayerNorm(gnn_hidden_dim * gnn_heads)

        self.conv2 = TransformerConv(gnn_hidden_dim * gnn_heads, gnn_hidden_dim, heads=gnn_heads,
                                     concat=False, dropout=dropout, edge_dim=2, beta=True)
        self.ln2 = LayerNorm(gnn_hidden_dim)

        self.pool = Set2Set(gnn_hidden_dim, processing_steps=3)
        self.pool_out_dim = gnn_hidden_dim * 2
        self.fc_g_out = nn.Sequential(
            nn.Linear(self.pool_out_dim, output_dim * 2),
            nn.ELU(),
            nn.Linear(output_dim * 2, output_dim)
        )

        # ---------- 2. ECFP 分支 ----------
        self.ecfp_proj_dim = 1024
        self.ecfp_reduction = nn.Sequential(
            nn.Linear(ecfp_dim, self.ecfp_proj_dim),
            nn.LayerNorm(self.ecfp_proj_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.ecfp_out_proj = nn.Sequential(
            nn.Linear(self.ecfp_proj_dim, output_dim),
            nn.GELU()
        )

        # ---------- 3. 蛋白质分支 ----------
        self.protein_encoder = AdvancedProteinEncoder(
            input_dim=esm_dim, 
            hidden_dim=1024, 
            output_dim=output_dim, 
            dropout=dropout
        )

        # ---------- 4. 融合组件 ----------
        self.drug_gate = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.Sigmoid()
        )

        self.learnable_fusion = LearnableFusion(3)
        # 注意：这里保留初始化是为了防止加载权重报错，但实际上我们用 mean 替代了它
        self.fusion_attention = MultiHeadAttention(output_dim, num_heads=8)

        # 最终分类器
        if fusion_strategy == 'full':
            # Concat(3) + Fused(1) + Attn(1) = 5
            fusion_dim_total = output_dim * 5
        else:
            fusion_dim_total = output_dim * 4

        self.classifier = nn.Sequential(
            SwiGLU(fusion_dim_total, 1024, dropout),
            ResidualBlock(1024, dropout),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ELU(),
            nn.Linear(512, n_output)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr

        # === 1. GNN ===
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = self.ln1(x, batch)
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        x = self.ln2(x, batch)
        x = self.pool(x, batch)
        x_gnn = self.fc_g_out(x)

        # === 2. ECFP ===
        ecfp = data.ecfp.float()
        x_ecfp_red = self.ecfp_reduction(ecfp)
        x_ecfp = self.ecfp_out_proj(x_ecfp_red)

        # === 3. Protein ===
        x_protein = self.protein_encoder(data.protein_features, x.device)

        # === 4. Gating ===
        gate_signal = self.drug_gate(x_protein)
        x_gnn_gated = x_gnn * gate_signal
        x_ecfp_gated = x_ecfp * gate_signal

        # === 5. Fusion ===
        # Weighted Sum
        x_fused = self.learnable_fusion([x_gnn_gated, x_ecfp_gated, x_protein])

        # Attention 替代方案 (稳健版)
        # 直接计算 Stacked 特征的平均值，作为一种简单的"全局交互"
        # 这避免了复杂的 Attention 计算导致的数值不稳定和报错
        stacked = torch.stack([x_gnn, x_ecfp, x_protein], dim=1)
        
        # [FIXED] 这里删除了之前报错的 attn_out = ... 行
        x_attn = stacked.mean(dim=1) 
        
        x_concat = torch.cat([x_gnn_gated, x_ecfp_gated, x_protein], dim=1)

        # 拼接所有特征
        xc = torch.cat([x_concat, x_fused, x_attn], dim=1)

        out = self.classifier(xc)
        return out