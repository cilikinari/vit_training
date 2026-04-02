import torch
import torch.nn as nn


#Patch embedding class
class PatchEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_embed = nn.Conv2d(
            config["num_channels"],
            config["embed_dim"],
            kernel_size=config["patch_size"],
            stride=config["patch_size"]
        )

    def forward(self, x):  # ✅ harus sejajar dengan __init__
        x = self.patch_embed(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x
        
#Transformer block class
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config["embed_dim"])
        self.multi_head_attention = nn.MultiheadAttention(config["embed_dim"], config["attention_heads"], batch_first=True)
        self.layer_norm2 = nn.LayerNorm(config["embed_dim"])
        self.mlp = nn.Sequential(
            nn.Linear(config["embed_dim"], config["mlp_nodes"]),
            nn.GELU(),
            nn.Linear(config["mlp_nodes"], config["embed_dim"])
        )

    def forward(self, x):
        residual1 = x
        x = self.layer_norm1(x)
        x = self.multi_head_attention(x, x, x)[0] + residual1
        residual2 = x
        x = self.layer_norm2(x)
        x = self.mlp(x) + residual2
        return x
    
#MLP class
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layernorm3 = nn.LayerNorm(config["embed_dim"])
        self.mlphead = nn.Sequential(
            nn.LayerNorm(config["embed_dim"]),
            nn.Linear(config["embed_dim"], config["num_classes"])
        )

    def forward(self, x):
        x = self.layernorm3(x)
        x = self.mlphead(x)
        return x
    
#Vision Transformer class
class VisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_embedding = PatchEmbedding(config)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config["embed_dim"]))
        self.position_embedding = nn.Parameter(torch.randn(1, config["patch_num"] + 1, config["embed_dim"]))
        self.transformer_blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config["transformer_blocks"])])
        self.mlp_head = MLP(config)

    def forward(self, x):
        x = self.patch_embedding(x)

        B = x.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embedding

        # 🔥 FIX DI SINI
        for block in self.transformer_blocks:
            x = block(x)

        x = x[:, 0]
        x = self.mlp_head(x)

        return x