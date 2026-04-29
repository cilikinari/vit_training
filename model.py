import torch
import torch.nn as nn

class VisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.img_size = config["img_size"]
        self.patch_size = config["patch_size"]
        self.embed_dim = config["embed_dim"]

        # =========================
        # HITUNG PATCH NUM DI SINI
        # =========================
        patch_num = (self.img_size // self.patch_size) ** 2

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels=config["num_channels"],
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim) * 0.02)

        # Positional embedding
        self.position_embedding = nn.Parameter(
            torch.randn(1, patch_num + 1, self.embed_dim) * 0.02
        )

        # Transformer encoder (simple version)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=config["attention_heads"],
            dim_feedforward=config["mlp_nodes"],
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config["transformer_blocks"]
        )

        # classifier
        self.mlp_head = nn.Linear(self.embed_dim, config["num_classes"])

    def forward(self, x):
        B = x.shape[0]

        x = self.patch_embed(x)  # (B, E, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, N, E)

        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        x = x + self.position_embedding

        x = self.transformer(x)

        x = x[:, 0]  # CLS token
        x = self.mlp_head(x)

        return x