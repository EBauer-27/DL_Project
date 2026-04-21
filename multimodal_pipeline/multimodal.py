"""Script to define the multimodal pipeline. This will include both the image and tabular data processing and model definition. Using cross attention to fuse the output form the resnet baseline and the tabtranaformer baseline."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ImageEncoder(nn.Module):
    """
    CNN-based image encoder.
    Returns:
        - pooled image feature vector [B, img_dim]
    """
    def __init__(self, backbone_name="resnet18", pretrained=True, out_dim=256):
        super().__init__()

        if backbone_name == "resnet18":
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            feat_dim = backbone.fc.in_features
            self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])  # remove fc
        elif backbone_name == "resnet50":
            backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            feat_dim = backbone.fc.in_features
            self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.proj = nn.Linear(feat_dim, out_dim)

    def forward(self, x):
        x = self.feature_extractor(x)          # [B, C, 1, 1]
        x = torch.flatten(x, 1)                # [B, C]
        x = self.proj(x)                       # [B, out_dim]
        return x


class TabularEncoder(nn.Module):
    """
    Custom tabular transformer encoder for categorical + continuous metadata.

    Returns:
        - token embeddings: [B, T, tab_dim]
        - pooled metadata vector: [B, tab_dim]
    """
    def __init__(
        self,
        categories,
        num_continuous,
        tab_dim=256,
        depth=2,
        heads=8,
        attn_dropout=0.1,
        ff_dropout=0.1,
    ):
        super().__init__()

        self.categories = categories
        self.num_categories = len(categories)
        self.num_continuous = num_continuous
        self.tab_dim = tab_dim

        # one embedding layer per categorical feature
        self.cat_embeds = nn.ModuleList([
            nn.Embedding(cardinality, tab_dim) for cardinality in categories
        ])

        # one projection per continuous feature -> each numeric feature becomes one token
        self.num_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, tab_dim),
                nn.ReLU(),
                nn.Linear(tab_dim, tab_dim),
            )
            for _ in range(num_continuous)
        ])

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=tab_dim,
            nhead=heads,
            dropout=attn_dropout,
            dim_feedforward=tab_dim * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.token_norm = nn.LayerNorm(tab_dim)
        self.pool_norm = nn.LayerNorm(tab_dim)

    def forward(self, x_categ, x_cont):
        tokens = []

        # categorical tokens
        for i, emb in enumerate(self.cat_embeds):
            tok = emb(x_categ[:, i])   # [B, tab_dim]
            tokens.append(tok)

        # continuous tokens
        for i, mlp in enumerate(self.num_mlps):
            val = x_cont[:, i].unsqueeze(1)    # [B, 1]
            tok = mlp(val)                     # [B, tab_dim]
            tokens.append(tok)

        x = torch.stack(tokens, dim=1)         # [B, T, tab_dim]
        x = self.transformer(x)                # [B, T, tab_dim]
        x = self.token_norm(x)

        pooled = x.mean(dim=1)                 # [B, tab_dim]
        pooled = self.pool_norm(pooled)

        return x, pooled


class CrossAttentionFusion(nn.Module):
    """
    Image-to-metadata cross-attention.
    Query = image token
    Key/Value = metadata tokens
    """
    def __init__(self, dim=256, heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, image_vec, metadata_tokens):
        """
        image_vec: [B, dim]
        metadata_tokens: [B, T, dim]
        """
        query = image_vec.unsqueeze(1)  # [B, 1, dim]

        attn_out, attn_weights = self.attn(
            query=query,
            key=metadata_tokens,
            value=metadata_tokens,
            need_weights=True,
            average_attn_weights=False,
        )  # attn_out: [B, 1, dim], attn_weights: [B, heads, 1, T]

        x = self.norm1(query + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        fused_vec = x.squeeze(1)  # [B, dim]
        return fused_vec, attn_weights


class MultimodalSkinClassifier(nn.Module):
    """
    Full multimodal model:
    image encoder + tabular encoder + cross-attention fusion + classifier
    """
    def __init__(
        self,
        categories,
        num_continuous,
        img_backbone="resnet18",
        pretrained=True,
        hidden_dim=256,
        tab_depth=2,
        tab_heads=8,
        fusion_heads=8,
        dropout=0.2,
    ):
        super().__init__()

        self.image_encoder = ImageEncoder(
            backbone_name=img_backbone,
            pretrained=pretrained,
            out_dim=hidden_dim,
        )

        self.tabular_encoder = TabularEncoder(
            categories=categories,
            num_continuous=num_continuous,
            tab_dim=hidden_dim,
            depth=tab_depth,
            heads=tab_heads,
            attn_dropout=dropout,
            ff_dropout=dropout,
        )

        self.fusion = CrossAttentionFusion(
            dim=hidden_dim,
            heads=fusion_heads,
            dropout=dropout,
        )

        # combine:
        # image vector
        # pooled metadata vector
        # fused vector
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, image, x_categ, x_cont, return_attention=False):
        img_vec = self.image_encoder(image)                    # [B, D]
        meta_tokens, meta_vec = self.tabular_encoder(x_categ, x_cont)  # [B, T, D], [B, D]
        fused_vec, attn_weights = self.fusion(img_vec, meta_tokens)    # [B, D], [B, H, 1, T]

        combined = torch.cat([img_vec, meta_vec, fused_vec], dim=1)    # [B, 3D]
        logits = self.classifier(combined)                              # [B, 1]

        if return_attention:
            return logits, attn_weights, meta_tokens
        return logits