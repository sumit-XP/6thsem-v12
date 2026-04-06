import torch
import torch.nn as nn

class TemporalViT(nn.Module):
    """
    Lightweight Video Vision Transformer (ViViT-lite).
    Expects input shape: [B, T, C, H, W]
    """
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 4,
        embed_dim: int = 192,
        depth: int = 4,
        num_heads: int = 3,
        num_frames: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert image_size % patch_size == 0, "Image size must be divisible by patch size"
        num_patches = (image_size // patch_size) ** 2
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        
        # Patch embedding: [B*T, C, H, W] -> [B*T, EmbedDim, H/P, W/P] -> [B*T, NumPatches, EmbedDim]
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Positional encodings
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)
        
        # Transformer Blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        # Sequence goes through `depth` layers
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a batch of temporal clips.
        Args:
            x: Tensor of shape [B, T, C, H, W]
        Returns:
            Tensor of shape [B, num_classes] (Logits)
        """
        B, T, C, H, W = x.shape
        # Only process up to `num_frames` limit
        T = min(T, self.num_frames)
        x = x[:, :T]
        
        # Flatten time into batch to process frame-wise patch embedding
        x = x.reshape(B * T, C, H, W)
        
        # Patch embedding
        x = self.patch_embed(x)                 # [B*T, EmbedDim, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)        # [B*T, NumPatches, EmbedDim]
        
        # Prepend class token
        cls_tokens = self.cls_token.expand(B * T, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)   # [B*T, NumPatches+1, EmbedDim]
        
        # Add spatial positional embedding
        x = x + self.spatial_pos_embed
        
        # Reshape to add temporal embedding
        _, N, D = x.shape
        x = x.view(B, T, N, D)
        
        # Add temporal embedding to all tokens inside the frame
        temp_embed = self.temporal_pos_embed[:, :T, :].unsqueeze(2) # [1, T, 1, D]
        x = x + temp_embed
        
        # Flatten time and space back to a single sequence per batch item
        x = x.view(B, T * N, D)
        
        # Apply dropout
        x = self.pos_drop(x)
        
        # Run through transformer
        x = self.transformer(x)
        
        # Extract the CLS token from each frame (located at indices 0, N, 2N...)
        x = x.view(B, T, N, D)
        cls_tokens_out = x[:, :, 0, :]          # [B, T, D]
        
        # Pool across time (e.g., mean pooling of CLS tokens)
        x = cls_tokens_out.mean(dim=1)          # [B, D]
        
        # Classify
        x = self.norm(x)
        x = self.head(x)                        # [B, NumClasses]
        
        return x

if __name__ == "__main__":
    # Smoke test
    model = TemporalViT(depth=2, num_heads=2)
    dummy_input = torch.randn(2, 8, 3, 224, 224) # Batch 2, 8 frames, RGB 224x224
    output = model(dummy_input)
    print(f"Output shape: {output.shape} | Expected: [2, 4]")
