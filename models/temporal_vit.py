import torch
import torch.nn as nn



# What: A projection layer that refines raw patch embeddings before the transformer blocks.
# Why: Standard patch embeddings can be noisy. This connector aligns features into the expected embedding space
#      and provides an initial non-linear transformation to help the model converge faster.
# How: Uses a Bottleneck-style MLP (Linear 1x -> 4x -> 1x) with GELU activation, Dropout, and a Residual Connection.
class FeatureExtractionConnector(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim * 4)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(embed_dim * 4, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.norm(x + residual)
        return x



# What: A core Transformer Encoder block consisting of Multi-Head Self-Attention and a Feed-Forward Network.
# Why: This allows tokens (patches from different frames) to interact with each other, capturing
#      both spatial (within a frame) and temporal (across frames) relationships.
# How: Implements a "Pre-Norm" architecture where LayerNorm is applied before the Attention and MLP layers.
#      It uses MultiheadAttention for context mapping followed by a two-layer Linear MLP.
class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )
        self.ffn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_input = self.norm1(x)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input)
        x = x + self.attn_drop(attn_output)

        ffn_input = self.norm2(x)
        ffn_output = self.ffn(ffn_input)
        x = x + self.ffn_drop(ffn_output)
        return x



# What: A specialized attention gate that learns specific temporal behaviors like "drinking" or "eating".
# Why: Simple transformers treat all tokens equally. This module uses learnable 'queries' to 
#      actively search for behavior-specific signals in the token stream and highlight them.
# How: Uses Cross-Attention between learnable 'behaviour_queries' and the input sequence. 
#      The resulting context is turned into a Sigmoid gate that modulates (scales) the original features.
class BehaviourAwarenessModule(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.behaviour_queries = nn.Parameter(torch.zeros(1, 4, embed_dim))
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.gate_proj = nn.Linear(embed_dim, embed_dim)
        self.gate_act = nn.Sigmoid()
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.behaviour_queries, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        queries = self.behaviour_queries.expand(batch_size, -1, -1)
        behaviour_context, _ = self.cross_attn(queries, x, x)
        pooled_context = behaviour_context.mean(dim=1, keepdim=True)
        gate = self.gate_act(self.gate_proj(pooled_context))
        x = x + (x * gate)
        x = self.norm(x)
        return x


class TemporalViT(nn.Module):
    """
    Lightweight Video Vision Transformer (ViViT-lite).
    Expects input shape: [B, T, C, H, W] / B=batch, T=8 frames, C=3 channels, 224×224
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

        self.connector = FeatureExtractionConnector(embed_dim=embed_dim, dropout=dropout)
        self.transformer = nn.ModuleList(
            [
                SelfAttentionBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.behaviour_awareness = BehaviourAwarenessModule(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        
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

    # What: The main processing logic that converts a video clip into a behavior classification.
    # Why: It handles the transformation from 5D (Batch, Time, Channel, H, W) to a 1D vector of logits.
    # How: 
    # 1. Patchify: Each frame is split into 16x16 pixels flattened into embeddings.
    # 2. Positional Encoding: Adds spatial (where in frame) and temporal (when in clip) information.
    # 3. Transformers: Runs multiple self-attention blocks to model motion.
    # 4. Gating: Uses BehaviourAwareness to focus on relevant action cues.
    # 5. Pooling: Averages the frame-wise 'Class Tokens' to get a single summary vector for classification.
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

        x = self.connector(x)

        # Apply dropout
        x = self.pos_drop(x)

        # Run through transformer
        for block in self.transformer:
            x = block(x)

        x = self.behaviour_awareness(x)

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
