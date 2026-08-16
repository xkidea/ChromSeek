import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List

# ================================================================
# 1. Basic Components
# ================================================================

class ConvBlock(nn.Module):
    """
    Standard convolution block: Conv1d -> BatchNorm -> GELU.
    Supports stride-based downsampling.
    """
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1, 
                 dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        # Compute padding:
        # If stride=1, this is same padding.
        # If stride>1, this preserves correct dimensional scaling when divisible.
        padding = (dilation * (kernel_size - 1)) // 2
        
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, 
                      padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.block(x)

class ResidualBlock(nn.Module):
    """
    Pre-activation residual block: preserves dimensions while increasing depth.
    """
    def __init__(self, channels: int, kernel_size: int = 3, 
                 dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        padding = (dilation * (kernel_size - 1)) // 2
        
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, 
                               padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, 
                               padding=padding, dilation=dilation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = F.gelu(self.bn1(x))
        out = self.conv1(out)
        out = F.gelu(self.bn2(out))
        out = self.conv2(out)
        out = self.dropout(out)
        return residual + out

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.
    max_len defaults to 10000, which covers length 2240 (448k/200).
    """
    def __init__(self, d_model, max_len=10000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Dim)
        # Dynamically slice PE to the corresponding sequence length.
        seq_len = x.size(1)
        if seq_len > self.pe.size(0):
            raise ValueError(f"Sequence length {seq_len} exceeds PositionalEncoding max_len {self.pe.size(0)}")
        
        x = x + self.pe[:seq_len, :]
        return self.dropout(x)

# ================================================================
# 2. Core Mechanism: Cell-Cross-Attention
# ================================================================

class CellAttention(nn.Module):
    """
    Use cell embeddings to gate and modulate genomic features.
    """
    def __init__(self, hidden_dim, cell_dim):
        super().__init__()
        self.to_q = nn.Linear(hidden_dim, hidden_dim)
        self.to_k = nn.Linear(cell_dim, hidden_dim)
        self.to_v = nn.Linear(cell_dim, hidden_dim)
        self.scale = hidden_dim ** -0.5
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, cell_embed):
        # x: (B, L, C)
        # cell_embed: (B, cell_dim)
        B, L, C = x.shape
        
        q = self.to_q(x)                    # (B, L, C)
        k = self.to_k(cell_embed).view(B, 1, C) # (B, 1, C)
        v = self.to_v(cell_embed).view(B, 1, C) # (B, 1, C)

        # Attention Score & Sigmoid Gating
        dots = (q @ k.transpose(-2, -1)) * self.scale
        attn = dots.sigmoid() 
        
        out = attn * v 
        out = self.out_proj(out)
        
        return x + out

# ================================================================
# 3. Main Model Class: CellSpecificOmicsModel_448k
# ================================================================

class CellSpecificOmicsModel_448k(nn.Module):
    """
    Version designed for 448kb input (224*2000) and 200bp output resolution.
    Total Downsampling: 200x
    Transformer Depth: 6 layers
    """
    def __init__(self, 
                 num_cells: int,
                 embed_dim: int = 64,
                 num_tasks: int = 1,
                 # Input length: 448,000
                 seq_len: int = 448000, 
                 encoder_channels: List[int] = [64, 128, 256, 384],
                 body_dim: int = 384
                 ):
        super().__init__()
        
        self.num_tasks = num_tasks
        
        # --- 1. Embedding ---
        self.cell_embedder = nn.Embedding(num_cells, embed_dim)

        # --- 2. Downsampler Tower (Total 200x) ---
        # Target: 448,000 -> 2,240 (scale factor 200)
        # Combination: 2 (Stem) * 5 * 5 * 4 = 200
        
        # Stem: Stride 2 (448k -> 224k)
        self.stem = ConvBlock(4, encoder_channels[0], kernel_size=15, stride=2) 
        
        self.down_blocks = nn.ModuleList()
        in_c = encoder_channels[0]
        
        # Strides are set to [5, 5, 4].
        factors = [5, 5, 4] 
        
        # Dynamically build the downsampling layers.
        # Layer 1: stride 5 (224k -> 44800)
        # Layer 2: stride 5 (44800 -> 8960)
        # Layer 3: stride 4 (8960 -> 2240) -> enter Body
        
        # Ensure encoder_channels is long enough; reuse the channel definition here.
        # encoder_channels[1:] corresponds to 128, 256, 384.
        for i, (out_c, pool) in enumerate(zip(encoder_channels[1:], factors)):
            self.down_blocks.append(
                ConvBlock(in_c, out_c, kernel_size=5, stride=pool)
            )
            in_c = out_c
            
        # --- 3. Body (Bottleneck L=2240) ---
        self.body_proj = ConvBlock(encoder_channels[-1], body_dim, kernel_size=1)
        
        # A. Local feature extraction (ResBlocks)
        self.body_res = nn.Sequential(
            ResidualBlock(body_dim, dilation=1),
            ResidualBlock(body_dim, dilation=2),
            ResidualBlock(body_dim, dilation=4)
        )
        
        # B. Global context (Transformer)
        # max_len=10000 safely covers 2240.
        self.pos_encoder = PositionalEncoding(body_dim, max_len=10000)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=body_dim,
            nhead=8,
            dim_feedforward=body_dim * 4,
            dropout=0.2,
            activation='gelu',
            batch_first=True # (B, L, C)
        )
        
        # Increase depth to 6 layers.
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        # C. Cell-specific modulation
        self.cell_attn1 = CellAttention(hidden_dim=body_dim, cell_dim=embed_dim)
        self.cell_attn2 = CellAttention(hidden_dim=body_dim, cell_dim=embed_dim)
        
        # --- 4. Decoder ---
        # Keep resolution unchanged (L=2240) and only refine features.
        self.decoder = nn.Sequential(
            ConvBlock(body_dim, 256, kernel_size=5, dropout=0.1),
            ConvBlock(256, 256, kernel_size=5, dropout=0.1),
            ConvBlock(256, 128, kernel_size=5, dropout=0.1),
            ConvBlock(128, 64, kernel_size=5, dropout=0.1)
        )
        
        # --- 5. Final Head ---
        self.final_head = nn.Sequential(
            nn.Conv1d(64, num_tasks, kernel_size=1)
        )

    def forward(self, dna_seq, cell_id):
        # dna_seq: (B, 4, 448000)
        
        cell_emb = self.cell_embedder(cell_id) 
        
        # --- Encoder ---
        x = self.stem(dna_seq)
        for block in self.down_blocks:
            x = block(x)
            
        # --- Body ---
        x = self.body_proj(x)
        x = self.body_res(x)
        
        # --- Transformer (B, L, C) ---
        x = x.permute(0, 2, 1) 
        x = self.pos_encoder(x)
        x = self.transformer(x) 
        
        # --- Cell Attention ---
        x = self.cell_attn1(x, cell_emb)
        x = self.cell_attn2(x, cell_emb)
        
        # --- Decoder (B, C, L) ---
        x = x.permute(0, 2, 1) 
        x = self.decoder(x)
        
        # Output
        out = self.final_head(x) # (B, num_tasks, 2240)
        
        return out

# ===============================================================
# Dimension and Logic Check
# ===============================================================
if __name__ == "__main__":
    # Mock parameters
    BATCH_SIZE = 2
    SEQ_LEN = 448000    # 224 * 2000 bp
    NUM_TASKS = 10      # Number of output channels
    
    model = CellSpecificOmicsModel_448k(num_cells=6, embed_dim=64, num_tasks=NUM_TASKS)
    
    # 1. Create dummy input
    dummy_dna = torch.randn(BATCH_SIZE, 4, SEQ_LEN)
    dummy_cell = torch.randint(0, 6, (BATCH_SIZE,))
    
    print(f"Input Shape: {dummy_dna.shape}")
    
    # 2. Forward Pass
    output = model(dummy_dna, dummy_cell)
    
    print(f"Output Shape: {output.shape}")
    
    # 3. Validate dimensions
    # Expected length: 448000 / 200 = 2240
    expected_len = SEQ_LEN // 200
    assert output.shape == (BATCH_SIZE, NUM_TASKS, expected_len), \
        f"Mismatch! Expected last dim {expected_len}, got {output.shape[-1]}"
        
    print(f"Test passed: output correctly downsampled to {output.shape[-1]} bins (200bp resolution).")
