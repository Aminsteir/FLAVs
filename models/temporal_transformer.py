import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights

class TemporalTransformer(nn.Module):
    def __init__(self, frame_dim=1280, num_frames=3, num_heads=4, hidden_dim=256, ff_dim=512, num_layers=4):
        """
        Temporal Transformer model for spatial-temporal tasks.

        Args:
            frame_dim (int): Dimensionality of the CNN-extracted features.
            num_frames (int): Number of frames in the temporal sequence.
            num_heads (int): Number of attention heads in the Transformer.
            hidden_dim (int): Hidden dimensionality for the Transformer.
            ff_dim (int): Feed-forward network dimensionality in the Transformer.
            num_layers (int): Number of Transformer encoder layers.
        """
        super(TemporalTransformer, self).__init__()
        
        # Lightweight CNN backbone for spatial feature extraction
        self.cnn_backbone = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT).features
        self.cnn_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(576, frame_dim),  # MobileNet's final feature size is 576
            nn.ReLU()
        )

        # Temporal Transformer
        self.embedding = nn.Linear(frame_dim, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, num_frames, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=ff_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Fully connected head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Predict single steering angle
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (Tensor): Input tensor of shape [batch_size, num_frames, 3, height, width].

        Returns:
            Tensor: Predicted steering angles of shape [batch_size, 1].
        """
        batch_size, num_frames, _, height, width = x.size()

        # Extract spatial features for each frame
        frame_features = []
        for i in range(num_frames):
            frame = x[:, i]  # Shape: [batch_size, 3, height, width]
            spatial_features = self.cnn_backbone(frame)  # Extract CNN features
            frame_features.append(self.cnn_fc(spatial_features))  # Shape: [batch_size, frame_dim]

        # Stack frame features and add positional encoding
        frame_features = torch.stack(frame_features, dim=1)  # Shape: [batch_size, num_frames, frame_dim]
        temporal_embeddings = self.embedding(frame_features) + self.positional_encoding  # Add positional encoding

        # Pass through Transformer
        temporal_features = self.transformer(temporal_embeddings.permute(1, 0, 2))  # Shape: [num_frames, batch_size, hidden_dim]
        temporal_features = temporal_features.mean(dim=0)  # Global average pooling over time

        # Prediction head
        return self.fc(temporal_features)
