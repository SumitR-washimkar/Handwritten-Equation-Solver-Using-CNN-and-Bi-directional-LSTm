import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class CNNEncoder(nn.Module):
    def __init__(self, output_dim, dropout=0.3):
        super(CNNEncoder, self).__init__()
        
        base_model = models.resnet18(pretrained=True)
        
        # Remove last avgpool and fc layer to keep feature maps
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-2]) 
        
        # Add BatchNorm after the final conv output to stabilize training
        self.bn = nn.BatchNorm1d(512)
        
        # Project 512-dim features to output_dim
        self.fc = nn.Linear(512, output_dim)
        
        # Dropout to regularize and reduce overfitting
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, images):
        features = self.feature_extractor(images)  # [B, 512, H', W']
        
        B, C, H, W = features.size()
        
        # Flatten spatial dimensions
        features = features.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, Seq, C]
        
        # BatchNorm1d expects (B * Seq, C), so reshape
        features_reshaped = features.contiguous().view(-1, C)  # [B*Seq, C]
        
        features_bn = self.bn(features_reshaped)  # Normalize features
        
        features_bn = self.dropout(features_bn)  # Apply dropout
        
        # Project features to output_dim
        projected = self.fc(features_bn)  # [B*Seq, output_dim]
        
        # Reshape back to sequence format
        projected = projected.view(B, H * W,-1)  # [B, Seq, output_dim]
        
        return projected
