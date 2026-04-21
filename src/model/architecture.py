import torch
import torch.nn as nn


class StonesCNN(nn.Module):
    """CNN model for audio classification using MFCC features."""
    
    def __init__(self, input_shape=torch.Size([101, 120]), num_classes=2):
        super(StonesCNN, self).__init__()
        
        time_frames, n_features = input_shape
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.0),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.0),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Dropout2d(0.0),
        )
        
        test_input = torch.randn(1, 1, time_frames, n_features)
        test_output = self.conv_layers(test_input)
        flat_size = test_output.view(1, -1).size(1)
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(flat_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x