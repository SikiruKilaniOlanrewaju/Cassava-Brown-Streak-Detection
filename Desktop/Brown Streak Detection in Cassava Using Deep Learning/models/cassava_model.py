import torch
import torch.nn as nn
from scripts.data_loader import get_data_loaders

class CassavaBrownStreakNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        from torchvision import models
        self.base_model = models.resnet18(weights='IMAGENET1K_V1')
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)

if __name__ == '__main__':
    # Dynamically get number of classes from data
    _, _, _, num_classes = get_data_loaders('./data', 32, 224)
    model = CassavaBrownStreakNet(num_classes=num_classes)
    print(model)
