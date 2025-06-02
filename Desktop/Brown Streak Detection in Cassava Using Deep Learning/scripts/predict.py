import torch
from models.cassava_model import CassavaBrownStreakNet
from scripts.data_loader import get_data_loaders

# Example: Load a saved model and run inference on a batch

def predict(model_path, images, data_dir='./data', img_size=224):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, _, class_names, num_classes = get_data_loaders(data_dir, 1, img_size)
    model = CassavaBrownStreakNet(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    with torch.no_grad():
        outputs = model(images.to(device))
        _, preds = torch.max(outputs, 1)
    return [class_names[p] for p in preds]

def load_model(model_path, data_dir='./data', img_size=224):
    _, _, class_names, num_classes = get_data_loaders(data_dir, 1, img_size)
    model = CassavaBrownStreakNet(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model, class_names

# Usage example (replace with real images and model path):
# images = ... # torch tensor of images
# preds = predict('models/best_model.pth', images)
