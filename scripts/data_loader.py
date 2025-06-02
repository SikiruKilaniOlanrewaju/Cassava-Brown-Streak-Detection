import torch
import torchvision
from torchvision import transforms, datasets
import os
from PIL import Image

def is_image_file(filepath):
    try:
        with Image.open(filepath) as img:
            img.verify()
        return True
    except Exception:
        return False

def get_valid_image_paths(root_dir):
    valid_paths = []
    for class_dir in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_dir)
        if not os.path.isdir(class_path):
            continue
        for fname in os.listdir(class_path):
            fpath = os.path.join(class_path, fname)
            if is_image_file(fpath):
                valid_paths.append((fpath, class_dir))
    return valid_paths

class FilteredImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        # Filter out samples that are not valid images
        self.samples = [(path, target) for path, target in self.samples if is_image_file(path)]
        self.imgs = self.samples

# Configuration
data_dir = './data'
batch_size = 32
img_size = 224

def get_data_loaders(data_dir, batch_size, img_size):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = FilteredImageFolder(data_dir, transform=transform)
    class_names = dataset.classes
    num_classes = len(class_names)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, class_names, num_classes

if __name__ == '__main__':
    train_loader, val_loader, class_names, num_classes = get_data_loaders(data_dir, batch_size, img_size)
    print(f'Train batches: {len(train_loader)} | Validation batches: {len(val_loader)}')
    print(f'Classes: {class_names}')
