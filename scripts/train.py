import torch
import torch.nn as nn
from models.cassava_model import CassavaBrownStreakNet
from scripts.data_loader import get_data_loaders

# Configuration
data_dir = './data'
batch_size = 32
img_size = 224
epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train():
    train_loader, val_loader, class_names, num_classes = get_data_loaders(data_dir, batch_size, img_size)
    model = CassavaBrownStreakNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total if total > 0 else 0
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Val Acc: {val_acc:.4f}")
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'models/best_model.pth')
    print(f"Training complete. Best Val Acc: {best_val_acc:.4f}. Classes: {class_names}")

if __name__ == '__main__':
    train()
