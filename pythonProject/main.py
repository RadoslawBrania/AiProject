import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from torchvision.models import resnet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

df = pd.read_csv('cards.csv')
df = df.dropna(subset=['filepaths'])

train_df = df[df['filepaths'].str.contains('train/')]
test_df = df[df['filepaths'].str.contains('test/')]
valid_df = df[df['filepaths'].str.contains('valid/')]

print(f"Train samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")
print(f"Validation samples: {len(valid_df)}")

class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 224x224
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Rozmiar po 3 poolingach: 224 → 112 → 56 → 28
        self.fc1 = nn.Linear(128 * 28 * 28, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def save_images(images, labels, folder="saved_images", prefix=""):
    os.makedirs(folder, exist_ok=True)
    for i, (img, label) in enumerate(zip(images, labels)):
        img_pil = transforms.ToPILImage()(img.cpu())
        class_name = str(label.item())
        class_folder = os.path.join(folder, class_name)
        os.makedirs(class_folder, exist_ok=True)
        filename = f"{prefix}sample_{i}.png"
        img_pil.save(os.path.join(class_folder, filename))


def plot_class_distribution(df, title):
    class_dist = df['labels'].value_counts()
    plt.figure(figsize=(12, 6))
    class_dist.plot(kind='bar')
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=90)
    plt.show()


plot_class_distribution(train_df, "Class Distribution - Training Set")
plot_class_distribution(test_df, "Class Distribution - Test Set")
plot_class_distribution(valid_df, "Class Distribution - Validation Set")


class CardDataset(Dataset):
    def __init__(self, dataframe, transform=None, save_samples=False):
        self.dataframe = dataframe
        self.transform = transform
        self.save_samples = save_samples
        if save_samples:
            self.samples_dir = "original_samples"
            os.makedirs(self.samples_dir, exist_ok=True)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['filepaths']
        label = self.dataframe.iloc[idx]['class']
        image = Image.open(img_path).convert('RGB')

        if self.save_samples and idx < 10:
            sample_path = os.path.join(self.samples_dir, f"sample_{idx}_class_{label}.png")
            image.save(sample_path)

        if self.transform:
            image = self.transform(image)

        #return image, label
        return image, torch.tensor(label, dtype=torch.long)


train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = CardDataset(train_df, transform=train_transforms, save_samples=True)
test_dataset = CardDataset(test_df, transform=test_transforms)
valid_dataset = CardDataset(valid_df, transform=test_transforms)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for batch in self.dl:
            yield to_device(batch, self.device)

    def __len__(self):
        return len(self.dl)


train_loader = DeviceDataLoader(train_loader, device)
test_loader = DeviceDataLoader(test_loader, device)
valid_loader = DeviceDataLoader(valid_loader, device)


def imshow(img, title=None):
    img = img.cpu().numpy().transpose((1, 2, 0))
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis('off')


images, labels = next(iter(train_loader))
save_images(images, labels, folder="transformed_samples", prefix="train_")

plt.figure(figsize=(12, 8))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    imshow(images[i], title=f"Class: {labels[i].item()}")
plt.show()

# ================================================
#model resnet
# ================================================

num_classes = df['class'].nunique()

# model = resnet18(weights=None)
# model.fc = nn.Linear(model.fc.in_features, num_classes)
model = CustomCNN(num_classes)
model = model.to(device)

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def evaluate(model, val_loader):
    model.eval()
    loss_total, acc_total = 0, 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            acc = accuracy(outputs, labels)
            loss_total += loss.item()
            acc_total += acc.item()
    return loss_total / len(val_loader), acc_total / len(val_loader)

def train_model(model, train_loader, val_loader, epochs=15, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss, train_acc = 0, 0
        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            acc = accuracy(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += acc.item()

        val_loss, val_acc = evaluate(model, val_loader)

        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"Train loss: {train_loss / len(train_loader):.4f}, acc: {train_acc / len(train_loader):.4f}")
        print(f"Val   loss: {val_loss:.4f}, acc: {val_acc:.4f}")
        print('-' * 40)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print("💾 Zapisano najlepszy model!")

train_model(model, train_loader, valid_loader, epochs=15, lr=0.001)

model.load_state_dict(torch.load('best_model.pth'))
test_loss, test_acc = evaluate(model, test_loader)
print(f"🧪 Test Accuracy: {test_acc:.4f}, Loss: {test_loss:.4f}")
