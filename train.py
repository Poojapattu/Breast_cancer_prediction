# ---------------------------
# BreastMNIST Deep Learning Pipeline
# ---------------------------
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns

# ---------------------------
# 1. Load BreastMNIST Dataset
# ---------------------------
from medmnist import BreastMNIST
import medmnist

# Download if not available
data_flag = 'breastmnist'
download = True
BATCH_SIZE = 64
NUM_EPOCHS = 5
LR = 0.001

# Data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

train_dataset = BreastMNIST(split='train', transform=transform, download=download)
val_dataset = BreastMNIST(split='val', transform=transform, download=download)
test_dataset = BreastMNIST(split='test', transform=transform, download=download)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

# ---------------------------
# 2. Define CNN Model
# ---------------------------
class BreastCancerCNN(nn.Module):
    def __init__(self):
        super(BreastCancerCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 2)  # 2 classes: Benign/Malignant

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BreastCancerCNN().to(device)

# ---------------------------
# 3. Training
# ---------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.squeeze().long().to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {running_loss/len(train_loader):.4f}")

# ---------------------------
# 4. Evaluation
# ---------------------------
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.squeeze().long().to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=["Benign","Malignant"]))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign","Malignant"], yticklabels=["Benign","Malignant"])
plt.title("Confusion Matrix")
plt.show()

# ---------------------------
# 5. Grad-CAM Visualization
# ---------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_image, target_class=None):
        input_image = input_image.unsqueeze(0).to(device)
        output = self.model(input_image)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_image.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam

# Pick one test image
test_img, label = test_dataset[0]
gradcam = GradCAM(model, model.conv2)
heatmap = gradcam.generate(test_img)

plt.subplot(1,2,1)
plt.imshow(test_img.squeeze(), cmap="gray")
plt.title(f"True: {label}")
plt.subplot(1,2,2)
plt.imshow(test_img.squeeze(), cmap="gray")
plt.imshow(heatmap, cmap="jet", alpha=0.5)
plt.title("Grad-CAM")
plt.show()
# Save the trained model
torch.save(model.state_dict(), "breastmnist_cnn.pth")
print("Model saved as breastmnist_cnn.pth")
