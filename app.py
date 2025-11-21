# app.py
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# 1. Model Definition
# ---------------------------
class BreastCancerCNN(nn.Module):
    def __init__(self):
        super(BreastCancerCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128,2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64*7*7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "breastmnist_cnn.pth"  # path to your trained model

# ---------------------------
# 2. Grad-CAM
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
        class_loss = output[0,target_class]
        class_loss.backward()
        weights = self.gradients.mean(dim=(2,3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_image.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

# ---------------------------
# 3. Load Model
# ---------------------------
@st.cache_resource
def load_model():
    model = BreastCancerCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()
    return model

model = load_model()
gradcam = GradCAM(model, model.conv2)
classes = ["Benign","Malignant"]

# ---------------------------
# 4. Streamlit UI
# ---------------------------
st.title("🩺 Breast Cancer Prediction (BreastMNIST)")
st.write("Upload a grayscale histopathology image (28x28) to predict Benign or Malignant.")

uploaded_file = st.file_uploader("Choose an image...", type=["png","jpg","jpeg"])

transform = transforms.Compose([
    transforms.Resize((28,28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("L")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    x = transform(img).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred = probs.argmax()
    
    st.subheader(f"Prediction: {classes[pred]}")
    st.write(f"Confidence: {probs[pred]:.3f}")
    
    st.bar_chart({classes[0]: probs[0], classes[1]: probs[1]})

    # Grad-CAM visualization
    heatmap = gradcam.generate(transform(img))
    plt.imshow(img, cmap="gray")
    plt.imshow(heatmap, cmap="jet", alpha=0.5)
    plt.axis("off")
    st.pyplot(plt.gcf())
