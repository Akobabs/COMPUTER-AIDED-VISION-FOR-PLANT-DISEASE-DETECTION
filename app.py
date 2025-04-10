import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from flask import Flask, request, render_template, jsonify
import cv2
import numpy as np

# Define paths
base_dir = "C:/Users/Akoba/Desktop/START up/COMPUTER-AIDED-VISION-FOR-PLANT-DISEASE-DETECTION"
model_path = os.path.join(base_dir, "models/plantvillage_hybrid_best.pth")
upload_dir = os.path.join(base_dir, "uploads")
static_dir = os.path.join(base_dir, "static")
os.makedirs(upload_dir, exist_ok=True)
os.makedirs(static_dir, exist_ok=True)

# Initialize Flask app
app = Flask(__name__)

# Define the hybrid model (same as in training)
class HybridCNNViT(nn.Module):
    def __init__(self, num_classes):
        super(HybridCNNViT, self).__init__()
        self.cnn = models.resnet50(pretrained=False)
        self.cnn.fc = nn.Identity()
        cnn_out_features = 2048
        self.vit = models.vit_b_16(pretrained=False)
        vit_out_features = self.vit.heads.head.in_features
        self.vit.heads = nn.Identity()
        self.fusion = nn.Linear(cnn_out_features + vit_out_features, 512)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        cnn_features = self.cnn(x)
        vit_features = self.vit(x)
        combined_features = torch.cat((cnn_features, vit_features), dim=1)
        fused = self.fusion(combined_features)
        fused = torch.relu(fused)
        output = self.classifier(fused)
        return output

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 15  # From your dataset
model = HybridCNNViT(num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Define label mapping (same as in training)
label_map = {
    "Pepper_bell___Bacterial_spot": 0, "Pepper_bell___healthy": 1,
    "Potato___Early_blight": 2, "Potato___healthy": 3, "Potato___Late_blight": 4,
    "Tomato___Target_Spot": 5, "Tomato___Tomato_mosaic_virus": 6,
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": 7, "Tomato___Bacterial_spot": 8,
    "Tomato___Early_blight": 9, "Tomato___healthy": 10, "Tomato___Late_blight": 11,
    "Tomato___Leaf_Mold": 12, "Tomato___Septoria_leaf_spot": 13,
    "Tomato___Spider_mites_Two_spotted_spider_mite": 14
}
reverse_label_map = {v: k for k, v in label_map.items()}

# Define transform for inference
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Simple Grad-CAM implementation for explainability
def generate_gradcam(model, image_tensor, target_class):
    model.eval()
    image_tensor = image_tensor.requires_grad_(True)
    output = model(image_tensor)
    model.zero_grad()
    output[0, target_class].backward()
    gradients = image_tensor.grad[0].cpu().data.numpy()
    weights = np.mean(gradients, axis=(1, 2))
    feature_maps = model.cnn.conv1(image_tensor)[0].cpu().data.numpy()
    cam = np.zeros(feature_maps.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * feature_maps[i]
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (256, 256))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    return cam

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    
    # Save the uploaded file
    img_path = os.path.join(upload_dir, file.filename)
    file.save(img_path)

    # Preprocess the image
    image = Image.open(img_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        predicted_class = predicted.item()
        confidence = confidence.item()

    # Get the predicted label
    predicted_label = reverse_label_map[predicted_class]

    # Generate Grad-CAM heatmap
    cam = generate_gradcam(model, image_tensor, predicted_class)
    heatmap_path = os.path.join(static_dir, "heatmap.jpg")
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    original_img = cv2.imread(img_path)
    original_img = cv2.resize(original_img, (256, 256))
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
    superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
    cv2.imwrite(heatmap_path, superimposed_img)

    return jsonify({
        "prediction": predicted_label,
        "confidence": confidence,
        "heatmap": "static/heatmap.jpg"
    })

if __name__ == "__main__":
    app.run(debug=True)