import gradio as gr
import torch
from transformers import AutoImageProcessor, SiglipForImageClassification, pipeline
from torchvision import transforms
from PIL import Image
import numpy as np

# -------------------------------
# Hugging Face models for Rice, Sugarcane, Tomato
# -------------------------------
hf_model_names = {
    "Rice": "prithivMLmods/Rice-Leaf-Disease",
    "Sugarcane": "dwililiya/sugarcane-plant-diseases-classification",
    "Tomato": "wellCh4n/tomato-leaf-disease-classification-resnet50"
}

# Load Rice model with processor
hf_processors = {}
hf_models = {}
hf_processors['Rice'] = AutoImageProcessor.from_pretrained(hf_model_names['Rice'])
hf_models['Rice'] = SiglipForImageClassification.from_pretrained(hf_model_names['Rice'])
print("Rice model loaded with image processor.")

# Load Sugarcane model without processor (manual preprocessing)
hf_models['Sugarcane'] = SiglipForImageClassification.from_pretrained(hf_model_names['Sugarcane'])
print("Sugarcane model loaded (manual preprocessing required).")

# Load Tomato model using pipeline (no processor needed)
hf_models['Tomato'] = pipeline("image-classification", model=hf_model_names['Tomato'])
print("Tomato model loaded with pipeline.")

# -------------------------------
# Sugarcane manual preprocessing
# -------------------------------
sugarcane_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------------------------------
# Disease mapping 
# -------------------------------
disease_dict = {
    "Rice": ["Bacterial Blight", "Blast", "Brown Spot", "Healthy", "Tungro"],
    "Sugarcane": ["Bacterial Blight", "Healthy", "Mosaic", "Red Rot", "Rust", "Yellow"],
    "Tomato": ["Early Blight", "Late Blight", "Healthy"]
}

# Remedies mapping
remedies = {
    "Early Blight": "Remove infected leaves, apply fungicide.",
    "Late Blight": "Use fungicides and remove infected plants.",
    "Bacterial Blight": "Use resistant varieties and avoid overhead watering.",
    "Blast": "Use balanced fertilizer, apply fungicide.",
    "Brown Spot": "Ensure proper field drainage and avoid overcrowding.",
    "Tungro": "Control green leafhoppers and remove infected plants.",
    "Mosaic": "Remove infected plants, avoid spread.",
    "Red Rot": "Remove infected plants, apply fungicide.",
    "Rust": "Use fungicide and resistant varieties.",
    "Yellow": "Monitor plant, apply preventive measures.",
    "Healthy": "No action needed."
}

# -------------------------------
# Prediction function
# -------------------------------
def predict_disease(crop, img):
    if img is None:
        return "No image uploaded", "Please upload a leaf image."

    img_pil = Image.fromarray(img).convert("RGB")

    if crop == "Rice":
        inputs = hf_processors[crop](images=img_pil, return_tensors="pt")
        with torch.no_grad():
            outputs = hf_models[crop](**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
            predicted_idx = int(np.argmax(probs))
        disease = disease_dict[crop][predicted_idx]
        advice = remedies.get(disease, "No advice available.")
        return disease, advice

    elif crop == "Sugarcane":
        img_tensor = sugarcane_transform(img_pil).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            outputs = hf_models[crop](img_tensor)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
            predicted_idx = int(np.argmax(probs))
        disease = disease_dict[crop][predicted_idx]
        advice = remedies.get(disease, "No advice available.")
        return disease, advice

    elif crop == "Tomato":
        result = hf_models[crop](img_pil)[0]
        disease = result['label']
        advice = remedies.get(disease, "No advice available.")
        return disease, advice

    else:
        return "Error", f"Model for {crop} is not available."

# -------------------------------
# Gradio Interface
# -------------------------------
custom_css = """
body, .gradio-container {
    background-image: url('https://media.istockphoto.com/id/1328004520/photo/healthy-young-soybean-crop-in-field-at-dawn.jpg?s=612x612&w=0&k=20&c=XRw20PArfhkh6LLgFrgvycPLm0Uy9y7lu9U7fLqabVY=');
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    min-height: 100vh !important;
}
.gradio-container > * {
    background-color: rgba(255, 255, 255, 0.88) !important;
    border-radius: 15px;
    padding: 20px;
}
"""

with gr.Blocks(css=custom_css) as app:
    gr.Markdown("## 🌿 Crop Disease Detector")
    gr.Markdown("Upload a leaf image of your crop and get AI-based disease prediction with remedies.")

    with gr.Row():
        with gr.Column():
            crop_input = gr.Dropdown(list(hf_model_names.keys()), label="Select Crop")
            img_input = gr.Image(type="numpy", label="Upload Leaf Image")
            predict_btn = gr.Button("🔍 Predict Disease")

        with gr.Column():
            disease_output = gr.Textbox(label="Predicted Disease")
            advice_output = gr.Textbox(label="Recommended Action")

    predict_btn.click(predict_disease, inputs=[crop_input, img_input], outputs=[disease_output, advice_output])

# Launch
app.launch(server_name="127.0.0.1", server_port=7860, share=True)
