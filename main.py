from fastapi import FastAPI
from pydantic import BaseModel
import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import base64
import os
import uuid

#INPUTS:
class RandomInput(BaseModel):
    imgID: int

class CustomInput(BaseModel):
    imgStr: str

#OUTPUTS:
class Output(BaseModel):
    classification: str

app = FastAPI()

MODEL_PATH = "trained-model.pth"

def load_model():
    # Create the model architecture
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    # Load saved weights into the model
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()  # Set the model to evaluation mode
    return model

def preprocess_image(img: Image.Image):
    preprocess_pipeline = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return preprocess_pipeline(img)

def classify_image(model, image_path):
    with Image.open(image_path) as img:  # Open image using PIL
        image_tensor = preprocess_image(img).unsqueeze(0)
    model.eval()
    image_tensor = image_tensor.to(next(model.parameters()).device)
    with torch.no_grad():
        outputs = model(image_tensor)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    _, predicted_class = torch.max(probabilities, 1)

    if predicted_class == 0: 
        return "cat"
    elif predicted_class == 1: 
        return "dog"

trained_model = load_model()

@app.get("/")
async def root():
    return {"message": "You were the chosen one..."}

@app.post("/classify/random/")
async def classify_random(inputs: RandomInput):
    # Classify the image
    image_path = "./testing-data/" + str(inputs.imgID) + ".jpg"
    result = classify_image(trained_model, image_path)

    return [
        Output(classification=result)
    ]

@app.post("/classify/custom/")
async def classify_custom(inputs: CustomInput):
    # Classify the image
    imgUUID = uuid.uuid4()
    image_file_name = str(imgUUID) + ".jpg"
    image_data = base64.b64decode(inputs.imgStr)
    with open(image_file_name, "wb") as f:
        f.write(image_data)
    image_path = "./" + image_file_name
    result = classify_image(trained_model, image_path)

    os.remove(image_file_name)

    return [
        Output(classification=result)
    ]