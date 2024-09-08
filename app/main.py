from pydantic import BaseModel # this has nothing to do my machine learning models
# this is just the parent class for everything that is strictly typed in Pydantic

from fastapi import FastAPI, Depends, UploadFile, File

from torchvision import transforms
from torchvision.models import ResNet

# we need this to upload images to fastAPI
# this is the Python image library
from PIL import Image

import torch

from app.model import load_model, load_transforms, CATEGORIES

import torch.nn.functional as F

import io

# This is what we use the BaseModel for
# the result is strictly typed so that it returns 
# a string for the category (label that we predict)
# and a float for the confidence (the probability for the label)
class Result(BaseModel):
    category: str
    confidence: float


# this creates an instance for the endpoint 
app = FastAPI()


# response_model is a pydantic BaseModel, not a machine learning model 
# is the POST response that we are defining with class Result(BaseModel)
@app.post('/predict', response_model=Result )
async def predict(
    # the output of File(...) is assigned to input_image, which is an UploadFile
    input_image: UploadFile = File(...),
    # the output of the load_model() function is assigned to model, which is of type ResNet 
    model: ResNet = Depends(load_model)
) -> Result: # this arrow specifies that predict() returns a Result object
    
    # Read the uploaded image
    image = Image.open(io.BytesIO(await input_image.read()))

    #convert of RGBA to RGB
    if image.mode == 'RGBA':
        image.convert('RGB')

    # apply the transformations to the image
    # we use unsqueeze(0) to define a batch size of 1 to feed the tensor to the model 
    image = transforms(image).unsqueeze(0)

    #Make the prediction
    with torch.no_grad():
        outputs = model(image)
        # todo: set up a breakpoint to understand outputs[0] and dim = 0
        probabilities = F.softmax(outputs[0], dim=0)
        confidence, predicted_class = torch.max(probabilities, 0)

    # Predicted label
    category = CATEGORIES[predicted_class.item()]

    return Result(category=category,
                   confidence=confidence.item())