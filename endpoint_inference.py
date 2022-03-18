import json
import logging
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import requests

#Basic config for logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

JPEG_CONTENT_TYPE = 'image/jpeg'
JSON_CONTENT_TYPE = 'application/json'
ACCEPTED_CONTENT_TYPE = [ JPEG_CONTENT_TYPE ] #Add support for jpeg images

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Call GPU

def net():
    model = models.resnet50(pretrained = True) #Use the pretrained resnet50 model with 50 layers
    
    for param in model.parameters():
        param.requires_grad = False #Freeze all the Conv layers
    
    num_features = model.fc.in_features
    model.fc = nn.Sequential( nn.Linear( num_features, 256), #Adding two fully connected layers
                             nn.ReLU(inplace = True),
                             nn.Linear(256, 133),
                             nn.ReLU(inplace = True) # output should have 133 nodes as we have 133 classes of dog breeds
                            )
    return model


#Override model_fn to load the model
def model_fn(model_dir):
    logger.info("Inside model_fn function!")
    logger.info(f"Reading model from dir: {model_dir}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on device: {device}")
    model = net().to(device)
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        logger.info("Starting to load the model...")
        model.load_state_dict(torch.load(f, map_location = device))
        logger.info("Successfully loaded the model")
    return model

# Override the default input_fn
def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
    logger.info('Deserializing the input data.')
    # Process an image uploaded to the endpoint
    logger.info(f'Incoming Requests Content-Type is: {content_type}')
    logger.info(f'Request body Type is: {type(request_body)}')
    if content_type in ACCEPTED_CONTENT_TYPE:
        logger.info(f"Returning an image of type {content_type}" )
        return Image.open(io.BytesIO(request_body))
    else:
        raise Exception(f"Requested an unsupported Content-Type: {content_type}, Accepted Content-Type are: {ACCEPTED_CONTENT_TYPE}")

# Override the default predict_fn
def predict_fn(input_object, model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Starting the prediction process...")
    test_transform =  transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor() ]
    )
    logger.info("Starting to apply Transforms to the input image")
    input_object=test_transform(input_object)
    if torch.cuda.is_available():
        input_object = input_object.cuda() #put data into GPU
    logger.info("Completed applying Transforms to the input image")
    model.eval()
    with torch.no_grad():
        logger.info("Starting the model invokation")
        prediction = model(input_object.unsqueeze(0))
    return prediction
