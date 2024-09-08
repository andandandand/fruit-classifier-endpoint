import wandb
import os
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet

#todo: remember to delete the use of the 
# loadotenv and os.getenv entries 
# when we use the Docker image later
from loadotenv import load_env

load_env(file_loc='/workspaces/fruit-classifier-endpoint/app/.env')

MODELS_DIR = 'models'
MODEL_FILE_NAME = 'model.pth'

CATEGORIES = ["freshapples", "freshbanana", "freshoranges", 
              "rottenapples", "rottenbanana", "rottenoranges"]

#todo: delete 
#print(os.getenv('WANDB_API_KEY'))

def download_artifact():
    assert 'WANDB_API_KEY' in os.environ, 'Please enter WANDB_API_KEY as an environmental variable.'

    wandb.login() # here we get access to the artifact registry
    # go to your artifact registry to find this full path
    # antonios-org/banana_apple_orange/resnet18:v1
    wandb_org = os.environ.get('WANDB_ORG')
    wandb_project = os.environ.get('WANDB_PROJECT')
    wandb_model_name = os.environ.get('WANDB_MODEL_NAME')
    wandb_model_version = os.environ.get('WANDB_MODEL_VERSION')

    #here we reconstruct the artifact path
    artifact_path = f'{wandb_org}/{wandb_project}/{wandb_model_name}:{wandb_model_version}'
    artifact = wandb.Api().artifact(artifact_path, type='model')
    artifact.download(root=MODELS_DIR)


def get_raw_model() -> ResNet:

    N_CLASSES = 6

    model = resnet18(weights=None)

    # check that this architecture is the same in your Kaggle notebook
    model.fc = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, N_CLASSES)
    )

    return model 

def load_model() -> ResNet:
    download_artifact()
    model = get_raw_model()
    model_state_dict_path = os.path.join(MODELS_DIR, MODEL_FILE_NAME)
    model_state_dict = torch.load(model_state_dict_path, map_location='cpu')
    # this was strict=False in the code here https://github.com/aihpi/fruit-classifier-gcloud-run/blob/main/app/model.py
    model.load_state_dict(model_state_dict, strict=True) 
    model.eval()

    return model

def load_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

