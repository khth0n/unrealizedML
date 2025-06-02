import torch
import numpy as np
from PIL import Image

from unrealizedML.nn import nca

def load_image(image_path):
    
    img = Image.open(image_path).convert('RGB')
    img = np.array(img).T / 255
    
    return img

def prepare_data(device):
    
    #TODO: Replace most of this with pickle loading for generalization
    seed_paths = []
    target_paths = []

    for i in range(3, 6):
        for j in range(20):
            seed_paths.append(f'seeds/med/{i}_{j}.png')
            target_paths.append(f'targets/med/{i}_{j}.png')

    seed_imgs = np.array([load_image(path) for path in seed_paths])
    seed_imgs = torch.tensor(seed_imgs, dtype=torch.float32, device=device)
    
    target_imgs = np.array([load_image(path) for path in target_paths])
    target_imgs = torch.tensor(target_imgs, dtype=torch.float32, device=device)
    
    return (seed_imgs, target_imgs)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Ensure pseudorandomness for testing and validation purposes
SEED = 0
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)

BATCH_SIZE = 4
GRID_SIZE = 64
INPUT_CHANNELS = 16

seeds, targets = prepare_data(DEVICE)

model = nca.NCA2d(INPUT_CHANNELS, 4, 32, DEVICE)

routine = model.train(INPUT_CHANNELS, GRID_SIZE, 100, DEVICE)
routine(BATCH_SIZE, 4000, seeds, targets, 64)

path = f'models/NCA_seed_{SEED}.pth'
torch.save(model.state_dict(), path)