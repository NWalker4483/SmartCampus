import argparse
import cv2
from utils import DeepSortMock
import torch
from model import SiameseEmbeddingModel
# connect to video 
video_cap = cv2.VideoCapture(0)

# Load Model 
model = SiameseEmbeddingModel(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()

# Define knn model wrapper
class Identity():
    def __init__(self):
        pass

class KNNWrapper():
    def __init__(self, model):
        pass
    def update(self):
        pass
    def predict(self, ):
        pass
    
# Run Detections
try:
    while True:
        pass
finally:
    pass