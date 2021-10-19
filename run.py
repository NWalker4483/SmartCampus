import argparse
import cv2
from utils import DeepSortMock
import torch
from model import SiameseEmbeddingModel
# connect to video 
video_cap = cv2.VideoCapture(0)

max_dt = 5 * 24
tracklets = dict()

def MultiCameraTimeConstraint():
    pass

def SingleCameraTimeConstraint():
    pass

def generate_tracklets():
    tracklets = dict()
    return tracklets

def match_tracklets(src_idx, tracklets):
    pass

grouped_tracklets = []
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