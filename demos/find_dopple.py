from tokenize import group
from torchreid.utils import FeatureExtractor
# from sklearn import pairwise_distance
from scipy.spatial import distance_matrix
import statistics as stats
from traceback import print_tb
import cv2
import numpy as np 
import csv 
from tqdm import tqdm
import math
class BOXLoader():
    def __init__(self, fname):
        self.detections = dict()
        with open(fname, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            next(reader) # Skip the header
            for row in tqdm(reader):
                f_no, _id, x1, y1, x2, y2 = [int(i) for i in row[0].split(",")]
                if f_no in self.detections:
                    self.detections[f_no].append((_id,x1,y1,x2,y2))
                else:
                    self.detections[f_no] = [(_id,x1,y1,x2,y2)]
        self.frame_num = 0 

    def update(self, frame = None):
        results = self.detections.get(self.frame_num, [])
        self.frame_num += 1 
        return results

class Identity:
    def __init__(self,cam_id, p_id, p1, p2):
        pass
    pass

extractor = FeatureExtractor(
            model_name='osnet_x1_0',
            model_path='/Users/nilez/.cache/torch/checkpoints/osnet_x1_0_imagenet.pth',
            device='cpu')

dataset = "/Users/walkenz1/Datasets/CAP_ONE/mta_test/"
IDs = [61,62]
assert(len(IDs) == 2)
feeds = [cv2.VideoCapture(dataset + f"cam_{ID}/cam_{ID}.mp4") for ID in IDs]
detectors = [BOXLoader(dataset + f"cam_{ID}/coords_fib_cam_{ID}.csv") for ID in IDs]
ret = True 

while ret:
    frames = []
    groups = []

    ids_ = []
    crop = None

    for i, (feed, detector) in enumerate(zip(feeds, detectors)): 
        crops = [] 
        ids_ = []
        ret, frame = feed.read()
        frames.append(frame)
        detections = detector.update(frame)

        for id_, x1, y1, x2, y2  in detections:
            crop = frame[y1:y2,x1:x2]
            ids_.append(id_)
            crops.append(crop)
            cv2.rectangle(frames[-1], (x1,y1), (x2,y2), (255,0,0), 5) 
            cv2.rectangle(frames[-1], (int(x1), int(y1-30)), (int(x1)+41, int(y1)), (255,0,0), -1)
            cv2.putText(frames[-1], f"{id_}:",(int(x1), int(y1-10)),0, 0.75, (255,255,255),2)

        if len(crops) > 0:
            groups.append(extractor(crops))
        else:
            groups.append([])

    dist_matrix = np.zeros((len(groups[0]),len(groups[1])))
    for i, feat_vec in enumerate(groups[0]):
        for j, sub_feat_vec in enumerate(groups[1]):
            dist_matrix[i][j] = math.dist(feat_vec,sub_feat_vec)

    vis = np.concatenate([*frames], axis=1)
    # if not isinstance(crop,type(None)):
    cv2.imshow(str(2),vis)
    cv2.waitKey(1)
    # extractor()