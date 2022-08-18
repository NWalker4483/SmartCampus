from multiprocessing.sharedctypes import Value
import statistics as stats
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

def closest_polyline(point, polylines):
    best_path_id = 0
    min_dist = np.inf 
    for i, line in enumerate(polylines): 
        for p2 in line: 
            distance = math.dist(point, p2)
            if distance < min_dist:
                min_dist = distance
                best_path_id = i
    return best_path_id, min_dist

def draw_info(img, p1, p2, ID, info = "", color = (255,0,0)):
    cv2.rectangle(img, p1, p2, color, 5)
    cv2.rectangle(img, (int(p1[0]), int(p1[1]-30)), (int(p1[0])+(len(str(ID))+len(str(info)) + 1)*12, int(p1[1])), color, -1)
    cv2.putText(img, f"{ID}: {info}",(int(p1[0]), int(p1[1]-10)),0, 0.75, (255,255,255),2)
    return img

def main():
    global z_cutoff
    global use_globals
    global global_mean
    global global_stddev 
    cam_id = 62
    video_file = f"/Users/walkenz1/Datasets/CAP_ONE/mta_test/cam_{cam_id}/cam_{cam_id}.mp4"
    video = cv2.VideoCapture(video_file)
    detector = BOXLoader(f"/Users/walkenz1/Datasets/CAP_ONE/mta_test/cam_{cam_id}/coords_fib_cam_{cam_id}.csv")
    data = np.load(f"QB_cam_{cam_id}.npz")
    all_paths = data["centroids"][:,:,:2].astype(int)
    
    allowed_path_ids = []
    removed_path_ids = [] 

    dead_box = [(0,0),(1,1)]#[(1000, 150),(1300, 650)]

    z_cutoff = .85
    use_globals = False
    global_mean = 0
    global_stddev = 0 
    draw = True

    anomoly_scores = dict()
    last_bbox = dict()

    for id_, line in enumerate(all_paths):
        moved = False
        for x, y in line:
            if ((dead_box[0][0] < x < dead_box[1][0]) and (dead_box[0][1] < y < dead_box[1][1])):
                moved = True
                break
        if not moved:
            allowed_path_ids.append(id_)
        else:
            removed_path_ids.append(id_)

    def set_globals(arg, value):
        global z_cutoff
        global use_globals
        global global_mean
        global global_stddev 
        constrain = lambda x, min_, max_: min_ if x < min_ else (max_ if x > max_ else x) 
        if arg == "z":
            z_cutoff = constrain(value / 100, 0.01, .99)
        if arg == "use_globals":
            use_globals = value
        if arg == "global_stddev":
            global_stddev = value
        if arg == "global_mean":
            global_mean = value
        pass

    cv2.namedWindow(demo_name)
    cv2.createTrackbar("Global Mean", demo_name , 0, 100, lambda update: set_globals("global_mean", update))
    cv2.createTrackbar("Z Cutoff", demo_name , 1, 100, lambda update: set_globals("z", update))

    ret = True 
    while ret:
        ret, frame = video.read()
        detections = detector.update(frame)

        if draw:
            # Draw Polylines
            frame = cv2.polylines(frame, all_paths[allowed_path_ids], False, (0,255,0), 3)
            frame = cv2.polylines(frame, all_paths[removed_path_ids], False, (0,0,255), 2)
            # Draw Deadbox
            # cv2.rectangle(frame, dead_box[0],dead_box[1], (0,0,255), 5)
            
            for id_ in allowed_path_ids:
                #rad = stats.NormalDist(mu = data["means"][id_], sigma = data["stddev"][id_]).inv_cdf(1 - z_cutoff)   
                for x,y in all_paths[id_]: # polylines.reshape(polylines.shape[0] * polylines.shape[1],2):
                    cv2.circle(frame, (x,y), 9, (255,255,0), -1)
                    #cv2.circle(frame, (x,y), int(abs(rad)), (0,0,255), 2)
                    #cv2.circle(frame, (x,y), int(abs(data["means"][id_])), (255,0,2), 2)
            
        for id_, x1, y1, x2, y2 in detections:
            x_mid = (x1 + x2)/2
            y_mid = (y1 + y2)/2
            
            path_id, dist = closest_polyline((x_mid, y_mid), all_paths[allowed_path_ids]) 
            
            if dist > data["means"][path_id]:
                norm = stats.NormalDist(mu = data["means"][path_id], sigma = data["stddev"][path_id])
                prob = norm.cdf(dist)
                if prob > z_cutoff: 
                    if id_ in anomoly_scores:
                        anomoly_scores[id_] += 10 if anomoly_scores[id_] < 40 else 0 
                    else: 
                        anomoly_scores[id_] = 10

            if anomoly_scores.get(id_, 0) > 25:
                frame = draw_info(frame, (x1,y1), (x2,y2), id_, color = (255, 165, 0))
            else:
                frame = draw_info(frame, (x1,y1), (x2,y2), id_, color = (255, 0, 0))

        for id_ in anomoly_scores:
            anomoly_scores[id_] -= 5 if anomoly_scores[id_] >= 5 else 0 
        cv2.imshow(demo_name,  frame)
        cv2.waitKey(5)

if __name__ == "__main__":
    global demo_name
    demo_name = "Anomalous Traveller Detection Demo #2"
    main()