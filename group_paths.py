
from clustering_algorithm import SegmentingAveragePointwiseEuclideanMetric
import cv2
import numpy as np
from demos.flag_paths import BOXLoader, draw_info

class Group(object):
    def __init__(self) -> None:
        pass
    def update(self):
        pass
class TrajectoryGroup():
    pass
class Trajectory():
    def __init__(self, id_) -> None:
        self.id = id_
        self.age = 0 
        self.points = []
    def append(self, point):
        self.points.append(point)
        pass
    
def main():
    groups = dict()
    unmatched = dict()


    video = cv2.VideoCapture(0)
    FPS = 0
    
    cam_id = 62
    video_file = f"/Users/walkenz1/Datasets/CAP_ONE/mta_test/cam_{cam_id}/cam_{cam_id}.mp4"
    video = cv2.VideoCapture(video_file)
    detector = BOXLoader(f"/Users/walkenz1/Datasets/CAP_ONE/mta_test/cam_{cam_id}/coords_fib_cam_{cam_id}.csv")
   
    metric = SegmentingAveragePointwiseEuclideanMetric()
    
    thresh = 200
    solitary_decay_rate = 5
    group_decay_rate = 2
    assignment_delay = 400

    while True:
        ret, frame = video.read()
        if not ret:
            break
        path_mask = None 
        # Log Detectionså
        for id_,x1,y1,x2,y2 in detector.update(frame):
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),4)
            assigned = False
            for group in groups:
                if assigned: break 
                if id_ in group: 
                    group[id_]
                    assigned = True
            if not assigned:
                if id_ in unmatched.keys():
                    unmatched[id_].append([int((x1+x2)/2),int((y1+y2)/2)])
                    unmatched[id_].age -= solitary_decay_rate
                    unmatched[id_].age = unmatched[id_].age if unmatched[id_].age > 0 else 0
                    pass
                else:
                    unmatched[id_] = Trajectory(id_)
                    unmatched[id_].append([int((x1+x2)/2),int((y1+y2)/2)])
        
        for num in list(unmatched.keys()):
            unmatched[num].age += 1
            traj = unmatched[num]
            if traj.age < assignment_delay:
                path = traj.points
                if len(path) > 1:
                    cv2.polylines(frame, [np.array(traj.points).astype(np.int32)], False, (255,0,0), int(50 * (1 -(traj.age/assignment_delay))))
            else:
                assigned = False
                for cluster in groups:
                    if assigned: break 
                    for path in cluster:
                        if assigned: break 
                        dist = metric.dist(traj.points, path)
                        if dist <= thresh:
                            print("fucl")
                            assigned = True
                        pass
                if not assigned:
                    # del unmatched[num]
                    pass #flag anomoly
            for 
            # traj.update()
            continue
            pass
        cv2.imshow("Camera", frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()