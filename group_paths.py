
from clustering_algorithm import SegmentingAveragePointwiseEuclideanMetric
import cv2
import numpy as np
import copy
from demos.flag_paths import BOXLoader, draw_info
from dipy.viz.app import distinguishable_colormap

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
    clusters = dict() # TODO: Replace with clustermap
    unmatched = dict()


    #video = cv2.VideoCapture(0)
    FPS = 0
    
    cam_id = 62
    video_file = f"/Users/walkenz1/Datasets/CAP_ONE/mta_test/cam_{cam_id}/cam_{cam_id}.mp4"
    video = cv2.VideoCapture(video_file)
    detector = BOXLoader(f"/Users/walkenz1/Datasets/CAP_ONE/mta_test/cam_{cam_id}/coords_fib_cam_{cam_id}.csv")
   
    metric = SegmentingAveragePointwiseEuclideanMetric()
    
    thresh = 200
    solitary_decay_rate = 2
    solitary_recovery_rate = 6
    group_decay_rate = 2
    assignment_delay = 400

    max_thickness = 25
    max_nb = 100
    cmap = distinguishable_colormap(nb_colors = max_nb)

    while True:
        ret, frame = video.read()
        if not ret:
            break

        path_mask = None # Empty Frame for visualization

        # Update Trajectory Detections
        for id_,x1,y1,x2,y2 in detector.update(frame):
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),4)
            assigned = False
            center = ([int((x1+x2)/2),int((y1+y2)/2)])
            for g_id, group in clusters.items():
                if assigned: break 
                if id_ in group: 
                    group[id_].append(center)
                    group[id_].age -= solitary_recovery_rate
                    assigned = True

            if not assigned:
                if id_ in unmatched.keys():
                    unmatched[id_].append(center)
                    unmatched[id_].age -= solitary_recovery_rate
                    unmatched[id_].age = unmatched[id_].age if unmatched[id_].age > 0 else 0
                    pass
                else:
                    unmatched[id_] = Trajectory(id_)
                    unmatched[id_].append(center)

        # Assign to Clusters 
        matched = set()
        for id_, traj in list(unmatched.items()):
            traj.age += solitary_decay_rate
            path = np.array(traj.points).astype(np.int32)

            if traj.age <= assignment_delay:
                if len(path) > 1:
                    cv2.polylines(frame, [path], False, (0,0,255), int(max_thickness))
                    cv2.polylines(frame, [path], False, (0,255,0), int(max_thickness * (1 -(traj.age/assignment_delay))))
            else:
                assigned = False
                group_copy = copy.deepcopy(clusters)

                for key, cluster in group_copy.items():
                    if id_ in matched: break 
                    for ckey, clus_traj in cluster.items():
                        if id_ in matched: break 
                        clus_path = np.array(clus_traj.points).astype(np.int32)
                        dist = metric.dist(path, clus_path)[2]
                        if dist <= thresh:
                            clusters[key][id_] = traj
                            matched.add(id_)

                if not assigned:
                    clusters[len(clusters)] = {id_: traj}
                    matched.add(id_)

        for id_ in matched:
            del unmatched[id_]
        

        # Create Cluster
        seen = set()
        new_assignments = dict()
        new_clusters = [] 

        for t_id, traj in unmatched.items():
            seen.add(t_id)
            if t_id in seen: continue
            for t_id, traj in unmatched.items():
                if t_id in seen: continue
            s = 0 
            if False:
                pass
            print(traj)
            pass
        # cv2.addWeighted(frame, 0.5, rgb, 0.5, 0)
        cv2.imshow("Camera", frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()