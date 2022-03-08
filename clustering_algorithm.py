from dipy.segment.clustering import QuickBundles, Clustering, ClusterMap, ClusterCentroid, Identity, ClusterMapCentroid, Cluster
from dipy.segment.metric import ResampleFeature, Metric
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from dipy.tracking.streamline import length
import numpy as np
from tqdm import tqdm
from dipy.viz.app import distinguishable_colormap

import cv2
import numpy as np

def draw_cluster_map(cluster_map):
    offset = 100
    cmap = distinguishable_colormap(nb_colors = len(cluster_map))
    points = np.concatenate(cluster_map.refdata)
    min_x, max_x, min_y, max_y = np.min(points[:,0]), np.max(points[:,0]), np.min(points[:,1]), np.max(points[:,1])   
    frame = np.zeros((int(max_y + 2 * offset),int(max_x + 2 * offset), 3), dtype=np.uint8)
    
    for cluster, color in zip(cluster_map, cmap):
        color = [int(i * 255) for i in color]
        cv2.polylines(frame, [(line + [offset, offset]).astype(np.int32) for line in cluster_map.refdata[cluster.indices]], False, color, 2)
    
    for cluster, color in zip(cluster_map, cmap):
        color = [int(i * 255) for i in color]
        cv2.polylines(frame, [cluster.centroid.astype(np.int32)+ [offset, offset]], False, (0,255,255), 15)
        cv2.polylines(frame, [cluster.centroid.astype(np.int32)+ [offset, offset]], False, color, 10)

    return frame

class SegmentingMetric(Metric):
    def __init__(self, *args, **kwargs):
        super(SegmentingMetric, self).__init__(*args, **kwargs)

class SegmentingCluster(ClusterCentroid):
    def __init__(self, centroid, id=0, indices=None, refdata=Identity()):
        super(ClusterCentroid, self).__init__(id, indices, refdata)
        self.centroid = centroid.copy()
        self.new_centroid = centroid.copy()

    def assign(self, id_datum, r1, r2, flipped, features):
        # Update Centroid
        pre_sample = ResampleFeature(r2[1] - r2[0])
        post_sample = ResampleFeature(r1[1] - r1[0])
        t1 = pre_sample.extract(self.centroid[r1[0]:r1[1]])
        t2 = features[r2[0]:r2[1]] if not flipped else features[r2[0]:r2[1]][::-1]
        new_features = post_sample.extract((t1 + t2) / 2)
        self.centroid[r1[0]:r1[1]] = new_features

        Cluster.assign(self, id_datum)

class SegmentingAveragePointwiseEuclideanMetric(SegmentingMetric):
    def __init__(self, nb_points = 36, sub_metric = "MDF_12points"):
        # For simplicity, features will be the vector between endpoints of a streamline.
        super(SegmentingAveragePointwiseEuclideanMetric, self).__init__(feature=ResampleFeature(nb_points = nb_points))
        if isinstance(sub_metric, Metric):
            self.metric = sub_metric
        elif sub_metric == "MDF_12points":
            feature = ResampleFeature(nb_points=12)
            self.seg_metric = AveragePointwiseEuclideanMetric(feature)
        else:
            raise ValueError("Unknown metric: {0}".format(sub_metric))
     
    def are_compatible(self, shape1, shape2):
        """ Checks if two features are vectors of same dimension.

        Basically this method exists so we don't have to do this check
        inside the `dist` method (speedup).
        """
        return (shape1[0] > 1 and shape2[0] > 1) and (shape1[1] == shape2[1])

    def dist(self, v1, v2):
        candidates = []
        for axis in [0,1]:
            range1, range2 = (max(v1[:,axis]), min(v1[:,axis])), (max(v2[:,axis]), min(v2[:,axis]))

            if False: # (range1[0] - range1[1]) < (range2[0] - range2[1]):
                v_split, max_, min_ = v2, *range1
                v1_is_base = True
            else:
                v_split, max_, min_ = v1, *range2
                v1_is_base = False

            segment_indexes = [[0]] 
            for i, point in enumerate(v_split):
                if min_ <= point[axis] <= max_: 
                    continue
                else:
                    segment_indexes[-1].append(i)
                    segment_indexes.append([i])
            segment_indexes[-1].append(len(v_split))

            for start, stop in segment_indexes:
                if stop - start <= 1:
                    continue

                if v1_is_base:
                    v1_seg, v2_seg = v1, v2[start:stop]
                else:
                    v1_seg, v2_seg = v1[start:stop], v2

                dist = self.seg_metric.dist(\
                    self.seg_metric.feature.extract(v1_seg),\
                    self.seg_metric.feature.extract(v2_seg)\
                    )

                if v1_is_base:
                    candidates.append((dist, (0,len(v1_seg)), (start, stop), v1_is_base, False))
                else:
                    candidates.append((dist, (start, stop), (0,len(v2_seg)), v1_is_base, False))

                if not self.seg_metric.feature.is_order_invariant:
                    dist = self.seg_metric.dist(\
                    self.seg_metric.feature.extract(v1_seg),\
                    self.seg_metric.feature.extract(v2_seg[::-1]))

                    if v1_is_base:
                        candidates.append((dist, (0,len(v1_seg)), (start, stop), v1_is_base, True))
                    else:
                        candidates.append((dist, (start, stop), (0,len(v2_seg)), v1_is_base, True))
                  
        if candidates:
            min_candidate = sorted(candidates, key = lambda x: x[0])[0]
            distance, r1, r2, _, flipped = min_candidate
          
            return r1, r2, distance, flipped
        else:
            return (-1,-1), (-1,-1), np.inf, False

class SegmentingQuickBundles(Clustering):
    def __init__(self, threshold, metric = "SMDF_36-12points", max_nb_clusters=np.iinfo('i4').max):
        self.threshold = threshold
        self.max_nb_clusters = max_nb_clusters
        if isinstance(metric, SegmentingMetric):
            self.metric = metric
        elif metric == "SMDF_36-12points":
            self.metric = SegmentingAveragePointwiseEuclideanMetric()
        else:
            raise ValueError("Unknown metric: {0}".format(metric))
        self.cluster_map = ClusterMapCentroid()

    def find_nearest_cluster(self, features):
        min_data = (-1,-1,-1)
        min_dist = np.inf
        min_idx = -1
        for idx, cluster in enumerate(self.cluster_map.clusters):
            r1, r2, dist, flipped = self.metric.dist(cluster.centroid, features)
            if dist < min_dist:
                min_dist = dist
                min_idx = idx
                min_data = r1, r2, flipped
        return min_idx, min_dist, *min_data

    def cluster(self, streamlines, ordering=None):
        # Get indexes sorted by length
        # for i in [1,0,1,0]:
        if ordering == None:
            ordering = sorted(range(len(streamlines)), key = lambda x: length(streamlines[x]), reverse=True)
        self.cluster_map.refdata = np.array(streamlines, dtype=object)
        for idx in ordering:
            streamline = streamlines[idx]
            if(length(streamline) < self.threshold / 2): 
                continue

            nearest_idx, dist, r1, r2, flipped = self.find_nearest_cluster(streamline)
            
            if (nearest_idx == -1) or (dist > self.threshold):
                self.cluster_map.add_cluster(SegmentingCluster(streamline,id = len(self.cluster_map), indices=[idx]))
            else:
                self.cluster_map[nearest_idx].assign(idx, r1, r2, flipped, streamline)
            cv2.imshow("Cluster Map", draw_cluster_map(self.cluster_map))
            cv2.waitKey(0)
        return self.cluster_map

   

if __name__ == "__main__":
    
    from utils import traclus_2_streamlines
    trails = np.load('paths.npy', allow_pickle=True)

    # fname = f"/Users/walkenz1/Datasets/CAP_ONE/mta_test/cam_62/trajectory_cam_62.json"
    # trails = traclus_2_streamlines(fname)
    # trails = np.array([np.array([[x,y] for x,y,_ in streamline]) for streamline in trails], dtype=object)

    # data = np.zeros((1000, 2))
    # trails = []
    # for i in range(24):
    #     data[:,0] = 0 #range(1000)
    #     data[:,0] += (30 * i) 
    #     data[:,1] = range(1000)
    #     # data[:,1] += (30 * i)
    #     trails.append(data.copy())
    # trails = np.array(trails, dtype= object)

    streamlines = trails
    test_thresh = 350
    sqb = SegmentingQuickBundles(threshold = test_thresh)
    clusters = sqb.cluster(streamlines)
    simg = draw_cluster_map(clusters)
    #cv2.imwrite(f"segmenting_{test_thresh}.jpg", simg)

    qb = QuickBundles(threshold = test_thresh)
    ordering = sorted(range(len(streamlines)), key = lambda x: length(streamlines[x]), reverse=True)

    clusters = qb.cluster(streamlines, ordering)
    nimg = draw_cluster_map(clusters)
    #cv2.imwrite(f"normal_{test_thresh}.jpg", nimg)

    vis = np.concatenate([nimg,simg], axis = 1)
    cv2.imwrite("both.jpg", vis)
    # print("Nb. clusters:", len(clusters))
    # print("Cluster sizes:", list(map(len, clusters)))
    # print("Clustered Paths:", sum(list(map(len, clusters))))
    # print("Small clusters:", clusters < 10)
    # print("Streamlines indices of the first cluster:\n", clusters[0].indices)
    # print("Centroid of the last cluster:\n", clusters[-1].centroid)
    
