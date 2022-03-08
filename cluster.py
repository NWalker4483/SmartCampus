from clustering_algorithm import SegmentingQuickBundles
from dipy.segment.clustering import QuickBundles
from utils import traclus_2_streamlines
from dipy.viz import window, actor, colormap
import numpy as np

def main(streamlines):
    qb = SegmentingQuickBundles(threshold=100)

    clusters = qb.cluster(streamlines)

    # if 0: 
    #     cluster_means = np.zeros(len(clusters))
    #     cluster_stddev = np.zeros(len(clusters))
    #     for i, cluster in enumerate(clusters): # For each cluster 
    #         distances = []
    #         for j in cluster.indices: # iterate over each contained trajectory,
    #             for (x1, y1, _) in streamlines[j]: # for each point in a trajectory,
    #                 min_dist = np.inf
    #                 for (x2, y2, _) in cluster.centroid: # determine its minimum distance from a centroid of the cluster NOTE: Resample centroids
    #                     dist = np.sqrt(((x2 - x1)**2) + ((y2 - y1) ** 2))
    #                     if dist < min_dist:
    #                         min_dist = dist
    #                 distances.append(min_dist)
    #         cluster_means[i] = np.mean(distances)
    #         cluster_stddev[i] = np.std(distances)

    #     np.savez(f'QB_cam_{ID}.npz', centroids = np.array([cluster.centroid for cluster in clusters]), means=cluster_means, stddev=cluster_stddev)

    if 1:
        scene = window.Scene()
        scene.SetBackground(1, 1, 1)
        cmap = colormap.create_colormap(np.arange(len(clusters)))

        scene.add(actor.streamtube(streamlines, window.colors.white))
        window.record(scene, out_path=f'streamlines_{ID}.png', size=(600, 600))

        scene.clear()
        # Color each streamline according to the cluster they belong to.
        scene.add(actor.streamtube(streamlines, window.colors.white, opacity=0.05))
        scene.add(actor.streamtube(clusters.centroids, cmap, linewidth=7))
        window.record(scene, out_path=f'centroids_{ID}.png', size=(600, 600))

        colormap_full = np.ones((len(streamlines), 3))
        for cluster, color in zip(clusters, cmap):
            colormap_full[cluster.indices] = color

        scene.add(actor.streamtube(streamlines, colormap_full))
        window.record(scene, out_path=f'clusters_{ID}.png', size=(600, 600))

if __name__ == "__main__":
    for ID in [62]:
        fname = f"/Users/walkenz1/Datasets/CAP_ONE/mta_test/cam_{ID}/trajectory_cam_{ID}.json"
        streamlines = traclus_2_streamlines(fname)

        # trails = np.load('paths.npy', allow_pickle=True)
        fname = f"/Users/walkenz1/Datasets/CAP_ONE/mta_test/cam_{ID}/trajectory_cam_{ID}.json"
        streamlines = traclus_2_streamlines(fname)

        # streamlines = trails # np.array([np.array([[x,y,0] for x,y in streamline]) for streamline in trails], dtype=object)

        main(streamlines)
        print(f"DONE: {ID}")
