
import cv2
import numpy as np
import json

import matplotlib.pyplot as plt

cmap = plt.get_cmap('tab20b')
colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

def random_bbox(H, W, max_H, max_W, min_width = 1, min_height = 1):
    x1 = np.random.randint(0,W)
    y1 = np.random.randint(0,H)
    x2 = np.random.randint(x1 + min_width, x1+max_W) + 1
    y2 = np.random.randint(y1 + min_height, y1+max_H)
    return (x1, y1), (x2, y2)
                
def drawDetection(frame, detection, info = ""):
    p1, p2, class_name, ID = detection
    
    color = colors[int(ID) % len(colors)]
    color = [i * 255 for i in color]

    cv2.rectangle(frame, p1, p2, color, 5)
    cv2.rectangle(frame, (int(p1[0]), int(p1[1]-30)), (int(p1[0])+(len(class_name)+len(str(ID))+len(str(info)) + 1)*17, int(p1[1])), color, -1)
    cv2.putText(frame, f"{class_name}-{ID}: {info}",(int(p1[0]), int(p1[1]-10)),0, 0.75, (255,255,255),2)
    return frame
    
def drawIdentitiesInfo():
    pass

def traclus_2_streamlines(fname):
    with open(fname) as f:
        streamlines = []
        data = json.load(f)

        for path in data['trajectories']:
            streamlines.append([])
            for coords in path:
                if (np.random.randint(0,100) < 1) and len(streamlines[-1]) > 500:
                    streamlines.append([]) 
                streamlines[-1].append([coords["x"], coords["y"], 0])
 
        for i, streamline in enumerate(streamlines):
            streamlines[i] = np.array(streamlines[i])

    streamlines = np.array(streamlines, dtype=object)

    return streamlines


cache = {}
def id_to_random_color(number):
    if not number in cache:
        r, g, b = np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255)
        cache[number]= (r, g, b)
        return r, g, b
    else:
        return cache[number]
