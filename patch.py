from PIL import Image
import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
import cv2

def tile_img(image, rows = 5, cols = 5):
    image = Image.fromarray(image)
    imgwidth, imgheight = image.size
    height = imgheight // rows
    width = imgwidth // cols
    tiles = [ ]
    for i in range(0, cols):
        for j in range(0, rows):
            box = (j * width, i * height, (j + 1) * width, (i + 1) * height)
            a = image.crop(box)
            tiles.append(a)
    return tiles 

def group_as_images():
    pass

def convolve_neighbors():
    shape = 5
    for row in range(shape):
        for col in range(shape):


    pass

X_components = []
Y_components = []

tiles = 5
rows, cols = tiles, tiles

interval = 24

cameras = [cv2.VideoCapture(f"raw_data/videos/MTA_ext_short/train/cam_{cam_id}/cam_{cam_id}.flow.mp4") for cam_id in [0, 1]]

rows, cols = 3,3 
for row in range(rows):
    for col in range(cols):
        for (x,y) in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]:
            if (0 <= (row + x) < rows) and (0 <= (col + y) < cols):
                idx  = ((row + x) * cols) + col + y
                print(idx, end = " ")
        print("")
        

for samples in range(15):
    X_components.append([])
    Y_components.append([])

    for i, feed in enumerate(cameras):
        ret, img = [feed.read() for _ in range(interval)][-1]
        if not ret:
            break

        tiles = tile_img(img, rows, cols)

        for tile in tiles:
            # Average all X and Y components within the block respectively 
            avg_val_per_row = np.average(tile, axis=0)
            avg_val = np.average(avg_val_per_row, axis=0)
            # Attach 
            X_components[-1].append(avg_val[0])
            Y_components[-1].append(avg_val[1])
            
X = np.array(X_components).T
cov = np.corrcoef(X)

print(cov)
