import cv2 
import numpy as np
from datetime import datetime
import argparse
import os
from tqdm import tqdm
from utils import BOXLoader

def dominant_flow_2_img(flow):
    # Creates an image filled with zero
    # intensities with the same dimensions 
    # as the frame
    print(flow.shape)
    mask = np.zeros((*flow.shape,3))

    # Sets image saturation to maximum
    mask[..., 1] = 255

    # Computes the magnitude and angle of the 2D vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Suppress high frequency noise by the edge of the frame
    border = 10
    magnitude[:border] = 0 
    magnitude[magnitude.shape[0] - border:] = 0 
    magnitude[...,:border] = 0 
    magnitude[...,magnitude.shape[1]-border:] = 0
    
    # Sets image hue according to the optical flow 
    # direction
    mask[..., 0] = angle * 180 / np.pi / 2
    
    norm_mag = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    # (T, mag) = cv2.threshold(mag, 50, 255, cv2.THRESH_BINARY)
    mask[..., 2] = norm_mag

    # Converts HSV to RGB (BGR) color representation
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
    return rgb 
    

def main(src_folder = "stream" , outfile = "motion_frame.npy", show = False, time_template = '%Y_%m_%d-%I:%M:%S_%p'):
    motion_frame = np.zeros((*frame_shape, 2)) # np.load("motion_frame.npy")


    try:
        for i, name in enumerate(os.listdir(src_folder)):
            vid = cv2.VideoCapture(os.path.join(src_folder, name))

            # start, stop = name[:-4].split("_to_")
            # start, stop = datetime.strptime(start, time_template), datetime.strptime(stop, time_template)
        
            # FPS = vid.get(cv2.CAP_PROP_FPS)
            # DT = (stop - start).total_seconds()
            FRAME_COUNT = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
            # TRUE_FPS = FRAME_COUNT/DT

            print(f"VIDEO: {name}") # \n\tFPS: {TRUE_FPS}\n\tLENGTH: {(FRAME_COUNT * TRUE_FPS) // 60} Minute(s)")

            ret, frame = vid.read()
            last_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            for j in tqdm(range(FRAME_COUNT)): 
                ret, frame = vid.read()
                if not ret:
                    break

                # Converts each frame to grayscale - we previously 
                # only converted the first frame to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Calculates dense optical flow by Farneback method
                flow = cv2.calcOpticalFlowFarneback(last_gray, gray, 
                                                None,
                                                0.5, 3, 15, 3, 5, 1.2, 0)
                last_gray = gray
                motion_frame += flow
                if show:
                    added_image = cv2.addWeighted(frame, 0.5, dominant_flow_2_img(motion_frame), 0.5, 0)
                    cv2.imshow("Long-Term Flow Frame + Current Image", added_image)
                    cv2.waitKey(1)
    except Exception as e:
        raise(e)
    finally:
        rgb = cv2.addWeighted(frame, 0.5, dominant_flow_2_img(motion_frame), 0.5, 0)
        cv2.imwrite('final-frame.png', frame)
        cv2.imwrite('final-combined.png', cv2.addWeighted(frame, 0.5, rgb, 0.5, 0))
        cv2.imwrite('final-flow.png', rgb)

        # reshaping the array from 3D
        # matrice to 2D matrice.
        with open(outfile, 'wb') as f:
            np.save(f, motion_frame)

#2021_09_18-03/50/21_PM_to_2021_09_18-04/05/21_PM.mp4
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('src_folder', type=str,
                        help='an integer for the accumulator')

    parser.add_argument('--use_cam', type=bool, default=False,
                        help='Show Updates while passing through Frames')

    parser.add_argument('--show', type=bool, default=True,
                        help='Show Updates while passing through Frames')

    args = parser.parse_args()
    main(args.src_folder, show = args.show)