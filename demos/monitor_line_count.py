import cv2 
import numpy as np
from datetime import datetime
import argparse
import os
from tqdm import tqdm
from utils import BOXLoader

def box_center(box):
    pass

def boxes_overlap(box_a, box_b):
    pass

def main(src_folder, target_file, show = False):
    motion_frame = np.zeros((*frame_shape, 2)) # 
    
    start_box = np.load("start-box.npy")
    start_box_lifetimes = dict()

    stop_box = np.load("stop-box.npy")
    stop_box_lifetimes = dict()
    
    vid = cv2.VideoCapture(src_folder)
    loader = BOXLoader()

    ret = True
    while ret:
        ret, frame = vid.read()
        if not ret:
            break
                
        detections = loader.update_fib(frame)

        last_gray = gray
        motion_frame += flow
        if show:
            added_image = cv2.addWeighted(frame, 0.5, dominant_flow_2_img(motion_frame), 0.5, 0)
            cv2.imshow("Long-Term Flow Frame + Current Image", added_image)
            cv2.waitKey(1)
# except Exception as e:
#     raise(e)
# finally:
#     rgb = cv2.addWeighted(frame, 0.5, dominant_flow_2_img(motion_frame), 0.5, 0)
#     cv2.imwrite('final-frame.png', frame)
#     cv2.imwrite('final-combined.png', cv2.addWeighted(frame, 0.5, rgb, 0.5, 0))
#     cv2.imwrite('final-flow.png', rgb)

#             # reshaping the array from 3D
#             # matrice to 2D matrice.
#             with open(outfile, 'wb') as f:
#                 np.save(f, motion_frame)

#2021_09_18-03/50/21_PM_to_2021_09_18-04/05/21_PM.mp4
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('src_folder', type=str, default = "/Users/walkenz1/Datasets/Starbucks",
                        help='')

    parser.add_argument('target_file', type=str, default = "2021_09_18-03/50/21_PM_to_2021_09_18-04/05/21_PM",
                        help='')
                        
    parser.add_argument('--use_cam', type=bool, default=False,
                        help='Show Updates while passing through Frames')

    args = parser.parse_args()
    main(args.src_folder,args.target_file, show = args.show)