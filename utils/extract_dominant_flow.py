import cv2 
import numpy as np
from datetime import datetime
import argparse
import os
from tqdm import tqdm

def dominant_flow_2_img(flow):
    # Creates an image filled with zero
    # intensities with the same dimensions 
    # as the frame
    h, w = flow.shape[:2]
    mask = np.zeros((h,w,3),dtype=np.uint8)

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
    
def main(src_folder, show = False):

    motion_frame = None # np.load("motion_frame.npy")

    try:
        for i, name in enumerate(os.listdir(src_folder)):
            if "mp4" not in name:
                continue
            vid = cv2.VideoCapture(os.path.join(src_folder, name))

            FRAME_COUNT = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

            print(f"VIDEO: {name}")

            ret, frame = vid.read()
            last_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for _ in range(12 * 60 * 10):
                vid.read()
            FRAME_COUNT -= (12 * 60 * 10)
            for j in tqdm(range(FRAME_COUNT)): 
                ret, new_frame = vid.read()
                if ret:
                    frame = new_frame
                else:
                    break
                
                # Converts each frame to grayscale - we previously 
                # only converted the first frame to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Calculates dense optical flow by Farneback method
                flow = cv2.calcOpticalFlowFarneback(last_gray, gray, 
                                                None,
                                                0.5, 3, 15, 3, 5, 1.2, 0)
                last_gray = gray

                if isinstance(motion_frame, type(None)):
                    motion_frame = flow
                else:
                    motion_frame += flow

                if show:
                    added_image = cv2.addWeighted(frame, 0.5, dominant_flow_2_img(motion_frame), 0.5, 0)
                    cv2.imshow("Long-Term Flow + Current Image", added_image)
                    cv2.waitKey(1)
    except Exception as e:
        raise(e)
    finally:
        with open("motion_frame.npy", 'wb') as f:
            np.save(f, motion_frame)

        rgb = cv2.addWeighted(frame, 0.5, dominant_flow_2_img(motion_frame), 0.5, 0)
      
        cv2.imwrite('final-frame.png', frame)
        cv2.imwrite('final-combined.png', cv2.addWeighted(frame, 0.5, rgb, 0.5, 0))
        cv2.imwrite('final-flow.png', rgb)

#2021_09_18-03/50/21_PM_to_2021_09_18-04/05/21_PM.mp4
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Help')

    parser.add_argument('src_folder', type=str, default="/Users/walkenz1/Datasets/CAP_ONE/test/",
                        help='an integer for the accumulator')

    parser.add_argument('--use_cam', type=bool, default=False,
                        help='Show Updates while passing through Frames')

    parser.add_argument('--show', type=bool, default=True,
                        help='Show Updates while passing through Frames')

    args = parser.parse_args()
    main(args.src_folder, show = args.show)
