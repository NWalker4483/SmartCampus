
import cv2
from tqdm import tqdm
import os.path as osp
import argparse
import numpy as np
import os 

def extract_flow(frame, last_frame):
    # TODO: Normalize as described 
    # As a preprocessing step images were converted to the 
    # YUV colour space, before being passed to the network, and each colour 
    # channel was normalised to have zero mean and unit variance. Horizontal 
    # and vertical optical flow channels were calculated between 
    # each pair of frames using the Lucas-Kanade algorithm [29]. 
    # The optical flow channels were then normalised to fall within the range -1 to 1. 
    # When training and testing with both optical flow and colour information, the first layer of 
    # the neural network used five input channels, three for colour and two for 
    # optical flow, and when training and testing with colour information only, three input channels were used.
    hsv = np.zeros_like(last_frame)
    hsv[...,1] = 255
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    last_frame = cv2.cvtColor(last_frame,cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(last_frame, frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang * 180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    return bgr

def thing(mta_dataset_path, camera_ids):

    for cam_id in camera_ids:
        print("processing cam_{}".format(cam_id))

        cam_path = os.path.join(mta_dataset_path,"cam_{}".format(cam_id))

        video_path = osp.join(cam_path, "cam_{}.mp4".format(cam_id))
        output_path = osp.join(cam_path, "cam_{}.flow.mp4".format(cam_id))

        video_capture = cv2.VideoCapture(video_path)
        ret, last_frame = video_capture.read()
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # note the lower case

        video_writer = cv2.VideoWriter()
        video_writer.open(output_path, fourcc, fps, last_frame.shape[:-1][::-1], True) 
        video_writer.write(np.zeros_like(last_frame))
        # TODO: Get Frame Count 
        pbar = tqdm(total=int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
        
        while ret:
            ret, frame = video_capture.read()
            if not ret: break 

            flow = extract_flow(frame, last_frame)
            video_writer.write(flow)

            last_frame = frame 
            pbar.update()
        video_writer.release()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mta_dataset_folder", type=str,default="raw_data/videos/grandma_me/test")
    parser.add_argument("--camera_ids", type=str, default="0,1,2,3,4,5")

    args = parser.parse_args()
    args.camera_ids = list(map(int,args.camera_ids.split(",")))
    return args

def main():
    args = parse_args()
    thing(mta_dataset_path=args.mta_dataset_folder
                        ,camera_ids=args.camera_ids)

if __name__ == '__main__':
    main()
