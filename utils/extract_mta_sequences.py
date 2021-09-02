
from extract_mta_flow import extract_flow
import pandas as pd
import numpy as np
import cv2
import gflags
import os
import torchvision
import sys

def grab_sequences(to_grab, camera_id, args):

    video_cap = cv2.VideoCapture(os.path.join(args.cameras_path, f"cam_{camera_id}/cam_{camera_id}.mp4"))
    flow_cap = cv2.VideoCapture(os.path.join(args.cameras_path, f"cam_{camera_id}/cam_{camera_id}.flow.mp4"))

    # Grab resize and modify all available sequences
    grabbed = dict() # {person_id: {seq_id: ([frames, ... ], info)}}
    for frame_num in to_grab:
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        _, frame = video_cap.read()

        if False:
            flow_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, flow = flow_cap.read()
            if not ret: print(f"Failed {frame_num} to Load Flow{os.path.join(args.cameras_path, f'cam_{camera_id}/cam_{camera_id}.flow.mp4')}"); exit()
        else:
            from extract_mta_flow import extract_flow
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
            _, last_frame = video_cap.read()
            flow = extract_flow(frame,last_frame)
        
        for person in to_grab[frame_num]:
            for (seq_id, person_id, p1, p2) in to_grab[frame_num][person]: # Should only be one but thats not a guarentee
                if person_id not in grabbed:
                    grabbed[person_id] = dict()
                if seq_id not in grabbed[person_id]:
                    grabbed[person_id][seq_id] = [[],[]]
                
                # Crop
                local_frame = frame[p1[1]:p2[1], p1[0]:p2[0]]
                local_flow = flow[p1[1]:p2[1], p1[0]:p2[0]]

                # Resize 
                local_frame = cv2.resize(local_frame, (128, 128), interpolation = cv2.INTER_AREA) 
                local_flow = cv2.resize(local_flow, (128, 128), interpolation = cv2.INTER_AREA) 

              
                # Normalize 
                img_transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[.5,.5,.5],
                    std=[.5, .5,.5],
                )])
                
                flow_transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0, 0, 0],
                    std =[1, 1, 1],
                )])

                local_frame = img_transform(local_frame).detach().cpu().numpy() 
                local_flow = flow_transform(local_flow).detach().cpu().numpy() 

                # Join
                joined = np.concatenate([local_frame, local_flow], axis = 2)
                
                # Save
                info = [person_id, frame_num, 0, 0, 0, 0]
                grabbed[person_id][seq_id][0].append(joined)
                grabbed[person_id][seq_id][1].append(info)
    imgs = [] 
    info = []
    for person_id in grabbed:
        for seq_id in grabbed[person_id]:
            imgs.append(grabbed[person_id][seq_id][0])
            info.append(grabbed[person_id][seq_id][1])
    np.savez_compressed(os.path.join("raw_data/datasets/sequence/grandma_me/test",f"cam_{camera_id}.npz"), np.asarray(imgs),  np.asarray(info))

def cap_transitions():
    pass
def cap_first_final(cameras_path, camera_id, length = 8, args = None):
        coords_path = os.path.join(cameras_path, f"cam_{camera_id}/coords_fib_cam_{camera_id}.csv")
    
        data = pd.read_csv(coords_path)
        out = None

        old_frame_num = 0 
        ids = dict()
        for index, row in data.iterrows():
            frame_num, ID = row['frame_no_cam'], row['person_id']
            if ID not in ids:
                ids[ID] = frame_num 
                out.write()
            # Expel old tracks
            if frame_num != old_frame_num:
                for ID in ids:
                    if ((ids[ID] - frame_num) // FPS) >= N:
                        out.write()
                        del ids[ID]
                old_frame_num = frame_num
def cap_strides(cameras_path, camera_id, stride = 2, length = 8, args = None):
    FPS = 24 
    samples = args.samples

    coords_path = os.path.join(cameras_path, f"cam_{camera_id}/coords_fib_cam_{camera_id}.csv")

    data = pd.read_csv(coords_path)
    dets = dict()
    # Group all occurances by person 
    for index, row in data.iterrows(): # ? Is this iterrows function slowing things down 
        
        p1 = (row["x_top_left_BB"], row["y_top_left_BB"])
        p1 = tuple([0 if x < 0 else x for x in p1])
        p2 = (row["x_bottom_right_BB"], row["y_bottom_right_BB"])
        if (p2[0] - p1[0]) * (p2[1] - p1[1]) > (64 ** 2):
            if row["person_id"] not in dets:
                dets[row["person_id"]] = dict() 
            dets[row["person_id"]][int(row["frame_no_cam"])] = (p1, p2)

    # Determine available sequences matching the criteria
    to_grab = dict() # {frame_num: [(seq_id, person_id, p1, p2), ...]}
    for person in dets: 
        first = min([i for i in dets[person].keys() if i > 0]) # First available sighting
        last = max([i for i in dets[person].keys() if i > 0]) # Last available sighting
        curr = first 
        seq_id = 0
        while (curr < last):
            seq = []
            for next_frame_num in np.linspace(curr, curr + (FPS * length) + 1, samples, dtype = np.int):
                if next_frame_num not in dets[person]:
                    # TODO: Look for a close enough detection 
                    break
                else:
                    seq.append((next_frame_num, seq_id, *dets[person][next_frame_num]))
                    
            if len(seq) == samples:
                for next_frame_num, seq_id, p1, p2 in seq:
                    if next_frame_num not in to_grab:
                        to_grab[next_frame_num] = dict()
                    if person not in to_grab[next_frame_num]:
                        to_grab[next_frame_num][person] = []
                    to_grab[next_frame_num][person].append((seq_id, person, p1, p2))
                    
            curr += (FPS * stride)
            seq_id += 1
    # Save to File
    grab_sequences(to_grab, camera_id, args)

if __name__ == "__main__":
    ["frame_no_cam",
    "person_id",
    "x_top_left_BB",
    "y_top_left_BB",
    "x_bottom_right_BB",
    "y_bottom_right_BB"]

    Flags = gflags.FLAGS
    gflags.DEFINE_string("cameras_path", "raw_data/videos/grandma_me/test", "training folder")
    gflags.DEFINE_string("camera_ids", "0,1", "camera ids used to train ex. 0,1,2,3")
    gflags.DEFINE_bool("multithreaded", False, "use multithreading")
    gflags.DEFINE_bool("calc_flow", False, "recalculate optical flow values")
    gflags.DEFINE_integer("samples", 4, "number of dataLoader workers")
  
    gflags.DEFINE_string("times", 400, "number of samples to test accuracy")
    gflags.DEFINE_integer("workers", 4, "number of dataLoader workers")
    gflags.DEFINE_float("lr", 0.00006, "learning rate")

    Flags(sys.argv)
    Flags.camera_ids = list([int(i) for i in Flags.camera_ids.split(',')])

    for camera_id in Flags.camera_ids:
        cap_strides(Flags.cameras_path, camera_id, args = Flags)
