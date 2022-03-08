import argparse
import cv2
import argparse
from datetime import datetime
import os

def capture(url, folder, minutes, template = '%Y_%m_%d-%I:%M:%S_%p'):
    stream = cv2.VideoCapture(f'{url}?action=stream')
    fps = stream.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # note the lower case
    video_writer = cv2.VideoWriter()
    opened = False
    try:
        start_time = datetime.now()
        now = datetime.now()
        filename = f"{folder}/{start_time.strftime(template)}.temp.mp4"
        while (now - start_time).total_seconds() <= (minutes * 60):
            ret, frame = stream.read()
            if ret:
                cv2.imshow('Video Stream Monitor', frame)
                cv2.waitKey(1)
                if not opened:
                    frame_shape = frame.shape
                    video_writer.open(filename, fourcc, fps, frame_shape[:-1][::-1]) # FPS is unkown and changes for stream 
                    opened = True
                else:
                    video_writer.write(frame)
            else:
                print("Failed to read the stream")
            now = datetime.now()
        
    finally:
        video_writer.release()
        stream.release()
        os.rename(filename, f"{folder}/{start_time.strftime(template)}_to_{now.strftime(template)}.mp4")
        print(f"{(now - start_time).total_seconds()//60} Minute(s) Capture Complete")

# Open a URL stream
test_url = "http://195.211.216.174/mjpg/video.mjpg"
starbucks_url = "http://131.95.3.162/mjpg/video.mjpg"
 
parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str,default="stream", help="")
parser.add_argument("--increment", type=int, default=15, help="")
parser.add_argument("--url", type=str, default=starbucks_url, help="")
parser.add_argument("--correct_fps", type=bool, default=False, help="")

args = parser.parse_args()
try:
    while True:
        capture(args.url, args.folder, args.increment)
finally:
    if args.correct_fps:
        pass
    pass