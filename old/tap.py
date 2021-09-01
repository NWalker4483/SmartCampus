
import numpy as np
import cv2

class DeepSortMock():
    def __init__(self, filename):
        self.frame_num = 0 
        self.detections = [ ]
        import pickle
        with open(filename, 'rb') as f:
            self.detections = pickle.load(f)
    def load():
        pass
    def update(self, frame):
        self.frame_num += 1 
        if self.frame_num >= len(self.detections):
            return []
        else:
            return self.detections[self.frame_num]

import matplotlib.pyplot as plt
#initialize color map
cmap = plt.get_cmap('tab20b')
colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

                
def drawDetection(frame, detection, info = ""):
    p1, p2, class_name, ID = detection
    
    color = colors[int(ID) % len(colors)]
    color = [i * 255 for i in color]

    cv2.rectangle(frame, p1, p2, color, 5)
    cv2.rectangle(frame, (int(p1[0]), int(p1[1]-30)), (int(p1[0])+(len(class_name)+len(str(ID))+len(str(info)) + 1)*17, int(p1[1])), color, -1)
    cv2.putText(frame, f"{class_name}-{ID}: {info}",(int(p1[0]), int(p1[1]-10)),0, 0.75, (255,255,255),2)
    return frame
    

class Identity():
    def __init__(self, ID, memory_length = 45): # Frames):
        self.ID = ID
        self.age = 0
        self.name = str(ID) 
        self.memory_length = memory_length
        self.__reference_images = []
    def push_image(self,frame):
        self.__reference_images.append(frame)
        self.__reference_images = self.__reference_images[-self.memory_length:]
    def get_images(self):
        return self.__reference_images
    def get_latest_image(self):
        if len(self.__reference_images) > 0:
            return self.__reference_images[-1]
        else:
            return None

img_grab_gap = 24 * .25 # Frames
max_id_age = 30 * 24 # Frames 

data_files = ["grandma_A", "grandma_B"]
feeds = [cv2.VideoCapture(f"raw_data/videos/grandma_me/{file}.mp4") for file in data_files]
feed_memories = [dict() for _ in feeds] # TODO: Store in a queue instead
detectors = [DeepSortMock(f"raw_data/videos/grandma_me/{file}.pb") for file in data_files]

# DEBUG 
output_frames = []
fps = feeds[0].get(cv2.CAP_PROP_FPS)
_, frame = feeds[0].read()
out_size = frame.shape[:-1][::-1]
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # note the lower case
out = cv2.VideoWriter()
out.open('output.flow.mp4',fourcc,fps,out_size,True) 


frames_left = True
frame_num = 0 
try:
    while frames_left:
        frame_num += 1
        output_frames = [ ]
        for feed_num, (feed, detector, identities) in enumerate(zip(feeds, detectors, feed_memories)):
            frames_left, frame = feed.read()

            if not frames_left: break

            # Query Deep Sort Algorithm
            people = detector.update(frame)

            # Grab Positive Person images to train seperators  
            for person in people:
                p1, p2, cat, ID = person
                if ID not in identities:
                    identities[ID] = Identity(ID)
                    if ID == 1 and feed_num == 1:
                        identities[ID].name = "Grandma"
                    if ID == 1 and feed_num == 1:
                        identities[ID].name = "Grandma"
                    if ID == 6 and feed_num == 0:
                        identities[ID].name = "Me"

                frame = drawDetection(frame, (p1, p2, cat, ID), info = identities[ID].name)
            output_frames.append(frame)
        ##################################################
  
        # Debug
        joined = np.concatenate(tuple(output_frame for output_frame in output_frames), axis = 1)
        joined = cv2.resize(joined, out_size, interpolation = cv2.INTER_AREA)

        out.write(output_frames[0])
        print(output_frames[0].dtype, output_frames[0].shape,joined.dtype, joined.shape)
        cv2.imshow("Joined Product",joined)
        cv2.waitKey(1)
                
finally:
    [cap.release() for cap in feeds]
    out.release()
    cv2.destroyAllWindows()


