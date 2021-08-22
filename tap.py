
import numpy as np
import cv2

from utils import DeepSortMock, drawDetection

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

# Works well with images of different dimensions
def orb_sim(img1, img2):
  # SIFT is no longer available in cv2 so using ORB
  orb = cv2.ORB_create()

  # detect keypoints and descriptors
  kp_a, desc_a = orb.detectAndCompute(img1, None)
  kp_b, desc_b = orb.detectAndCompute(img2, None)

  # define the bruteforce matcher object
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
  # perform matches. 
  matches = bf.match(desc_a, desc_b)
  # Look for similar regions with distance < 50. Goes from 0 to 100 so pick a number between.
  similar_regions = [j for j in matches if j.distance < 50]  
  if len(matches) == 0:
    return 0
  return len(similar_regions) / len(matches)

def shirt_sim(img1, img2):
    pass
img_grab_gap = 24 * .25 # Frames
max_id_age = 30 * 24 # Frames 

data_files = ["grandma_A", "grandma_B"]
feeds = [cv2.VideoCapture(f"raw_data/videos/{file}.mp4") for file in data_files]
feed_memories = [dict() for _ in feeds] # TODO: Store in a queue instead
detectors = [DeepSortMock(f"raw_data/tracks/{file}.pb") for file in data_files]

# DEBUG 
output_frames = []
fps = feeds[0].get(cv2.CAP_PROP_FPS)
out_size = (640 * 2, 480 * 2)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # note the lower case
out = cv2.VideoWriter()
out.open('output.mov',fourcc,fps,out_size,True) 


frames_left = True
frame_num = 0 
try:
    while frames_left:
        frame_num += 1
        output_frames = [ ]
        for feed_num, (feed, detector, identities) in enumerate(zip(feeds, detectors, feed_memories)):
            updated_ids = set()
            frames_left, frame = feed.read()

            if not frames_left: break

            # Query Deep Sort Algorithm
            people = detector.update(frame)

            # Grab Positive Person images to train seperators  
            for person in people:
                p1, p2, cat, ID = person
                updated_ids.add(ID)
                if ID not in identities:
                    identities[ID] = Identity(ID)
                    if ID == 1 and feed_num == 1:
                        identities[ID].name = "Grandma"
                    if ID == 1 and feed_num == 1:
                        identities[ID].name = "Grandma"
                    if ID == 6 and feed_num == 0:
                        identities[ID].name = "Me"

                frame = drawDetection(frame, (p1, p2, cat, ID), info = identities[ID].name)
                
                # Retrain the ORB Based on recent image captures
                if frame_num % img_grab_gap == 0:
                    crop = frame[p1[1]:p2[1], abs(p1[0]):p2[0]]
                    dim = 128
                    crop = cv2.resize(crop, (dim, dim), interpolation = cv2.INTER_AREA)

                    identities[ID].push_image(crop) 
 
            # TODO: Age Identity Objects 
            for identities in feed_memories:
                for identity in [i for i in identities.keys()]:# Deep Copy
                    if identity in updated_ids:
                        identities[identity].age = 0
                    else: 
                        if identities[identity].age >= max_id_age:
                            print(f"Removing ID {identity} from Feed {feed_num}")
                            del identities[identity]
                        else:
                            identities[identity].age += 1        
            output_frames.append(frame)   
        ##################################################
        # TODO: 
        ids = []
        for i, identities in enumerate(feed_memories):
                for identity in identities:
                    ids.append((i,identity))
        # Match Young Local Identities with Old Global Identities
        scores = [] 
        seen = set()
        for feedA, IdA in ids:
            seen.add((feedA, IdA))
            img_A = feed_memories[feedA][IdA].get_latest_image()
            if type(img_A) == type(None): continue
            best_comp = (feedA, IdA, 0) # Feed, ID, Score
            for feedB, IdB in ids:
                score = 0
                if (feedB == feedA) and (IdA == IdB): continue
                
                for img_B in feed_memories[feedB][IdB].get_images():
                    score += orb_sim(img_A, img_B)
                if score > 0:
                    score /= len(feed_memories[feedB][IdB].get_images())
                if score > best_comp[2]:
                    best_comp = (feedB, IdB, score)

            # Swap/Join Names
            if best_comp[2] >= .25:
                orig_name, new_name = feed_memories[feedA][IdA].name, feed_memories[best_comp[0]][best_comp[1]].name
                if not new_name.strip().isnumeric(): # is real Name
                    feed_memories[feedA][IdA].name = f"{IdA} or {best_comp[1]}"
                else:
                    feed_memories[feedA][IdA].name = new_name 
            # gma, me    
            pairings = [[{7, 10, 11}, {1, 7, 10, 11}], 
                            [{6, 9}, {7, 9}]]

            skip = False
            feed, ID, score = best_comp
            if best_comp[2] < .1:
                skip = True 
            if ((IdA in set({7, 10, 11})) or (IdA in set({1, 7, 10, 11}))) and ((ID in set({7, 10, 11})) or (ID in set({1, 7, 10, 11}))):
                skip = True
            if (((IdA in set({6, 9})) or (IdA in set({7, 9})))) and ((ID in set({6, 9})) or (ID in set({7, 9}))):
                skip = True
            elif IdA == ID:
                skip = True
            elif IdA not in set({1, 7, 9, 10, 11}):
                skip = True
            if not skip: 
                print(f"Feed: {feedA}, Main: {IdA}, Best: {best_comp[1]} Score: {score}")
        # Debug
        joined = np.concatenate(tuple(output_frame for output_frame in output_frames), axis = 1)
        joined = cv2.resize(joined, out_size, interpolation = cv2.INTER_AREA)

        out.write(joined)
        cv2.imshow("Joined Product",joined)
        cv2.waitKey(1)
                
finally:
    [cap.release() for cap in feeds]
    out.release()
    cv2.destroyAllWindows()


