import numpy as np 
import cv2

def arc_length(line):
    dist = lambda p1, p2: ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**.5
    traveled = 0 
    for i in range(len(line) - 1):
        traveled += dist(line[i], line[i + 1])
    return traveled

def main(scene_img = "final-frame.png", 
    particle_spacing = 50, max_iterations = 2500, max_path_diameter= 15, show = False):

    motion_frame = np.load("motion_frame.npy")
    # Computes the magnitude and angle of the 2D vectors
    magnitude, angle = cv2.cartToPolar(motion_frame[..., 0], motion_frame[..., 1])
    magnitude = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)
    dx, dy = cv2.polarToCart(magnitude, angle)

    # Init Particles
    trails = []
    for x in range(0,dx.shape[1], particle_spacing):
        for y in range(0,dy.shape[0], particle_spacing):
            trails.append([(x,y)])

    # Calculate Paths
    trail_copy = trails
    for i, trail in enumerate(trail_copy):
        x, y = trail[-1]
        for _ in range(max_iterations):
            x, y = x + dx[int(y)][int(x)], y + dy[int(y)][int(x)]
            done = False
            if (0 > x) or (x >= dx.shape[1]) or (0 > y) or (y >= dy.shape[0]): # Path has exited the Image
                done = True
            if False:
                done = True
            if done:
                print("Trail Complete")
                break
            trails[i].append((x,y))

    lengths = np.array([arc_length(trail) for trail in trails])
    sizes = lengths * (1/lengths.max())

    sorted_trails = sorted(zip(trails, sizes), key= lambda x: x[1], reverse=False)
    final = []

    frame = cv2.imread(scene_img)
    image = frame.copy()

    # TODO: Make Not terrible
    for trail, size in sorted_trails:
        trail = np.array(trail, dtype=np.int32)
        cv2.circle(image, tuple(trail[0]), 3, (0,255,0), -1)
        cv2.circle(image, tuple(trail[-1]), 3, (0,0,255), -1)
        try:
            if size > .5:
                color = (0, int(size*255), 255)
                final.append(trail)
            else:
                color = (255, int(size*255), 0)

            trail = trail.reshape((-1, 1, 2))
            cv2.polylines(image, [trail], 
                                False, color, int(size * max_path_diameter) if int(size * max_path_diameter) != 0 else 1)
        except Exception as e:
            print(e)

    good_only = frame.copy()
    mask = np.zeros_like(frame)
    for trail in final:
        good_only = cv2.polylines(good_only, [trail], 
                                False, color, int(size * 10) if int(size * 10) != 0 else 1 )
        cv2.circle(mask, tuple(trail[0].astype(int)), 1+(particle_spacing//2), (0,255,0), -1)
        cv2.circle(mask, tuple(trail[-1].astype(int)), 1+(particle_spacing//2), (0,0,255), -1)

    show_image = image.copy()
    b, g, r = cv2.split(mask)

    ret, start_thresh = cv2.threshold(g, 40, 255, 0)
    ret, stop_thresh = cv2.threshold(r, 40, 255, 0)

    contours, hierarchy = cv2.findContours(start_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) != 0:
        # find the biggest countour (c) by the area
        c = max(contours, key = cv2.contourArea)

        x,y,w,h = start = cv2.boundingRect(c)
        # draw the biggest contour (c) in green
        cv2.rectangle(show_image,(x,y),(x+w,y+h),(0,255,0),5)

    contours, hierarchy = cv2.findContours(stop_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) != 0:
        # find the biggest countour (c) by the area
        c = max(contours, key = cv2.contourArea)
        
        x,y,w,h = stop = cv2.boundingRect(c)
        # draw the biggest contour (c) in green
        cv2.rectangle(show_image,(x,y),(x+w,y+h),(0,0,255),5)
    
    trails = np.array([np.array(streamline) for streamline in trails if len(streamline) > 1], dtype=object)
    with open('paths.npy', 'wb') as f:
        np.save(f, trails)
    # with open('stop-box.npy', 'wb') as f:
    #     np.save(f, stop)
    # with open('start-box.npy', 'wb') as f:
    #     np.save(f, start)
        
    # cv2.imwrite('start-stop-mask.png', mask)
    cv2.imwrite('good-only.png', good_only)
    cv2.imwrite('start-stop-final.png', np.hstack((good_only, mask, show_image)))
    cv2.imwrite('path-traveled.png', image)
                

if __name__ == "__main__":
    main()
