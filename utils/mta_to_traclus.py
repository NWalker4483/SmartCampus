import csv
import json 
import cv2
import numpy as np

def main(in_file,out_file):
    trajectories = dict()
    img = np.zeros((2000,2000,3), np.uint8)
    with open(in_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        next(reader) # Skip the header
        x_, y_ = 0, 0 
        for row in reader:
            f_no,_id,x1,y1,x2,y2 = [int(i) for i in row[0].split(",")]
            x1,y1 = (x1 if x1 > 0 else 0), (y1 if y1 > 0 else 0)
            x_mid = x1 + ((x2-x1)/2)
            y_mid = y2 # y1 + ((y2-y1)/2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
            # cv2.circle(img, (int(x_mid), int(y_mid)), 5, (0,255,0), 2)
  
            x_, y_ = int(x_mid), int(y_mid)
            if _id in trajectories:
                x, y = trajectories[_id][-1]["x"], trajectories[_id][-1]["y"]
                if (((x - x_mid) ** 2) + ((y - y_mid) ** 2) ) **.5 > 200: # Jump Limit
                    pass
                else:
                    trajectories[_id].append({"x": x_mid, "y": y_mid})
            else:
                trajectories[_id] = [{"x": x_mid, "y": y_mid}]
    for ID in trajectories:
        x_ , y_ = 0, 0
        color = tuple(np.random.randint(0,255,3).tolist())
        for coords in trajectories[ID]:
            if x_ != 0:
                cv2.line(img, (int(coords["x"]), int(coords["y"])), (int(x_), int(y_)), color, 1)
            x_ , y_ = int(coords["x"]), int(coords["y"])

    cv2.imshow("raw paths", img)
    cv2.waitKey(0)
    if 1:
        # Add Parameters for traclus algorithm 
        out_data = dict()
        out_data["trajectories"] = list(trajectories.values())
        out_data["epsilon"] = 0.00016
        out_data["min_neighbors"] = 2 
        out_data["min_num_trajectories_in_cluster"] = 3 
        out_data["min_vertical_lines"] = 2 
        out_data["min_prev_dist"] = 0.0002 

        # Directly from dictionary
        with open(out_file, 'w') as outfile:
            json.dump(out_data, outfile)

if __name__ == "__main__":
    for ID in [61, 62]:
        fname = f"/Users/walkenz1/Datasets/CAP_ONE/mta_test/cam_{ID}"
        in_file = f"{fname}/coords_fib_cam_{ID}.csv"
        out_file = f'{fname}/trajectory_cam_{ID}.json'
        main(in_file, out_file)
        d_in, d_out = in_file.split("/")[-1], out_file.split("/")[-1] 
        print(f"DONE: {d_in} to {d_out}")
