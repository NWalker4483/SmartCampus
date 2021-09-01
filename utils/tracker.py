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

if __name__ == "__main__":
    a = DeepSortMock("test.pb")
    headers = ["frame_no_cam",
    "person_id",
    "x_top_left_BB",
    "y_top_left_BB",
    "x_bottom_right_BB",
    "y_bottom_right_BB"]
    print(*headers, sep =", ")
    for frame_num, detections in enumerate(a.detections):
        for p1, p2, _, person_id in detections:
            print(f"{frame_num}, {person_id}, {p1[0]}, {p1[1]}, {p2[0]}, {p2[1]}")
