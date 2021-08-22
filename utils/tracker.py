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
    print(a.detections)
