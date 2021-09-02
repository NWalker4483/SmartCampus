from torch.utils.data import Dataset
import torch
import numpy as np
import os


class SequenceDataset(Dataset):
    def __init__(self, cameras_path="raw_data/datasets/sequence/mta_ext_short/train", camera_ids=[1], use_onehot=True, add_cam_id=False):
        self.sequences = []
        self.handcrafted = []
        self.labels = []
        for ID in camera_ids:
            data = np.load(os.path.join(
                cameras_path, f"cam_{ID}.npz"), allow_pickle=True)
            self.sequences.append(data["arr_0"])
            self.labels.append(data["arr_1"][..., :1][:, 0].reshape(-1))

            hand_features = data["arr_1"][..., 1:][..., 0]
            if use_onehot:
                one_hot = np.zeros((len(hand_features), max(camera_ids) + 1))
                one_hot[..., ID] = 1
                hand_features = np.concatenate([hand_features, one_hot], 1)
            self.handcrafted.append(hand_features)

        self.handcrafted = np.concatenate(self.handcrafted)
        self.sequences = np.concatenate(self.sequences)
        self.labels = np.concatenate(self.labels)

        self.handcrafted = torch.tensor(self.handcrafted.astype(np.float32))
        self.sequences = torch.tensor(self.sequences.astype(np.float32))
        self.labels = torch.tensor(self.labels.astype(np.float32))

        self.length = len(self.labels)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.sequences[idx], self.handcrafted[idx], self.labels[idx]


class RandomlyExploredDataset(Dataset):
    def __init__(self, idx_len, sample_size, cache_path=None, seed=42, *args, **kwargs):
        np.random.seed(seed)
        if type(cache_path) != type(None):
            self.load_cache(cache_path)
        self.idx_len = idx_len
        self.sub_length = sample_size
        self.idx_cache = dict()
        self.idx_mapped = set()
        self.idx_valid = set()
        self.idx_invalid = set()
        self.length = sample_size

    def load_cache(self, filename):
        pass

    def save_cache(self, filename):
        pass

    def __len__(self):
        return self.length

    def getindex(self, idx):
        if idx not in self.idx_cache:
            while True:
                idxs = tuple(np.random.randint(
                    0, self.sub_length, self.idx_len))
                if self.is_valid(*(idxs)):
                    self.idx_valid.add(idxs)
                    self.idx_mapped.add(idxs)
                    self.length += 1
                    self.idx_cache[idx] = idxs
                    break
                else:
                    self.idx_invalid.add(idxs)
        return self.idx_cache[idx]


class SequencePairDataset(RandomlyExploredDataset):
    def __init__(self, cameras_path="raw_data/datasets/sequence/mta_ext_short/train", camera_ids=[0, 1], max_dt=(30 * 24), *args, **kwargs):
        self.max_dt = max_dt
        self.dataset = SequenceDataset(
            cameras_path=cameras_path, camera_ids=camera_ids)
        RandomlyExploredDataset.__init__(
            self, 2, self.dataset.length, *args, **kwargs)

    def is_valid(self, main_idx, other_idx):
        if (main_idx, other_idx) in self.idx_invalid:
            return False

        main_imgs, main_data, main_label = self.dataset[main_idx]
        other_imgs, other_data, other_label = self.dataset[other_idx]

        if (abs(main_data[0] - other_data[0]) >= self.max_dt):
            self.idx_invalid.add((main_idx, other_idx))
            return False
        return True

    def __getitem__(self, idx):
        main_idx, other_idx = self.getindex(idx)
        return self.dataset[main_idx], self.dataset[other_idx]


class SequenceTripletDataset(RandomlyExploredDataset):
    def __init__(self, cameras_path="raw_data/datasets/sequence/mta_ext_short/train", camera_ids=[0, 1, 2], max_dt=(10 * 24), pos_neg_split=.5, per_anchor=10, *args, **kwargs):
        self.dataset = SequenceDataset(
            cameras_path=cameras_path, camera_ids=camera_ids)
        self.max_dt = max_dt
        RandomlyExploredDataset.__init__(
            self, 3, self.dataset.length, *args, **kwargs)

        self.per_anchor = 5
        self.build_idx()

    def build_idx(self):
        return

    def is_valid(self, negative_idx, anchor_idx, positive_idx):
        positive_imgs, positive_data, positive_label = self.dataset[positive_idx]
        anchor_imgs, anchor_data, anchor_label = self.dataset[anchor_idx]
        negative_imgs, negative_data, negative_label = self.dataset[negative_idx]

        if ((anchor_idx, positive_idx, True) in self.idx_valid) or all([
                ((anchor_idx, positive_idx, True) not in self.idx_invalid),
                (abs(anchor_data[0] - positive_data[0]) <= self.max_dt),
                (anchor_label == positive_label)]):
            self.idx_valid.add((anchor_idx, positive_idx, True))
        else:
            self.idx_invalid.add((anchor_idx, positive_idx, True))
            return False

        if ((anchor_idx, negative_idx, False) in self.idx_valid) or any([
                ((anchor_idx, negative_idx, False) not in self.idx_invalid),
                (abs(anchor_data[0] - negative_data[0]) <= self.max_dt),
                (anchor_label != negative_label)]):
            self.idx_valid.add((anchor_idx, negative_idx, False))
        else:
            self.idx_invalid.add((anchor_idx, negative_idx, False))
            return False

        return True

    def __getitem__(self, idx):
        negative_idx, anchor_idx, positive_idx = self.getindex(idx)
        return self.dataset[negative_idx][:-1], self.dataset[anchor_idx][:-1], self.dataset[positive_idx][:-1]


if __name__ == "__main__":
    a = SequenceDataset("raw_data/datasets/sequence/grandma_me/test", [0, 1])
    print(
        f"Num Sequences: {a.length}\nImg Shape: {a[0][0].shape} \nInfo Shape: {a[0][1].shape}")
    b = SequenceTripletDataset(
        "raw_data/datasets/sequence/grandma_me/test", [0, 1], max_dt=(24 * 30))

    print(f"Num Sequences: {b.length}\nImg Shape: {b[0]}")
