from torch.utils.data import Dataset
import torch
import numpy as np
import os

class SequenceDataset(Dataset):
	def __init__(self, cameras_path = "raw_data/datasets/sequence/mta_ext_short/train", camera_ids = [1]):
		self.sequences = []
		self.handcrafted = [] 
		for ID in camera_ids:
			data = np.load(os.path.join(cameras_path, f"cam_{ID}.npz"))
			self.sequences += data["arr_0"]
			self.handcrafted += data["arr_1"]

		self.handcrafted = torch.from_numpy(self.handcrafted, requires_grad = True)
		self.sequences = torch.from_numpy(self.sequences, requires_grad = True)

		self.length = len(self.sequences)

	def __len__(self):
		return self.length

	def __getitem__(self, idx):
		return self.sequences[idx], self.handcrafted[idx]

class SequencePairDataset(Dataset):
	def __init__(self, cameras_path, camera_ids = None):
		pass

	def __len__(self):
		return 1

	def __getitem__(self, idx):
		return 1
		
def triplet_index(n):
	pass

class SequenceTripletDataset(Dataset):
	def __init__(self, cameras_path = "raw_data/datasets/sequence/mta_ext_short/train", camera_ids = [0,1,2], cache_path = None):
		self.dataset = SequenceDataset(cameras_path = cameras_path, camera_ids = camera_ids)
		if type(cache_path) != type(None):
			self.__idx_cache = dict()
			self.__idx_mapped = set()
			pass
		else:
			self.__idx_cache = dict()

	def is_valid(self, anchor_data, other_data, positive = True):
		(abs(anchor_data[0] - other_data[0]) <= (24 * 10))
		((anchor_data[1] == other_data[1]) == positive)
		return True

	def __len__(self):
		return self.length

	def __getitem__(self, idx):
		if idx not in self.__idx_cache:
			while True:
				negative_idx, anchor_idx, positive_idx = np.random.randint(0, self.length, 3)
				if (negative_idx, anchor_idx, positive_idx) in self.__idx_mapped: continue
				if self.is_valid(self.dataset[anchor_idx][1], self.dataset[negative_idx][1], False) and self.is_valid(self.dataset[anchor_idx][1], self.dataset[positive_idx][1]):
					self.__idx_cache[idx] = (negative_idx, anchor_idx, positive_idx)
					self.__idx_mapped.add((negative_idx, anchor_idx, positive_idx))
					break
		return self.dataset[negative_idx], self.dataset[anchor_idx], self.dataset[positive_idx] 
		
if __name__ == "__main__":
	pass
