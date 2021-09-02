from torch.utils.data import Dataset
import torch
import numpy as np
import os

class SequenceDataset(Dataset):
	def __init__(self, cameras_path = "raw_data/datasets/sequence/mta_ext_short/train", camera_ids = [1], add_onehot = True, add_cam_id= False):
		self.sequences = []
		self.handcrafted = [] 
		self.labels = []
		for ID in camera_ids:
			data = np.load(os.path.join(cameras_path, f"cam_{ID}.npz"), allow_pickle=True)
			self.sequences.append(data["arr_0"])
			self.labels.append(data["arr_1"][...,:1][:,0].reshape(-1))

			hand_features = data["arr_1"][...,1:][...,0]
			if add_onehot:
				one_hot = np.zeros((len(hand_features), max(camera_ids) + 1))
				one_hot[...,ID] = 1 
				hand_features = np.concatenate([hand_features, one_hot], 1)
			self.handcrafted.append(hand_features)

		self.handcrafted = np.concatenate(self.handcrafted)
		self.sequences = np.concatenate(self.sequences)
		self.labels = np.concatenate(self.labels)

		self.handcrafted = torch.tensor(self.handcrafted.astype(np.float32))
		self.sequences = torch.tensor(self.sequences.astype(np.float32))
		self.labels = torch.tensor(self.labels.astype(np.float32))
		
		self.length = len(self.sequences)

	def __len__(self):
		return self.length

	def __getitem__(self, idx):
		return self.sequences[idx], self.handcrafted[idx], self.labels[idx]


class RandomlyExploredDataset(Dataset):
	def __init__(self, idx_size):
		np.random.seed(42)
		self.idx_size = idx_size
		self.__idx_cache = dict()
		self.__idx_valid = set()
		self.__idx_invalid = set()
	
	def __len__(self):
		return 1

	def __getitem__(self, idx):
		return 2

class SequencePairDataset(Dataset):
	def __init__(self, cameras_path = "raw_data/datasets/sequence/mta_ext_short/train", camera_ids = [0,1,2], cache_path = None, max_dt = (7 * 24)):
		super(RandomlyExploredDataset, self).__init__(3)
		self.dataset = SequenceDataset(cameras_path = cameras_path, camera_ids = camera_ids)
		self.max_dt = max_dt
		self.sub_length = self.dataset.length
		if type(cache_path) != type(None):
			self.__idx_cache = dict()
			self.__idx_mapped = set()
		self.length = self.dataset.length * 3

	def is_valid(self, main_idx, other_idx, positive = True):
		main_imgs, main_data, main_label = self.dataset[main_idx]
		other_imgs, other_data, other_label = self.dataset[other_idx]

		if (abs(main_data[0] - other_data[0]) <= self.max_dt):
			return True
		return False

	def __len__(self):
		return self.length

	def save(self):
		pass
	def load(self):
		pass


	def __getitem__(self, idx):
		if idx not in self.__idx_cache:
			while True:
				main_idx, other_idx = np.random.randint(0, self.sub_length, 2)
				if (main_idx, other_idx) in self.__idx_mapped: continue
				if self.is_valid(main_idx, other_idx):
					self.__idx_cache[idx] = (main_idx, other_idx)
					self.__idx_mapped.add((main_idx, other_idx))
					break
		else:
			main_idx, other_idx  = self.__idx_cache[idx] 
		main_imgs, main_data, main_label = self.dataset[main_idx]
		other_imgs, other_data, other_label = self.dataset[other_idx]

		return main_imgs, main_data, main_label, other_imgs, other_data, other_label

def triplet_index(n):
	pass

class SequenceTripletDataset(Dataset):
	def __init__(self, cameras_path = "raw_data/datasets/sequence/mta_ext_short/train", camera_ids = [0,1,2], cache_path = None, max_dt = (10 * 24), pos_neg_split = .5, per_anchor = 10):
		self.dataset = SequenceDataset(cameras_path = cameras_path, camera_ids = camera_ids)
		self.max_dt = max_dt
		self.per_anchor = per_anchor

		self.pos_neg_split = pos_neg_split
		self.sub_length = self.dataset.length
		if type(cache_path) != type(None):
			self.__idx_cache = dict()
			self.__idx_mapped = set()
			self.__idx_valid = set()
			self.__idx_invalid = set()
			pass
		else:
			self.__idx_cache = dict()
			self.__idx_mapped = set()
			self.__idx_valid = set()
			self.__idx_invalid = set()
		self.length = 0
		self.build_idx()
		
	def build_idx(self):
		self.length = 0
		print("Building Triplet Index...")
		for anchor_idx in range(self.sub_length):
			got = 0
			tried = 0
			while got <= self.per_anchor:
				if tried >= 10000:
					break
				negative_idx, positive_idx = np.random.randint(0,self.sub_length,2)
				if self.is_valid(anchor_idx, positive_idx, True) and self.is_valid(anchor_idx, negative_idx, False):
					self.__idx_cache[self.length] = (negative_idx, anchor_idx, positive_idx)
					self.__idx_mapped.add((negative_idx, anchor_idx, positive_idx))
					
					self.length += 1
					got += 1
				else:
					tried += 1 
					self.__idx_invalid.add((negative_idx, anchor_idx, positive_idx))
		
		print(f"Done {self.length}")
			
	def save(self):
		pass

	def load(self):
		pass

	def is_valid(self, anchor_idx, other_idx, positive = True):
		anchor_imgs, anchor_data, anchor_label = self.dataset[anchor_idx]
		other_imgs, other_data, other_label = self.dataset[other_idx]

		if (abs(anchor_data[0] - other_data[0]) <= self.max_dt) \
		 and ((anchor_label == other_label) == positive):
			return True
		return False

	def __len__(self):
		return self.length

	def __getitem__(self, idx):
		negative_idx, anchor_idx, positive_idx = self.__idx_cache[idx] 
		return self.dataset[negative_idx][:-1], self.dataset[anchor_idx][:-1], self.dataset[positive_idx][:-1]
		
if __name__ == "__main__":
	a = SequenceDataset("raw_data/datasets/sequence/grandma_me/test",[0,1])
	print(f"Num Sequences: {a.length}\nImg Shape: {a[0][0].shape} \nInfo Shape: {a[0][1].shape}")
	b = SequenceTripletDataset("raw_data/datasets/sequence/grandma_me/test",[0,1], max_dt = (24 * 30))

	print(f"Num Sequences: {b.length}\nImg Shape: {b[0]}")
	