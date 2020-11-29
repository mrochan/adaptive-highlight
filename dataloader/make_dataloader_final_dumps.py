"""
1. Use this loader when BOTH user "train video" and "processed history" are stored in .csv and .json files respectively
and train/val/text vid info available in a dictionary (../data/).
"""

import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import array
import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F
import json

def read_json(file_path):
    """ Parsing a .json file
    input:
    ------
        file_path: file_path of .json file as str
    
    returns:
    --------
        data: contains the data as dict
    """
    with open(file_path,"r") as f:
        data = json.loads(f.read())

    return data

class Highlight_Dataset(Dataset):
	""" Dataset class
	input:
	------
			data_path: path to directory that contains users video, V
			is_history: flag to indicate whether or not to read user's highlight history information from different videos (G), G != V
			hist_gt_path: dir path where user history.json and ground-truth for train video (example folder layout=> ground_truth_final/[train or test]/user_id/train/gt/[user_vid ID]/combined_strided_train.csv; ground_truth_final/[train or test]/user_id/history/history.json)
			mode: 'train' or 'test'
			l2_norm: flag (True/False) l2-normalize stride features of user video and each stride feat of user hist_vids
	returns:
	--------
			vid_feat_tensor: Tensor of shape (nFrames X 8192)
			gt_strided_binary: Binary indicator tensor of shape ([nFrames])
			user_history_collection (if is_history = True): a list of size N with each element is a tensor of shape [8192];
			user_path: user training video as a string (e.g., ('e.g. path/to/phd-gif/ground_truth_final/train/6324',))
			gt_nframes: a tensor ([num_frames]) with original number of frames in the user train video (required in evaluation)
	"""
	def __init__(self, data_path, is_history=False, hist_gt_path=None, l2_norm=False, mode = None):
		self.data_path = data_path
		self.is_history = is_history
		self.users_hist_gt_path = hist_gt_path
		self.l2_norm = l2_norm
		self.mode = mode

		# load all the users from the data path (not sorted currently as users are numbered
		self.users_vid_hist_dict = read_json(self.data_path)
		self.users = list(self.users_vid_hist_dict.keys())

	def __len__(self):
		return len(self.users)

	def __getitem__(self, index):
		# fetch the user_id
		user_id = self.users[index]

		# path to user train video feature
		if self.mode == 'train':
			user_train_vid_feat_path = os.path.join('/path/to/datasets/highlight/phd-gif/usr_vid_features_csv', self.mode)

		if self.mode == 'test':
			user_train_vid_feat_path = os.path.join('/path/to/datasets/highlight/phd-gif/usr_vid_features_csv', self.mode)

		# 1.1 read the training video feature and norm
		vid_feat_tensor = torch.from_numpy(np.loadtxt(user_train_vid_feat_path + '/'+ str(user_id)+'.csv', delimiter=',')).float()

		# 1.2 normalize the video features
		if self.l2_norm:
			vid_feat_tensor = F.normalize(vid_feat_tensor, p=2, dim=1)
		else:# no l2-normalization of features
			pass

		# 1.3 read the training video ground-truth
		user_path = os.path.join(self.users_hist_gt_path, self.mode, user_id)
		user_vid_gt_path = os.path.join(user_path, 'train/gt')
		user_gt_list = [os.path.join(user_vid_gt_path, gt) for gt in os.listdir(user_vid_gt_path)]

		# looping through user's ground-truth folder
		for ugt in user_gt_list:
			gt_strided_file_path = os.path.join(ugt, 'combined_strided_train.csv')
			# read gt strided file path
			gt_strided_data = pd.read_csv(gt_strided_file_path, delimiter=',') # csv with fields (user_id, youtube_id, stride, #frames, fps, gt_value (0 or 1))
			gt_strided_binary = torch.from_numpy(gt_strided_data['gt_value'].values)
			gt_nframes = torch.from_numpy(gt_strided_data['num_frames'].values)

		assert vid_feat_tensor.shape[0] == gt_strided_binary.shape[0], "Video and GT shape does not match...."

		if self.is_history:

			# user history collection, contains all the history tensors per each video
			user_history_collection = []

			# creating user history path (is_last=False)
			user_history_path = os.path.join(user_path, 'history', 'history.json')

			# read the user history.json
			hist_json = read_json(user_history_path)

			if len(hist_json.keys()) > 0:
				# process each history video for the user
				for hist_vid in hist_json.keys():
					hist_vid_data = hist_json.get(hist_vid, [])
					hist_vid_feats_list = hist_vid_data['stride_feat_list']
					if self.l2_norm:
						hist_vid_tensor_list = [F.normalize(torch.from_numpy(np.asarray(feat)), p=2, dim=1) for feat in hist_vid_feats_list]
					else:
						hist_vid_tensor_list = [torch.from_numpy(np.asarray(feat)) for feat in hist_vid_feats_list]
					if len(hist_vid_tensor_list)>0:
						hist_vid_feat_tensor = torch.cat(hist_vid_tensor_list, dim=0)
						# perform averaging over features of that history video
						hist_vid_feat_tensor = torch.mean(hist_vid_feat_tensor, dim=0)
						# adding averaged features of a highlight video to the history video collection
						user_history_collection.append(hist_vid_feat_tensor)
					else:
						print('data_loader: hist_vid_feat_tensor is empty for user:{}'.format(user_path))
			else:
				pass

			# history data
			return vid_feat_tensor, gt_strided_binary, user_history_collection, user_path, gt_nframes[0]
		else:
			# no history data
			return vid_feat_tensor, gt_strided_binary, user_path, gt_nframes[0]


def get_loader(config):
	"""
	input:
	------
			config: object of Config() representing train/val config or test config
	"""
	if config.train_data_path is not None and config.val_data_path is not None and config.test_data_path is not None:			
		train_dataset = Highlight_Dataset(config.train_data_path, is_history=config.is_history, 
						hist_gt_path=config.history_gt_dir, l2_norm=config.is_l2_normalize, mode='train')
		val_dataset = Highlight_Dataset(config.val_data_path, is_history=config.is_history, 
						hist_gt_path=config.history_gt_dir, l2_norm=config.is_l2_normalize, mode='train')
		test_dataset = Highlight_Dataset(config.test_data_path, is_history=config.is_history, 
						hist_gt_path=config.history_gt_dir, l2_norm=config.is_l2_normalize, mode='test')
		
		train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
		val_loader = DataLoader(val_dataset, batch_size=1, num_workers=0)
		test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0)

		return train_loader, val_loader, test_loader
	else:
		raise ValueError("data_path_train or val or test is None -- check dataloader!")

