import os
import random
import skimage
import numpy as np
from glob import glob
import cv2
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from skimage import transform as tf


def load_dataset(cfg, shuffle=False, num_workers=0):
	print("[Data] Load datasets: {}".format(cfg['train_dataset']['name']))
	dataset = PRNet_dataset(cfg['train_dataset']['path'], is_aug=True)
	data_loader = DataLoader(dataset, 
		batch_size=cfg['batch_size'], drop_last=True, shuffle=shuffle, 
		num_workers=num_workers)
	print("[*] Load Success, dataset size: %d" %len(dataset))
	return data_loader
# return len(data_loader.dataset), loader(num_epoch, data_loader)


def take(num_step, data_loader):
	num_epoch = num_step * data_loader.batch_size // len(data_loader.dataset)
	if num_epoch >= 1:
		for _ in range(num_epoch):
			for data in data_loader:
				data_dict = {'Image': data['Image'].numpy(),
							'Posmap': data['Posmap'].numpy()}
				yield data_dict
	else:
		step = 0
		for data in data_loader:
			if step < num_step:
				data_dict = {'Image': data['Image'].numpy(),
							'Posmap': data['Posmap'].numpy()}
				yield data_dict
				step += 1
			else:
				break


class PRNet_dataset(Dataset):
	def __init__(self, data_root, is_aug=False, img_size=256, 
				 posmap_size=256, min_blur_resize=75, max_noise_var=0.01, 
				 max_rot=45, min_scale=0.95, max_scale=1.05, max_shift=0.):
		self.data_root = data_root
		self.datas_list = glob(os.path.join(self.data_root, '*/*.npy'))
		self.img_size = img_size
		self.posmap_size = posmap_size

		self.is_aug = is_aug
		self.min_blur_resize = min_blur_resize
		self.max_noise_var = max_noise_var
		self.max_rot = max_rot
		self.min_scale = min_scale
		self.max_scale = max_scale
		self.max_shift = max_shift

	def data_aug(self, data_dict):
		### image data augmentation ###
		new_img = data_dict['Image']

		angle_aug = random.random() * self.max_rot * 2 - self.max_rot
		scale_aug = random.random() * (self.max_scale - self.min_scale) + \
								self.min_scale

		shift_aug_x = random.random() * \
				(self.max_shift * self.posmap_size) * 2 \
				- (self.max_shift * self.posmap_size)
		shift_aug_y = random.random() * \
				(self.max_shift * self.posmap_size) * 2 \
				- (self.max_shift * self.posmap_size)

		tform = tf.SimilarityTransform(
			scale=scale_aug, 
			rotation=np.deg2rad(angle_aug),
	     	translation=(shift_aug_x, shift_aug_y))

		shift_y, shift_x = np.array(new_img.shape[:2]) / 2.
		tf_shift = tf.SimilarityTransform(translation=[-shift_x, -shift_y])
		tf_shift_inv = tf.SimilarityTransform(translation=[shift_x, shift_y])

		new_img = tf.warp(new_img, (tf_shift + (tform + tf_shift_inv)).inverse) 

		# fill blank
		border_value = np.mean(new_img[ :3, :], axis=(0, 1)) * 0.25 + \
					   np.mean(new_img[-3:, :], axis=(0, 1)) * 0.25 + \
					   np.mean(new_img[:, -3:], axis=(0, 1)) * 0.25 + \
					   np.mean(new_img[:,  :3], axis=(0, 1)) * 0.25

		mask = np.sum(new_img.reshape(-1, 3), axis=1) == 0
		mask = mask[:, np.newaxis]
		mask = np.concatenate((mask, mask, mask), axis=1)
		border_value = np.repeat(border_value[np.newaxis, :], mask.shape[0], axis=0)
		border_value *= mask
		new_img = (new_img.reshape(-1, 3) + border_value)
		new_img = new_img.reshape(self.img_size, self.img_size, 3)
		new_img = (new_img * 255.).astype('uint8')

		# gamma correlation
		gamma = random.random() * (1.8 - 1.0) + 1.0
		invGamma = 1.0 / gamma
		table = np.array([((i / 255.0) ** invGamma) * 255
											for i in np.arange(0, 256)]).astype(np.uint8)
		new_img = cv2.LUT(new_img, table)

		# noise
		noise_aug = random.random() * self.max_noise_var
		new_img = (skimage.util.random_noise(
				new_img, mode="gaussian", var=noise_aug) * 255).astype(np.uint8)

		# blur
		blur_aug = random.randint(self.min_blur_resize, self.img_size)
		new_img = cv2.resize(cv2.resize(new_img, (blur_aug, blur_aug)),
				(self.img_size, self.img_size))
		
		# gray
		if random.random() < 0.2:
			new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
			new_img = np.stack((new_img,) * 3, axis=-1)

		data_dict['Image'] = new_img

		# aug posmap
		posmap = data_dict['Posmap']
		vertices = np.reshape(posmap, [-1, 3]).T
		z = vertices[2,:].copy() / tform.params[0, 0]
		vertices[2,:] = 1
		vertices = np.dot((tf_shift + (tform + tf_shift_inv)).params, vertices)
		vertices = np.vstack((vertices[:2,:], z))
		posmap = np.reshape(vertices.T, [self.posmap_size, self.posmap_size, 3])
		data_dict['Posmap'] = posmap

		return data_dict

	def __len__(self):
		return len(self.datas_list)

	def __getitem__(self, index):
		# data = np.load(self.datas_list[index], allow_pickle=True).item()
		data = {'Image': cv2.imread(self.datas_list[index].replace('.npy', '.jpg')),
				'Posmap': np.load(self.datas_list[index])}
		if self.is_aug:
			data = self.data_aug(data)

		data['Image'] = (data['Image'] / 255.).astype(np.float32) 
		data['Posmap'] = (data['Posmap'] / 255.).astype(np.float32) 
		
		return data