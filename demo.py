from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path as osp
import sys
import os
import time
import torch
import torchvision.transforms as transforms
import torchvision
import numpy as np 
import cv2
from time import time

from write_video import write_images_to_video

# from bvh_skeleton import openpose_skeleton, h36m_skeleton, cmu_skeleton, smartbody_skeleton

from MHFormer.common.skeleton import Skeleton

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device('cpu')

joints_left, joints_right = [4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]
kps_left, kps_right = [4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]

skeleton = Skeleton(parents=[-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15],
					joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
					joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31])

keypoints_metadata = {'keypoints_symmetry': (joints_left, joints_right), 'layout_name': 'Human3.6M', 'num_joints': 17}



def coco_h36m(keypoints):
	h36m_coco_order = [9, 11, 14, 12, 15, 13, 16, 4, 1, 5, 2, 6, 3]
	coco_order = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
	spple_keypoints = [10, 8, 0, 7]
	# keypoints: (T, N, 2) or (M, N, 2)

	temporal = keypoints.shape[0]
	keypoints_h36m = np.zeros_like(keypoints, dtype=np.float32)
	htps_keypoints = np.zeros((temporal, 4, 2), dtype=np.float32)

	# htps_keypoints: head, thorax, pelvis, spine
	htps_keypoints[:, 0, 0] = np.mean(keypoints[:, 1:5, 0], axis=1, dtype=np.float32)
	htps_keypoints[:, 0, 1] = np.sum(keypoints[:, 1:3, 1], axis=1, dtype=np.float32) - keypoints[:, 0, 1]
	htps_keypoints[:, 1, :] = np.mean(keypoints[:, 5:7, :], axis=1, dtype=np.float32)
	htps_keypoints[:, 1, :] += (keypoints[:, 0, :] - htps_keypoints[:, 1, :]) / 3

	htps_keypoints[:, 2, :] = np.mean(keypoints[:, 11:13, :], axis=1, dtype=np.float32)
	htps_keypoints[:, 3, :] = np.mean(keypoints[:, [5, 6, 11, 12], :], axis=1, dtype=np.float32)

	keypoints_h36m[:, spple_keypoints, :] = htps_keypoints
	keypoints_h36m[:, h36m_coco_order, :] = keypoints[:, coco_order, :]

	keypoints_h36m[:, 9, :] -= (keypoints_h36m[:, 9, :] - np.mean(keypoints[:, 5:7, :], axis=1, dtype=np.float32)) / 4
	keypoints_h36m[:, 7, 0] += 0.3*(keypoints_h36m[:, 7, 0] - np.mean(keypoints_h36m[:, [0, 8], 0], axis=1, dtype=np.float32))
	keypoints_h36m[:, 8, 1] -= (np.mean(keypoints[:, 1:3, 1], axis=1, dtype=np.float32) - keypoints[:, 0, 1])*2/3

	# half body: the joint of ankle and knee equal to hip
	# keypoints_h36m[:, [2, 3]] = keypoints_h36m[:, [1, 1]]
	# keypoints_h36m[:, [5, 6]] = keypoints_h36m[:, [4, 4]]
	keypoints_h36m[:, [joints_left, joints_right],:] = keypoints_h36m[:, [joints_right, joints_left],:]
	return keypoints_h36m


def parse_args():
	parser = argparse.ArgumentParser(description='Train keypoints network')
	parser.add_argument('--video', type=str)
	
	parser.add_argument('--ckpt_path', type=str, default='MHFormer/checkpoint/pretrained/model_8_1166.pth')

	parser.add_argument('--output', type=str, default='output/' )
	
	args = parser.parse_args()
	return args



if __name__ == '__main__':
	args = parse_args()
	output_dir = args.output


	kpts_2d_dir = osp.join(output_dir, 'kpts2d') # output for detected 2D keypoints

	lifting_models = ['MHFormer', 'GAST', 'ST']
	lifting_model = lifting_models[0]

	output_dir = osp.join(output_dir, lifting_model)

	# if lifting_model == 'MHFormer':
	# 	mhformer_pretrained_path = args.ckpt_path
	# 	mh_para = mhformer_pretrained_path.split('/')[-2]
	# 	output_dir = osp.join(output_dir, mh_para)


	
	joints_3d_dir = osp.join(output_dir, 'joints3d') # output for detected 3D keypoints
	visualize_dir = osp.join(output_dir, 'visualize') 
	bvh_dir = osp.join(output_dir, 'bvh')  # output for bvh files
	os.makedirs(kpts_2d_dir, exist_ok=True)
	os.makedirs(joints_3d_dir, exist_ok=True)
	os.makedirs(visualize_dir, exist_ok=True)
	os.makedirs(bvh_dir, exist_ok=True)
	
	cap = cv2.VideoCapture(args.video)
	if cap.isOpened(): 
		# get vcap property 
		image_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
		image_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float `height`
		fps = cap.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
		frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		duration = frame_count/fps
		print('the video duration is ', str(duration))
		start_time = time()
		cap.release()
	else:
		print('no video file exist')
		exit(0)

	start_time = time()
	from estimate_2d import estimate_2d
	# estimate 2d kpts in COCO format ( the estimator is TransPose )
	joints_2d, images_to_save = estimate_2d(args.video, device=device) # .squeeze(1)

	# write_images_to_video(images_to_save, osp.join(visualize_dir, osp.basename(args.video).split('.')[0] + '_2d.mp4') )
	
	joints_2d = joints_2d.squeeze(1)
	## transform to Human3.6M format
	joints_2d = coco_h36m(joints_2d)
	np.save(  osp.join(kpts_2d_dir, osp.basename(args.video).split('.')[0] + '.npy') , joints_2d)
	print('2d keypoints estimation takes: ', time() - start_time)
	####=======================================================================================================
	joints_2d = np.load( osp.join(kpts_2d_dir, osp.basename(args.video).split('.')[0] + '.npy') )
	
	if lifting_model == 'MHFormer':
		from MHFormer.demo.my_demo import pose_lift 
		joints_3d = pose_lift(joints_2d, args.video, mhformer_pretrained_path, frames=27, device=device)
		np.save(  osp.join(joints_3d_dir, osp.basename(args.video).split('.')[0] + '.npy') , joints_3d)
	####=======================================================================================================
	



