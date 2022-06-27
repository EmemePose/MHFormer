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

# sys.path.insert(0, os.path.abspath('TransPose'))

# from TransPose.demo import estimate_2d
from estimate_2d import estimate_2d
# from bvh_skeleton import openpose_skeleton, h36m_skeleton, cmu_skeleton, smartbody_skeleton
from MHFormer.demo.my_demo import pose_lift as mhformer_estimate_3d
from MHFormer.common.skeleton import Skeleton

# from GAST.my_3d import pose_lift
# from GAST.tools.vis_h36m import render_animation

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


joints_left, joints_right = [4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]
kps_left, kps_right = [4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]

skeleton = Skeleton(parents=[-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15],
					joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
					joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31])

keypoints_metadata = {'keypoints_symmetry': (joints_left, joints_right), 'layout_name': 'Human3.6M', 'num_joints': 17}


keypoint2index = {
		'Hips': 0,
		'RightUpLeg': 1,
		'RightLeg': 2,
		'RightFoot': 3,
		'LeftUpLeg': 4,
		'LeftLeg': 5,
		'LeftFoot': 6,
		'Spine': 7,
		'Spine3': 8,
		'Neck': 9,
		'Head': 10,
		'LeftArm': 11,
		'LeftForeArm': 12,
		'LeftHand': 13,
		'RightArm': 14,
		'RightForeArm': 15,
		'RightHand': 16
	}

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
	keypoints_h36m[:, [joints_left, joints_right],:] = keypoints_h36m[:, [joints_right, joints_left],:] #  to comment out??
	return keypoints_h36m

def visualize( keypoints, prediction, visualize_dir, video, output_animation=True):
		'''
		prediction: (1, N, 17 3)
		'''
		cap = cv2.VideoCapture(video)
		width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
		height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

		print('width: ', width)
		print('height: ', height)

		keypoints = np.expand_dims(keypoints, 1)
		re_kpts = np.transpose(keypoints, [1,0,2,3])
		prediction = [prediction] #np.expand_dims(keypoints, 0)

		anim_output = {}
		for i, anim_prediction in enumerate(prediction):
			anim_output.update({'Reconstruction %d' % (i+1): anim_prediction})

		if output_animation:
			viz_output = visualize_dir + '/animation_' + video.split('/')[-1].split('.')[0] + '.mp4'
			print('Generating animation ...')
			# re_kpts: (M, T, N, 2) --> (T, M, N, 2)
			re_kpts = re_kpts.transpose(1, 0, 2, 3)
			render_animation(re_kpts, keypoints_metadata, anim_output, skeleton, 25, 30000, np.array(70., dtype=np.float32),
							viz_output, input_video_path=video, viewport=(width, height), com_reconstrcution=False)
		else:
			print('Saving 3D reconstruction...')
			output_npz = visualize_dir + '/' + video.split('/')[-1].split('.')[0] + '.npz'
			np.savez_compressed(output_npz, reconstruction=prediction)
			print('Completing saving...')




def parse_args():
	parser = argparse.ArgumentParser(description='Train keypoints network')
	parser.add_argument('--video', type=str)
	
	parser.add_argument('--ckpt_path', type=str, default='checkpoint/pretrained/model_8_1166.pth')

	parser.add_argument('--output', type=str, default='output/' )
	
	args = parser.parse_args()
	return args
if __name__ == '__main__':
	args = parse_args()
	# output_dir = '/home/jqin/wk/pose/new_pipeline/outputs_0613'
	output_dir = args.output
	kpts_2d_dir = osp.join(output_dir, 'kpts2d') #'/home/jqin/wk/pose/pipeline/outputs/kpts2d'


	lifting_models = ['MHFormer', 'GAST', 'ST']
	lifting_model = lifting_models[0]


	output_dir = osp.join(output_dir, lifting_model)

	# if lifting_model == 'MHFormer':
	# 	mhformer_pretrained_path = '/home/jqin/wk/pose/new_pipeline/MHFormer/checkpoint/pretrained/model_4294.pth'
	# 	mhformer_pretrained_path = '/home/jqin/wk/pose/new_pipeline/MHFormer/checkpoint/0418_0120_01_27/model_18_929.pth'
	# 	# mhformer_pretrained_path = '/home/jqin/wk/pose/new_pipeline/MHFormer/checkpoint/0424_0217_07_9/model_17_1097.pth'
	# 	mh_para = mhformer_pretrained_path.split('/')[-2]
	# 	output_dir = osp.join(output_dir, mh_para)
	if lifting_model == 'MHFormer':
		mhformer_pretrained_path = args.ckpt_path
		mh_para = mhformer_pretrained_path.split('/')[-2]
		output_dir = osp.join(output_dir, mh_para)

	joints_3d_dir = osp.join(output_dir, 'joints3d') #'/home/jqin/wk/pose/pipeline/outputs/joints3d'
	visualize_dir = osp.join(output_dir, 'visualize') #'/home/jqin/wk/pose/pipeline/outputs/visualize'
	bvh_dir = osp.join(output_dir, 'bvh')  #'/home/jqin/wk/pose/pipeline/outputs/bvh'
	os.makedirs(kpts_2d_dir, exist_ok=True)
	os.makedirs(joints_3d_dir, exist_ok=True)
	os.makedirs(visualize_dir, exist_ok=True)
	os.makedirs(bvh_dir, exist_ok=True)
	
	from time import time
	cap = cv2.VideoCapture(args.video)
	fps = cap.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
	frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	duration = frame_count/fps
	print('the video duration is ', str(duration))
	start_time = time()
	cap.release()


	# ###======================================================================================================
	# estimate 2d kpts in COCO format ( the estimator is TransPose )
	joints_2d, images_to_save = estimate_2d(args.video, device=device)
	print('joint 2d: ', joints_2d.shape)
	joints_2d = joints_2d.squeeze(1)
	# write_images_to_video(images_to_save, osp.join(visualize_dir, osp.basename(args.video).split('.')[0] + '_2d.mp4') )
	

	print('joints_2d shape:', joints_2d.shape)
	## transform to Human3.6M format
	joints_2d = coco_h36m(joints_2d)
	np.save(  osp.join(kpts_2d_dir, osp.basename(args.video).split('.')[0] + '.npy') , joints_2d)
	print('2d keypoints estimation takes: ', time() - start_time)
	# ###=======================================================================================================

	joints_2d = np.load( osp.join(kpts_2d_dir, osp.basename(args.video).split('.')[0] + '.npy') )
	print('joints_2d shape:', joints_2d.shape)

	# ###=======================================================================================================
	
	if lifting_model == 'MHFormer':
		from MHFormer.demo.my_demo import pose_lift 
		joints_3d = pose_lift(joints_2d, args.video, mhformer_pretrained_path, frames=27, device=device)
	


	np.save(  osp.join(joints_3d_dir, osp.basename(args.video).split('.')[0] + '.npy') , joints_3d)


	total_time_taken = time() - start_time
	print('total takes: ', total_time_taken)


	print('the video duration is ', str(duration))
	print('the time/duration = ', total_time_taken/duration)
	# ###=======================================================================================================



	# joints_3d = np.load(osp.join(joints_3d_dir, osp.basename(args.video).split('.')[0] + '.npy'), allow_pickle=True)



	# def joints_process(joints_3d):
	# 	from scipy.spatial.transform import Rotation
	# 	from MHFormer.common.camera import camera_to_world
	# 	joints_left =  [4, 5, 6, 11, 12, 13]
	# 	joints_right = [1, 2, 3, 14, 15, 16]
	# 	joints_3d_trans = joints_3d.copy()
	# 	r_manual = Rotation.from_euler('zyx', [-90, 0, 0], degrees=True)
	# 	print('r rotation matrix: ', r_manual.as_matrix())
	# 	rot = r_manual.as_quat()
	# 	rot = np.array(rot, dtype='float32')
	# 	print('camera to world')
	# 	# print('rot: ', rot)

	# 	joints_3d_trans = camera_to_world(joints_3d_trans, R=rot, t=0)

	# 	for key, idx in keypoint2index.items():
	# 		print(key, idx, ' : ', joints_3d_trans[0][idx])
		
		
	# 	## to the ground
	# 	z_leftFoot_frame0 = joints_3d_trans[0, 6, 2]
	# 	z_rightFoot_frame0 = joints_3d_trans[0, 3, 2]
	# 	z_Foot_frame0 = np.minimum(z_leftFoot_frame0, z_rightFoot_frame0)
	# 	joints_3d_trans -= np.array([0,0, z_Foot_frame0])

	# 	# ## flip and change the right and left arm/leg
	# 	joints_3d_trans[:, : , 0] *= -1
	# 	joints_3d_trans[:, joints_left + joints_right] = joints_3d_trans[:, joints_right + joints_left]
		
	# 	return joints_3d_trans

	# joints_3d = joints_process(joints_3d)

	# visualize(joints_2d, joints_3d, visualize_dir, args.video)
	
