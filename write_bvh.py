def write_standard_bvh(outbvhfilepath,prediction3dpoint):
	'''
	:param outbvhfilepath: 输出bvh动作文件路径
	:param prediction3dpoint: 预测的三维关节点
	:return:
	'''
	def rotation_matrix(x, y, z):
		'''
		x, y, z: roll, pitch, yaw, (radians)
		'''
		Rx = np.array([[1,0,0],
					[0, np.cos(x), -np.sin(x)],
					[0, np.sin(x), np.cos(x)]])

		Ry = np.array([[ np.cos(y), 0, np.sin(y)],
					[ 0,         1,         0],
					[-np.sin(y), 0, np.cos(y)]])

		Rz = np.array([[np.cos(z), -np.sin(z), 0],
					[np.sin(z),  np.cos(z), 0],
					[0,0,1]])
		return Rz@Ry@Rx
	# prediction3dpoint_ = prediction3dpoint.copy()
	for frame in prediction3dpoint:
		
		
		for point3d in frame:
			# point3d[0] *= 100
			# point3d[1] *= 100
			# point3d[2] *= 100

			# change Y and Z 
			X = point3d[0]
			Y = point3d[1]
			Z = point3d[2]
			point3d[0] = -X
			point3d[1] = Z
			point3d[2] = Y
			
			
			# point3d = point3d.reshape(1,3)
			# point3d = point3d @ R.T
		
	# dir_name = os.path.dirname(outbvhfilepath)
	# basename = os.path.basename(outbvhfilepath)
	# video_name = basename[:basename.rfind('.')]
	# human36m_skeleton = h36m_skeleton.H36mSkeleton()
	# human36m_skeleton.poses2bvh(prediction3dpoint,output_file=outbvhfilepath)

	R = rotation_matrix(0, np.pi, 0)

	# prediction3dpoint *= np.array([-1,-1,-1])

	SmartBody_skeleton = smartbody_skeleton.SmartBodySkeleton()
	SmartBody_skeleton.poses2bvh(prediction3dpoint @ R.T ,output_file=outbvhfilepath)

	# CMU_skeleton = cmu_skeleton.CMUSkeleton()
	# # CMU_skeleton.poses2bvh(prediction3dpoint @ R.T ,output_file=outbvhfilepath)
	# CMU_skeleton.poses2bvh(prediction3dpoint,output_file=outbvhfilepath)

	# H36M_skeleton = h36m_skeleton.H36mSkeleton()
	# H36M_skeleton.poses2bvh(prediction3dpoint @ R.T ,output_file=outbvhfilepath)