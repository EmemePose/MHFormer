import numpy as np
import cv2
import os
import os.path as osp


metadata = {'layout_name': 'coco', 'num_joints': 17, 'keypoints_symmetry': [[1, 3, 5, 7, 9, 11, 13, 15], [2, 4, 6, 8, 10, 12, 14, 16]]}
keypoints_symmetry = metadata['keypoints_symmetry']
kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
joints_left, joints_right = list([4, 5, 6, 11, 12, 13]), list([1, 2, 3, 14, 15, 16])

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


def to_cv_coord(joints_3d):

    joints_left =  [4, 5, 6, 11, 12, 13]
    joints_right = [1, 2, 3, 14, 15, 16]
    joints_3d_trans = joints_3d.copy()
    
#     joints_3d_trans[:,:, [1,2]] = joints_3d_trans[:,:, [2,1]]
    joints_3d_trans[:,:,2] *= -1
#     joints_3d_trans[:,:,1] *= -1
    joints_3d_trans[:, joints_left + joints_right] = joints_3d_trans[:, joints_right + joints_left]
    return joints_3d_trans
def estimatePose(points_2d, points_3d, camera, distortion, iterate=True):
    ret, rvec, tvec = cv2.solvePnP(points_3d, points_2d, camera, distortion, flags=cv2.SOLVEPNP_EPNP)
    ## further optimize
    if iterate:
        ret, rvec, tvec = cv2.solvePnP(points_3d, points_2d, camera, distortion, rvec, tvec, True)
    return rvec, tvec



if __name__ == '__main__':
    joints_2d[ :, joints_left + joints_right] = joints_2d[ :, joints_right + joints_left]
    fx,fy, cx,cy = 1000, 1000 , image_width/2.0, image_height/2.0
    camera_matrix = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]] )

    print('camera_matrix: ', camera_matrix)


    points_2d = joints_2d[frame].reshape(-1, 1, 2).astype(np.float32)
    points_3d = joints_3d_cv[frame].reshape(-1, 1, 3).astype(np.float32)

    hr, ht = estimatePose(points_2d, points_3d, camera_matrix, np.zeros(5,))

    hR = cv2.Rodrigues(hr)[0]
    print(hR)
    print(ht)
    Fc = joints_3d_cv[frame]  @ hR.T + ht.reshape(-1,3)

    plot_3d(Fc) 

