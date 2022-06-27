import torch
import time
import cv2

import argparse
import os.path as osp
import sys
import os
import time


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('device: ', device)
# Model



video_path = '/home/jqin/wk/pose/pipeline/inputs/test0619_2.mov'

v5 = True
if v5:
    ################################################################################
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.to(device)
    vidcap = cv2.VideoCapture(video_path)
    while True:
        ret, image_bgr = vidcap.read()
        if ret:
            ## human detection
            last_time = time.time()
            
            results = model(image_bgr)

            print('results: ', results)

            fps = 1/(time.time()-last_time)
            print('FPS: ', fps)
        else:
            print('cannot load the video.')
            break
    vidcap.release()
    ################################################################################
else:
    from visualize import update_config, add_path
    sys.path.insert(0, os.path.dirname(__file__))
    from visualize import update_config, add_path
    lib_path = osp.join(os.path.dirname(__file__), 'lib')
    add_path(lib_path)
    from lib.core.inference import get_final_preds
    from lib.yolo.human_detector import load_model as yolo_model
    from lib.yolo.human_detector import main as yolo_det


    def set_yolo_model():
        ### detection model
        box_model = yolo_model()
        return box_model

    box_model = set_yolo_model()
    box_model.to(device)
    box_model.eval()

    vidcap = cv2.VideoCapture(video_path)
    while True:
        ret, image_bgr = vidcap.read()
        if ret:
            ## human detection
            last_time = time.time()
            
            pred_boxes, scores = yolo_det(image_bgr, box_model )
            
            pred_boxes = [  [(box[0], box[1]), (box[2], box[3])]   for box in pred_boxes]
            image = image_bgr[:, :, [2, 1, 0]]


            fps = 1/(time.time()-last_time)
            print('FPS: ', fps)
        else:
            print('cannot load the video.')
            break
    vidcap.release()


# Images
# imgs = ['https://ultralytics.com/images/zidane.jpg']  # batch of images

# # Inference
# results = model(imgs)

# # Results
# results.print()
# results.save()  # or .show()

# results.xyxy[0]  # img1 predictions (tensor)

# print('results: ', results)