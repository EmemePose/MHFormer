## 
```
git clone --recursive https://github.com/EmemePose/pose3D.git
```

## requirements
```
pip install -r requirements.txt
cd TransPose/lib
make
```

```

## MHFormer checkpoint

```
cd MHFormer
mkdir -p checkpoint/pretrained
```
Download [pretrained model (model_8_1166.pth)](https://drive.google.com/file/d/199FalebIXUOWgkS_1m4BhHlLiNqYFyAQ/view?usp=sharing) into `MHFormer/checkpoint/pretrained`

## Demo
#### video --> joints_2d + joints_3d
The 3D joints output will be stored in `./output/joints3d`


<!-- 
#### joints_2d + joints_3d --> 8bit
The 8 bit mp4 output will be stored in `./output/bit_output`

```
VIDEO_PATH='samples/sample_video.mp4'
OUTPUT_PATH='output'
python demo.py --video VIDEO_PATH
python bit8.py --video VIDEO_PATH
``` 
-->



