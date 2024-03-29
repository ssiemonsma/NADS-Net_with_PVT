[param]

# CPU mode or GPU mode
use_gpu = 1

# GPU device number (doesn't matter for CPU mode)
GPUdeviceNumber = 0

# Select model (default: 1)
modelID = 1

# Look in matlab counterpart for explanation
octave = 3
starting_range = 0.8
ending_range = 2
scale_search = 0.5, 1, 1.5, 2
thre1 = 0.1
thre2 = 0.05
thre3 = 0.5
min_num = 4
mid_num = 10
crop_ratio = 2.5
bbox_ratio = 0.25

[models]

[[1]]
caffemodel = './model/_trained_COCO/pose_iter_440000.caffemodel'
deployFile = './model/_trained_COCO/pose_deploy.prototxt'
description = 'COCO Pose56 Two-level Linevec'
boxsize = 384
padValue = 128
np = 12
stride = 32
part_str = [nose, neck, Rsho, Relb, Rwri, Lsho, Lelb, Lwri, Rhip, Rkne, Rank, Lhip, Lkne, Lank, Leye, Reye, Lear, Rear, pt19]
